"""把数值焊接参数按需转换为运行时语言提示。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    ProcessorStepRegistry,
)
from lerobot.processor.converters import TransitionKey, batch_to_transition

from welding_path_vla.dataset.task_parameters import TASK_DIRECTION, TASK_PARAMETERS

WELDING_PROMPT_FIELDS = (
    "direction",
    "welding_speed",
    "work_angle",
    "travel_angle",
    "tool_roll",
)
DIRECTION_PROMPTS = {0: "forward", 1: "reverse"}


def numpy_value(value: Any) -> np.ndarray:
    """把 Tensor、数组或列表转换为 CPU NumPy 数组。"""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def batch_rows(value: Any, count: int, width: int | None = None) -> np.ndarray:
    """把单样本或 batch 数值统一整理成首维为 batch 的数组。"""
    array = numpy_value(value)
    if width is None:
        array = array.reshape(-1)
    elif array.ndim == 1:
        array = array[None, :]
    if len(array) == 1 and count > 1:
        array = np.repeat(array, count, axis=0)
    if len(array) != count or (width is not None and array.shape[-1] != width):
        raise ValueError(f"welding prompt batch shape mismatch: {array.shape}, batch={count}")
    return array


def format_number(value: float) -> str:
    """使用紧凑且稳定的十进制表示，避免无意义的尾随零。"""
    return f"{float(value):.6g}"


def angle_prompt(fields: tuple[str, ...], parameters: np.ndarray) -> str | None:
    """按选中的角度字段生成语法完整的单句或多行描述。"""
    descriptions = {
        "work_angle": f"a work angle of {format_number(parameters[1])} degrees",
        "travel_angle": f"a travel angle of {format_number(parameters[2])} degrees",
        "tool_roll": f"a tool roll of {format_number(parameters[3])} degrees",
    }
    selected = [descriptions[name] for name in descriptions if name in fields]
    if not selected:
        return None
    if len(selected) == 1:
        return f"Maintain {selected[0]}."
    if len(selected) == 2:
        return f"Maintain {selected[0]}\nand {selected[1]}."
    return f"Maintain {selected[0]},\n{selected[1]},\nand {selected[2]}."


@dataclass
@ProcessorStepRegistry.register(name="welding_prompt_builder")
class WeldingPromptBuilder(ComplementaryDataProcessorStep):
    """根据配置把结构化任务参数追加到原始任务文本。

    ``fields`` 为空时完全保留原任务文本；其余情况下可独立选择方向、速度、
    工作角、行走角和工具滚转角，便于使用同一数据集开展 prompt 消融实验。
    """

    fields: tuple[str, ...] = WELDING_PROMPT_FIELDS

    def __post_init__(self) -> None:
        """规范化从 checkpoint JSON 恢复的列表并验证字段名。"""
        self.fields = tuple(self.fields)
        unknown = set(self.fields).difference(WELDING_PROMPT_FIELDS)
        if unknown:
            raise ValueError(f"unknown welding prompt fields: {sorted(unknown)}")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("welding prompt fields must be unique")

    def build(self, task: str, direction: int | None, parameters: np.ndarray | None) -> str:
        """为一个样本构造未包装 chat template 的焊接提示。"""
        lines = [task.strip()]
        if "direction" in self.fields:
            if direction not in DIRECTION_PROMPTS:
                raise ValueError(f"unknown welding direction code: {direction}")
            lines.append(f"Move in the {DIRECTION_PROMPTS[direction]} direction.")
        if "welding_speed" in self.fields:
            if parameters is None:
                raise ValueError(f"{TASK_PARAMETERS} is required by welding prompt fields")
            lines.append(f"Use a welding speed of {format_number(parameters[0])} m/s.")
        angles = angle_prompt(self.fields, parameters) if parameters is not None else None
        if any(name in self.fields for name in ("work_angle", "travel_angle", "tool_roll")):
            if angles is None:
                raise ValueError(f"{TASK_PARAMETERS} is required by welding prompt fields")
            lines.append(angles)
        return "\n".join(lines)

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        """处理单样本或 batch task，并在使用后移除结构化临时字段。"""
        result = dict(complementary_data)
        task = result.get("task")
        result.pop(TASK_DIRECTION, None)
        result.pop(TASK_PARAMETERS, None)
        if task is None or not self.fields:
            return result
        tasks = [task] if isinstance(task, str) else list(task)
        needs_direction = "direction" in self.fields
        needs_parameters = any(name != "direction" for name in self.fields)
        if needs_direction and TASK_DIRECTION not in complementary_data:
            raise ValueError(f"{TASK_DIRECTION} is required by welding prompt fields")
        if needs_parameters and TASK_PARAMETERS not in complementary_data:
            raise ValueError(f"{TASK_PARAMETERS} is required by welding prompt fields")
        directions = (
            batch_rows(complementary_data[TASK_DIRECTION], len(tasks))
            if needs_direction
            else np.zeros(len(tasks), dtype=np.int64)
        )
        parameters = (
            batch_rows(complementary_data[TASK_PARAMETERS], len(tasks), 4)
            if needs_parameters
            else None
        )
        prompts = [
            self.build(
                text,
                int(directions[index]) if needs_direction else None,
                parameters[index] if parameters is not None else None,
            )
            for index, text in enumerate(tasks)
        ]
        result["task"] = prompts[0] if isinstance(task, str) else prompts
        return result

    def get_config(self) -> dict[str, Any]:
        """把消融字段写入 checkpoint processor 配置。"""
        return {"fields": list(self.fields)}

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """仅改变语言内容，不改变模型 feature 形状。"""
        return features


def welding_batch_to_transition(batch: dict[str, Any]) -> Any:
    """使用官方转换器，并额外保留 prompt builder 需要的两个字段。"""
    transition = batch_to_transition(batch)
    complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
    for key in (TASK_DIRECTION, TASK_PARAMETERS):
        if key in batch:
            complementary[key] = batch[key]
    cast(Any, transition)[TransitionKey.COMPLEMENTARY_DATA] = complementary
    return transition


def configure_welding_prompt_builder(
    preprocessor: Any,
    fields: tuple[str, ...] = (),
) -> None:
    """在所有 VLM 共用的 batch 边界后插入或恢复 prompt builder。

    该位置早于 SmolVLM、Qwen 和 PaliGemma 各自的语言模板与 tokenizer，
    因而无需感知具体 VLM，同时也兼容从 checkpoint 恢复的 processor。
    """
    builder = next(
        (step for step in preprocessor.steps if isinstance(step, WeldingPromptBuilder)),
        None,
    )
    if builder is None and fields:
        index = next(
            index + 1
            for index, step in enumerate(preprocessor.steps)
            if isinstance(step, AddBatchDimensionProcessorStep)
        )
        builder = WeldingPromptBuilder(fields=fields)
        preprocessor.steps = [
            *preprocessor.steps[:index],
            builder,
            *preprocessor.steps[index:],
        ]
    if builder is not None:
        preprocessor.to_transition = welding_batch_to_transition


__all__ = [
    "WELDING_PROMPT_FIELDS",
    "WeldingPromptBuilder",
    "configure_welding_prompt_builder",
    "welding_batch_to_transition",
]
