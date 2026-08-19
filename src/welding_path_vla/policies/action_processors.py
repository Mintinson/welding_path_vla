"""焊接末端相对轨迹的 LeRobot processor。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from lerobot.processor import (
    NormalizerProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from welding_path_vla.core.geometry import (
    absolute_ee_actions_from_relative,
    relative_ee_actions_from_absolute,
)

RELATIVE_PROCESSOR = "welding_relative_ee_actions"
ABSOLUTE_PROCESSOR = "welding_absolute_ee_actions"


@ProcessorStepRegistry.register(RELATIVE_PROCESSOR)
@dataclass
class RelativeEEActionsProcessorStep(ProcessorStep):
    """训练前将绝对目标变为共享锚点 relative actions，并缓存当前 TCP。"""

    anchor_positions: torch.Tensor | None = field(default=None, init=False, repr=False)
    anchor_quaternions: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """缓存观测 TCP；存在动作时在归一化之前转换整段动作。"""
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is None:
            return transition
        anchor_positions = state[..., 6:9].detach().clone()
        anchor_quaternions = state[..., 9:13].detach().clone()
        self.anchor_positions = anchor_positions
        self.anchor_quaternions = anchor_quaternions
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        converted = transition.copy()
        cast(Any, converted)[TransitionKey.ACTION] = relative_ee_actions_from_absolute(
            action, anchor_positions, anchor_quaternions
        )
        return converted

    def reset(self) -> None:
        """清除上一轮推理使用的 TCP 锚点。"""
        self.anchor_positions = None
        self.anchor_quaternions = None

    def transform_features(self, features: dict[Any, Any]) -> dict[Any, Any]:
        """动作维数和类型不变。"""
        return features


@ProcessorStepRegistry.register(ABSOLUTE_PROCESSOR)
@dataclass
class AbsoluteEEActionsProcessorStep(ProcessorStep):
    """反归一化后将 relative action chunk 恢复为世界系绝对目标。"""

    relative_step: Any = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """使用同一次预测缓存的 TCP 解码完整动作块。"""
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        step = self.relative_step
        if step is None or step.anchor_positions is None or step.anchor_quaternions is None:
            raise RuntimeError("relative action postprocessor 缺少当前 TCP 锚点")
        anchor_positions = step.anchor_positions.to(action)
        anchor_quaternions = step.anchor_quaternions.to(action)
        converted = transition.copy()
        cast(Any, converted)[TransitionKey.ACTION] = absolute_ee_actions_from_relative(
            action, anchor_positions, anchor_quaternions
        )
        return converted

    def transform_features(self, features: dict[Any, Any]) -> dict[Any, Any]:
        """动作维数和类型不变。"""
        return features


def connect_relative_processors(preprocessor: Any, postprocessor: Any, require_saved: bool) -> None:
    """插入或连接焊接 relative action processor。"""
    relative = next(
        (step for step in preprocessor.steps if isinstance(step, RelativeEEActionsProcessorStep)),
        None,
    )
    absolute = next(
        (step for step in postprocessor.steps if isinstance(step, AbsoluteEEActionsProcessorStep)),
        None,
    )
    if require_saved and (relative is None or absolute is None):
        raise ValueError("checkpoint 不包含 relative_action processor, 请重新训练或转换 checkpoint")
    if relative is None:
        relative = RelativeEEActionsProcessorStep()
        index = next(
            (
                index
                for index, step in enumerate(preprocessor.steps)
                if isinstance(step, NormalizerProcessorStep)
            ),
            len(preprocessor.steps),
        )
        preprocessor.steps = [*preprocessor.steps[:index], relative, *preprocessor.steps[index:]]
    if absolute is None:
        absolute = AbsoluteEEActionsProcessorStep()
        index = next(
            (
                index + 1
                for index, step in enumerate(postprocessor.steps)
                if isinstance(step, UnnormalizerProcessorStep)
            ),
            0,
        )
        postprocessor.steps = [*postprocessor.steps[:index], absolute, *postprocessor.steps[index:]]
    absolute.relative_step = relative


def require_relative_checkpoint(path: str | Path) -> None:
    """拒绝继续训练动作语义不明确的旧项目 checkpoint。"""
    root = Path(path)
    config = root / "policy_preprocessor.json"
    if not config.exists():
        raise ValueError(f"checkpoint 缺少 relative_action processor: {root}")
    steps = json.loads(config.read_text(encoding="utf-8")).get("steps", [])
    names = {step.get("registry_name") for step in steps}
    if RELATIVE_PROCESSOR not in names:
        raise ValueError(f"checkpoint 使用旧 delta action 语义, 不能恢复训练: {root}")


def make_relative_pre_post_processors(
    policy_cfg: Any,
    pretrained_path: str | None = None,
    require_saved: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """调用 LeRobot 官方工厂，再接入项目的 SE(3) relative action 语义。"""
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg, pretrained_path=pretrained_path, **kwargs
    )
    connect_relative_processors(preprocessor, postprocessor, require_saved)
    return preprocessor, postprocessor


@contextmanager
def relative_processor_factory() -> Iterator[None]:
    """仅在官方训练调用期间替换其 processor 工厂入口。"""
    import lerobot.scripts.lerobot_train as train_module

    original = train_module.make_pre_post_processors

    def factory(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        preprocessor, postprocessor = original(*args, **kwargs)
        connect_relative_processors(preprocessor, postprocessor, require_saved=False)
        return preprocessor, postprocessor

    train_module.make_pre_post_processors = factory
    try:
        yield
    finally:
        train_module.make_pre_post_processors = original


__all__ = [
    "ABSOLUTE_PROCESSOR",
    "RELATIVE_PROCESSOR",
    "AbsoluteEEActionsProcessorStep",
    "RelativeEEActionsProcessorStep",
    "connect_relative_processors",
    "make_relative_pre_post_processors",
    "relative_processor_factory",
    "require_relative_checkpoint",
]
