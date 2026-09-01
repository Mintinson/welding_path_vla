"""焊接任务参数动态提示词的回归测试。"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from lerobot.processor import AddBatchDimensionProcessorStep

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.action_processors import (
    AbsoluteEEActionsProcessorStep,
    RelativeEEActionsProcessorStep,
    relative_processor_factory,
)
from welding_path_vla.policies.base import Observation
from welding_path_vla.policies.runtime import LeRobotRuntime
from welding_path_vla.policies.spec import PI0, PI05, SMOLVLA, TRAJ_VLA_QWEN, TRAJECTORY_VLA
from welding_path_vla.policies.welding_prompt import (
    WeldingPromptBuilder,
    configure_welding_prompt_builder,
    welding_batch_to_transition,
)


def test_welding_prompt_builder_formats_batch_parameters() -> None:
    """数值字段应在运行时转换为自然语言，并支持正反两个方向。"""
    task = "Weld along the periodic curved seam on the flat plate."
    result = WeldingPromptBuilder().complementary_data(
        {
            "task": [task, task],
            "task.direction": torch.tensor([0, 1]),
            "task.parameters": torch.tensor([[0.038, 46.0, 8.0, -23.0], [0.02, 45.0, 10.0, -20.0]]),
        }
    )

    assert result["task"][0] == (
        f"{task}\n"
        "Move in the forward direction.\n"
        "Use a welding speed of 0.038 m/s.\n"
        "Maintain a work angle of 46 degrees,\n"
        "a travel angle of 8 degrees,\n"
        "and a tool roll of -23 degrees."
    )
    assert "Move in the reverse direction." in result["task"][1]
    assert "task.direction" not in result
    assert "task.parameters" not in result


def test_welding_prompt_builder_fields_are_independently_selectable() -> None:
    """消融配置可独立启用方向和任意工艺参数。"""
    result = WeldingPromptBuilder(fields=("direction", "tool_roll")).complementary_data(
        {
            "task": "Weld the seam.",
            "task.direction": 1,
            "task.parameters": [0.038, 46.0, 8.0, -23.0],
        }
    )
    prompt = result["task"]
    assert "reverse direction" in prompt
    assert "tool roll of -23 degrees" in prompt
    assert "welding speed" not in prompt
    assert "work angle" not in prompt
    assert "travel angle" not in prompt
    assert WeldingPromptBuilder(fields=("direction",)).get_config() == {"fields": ["direction"]}
    with pytest.raises(ValueError, match="unknown welding prompt fields"):
        WeldingPromptBuilder(fields=("unknown",))


def test_welding_converter_preserves_parameter_fields_for_prompt_builder() -> None:
    """LeRobot batch 转换不得在 prompt builder 之前丢弃任务参数。"""
    transition = welding_batch_to_transition(
        {
            "task": ["Weld the seam."],
            "task.direction": torch.tensor([0]),
            "task.parameters": torch.tensor([[0.038, 46.0, 8.0, -23.0]]),
        }
    )
    complementary = transition["complementary_data"]
    assert set(complementary) == {"task", "task.direction", "task.parameters"}


@pytest.mark.parametrize("vlm", ["smolvlm", "qwen", "paligemma-pi0", "paligemma-pi05"])
def test_builder_uses_vlm_independent_batch_boundary(vlm: str) -> None:
    """所有 VLM 都应在各自语言模板之前通过同一个公共插入点。"""
    language_step = object()
    preprocessor = type(
        "Preprocessor",
        (),
        {
            "steps": [AddBatchDimensionProcessorStep(), language_step],
            "to_transition": None,
        },
    )()
    configure_welding_prompt_builder(preprocessor, ("direction",))

    assert preprocessor.steps[2] is language_step, vlm
    assert isinstance(preprocessor.steps[1], WeldingPromptBuilder)
    assert preprocessor.to_transition is welding_batch_to_transition


def test_shared_training_factory_injects_welding_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """所有 LeRobot policy 共用的训练工厂都应保存 builder 到 checkpoint。"""
    import lerobot.scripts.lerobot_train as train_module

    preprocessor = SimpleNamespace(
        steps=[AddBatchDimensionProcessorStep(), RelativeEEActionsProcessorStep()],
        to_transition=None,
    )
    postprocessor = SimpleNamespace(steps=[AbsoluteEEActionsProcessorStep()])
    monkeypatch.setattr(
        train_module,
        "make_pre_post_processors",
        lambda *args, **kwargs: (preprocessor, postprocessor),
    )

    with relative_processor_factory(("direction", "welding_speed")):
        result, _ = train_module.make_pre_post_processors(object())

    builder = next(step for step in result.steps if isinstance(step, WeldingPromptBuilder))
    assert builder.fields == ("direction", "welding_speed")
    assert result.to_transition is welding_batch_to_transition


@pytest.mark.parametrize("spec", [SMOLVLA, TRAJECTORY_VLA, TRAJ_VLA_QWEN, PI0, PI05])
def test_vlm_runtime_exposes_current_task_parameters(spec: Any) -> None:
    """所有 VLM 部署都应把当前任务参数以与训练数据相同的键送入 processor。"""
    runtime = LeRobotRuntime(None, None, None, "cpu", spec)
    observation = Observation(
        0.0,
        {"global": torch.zeros(8, 8, 3).numpy()},
        torch.zeros(13).numpy(),
        "Weld the seam.",
        "reverse",
        0.038,
        46.0,
        8.0,
        -23.0,
    )
    sample = runtime.observation_sample(observation)
    expected_task = "Weld the seam." if spec.processor_adds_batch else ["Weld the seam."]
    assert sample["task"] == expected_task
    assert sample["task.direction"].reshape(-1).tolist() == [1]
    torch.testing.assert_close(
        sample["task.parameters"].reshape(1, 4),
        torch.tensor([[0.038, 46.0, 8.0, -23.0]]),
    )


@pytest.mark.parametrize(
    ("path", "token_limit"),
    [
        ("configs/policies/smolvla.yaml", 160),
        ("configs/policies/trajectory_vla.yaml", 160),
        ("configs/policies/traj_vla_qwen.yaml", 160),
        ("configs/policies/pi0.yaml", 160),
        ("configs/policies/pi0_5.yaml", 200),
    ],
)
def test_all_vlm_configs_enable_welding_prompt_fields(path: str, token_limit: int) -> None:
    """SmolVLM、Qwen 与 PI 系列应共享项目级 prompt 消融配置。"""
    raw = yaml.safe_load(Path(path).read_text())
    assert raw["policy"]["welding_prompt_fields"] == [
        "direction",
        "welding_speed",
        "work_angle",
        "travel_angle",
        "tool_roll",
    ]
    assert raw["policy"]["parameters"]["tokenizer_max_length"] == token_limit
    assert AppConfig.load(path).policy.welding_prompt_fields == (
        "direction",
        "welding_speed",
        "work_angle",
        "travel_angle",
        "tool_roll",
    )
