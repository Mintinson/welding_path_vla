from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
import torch

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.base import Observation
from welding_path_vla.policies.factory import get_policy_pipeline
from welding_path_vla.policies.lerobot_training import make_policy_config
from welding_path_vla.policies.runtime import LeRobotRuntime
from welding_path_vla.policies.spec import PI0, PI05
from welding_path_vla.policies.training import TrainingRequest


@pytest.mark.parametrize(
    ("path", "family", "batch_size", "horizon"),
    [
        ("configs/pi0.yaml", "pi0", 1, 15),
        ("configs/pi0_5.yaml", "pi0_5", 1, 15),
        ("configs/pi0_a100.yaml", "pi0", 2, 30),
        ("configs/pi0_5_a100.yaml", "pi0_5", 2, 30),
    ],
)
def test_pi_training_configs_compose_hardware_profiles(
    path: str,
    family: str,
    batch_size: int,
    horizon: int,
) -> None:
    """本机和服务器入口应只覆盖硬件相关参数。"""
    config = AppConfig.load(path)
    assert config.policy.family == family
    assert config.training.batch_size == batch_size
    assert config.policy.action_horizon == horizon
    assert config.task.seam_id == "straight_fillet"


@pytest.mark.parametrize(
    ("path", "family", "seam"),
    [
        ("configs/deploy/pi0_l_joint.yaml", "pi0", "straight_fillet"),
        ("configs/deploy/pi0_pipe_bottom.yaml", "pi0", "pipe_bottom"),
        ("configs/deploy/pi0_pipe_top.yaml", "pi0", "pipe_top"),
        ("configs/deploy/pi0_5_l_joint.yaml", "pi0_5", "straight_fillet"),
        ("configs/deploy/pi0_5_pipe_bottom.yaml", "pi0_5", "pipe_bottom"),
        ("configs/deploy/pi0_5_pipe_top.yaml", "pi0_5", "pipe_top"),
    ],
)
def test_pi_deployment_configs_switch_complete_tasks(path: str, family: str, seam: str) -> None:
    """切换部署入口应同时切换工件、焊缝和策略。"""
    config = AppConfig.load(path)
    assert config.policy.family == family
    assert config.task.seam_id == seam
    assert config.policy.checkpoint is not None


def test_pi_pipelines_use_official_lerobot_names() -> None:
    """项目名 pi0_5 应正确映射到 LeRobot 的 pi05。"""
    pi0 = AppConfig.load("configs/pi0.yaml")
    pi05 = AppConfig.load("configs/pi0_5.yaml")
    pi0_command = TrainingRequest(pi0.policy, pi0.training).command()
    pi05_command = TrainingRequest(pi05.policy, pi05.training).command()
    assert "--policy.path=lerobot/pi0_base" in pi0_command
    assert "--policy.path=lerobot/pi05_base" in pi05_command
    assert get_policy_pipeline("pi0").spec.policy_type == "pi0"
    assert get_policy_pipeline("pi0_5").spec.policy_type == "pi05"


def test_pi_configs_reuse_official_pretrained_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """训练配置应清空原机器人 feature，并保留官方架构。"""
    from lerobot.configs import NormalizationMode, PreTrainedConfig
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    models = {
        "lerobot/pi0_base": PI0Config(),
        "lerobot/pi05_base": PI05Config(),
    }
    monkeypatch.setattr(
        PreTrainedConfig,
        "from_pretrained",
        lambda source: models[str(source)],
    )

    pi0 = make_policy_config(AppConfig.load("configs/pi0.yaml").policy, PI0)
    pi05 = make_policy_config(AppConfig.load("configs/pi0_5.yaml").policy, PI05)
    assert pi0.input_features == {}
    assert pi0.output_features == {}
    assert pi0.chunk_size == 15
    assert pi0.train_expert_only
    assert pi05.type == "pi05"
    assert pi05.normalization_mapping["ACTION"] == NormalizationMode.QUANTILES


class CaptureProcessor:
    """记录 π0 runtime 送入官方 processor 的未批处理观测。"""

    def __init__(self) -> None:
        self.sample: dict[str, object] = {}

    def __call__(self, sample: dict[str, object]) -> dict[str, object]:
        self.sample = sample
        converted = dict(sample)
        state = cast(torch.Tensor, sample["observation.state"])
        converted["observation.state"] = state.unsqueeze(0)
        for name in ("global", "wrist"):
            key = f"observation.images.{name}"
            converted[key] = cast(torch.Tensor, sample[key]).unsqueeze(0)
        converted["task"] = [sample["task"]]
        return converted


class StaticPIPolicy:
    """返回固定 9D 动作的最小 π0 替身。"""

    def reset(self) -> None:
        pass

    def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
        assert batch["observation.state"].shape == (1, 13)
        return torch.zeros((1, 9), dtype=torch.float32)


@pytest.mark.parametrize("family", [PI0, PI05])
def test_pi_runtime_keeps_language_and_dual_cameras(family: Any) -> None:
    """π0 与 π0.5 都必须保留语言、状态和双相机观测。"""
    processor = CaptureProcessor()
    runtime = LeRobotRuntime(
        cast(Any, StaticPIPolicy()),
        processor,
        lambda action: action,
        "cpu",
        family,
    )
    observation = Observation(
        0.0,
        {
            "global": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        },
        np.zeros(13, dtype=np.float32),
        "沿圆管上沿完成整圆焊接。",
    )
    action = runtime.select_action(observation)
    assert action.shape == (9,)
    assert processor.sample["task"] == observation.instruction
    assert processor.sample["observation.state"].shape == (13,)
    assert processor.sample["observation.images.global"].shape == (3, 8, 8)


def test_server_profile_preserves_command_line_overrides() -> None:
    """Draccus 覆盖仍应作用于组合后的服务器配置。"""
    config = AppConfig.load("configs/pi0_a100.yaml")
    overridden = replace(config.training, steps=10)
    command = TrainingRequest(config.policy, overridden).command()
    assert "--steps=10" in command
    assert "--policy.train_expert_only=false" in command
