from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
import torch

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.base import Observation
from welding_path_vla.policies.factory import get_policy_pipeline
from welding_path_vla.policies.runtime import LeRobotRuntime
from welding_path_vla.policies.simulation_rollout import deployment_output_dir
from welding_path_vla.policies.spec import TRAJECTORY_VLA
from welding_path_vla.policies.training import TrainingRequest
from welding_path_vla.policies.trajectory_vla.configuration_trajectory_vla import (
    TrajectoryVLAConfig,
)
from welding_path_vla.policies.trajectory_vla.flow_matching import (
    make_attention_masks,
    pad_sequence,
    pad_vector,
    resize_with_pad,
    sinusoidal_time_embedding,
)


@pytest.mark.parametrize(
    ("path", "seam"),
    [
        ("configs/deploy/trajectory_vla_l_joint.yaml", "straight_fillet"),
        ("configs/deploy/trajectory_vla_pipe_bottom.yaml", "pipe_bottom"),
        ("configs/deploy/trajectory_vla_pipe_top.yaml", "pipe_top"),
        ("configs/deploy/trajectory_vla_curve_plate.yaml", "curve_seam"),
    ],
)
def test_trajectory_vla_configs_switch_complete_tasks(path: str, seam: str) -> None:
    """部署配置应同时切换工件、焊缝并自动生成输出目录。"""
    config = AppConfig.load(path)
    assert config.policy.family == "trajectory_vla"
    assert config.task.seam_id == seam
    assert config.policy.checkpoint is not None
    task_name = {"straight_fillet": "l_joint", "curve_seam": "curve_plate"}.get(seam, seam)
    assert deployment_output_dir(config).name == f"trajectory_vla_{task_name}"


def test_trajectory_vla_uses_local_registered_policy() -> None:
    """训练命令和注册表必须指向本地 policy 类型。"""
    config = AppConfig.load("configs/trajectory_vla.yaml")
    command = TrainingRequest(config.policy, config.training).command()
    assert "--policy.type=trajectory_vla" in command
    assert "--policy.pretrained_path=lerobot/smolvla_base" in command
    assert get_policy_pipeline("trajectory_vla").spec is TRAJECTORY_VLA


def test_attention_mask_preserves_prefix_and_causal_blocks() -> None:
    """上下文不能读取动作，而动作可以读取上下文和过去动作。"""
    padding = torch.tensor([[True, True, True, True]])
    blocks = torch.tensor([[False, False, True, True]])
    mask = make_attention_masks(padding, blocks)
    expected = torch.tensor(
        [
            [
                [True, True, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        ]
    )
    assert torch.equal(mask, expected)


def test_public_tensor_helpers_keep_shapes_and_values() -> None:
    """公开的图像、向量和序列 helper 应保持数据含义。"""
    image = torch.ones(1, 3, 4, 8)
    resized = resize_with_pad(image, 8, 8)
    assert resized.shape == (1, 3, 8, 8)
    assert torch.all(resized[:, :, 4:] == 1)
    assert torch.equal(pad_vector(torch.ones(2, 3), 5)[:, :3], torch.ones(2, 3))
    assert pad_sequence(torch.ones(2, 3, 4), 5).shape == (2, 5, 4)
    embedding = sinusoidal_time_embedding(torch.tensor([0.5]), 8, 0.004, 4.0)
    assert embedding.shape == (1, 8)
    assert torch.isfinite(embedding).all()


class CaptureProcessor:
    """记录 runtime 送入 processor 的原始观测。"""

    def __init__(self) -> None:
        self.sample: dict[str, object] = {}

    def __call__(self, sample: dict[str, object]) -> dict[str, object]:
        self.sample = sample
        return sample


class StaticTrajectoryPolicy:
    """返回固定动作的最小本地 policy 替身。"""

    config = type("PolicyConfig", (), {"n_action_steps": 1})()

    def reset(self) -> None:
        pass

    def predict_action_chunk(self, batch: dict[str, Any]) -> torch.Tensor:
        return torch.zeros((1, 1, 9), dtype=torch.float32)


def test_runtime_keeps_language_state_and_dual_cameras() -> None:
    """在线运行时必须保留语言、13D 状态和双相机观测。"""
    processor = CaptureProcessor()
    runtime = LeRobotRuntime(
        cast(Any, StaticTrajectoryPolicy()),
        processor,
        lambda action: action,
        "cpu",
        TRAJECTORY_VLA,
    )
    observation = Observation(
        0.0,
        {
            "global": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        },
        np.zeros(13, dtype=np.float32),
        "Weld around the top rim of the pipe.",
    )
    action = runtime.select_action(observation)
    assert action.shape == (9,)
    assert processor.sample["task"] == [observation.instruction]
    assert processor.sample["observation.state"].shape == (1, 13)
    assert processor.sample["observation.images.global"].shape == (1, 3, 8, 8)


def test_config_exposes_flow_and_backbone_parameters() -> None:
    """主要研究开关应为公开配置，而不是隐藏常量。"""
    config = TrajectoryVLAConfig(chunk_size=30, n_action_steps=8)
    assert config.num_steps == 10
    assert config.flow_beta_alpha == 1.5
    assert config.attention_mode == "cross_attn"
    assert config.expert_width_multiplier == 0.75
    with pytest.raises(ValueError):
        replace(config, n_action_steps=31)
