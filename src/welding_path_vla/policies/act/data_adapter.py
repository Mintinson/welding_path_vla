"""把 weldpath LeRobot 数据集适配为 ACT 的动作块输入。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from welding_path_vla.core.config import PolicyConfig, TrainingConfig

CAMERA_KEYS = ("observation.images.global", "observation.images.wrist")


@dataclass(frozen=True, slots=True)
class ACTDataReport:
    """ACT 数据 schema 检查结果。"""

    episodes: int
    frames: int
    fps: int
    state_dimension: int
    action_dimension: int
    camera_keys: tuple[str, ...]


def metadata(training: TrainingConfig) -> LeRobotDatasetMetadata:
    """用 LeRobot 官方元数据读取器打开本地数据集。"""
    if not training.dataset_repo_id or not training.dataset_root:
        raise ValueError("ACT requires training.dataset_repo_id and training.dataset_root")
    return LeRobotDatasetMetadata(training.dataset_repo_id, root=training.dataset_root)


def validate_dataset(training: TrainingConfig) -> ACTDataReport:
    """确认相机、状态和 9D 动作满足 ACT 输入约定。"""
    meta = metadata(training)
    missing = [
        key for key in (*CAMERA_KEYS, "observation.state", "action") if key not in meta.features
    ]
    if missing:
        raise ValueError(f"ACT dataset is missing features: {missing}")
    state_dimension = int(meta.features["observation.state"]["shape"][0])
    action_dimension = int(meta.features["action"]["shape"][0])
    if state_dimension != 13 or action_dimension != 9:
        raise ValueError(
            f"ACT expects state/action dimensions 13/9, got {state_dimension}/{action_dimension}"
        )
    return ACTDataReport(
        meta.total_episodes,
        meta.total_frames,
        meta.fps,
        state_dimension,
        action_dimension,
        CAMERA_KEYS,
    )


def held_out_episode_indices(training: TrainingConfig, count: int) -> list[int]:
    """返回数据集末尾的 episode 作为确定性测试集。"""
    total = validate_dataset(training).episodes
    return list(range(max(0, total - min(count, total)), total))


def make_dataset(
    policy: PolicyConfig,
    training: TrainingConfig,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    """创建含 future action chunk 和 padding mask 的 LeRobotDataset。"""

    meta = metadata(training)
    repo_id = training.dataset_repo_id
    if repo_id is None:
        raise ValueError("training.dataset_repo_id is required")
    delta_timestamps = {"action": [index / meta.fps for index in range(policy.action_horizon)]}
    return LeRobotDataset(
        repo_id,
        root=Path(training.dataset_root or ""),
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend=training.video_backend,
        return_uint8=True,
    )


def scale_uint8_images(batch: dict[str, Any]) -> dict[str, Any]:
    """把 LeRobot 0.6 返回的 uint8 RGB 转为 ACT 所需的 `[0,1]`。"""
    import torch

    converted = dict(batch)
    for key in CAMERA_KEYS:
        image = converted[key]
        if image.dtype == torch.uint8:
            converted[key] = image.to(torch.float32).div(255)
    return converted


__all__ = [
    "CAMERA_KEYS",
    "ACTDataReport",
    "held_out_episode_indices",
    "make_dataset",
    "metadata",
    "scale_uint8_images",
    "validate_dataset",
]
