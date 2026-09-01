"""不同模仿学习策略共享的 LeRobot 数据访问工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from welding_path_vla.core.config import PolicyConfig, TrainingConfig
from welding_path_vla.dataset.actions import ABSOLUTE_ACTION_NAMES
from welding_path_vla.dataset.task_parameters import TASK_DIRECTION, TASK_PARAMETERS

MANIFEST_PATH = Path("meta/welding_path_vla_export.json")

CAMERA_KEYS = ("observation.images.global", "observation.images.wrist")


@dataclass(frozen=True, slots=True)
class PolicyDataReport:
    """策略数据 schema 检查结果。

    Attributes:
        episodes: episode 总数。
        frames: 视频帧总数。
        fps: 数据集采样帧率。
        state_dimension: 机器人状态维数。
        action_dimension: 动作维数。
        camera_keys: 策略使用的相机 feature 名称。
        tasks: 数据集包含的任务数。
        action_representation: 模型实际学习的动作表示。
        action_horizon: relative action 统计对应的 horizon。
        action_stride: relative action 统计对应的采样间隔。
    """

    episodes: int
    frames: int
    fps: int
    state_dimension: int
    action_dimension: int
    camera_keys: tuple[str, ...]
    tasks: int
    action_representation: str
    action_horizon: int
    action_stride: int


def metadata(training: TrainingConfig) -> LeRobotDatasetMetadata:
    """用 LeRobot 官方元数据读取器打开本地数据集。

    Args:
        training: 包含本地路径和稳定 repo id 的训练配置。

    Returns:
        LeRobot 数据集元数据。
    """
    if not training.dataset_repo_id or not training.dataset_root:
        raise ValueError("training requires dataset_repo_id and dataset_root")
    return LeRobotDatasetMetadata(training.dataset_repo_id, root=training.dataset_root)


def validate_dataset(training: TrainingConfig, policy: PolicyConfig) -> PolicyDataReport:
    """确认数据 schema 与 relative action 训练契约一致。"""
    meta = metadata(training)
    prompt_fields = policy.welding_prompt_fields
    prompt_features = (
        *((TASK_DIRECTION,) if "direction" in prompt_fields else ()),
        *((TASK_PARAMETERS,) if set(prompt_fields).difference({"direction"}) else ()),
    )
    required = (*CAMERA_KEYS, "observation.state", "action", *prompt_features)
    missing = [key for key in required if key not in meta.features]
    if missing:
        raise ValueError(f"policy dataset is missing features: {missing}")
    state_dimension = int(meta.features["observation.state"]["shape"][0])
    action_dimension = int(meta.features["action"]["shape"][0])
    if state_dimension != 13 or action_dimension != 9:
        raise ValueError(
            f"policies expect state/action dimensions 13/9, got "
            f"{state_dimension}/{action_dimension}"
        )
    action_names = meta.features["action"].get("names")
    if action_names != list(ABSOLUTE_ACTION_NAMES):
        raise ValueError("数据集必须保存 absolute EE targets, 请用新版 export-lerobot 重新导出")
    manifest_path = Path(training.dataset_root or "") / MANIFEST_PATH
    if not manifest_path.exists():
        raise ValueError("数据集缺少 relative_action manifest, 请重新导出")
    representation = json.loads(manifest_path.read_text(encoding="utf-8")).get(
        "action_representation", {}
    )
    expected = (policy.action_representation, policy.action_horizon, policy.action_stride)
    actual = (
        representation.get("type"),
        representation.get("horizon"),
        representation.get("stride"),
    )
    if actual != expected:
        raise ValueError(f"动作契约不匹配: policy={expected}, dataset={actual}")
    tasks = len(meta.tasks)
    if tasks < 1:
        raise ValueError("policy dataset must contain at least one task")
    return PolicyDataReport(
        meta.total_episodes,
        meta.total_frames,
        meta.fps,
        state_dimension,
        action_dimension,
        CAMERA_KEYS,
        tasks,
        representation["type"],
        representation["horizon"],
        representation["stride"],
    )


def held_out_episode_indices(training: TrainingConfig, count: int) -> list[int]:
    """从各任务尾部均衡选取 episode，供确定性离线测试使用。

    Args:
        training: 数据集配置。
        count: 最多选取的 episode 总数。

    Returns:
        按 episode 索引排序的留出集；任务数不整除时，前面的任务多取一条。
    """
    meta = metadata(training)
    groups: dict[str, list[int]] = {}
    for raw_episode in meta.episodes:
        episode = cast(dict[str, Any], raw_episode)
        tasks = episode["tasks"]
        task = tasks[0] if tasks else ""
        groups.setdefault(task, []).append(int(episode["episode_index"]))
    count = min(count, meta.total_episodes)
    base, remainder = divmod(count, len(groups))
    selected: list[int] = []
    for index, episodes in enumerate(groups.values()):
        take = base + int(index < remainder)
        selected.extend(episodes[-take:] if take else [])
    return sorted(selected)


def balanced_frame_indices(dataset: LeRobotDataset, count: int) -> list[int]:
    """从各任务均衡抽取分布在完整轨迹上的帧。

    Args:
        dataset: 已限定到留出 episode 的 LeRobot 数据集。
        count: 最多抽取的帧数。

    Returns:
        按数据集索引排序的帧索引。
    """
    task_indices = np.asarray(dataset.hf_dataset["task_index"])
    tasks = np.unique(task_indices)
    count = min(count, len(dataset))
    base, remainder = divmod(count, len(tasks))
    selected: list[int] = []
    for index, task in enumerate(tasks):
        candidates = np.flatnonzero(task_indices == task)
        take = base + int(index < remainder)
        selected.extend(candidates[np.linspace(0, len(candidates) - 1, take, dtype=int)].tolist())
    return sorted(selected)


def make_dataset(
    policy: PolicyConfig,
    training: TrainingConfig,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    """创建含 future action chunk、语言任务和 padding mask 的数据集。"""
    meta = metadata(training)
    repo_id = training.dataset_repo_id
    if repo_id is None:
        raise ValueError("training.dataset_repo_id is required")
    delta_timestamps = {
        "action": [
            index * policy.action_stride / meta.fps for index in range(policy.action_horizon)
        ]
    }
    return LeRobotDataset(
        repo_id,
        root=Path(training.dataset_root or ""),
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend=training.video_backend,
        return_uint8=True,
    )


def scale_uint8_images(batch: dict[str, Any]) -> dict[str, Any]:
    """把 LeRobot 的 uint8 RGB 转为策略 processor 使用的 ``[0, 1]``。"""
    import torch

    converted = dict(batch)
    for key in CAMERA_KEYS:
        image = converted[key]
        if image.dtype == torch.uint8:
            converted[key] = image.to(torch.float32).div(255)
    return converted


__all__ = [
    "CAMERA_KEYS",
    "PolicyDataReport",
    "balanced_frame_indices",
    "held_out_episode_indices",
    "make_dataset",
    "metadata",
    "scale_uint8_images",
    "validate_dataset",
]
