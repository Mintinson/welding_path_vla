"""不同模仿学习策略共享的 LeRobot 数据访问工具。"""

from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, cast

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


@dataclass(frozen=True, slots=True)
class DatasetTaskSummary:
    """LeRobot metadata 中一个语言任务的数据规模。"""

    task_index: int
    name: str
    episodes: int
    frames: int
    episode_indices: tuple[int, ...]


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


def dataset_task_summaries(training: TrainingConfig) -> list[DatasetTaskSummary]:
    """只读取 metadata，按任务汇总 episode 和帧数。"""
    meta = metadata(training)
    indices = {str(name): int(row.task_index) for name, row in meta.tasks.iterrows()}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw_episode in meta.episodes:
        episode = cast(dict[str, Any], raw_episode)
        task = str(episode["tasks"][0]) if episode["tasks"] else ""
        grouped.setdefault(task, []).append(episode)
    return [
        DatasetTaskSummary(
            indices[name],
            name,
            len(episodes),
            sum(int(episode["length"]) for episode in episodes),
            tuple(int(episode["episode_index"]) for episode in episodes),
        )
        for name, episodes in grouped.items()
    ]


def filtered_episode_indices(training: TrainingConfig) -> list[int] | None:
    """把任务名称过滤器解析为 LeRobot 原生 episode 子集。"""
    if not training.include_tasks and not training.exclude_tasks:
        return None
    summaries = dataset_task_summaries(training)
    available = {summary.name for summary in summaries}
    requested = set(training.include_tasks) | set(training.exclude_tasks)
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f"unknown LeRobot tasks: {unknown}; available={sorted(available)}")
    selected = set(training.include_tasks) if training.include_tasks else available
    selected -= set(training.exclude_tasks)
    episodes = [
        episode
        for summary in summaries
        if summary.name in selected
        for episode in summary.episode_indices
    ]
    if not episodes:
        raise ValueError("training task filters selected no episodes")
    return sorted(episodes)


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
    selected = filtered_episode_indices(training)
    selected_set = set(selected) if selected is not None else None
    episodes = [
        cast(dict[str, Any], episode)
        for episode in meta.episodes
        if selected_set is None or int(episode["episode_index"]) in selected_set
    ]
    tasks = len({episode["tasks"][0] for episode in episodes if episode["tasks"]})
    if tasks < 1:
        raise ValueError("policy dataset must contain at least one task")
    return PolicyDataReport(
        len(episodes),
        sum(int(episode["length"]) for episode in episodes),
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
    selected = filtered_episode_indices(training)
    selected_set = set(selected) if selected is not None else None
    groups: dict[str, list[int]] = {}
    for raw_episode in meta.episodes:
        episode = cast(dict[str, Any], raw_episode)
        if selected_set is not None and int(episode["episode_index"]) not in selected_set:
            continue
        tasks = episode["tasks"]
        task = tasks[0] if tasks else ""
        groups.setdefault(task, []).append(int(episode["episode_index"]))
    count = min(count, sum(len(episodes) for episodes in groups.values()))
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


def filtered_training_stats(dataset: LeRobotDataset, policy: PolicyConfig) -> dict[str, Any]:
    """用筛选后的训练 episode 重算状态和 relative EE action 统计。"""
    from lerobot.datasets.compute_stats import RunningQuantileStats, get_feature_stats

    from welding_path_vla.core.geometry import relative_ee_actions_from_absolute

    states = np.asarray(dataset.hf_dataset["observation.state"], dtype=np.float32)
    actions = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
    episode_indices = np.asarray(dataset.hf_dataset["episode_index"])
    stats = deepcopy(dataset.meta.stats)
    stats["observation.state"] = get_feature_stats(states, axis=0, keepdims=False)

    # 动作统计容器
    action_stats = RunningQuantileStats()
    offsets = np.arange(policy.action_horizon) * policy.action_stride
    sampled = False
    for episode in np.unique(episode_indices):
        # 该 episode 所有帧在全局数据集中的行索引
        rows = np.flatnonzero(episode_indices == episode)
        sample_count = len(rows) - int(offsets[-1])
        for start in range(0, max(sample_count, 0), 1024):
            starts = np.arange(start, min(start + 1024, sample_count))
            chunks = actions[rows[starts[:, None] + offsets]]
            relative = relative_ee_actions_from_absolute(
                chunks,
                states[rows[starts], 6:9],
                states[rows[starts], 9:13],
            )
            action_stats.update(relative.reshape(-1, actions.shape[1]))
            sampled = True
    if not sampled:
        raise ValueError("filtered training episodes contain no complete action chunks")
    stats["action"] = action_stats.get_statistics()
    return stats


@contextmanager
def lerobot_task_filter(policy: PolicyConfig) -> Generator[None, None, None]:
    """保留 LeRobot 数据集工厂，只为 episode 子集提供训练集统计。"""
    from lerobot.scripts import lerobot_train

    make_datasets = lerobot_train.make_train_eval_datasets

    def make_filtered_datasets(config: Any) -> tuple[Any, Any]:
        train_dataset, eval_dataset = make_datasets(config)
        if config.dataset.episodes is None:
            return train_dataset, eval_dataset
        stats = filtered_training_stats(train_dataset, policy)
        train_dataset.meta.stats = stats
        if eval_dataset is not None:
            eval_dataset.meta.stats = deepcopy(stats)
        return train_dataset, eval_dataset

    lerobot_train.make_train_eval_datasets = make_filtered_datasets
    try:
        yield
    finally:
        lerobot_train.make_train_eval_datasets = make_datasets


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
    "DatasetTaskSummary",
    "PolicyDataReport",
    "balanced_frame_indices",
    "dataset_task_summaries",
    "filtered_episode_indices",
    "filtered_training_stats",
    "held_out_episode_indices",
    "lerobot_task_filter",
    "make_dataset",
    "metadata",
    "scale_uint8_images",
    "validate_dataset",
]
