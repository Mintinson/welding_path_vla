"""焊接 EE 动作的 absolute、relative 与 delta 表示。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from welding_path_vla.core.geometry import (
    quaternion_to_matrix,
    relative_ee_actions_from_absolute,
    rotation_to_6d_rows,
)
from welding_path_vla.dataset.raw_schema import EpisodeReader

RELATIVE_ACTION_NAMES = ("dx", "dy", "dz", "r1x", "r1y", "r1z", "r2x", "r2y", "r2z")
ABSOLUTE_ACTION_NAMES = ("x", "y", "z", "r1x", "r1y", "r1z", "r2x", "r2y", "r2z")


@dataclass(frozen=True, slots=True)
class RelativeActionChunk:
    """以预测时刻 TCP 为共同锚点的未来 EE 轨迹。

    Attributes:
        values: `[horizon, 9]` 相对动作；前三维为米，后六维为相对旋转 6D。
        valid_mask: `[horizon]` 有效位；越过 episode 末尾的复制值为 `False`。
    """

    values: np.ndarray
    valid_mask: np.ndarray


def action_targets(episode: EpisodeReader, source: str) -> tuple[np.ndarray, np.ndarray]:
    """返回与每个动作时刻对齐的世界系绝对目标位姿。

    Args:
        episode: 原始 episode。
        source: `safe_command`、`reference` 或 `executed`。

    Returns:
        位置 `[N, 3]` 和 wxyz 四元数 `[N, 4]`。`executed` 使用动作后的状态。
    """
    names = {
        "safe_command": ("safe_command_position", "safe_command_quaternion_wxyz", 0),
        "reference": ("reference_position", "reference_quaternion_wxyz", 0),
        "executed": ("tcp_position", "tcp_quaternion_wxyz", 1),
    }
    if source not in names:
        raise ValueError(f"unknown action source: {source}")
    position_name, quaternion_name, offset = names[source]
    count = episode.action_count
    trajectory = episode.trajectory
    return (
        np.asarray(trajectory[position_name][offset : offset + count]),
        np.asarray(trajectory[quaternion_name][offset : offset + count]),
    )


def build_absolute_actions(episode: EpisodeReader, source: str = "safe_command") -> np.ndarray:
    """构造供 LeRobot 存储、再动态转换为 relative action 的绝对 EE 目标。

    Args:
        episode: 原始 episode。
        source: 绝对目标来源。

    Returns:
        `[N, 9]` 世界系动作 `[x, y, z, rotation_6d_rows]`。
    """
    positions, quaternions = action_targets(episode, source)
    rotations = quaternion_to_matrix(quaternions)
    rotation_6d = rotation_to_6d_rows(rotations)
    return np.concatenate((positions, rotation_6d), axis=1).astype(np.float32)


def build_relative_actions(
    episode: EpisodeReader,
    frame_index: int,
    horizon: int,
    stride: int = 1,
    source: str = "safe_command",
) -> RelativeActionChunk:
    """构造符合 LeRobot 官方命名的共同锚点 relative action chunk。

    所有 future target 都相对预测时刻 `frame_index` 的实际 TCP，而不是相对各自
    时刻的 TCP，也不是相对前一个动作。这与 UMI / LeRobot 的 relative trajectory
    定义一致。

    Args:
        episode: 原始 episode。
        frame_index: 当前预测帧。
        horizon: future chunk 长度。
        stride: 相邻 future target 的帧间隔。
        source: `safe_command`、`reference` 或 `executed`。

    Returns:
        共同锚点动作及 episode 尾部 padding mask。
    """
    if horizon < 1 or stride < 1:
        raise ValueError("horizon and stride must be positive")
    if not 0 <= frame_index < episode.action_count:
        raise IndexError(frame_index)

    absolute = build_absolute_actions(episode, source)
    raw_indices = frame_index + np.arange(horizon) * stride
    valid = raw_indices < len(absolute)
    indices = np.clip(raw_indices, 0, len(absolute) - 1)
    trajectory = episode.trajectory
    values = relative_ee_actions_from_absolute(
        absolute[indices],
        np.asarray(trajectory["tcp_position"][frame_index], dtype=absolute.dtype),
        np.asarray(trajectory["tcp_quaternion_wxyz"][frame_index], dtype=absolute.dtype),
    )
    return RelativeActionChunk(values, valid)


def build_delta_action(episode: EpisodeReader, source: str = "safe_command") -> np.ndarray:
    """构造逐时刻局部差分动作，仅用于旧数据迁移和表示消融。

    第 `t` 个动作以 `T_tcp[t]` 为参考，future chunk 中每一步参考点不同。按照
    LeRobot 官方术语，这属于 sequential delta，而不是 relative trajectory。

    Args:
        episode: 原始 episode。
        source: 绝对目标来源。

    Returns:
        `[N, 9]` delta actions。
    """
    absolute = build_absolute_actions(episode, source)
    count = episode.action_count
    trajectory = episode.trajectory
    return relative_ee_actions_from_absolute(
        absolute,
        np.asarray(trajectory["tcp_position"][:count], dtype=absolute.dtype),
        np.asarray(trajectory["tcp_quaternion_wxyz"][:count], dtype=absolute.dtype),
    )


__all__ = [
    "ABSOLUTE_ACTION_NAMES",
    "RELATIVE_ACTION_NAMES",
    "RelativeActionChunk",
    "action_targets",
    "build_absolute_actions",
    "build_delta_action",
    "build_relative_actions",
]
