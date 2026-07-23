from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from welding_path_vla.dataset.raw_schema import EpisodeReader
from welding_path_vla.geometry import (
    frame_delta,
    pose_delta,
    quaternion_to_matrix,
    yaw_degrees_to_matrix,
)


@dataclass(frozen=True, slots=True)
class RelativeActionChunk:
    """相对当前末端坐标系的未来累计轨迹块。"""

    values: np.ndarray
    valid_mask: np.ndarray


def build_action_chunk(
    episode: EpisodeReader,
    frame_index: int,
    horizon: int,
    stride: int = 1,
    source: str = "executed_tcp",
) -> np.ndarray:
    """Build a policy-specific action window from the complete raw trajectory."""
    if source == "command_seam":
        actions = episode.trajectory["command_delta_pose_seam"]
        indices = np.minimum(frame_index + np.arange(horizon) * stride, len(actions) - 1)
        return np.asarray(actions[indices], dtype=np.float32)
    if source not in {"executed_tcp", "executed_base", "executed_world"}:
        raise ValueError(f"unknown action source: {source}")
    positions = episode.trajectory["tcp_position"]
    quaternions = episode.trajectory["tcp_quaternion_wxyz"]
    current_position = positions[frame_index]
    current_quaternion = quaternions[frame_index]
    indices = np.minimum(frame_index + (np.arange(horizon) + 1) * stride, len(positions) - 1)
    world_from_base = yaw_degrees_to_matrix(
        episode.metadata["resolved_config"]["scene"].get("robot_base_yaw_deg", 0.0)
    )
    transform = (
        (lambda delta: delta)
        if source == "executed_world"
        else (lambda delta: frame_delta(delta, world_from_base))
    )
    return np.asarray(
        [
            transform(
                pose_delta(
                    current_position,
                    current_quaternion,
                    positions[index],
                    quaternions[index],
                ),
            )
            for index in indices
        ],
        dtype=np.float32,
    )


def pose_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """把位置和 wxyz 四元数组合为齐次变换矩阵。"""
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_matrix(quaternion)
    matrix[:3, 3] = position
    return matrix


def build_relative_action_chunk(
    episode: EpisodeReader,
    frame_index: int,
    horizon: int,
    stride: int = 1,
    source: str = "safe_command",
    include_current: bool = False,
) -> RelativeActionChunk:
    """从绝对轨迹构造 Hy-VLA 风格的 9D future chunk。

    Args:
        episode: 原始 episode 读取器。
        frame_index: 当前策略观测帧索引。
        horizon: future chunk 长度。
        stride: 相邻 future target 的帧间隔。
        source: `safe_command`、`reference` 或 `executed`。
        include_current: 是否把单位变换作为 chunk 第一个元素。

    Returns:
        9D 动作 `[xyz, rotation_6d_rows]` 和末尾有效掩码。
    """
    if horizon < 1 or stride < 1:
        raise ValueError("horizon and stride must be positive")
    if not 0 <= frame_index < episode.state_count:
        raise IndexError(frame_index)
    trajectory = episode.trajectory
    current = pose_matrix(
        trajectory["tcp_position"][frame_index],
        trajectory["tcp_quaternion_wxyz"][frame_index],
    )
    source_names = {
        "safe_command": ("safe_command_position", "safe_command_quaternion_wxyz"),
        "reference": ("reference_position", "reference_quaternion_wxyz"),
        "executed": ("tcp_position", "tcp_quaternion_wxyz"),
    }
    if source not in source_names:
        raise ValueError(f"unknown relative action source: {source}")
    position_name, quaternion_name = source_names[source]
    positions = trajectory[position_name]
    quaternions = trajectory[quaternion_name]
    first_offset = 0 if source != "executed" else 1
    raw_indices = frame_index + (np.arange(horizon) + first_offset) * stride
    if include_current:
        raw_indices = np.concatenate(([-1], raw_indices[: horizon - 1]))
    valid = (raw_indices >= 0) & (raw_indices < len(positions))
    indices = np.clip(raw_indices, 0, len(positions) - 1)
    targets = np.asarray([pose_matrix(positions[index], quaternions[index]) for index in indices])
    if include_current:
        targets[0] = current
        valid[0] = True
    relative = np.linalg.inv(current)[None] @ targets
    rotation_6d = relative[:, :2, :3].reshape(horizon, 6)
    values = np.concatenate((relative[:, :3, 3], rotation_6d), axis=1).astype(np.float32)
    return RelativeActionChunk(values, valid)
