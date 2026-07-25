"""动作构造模块: 为策略训练提供多种坐标系下的动作表示。

本模块解决了策略动作空间的两个核心问题:
  1. 动作的坐标系选择 — 世界系/基座系/焊缝系/末端系各有优劣
  2. 动作的时间跨度 — 单步 vs future chunk

动作表示约定:
  - 9D 向量: [dx, dy, dz, r1x, r1y, r1z, r2x, r2y, r2z]
  - 前三维是相对当前 TCP 末端的位置平移
  - 后六维是旋转矩阵的前两行 (rotation_6d), 相对于当前 TCP 末端
  - 这种 6D 旋转表示避免了欧拉角的万向锁和四元数的符号歧义
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from welding_path_vla.core.geometry import (
    frame_delta,
    pose_delta,
    quaternion_to_matrix,
    yaw_degrees_to_matrix,
)
from welding_path_vla.dataset.raw_schema import EpisodeReader


@dataclass(frozen=True, slots=True)
class RelativeActionChunk:
    """相对于当前 TCP 末端的未来动作块。

    每个动作是 9D 向量, 描述未来某个时刻目标位姿相对于当前 TCP 的变换。
    适用于预测-执行为范式的策略模型, 使动作与物理世界解耦。

    Attributes:
        values: [horizon, 9] 的数组, 每行是 [dx, dy, dz, r1x, r1y, r1z, r2x, r2y, r2z]。
                前 3 维是相对位置平移, 后 6 维是相对旋转矩阵的前两行。
        valid_mask: [horizon] 的布尔掩码, False 表示该索引超出 episode 边界。
    """

    values: np.ndarray
    valid_mask: np.ndarray


def build_action_chunk(
    episode: EpisodeReader,
    frame_index: int,
    horizon: int,
    stride: int = 1,
    source: str = "executed_tcp",
) -> np.ndarray:
    """从原始轨迹构造策略专用的动作窗口。

    支持两种模式:
      1. command_seam: 直接取录制时已算好的焊缝坐标系 delta pose。
         这是离线时从世界 delta 变换到焊缝标架的结果, 无需重复计算。
      2. executed_*: 基于实际执行的 TCP 轨迹计算 delta pose,
         再按需变换到基座标架。

    数学原理:
      delta_pose_{world} = diff(current_tcp_pose, future_tcp_pose)
      delta_pose_{base}  = R_base^T @ delta_pose_{world}
      其中 R_base 是机器人基座在世界系下的偏航旋转。

    Args:
        episode: 原始 episode 读取器。
        frame_index: 当前策略观测帧索引。
        horizon: 动作窗口长度。
        stride: 相邻动作帧的间隔步长。
        source: 动作来源:
            - "command_seam": 录制时存储的焊缝坐标系指令 (不需要当前帧 +1 偏移)。
            - "executed_tcp": 实际执行 TCP 轨迹的世界坐标系 delta。
            - "executed_base": 变换到基座标架后的实际执行 delta。
            - "executed_world": 世界坐标系下的实际执行 delta。

    Returns:
        [horizon, 6] 数组, 每行是 [dx, dy, dz, rx, ry, rz] 的 delta pose。
        注意: 与 9D 相对动作不同, 这里返回的是 delta pose 的轴角表示,
        而不是 rotation_6d。
    """
    # ---- 模式 1: 直接使用已存储的焊缝系指令 ----
    # 这些指令在采集时已计算好, 包含当前到目标的世界 delta 再变换到 seam 标架
    if source == "command_seam":
        actions = episode.trajectory["command_delta_pose_seam"]
        indices = np.minimum(frame_index + np.arange(horizon) * stride, len(actions) - 1)
        return np.asarray(actions[indices], dtype=np.float32)

    # ---- 模式 2: 基于实际执行 TCP 计算 delta pose ----
    if source not in {"executed_tcp", "executed_base", "executed_world"}:
        raise ValueError(f"unknown action source: {source}")

    positions = episode.trajectory["tcp_position"]
    quaternions = episode.trajectory["tcp_quaternion_wxyz"]
    current_position = positions[frame_index]
    current_quaternion = quaternions[frame_index]

    # (horizon + 1) 偏移是因为 action 数比 state 数少 1:
    # 第 N 帧的状态对应第 N 帧观察, 但动作指向 N+1 帧的目标
    indices = np.minimum(frame_index + (np.arange(horizon) + 1) * stride, len(positions) - 1)

    world_from_base = yaw_degrees_to_matrix(
        episode.metadata["resolved_config"]["scene"].get("robot_base_yaw_deg", 0.0)
    )

    # 选择坐标变换: executed_world → 不变换; executed_{base,tcp} → 旋转到目标系
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
    """将位置和 wxyz 四元数组合为 4x4 齐次变换矩阵。

    齐次变换矩阵形式:
        T = | R_{3x3}  t_{3x1} |
            |   0^T       1     |

    其中 R 是四元数对应的旋转矩阵, t 是位置向量。这个表示允许通过矩阵
    乘法 (T_1^{-1} @ T_2) 便捷地计算相对变换, 而不需要手动处理旋转和平移
    的耦合。

    Args:
        position: 三维位置 [x, y, z]。
        quaternion: 四元数 (w, x, y, z)。

    Returns:
        4x4 齐次变换矩阵。
    """
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
    """从绝对轨迹构造 Hy-VLA 风格的 9D 相对动作块。

    动机:
      策略网络需要在一个统一的局部坐标系中预测动作, 而不是在世界坐标系中,
      因为世界坐标系的预测随工件位姿变化剧烈。将目标位姿转换到当前 TCP
      末端坐标系下, 使动作表示与全局位姿解耦, 提高泛化性。

    数学原理:
      给定当前 TCP 的齐次变换矩阵 T_current, 和目标位姿 T_target, 相对变换为:
        T_rel = T_current^{-1} @ T_target

      展开形式:
        T_rel = | R_current^T  -R_current^T @ t_current |   | R_target  t_target |
                |     0^T              1                 | @ |    0^T       1      |

      结果 T_rel 包含:
        - 位置分量: t_rel = R_current^T @ (t_target - t_current)
        - 旋转分量: R_rel = R_current^T @ R_target

      9D 编码方式:
        前 3 维 = t_rel (相对位置平移)
        后 6 维 = R_rel 的前两行展平 (rotation_6d)
        使用 6D 旋转表示 Zhou et al. "On the Continuity of Rotation Representations
        in Neural Networks" 避免欧拉角不连续性和四元数符号歧义。

    时间步对齐说明:
      录制格式为 N 个动作 + N+1 个状态 (动作发生在状态之间)。
      - 对于 "safe_command"/"reference": frame_index 对应的目标与当前观测同一步。
      - 对于 "executed": 第 0 个动作对应 frame_index=0 到 frame_index=1 的执行结果,
        因此需要 +1 偏移来对齐。

    Args:
        episode: 原始 episode 读取器。
        frame_index: 当前策略观测帧索引。
        horizon: future chunk 长度。
        stride: 相邻 future target 的帧间隔。
        source: 目标位姿来源:
            - "safe_command": IK 求解并限幅后的安全位姿 (训练常用)。
            - "reference": 专家轨迹参考帧的期望位姿。
            - "executed": 物理引擎实际执行后的 TCP 位姿。
        include_current: 是否将单位矩阵 (零位移、零旋转) 作为 chunk 第一个元素,
            用于标识"当前位置不动的基线动作"。

    Returns:
        RelativeActionChunk, values 为 [horizon, 9]:
            values[:, :3]  = [dx, dy, dz] — 相对当前 TCP 的位置平移
            values[:, 3:]  = [r1x, r1y, r1z, r2x, r2y, r2z] — 相对旋转的 6D 表示
            valid_mask      = 各索引是否在有效范围内的布尔掩码。
    """
    if horizon < 1 or stride < 1:
        raise ValueError("horizon and stride must be positive")
    if not 0 <= frame_index < episode.state_count:
        raise IndexError(frame_index)

    trajectory = episode.trajectory

    # 当前 TCP 的齐次变换矩阵 (世界系)
    current = pose_matrix(
        trajectory["tcp_position"][frame_index],
        trajectory["tcp_quaternion_wxyz"][frame_index],
    )

    # 选择目标位姿的字段
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

    # 索引偏移: executed 需跳过第 0 帧 (它是已执行动作的起点)
    first_offset = 0 if source != "executed" else 1
    raw_indices = frame_index + (np.arange(horizon) + first_offset) * stride

    # 可选: 在块头部插入当前位姿 (单位变换) 作为基线
    if include_current:
        raw_indices = np.concatenate(([-1], raw_indices[: horizon - 1]))

    # 裁剪越界索引, 保留有效掩码
    valid = (raw_indices >= 0) & (raw_indices < len(positions))
    indices = np.clip(raw_indices, 0, len(positions) - 1)

    # 构造目标位姿的齐次矩阵序列
    targets = np.asarray([pose_matrix(positions[index], quaternions[index]) for index in indices])

    # 插入的当前帧使用实际当前位姿矩阵 (而不是从轨迹中取)
    if include_current:
        targets[0] = current
        valid[0] = True

    # 核心计算: T_rel = T_current^{-1} @ T_target
    # np.linalg.inv(current)[None] 增加 batch 维度以便广播
    relative = np.linalg.inv(current)[None] @ targets

    # 提取 6D 旋转表示: 旋转矩阵的前两行展平 [0,0][0,1][0,2][1,0][1,1][1,2]
    rotation_6d = relative[:, :2, :3].reshape(horizon, 6)

    # 拼接: [位置3D | 旋转6D] → 9D 动作向量
    values = np.concatenate((relative[:, :3, 3], rotation_6d), axis=1).astype(np.float32)

    return RelativeActionChunk(values, valid)


def build_relative_actions(episode: EpisodeReader, source: str = "safe_command") -> np.ndarray:
    """批量构造整条 episode 的 9D 单步局部动作。

    与 build_relative_action_chunk 的区别:
      - 本函数一次性处理所有时间步, 返回完整 episode 的动作数组
      - 每步动作 horizon=1 (单步), 没有 future chunk 的概念
      - 适用于需要全轨迹动作的离线分析或传统 BC 训练

    数学原理 (逐帧):
      R_rel = R_current^T @ R_target        ← 相对旋转矩阵
      t_rel = R_current^T @ (t_target - t_current)  ← 相对位置平移

      其中 R_current 是当前 TCP 在世界系下的旋转矩阵, R_target 是目标
      位姿在世界系下的旋转矩阵。R_current^T 将世界系下的向量旋转到局部
      TCP 坐标系。

    Args:
        episode: 原始 episode 读取器。
        source: 与 build_relative_action_chunk 相同,
            "safe_command" / "reference" / "executed"。

    Returns:
        [action_count, 9] 数组, 每行 [dx, dy, dz, r1x, r1y, r1z, r2x, r2y, r2z]。
    """
    trajectory = episode.trajectory

    # 各 source 的字段名和时间偏移 (executed 需要 +1 跳过初始帧)
    source_names = {
        "safe_command": ("safe_command_position", "safe_command_quaternion_wxyz", 0),
        "reference": ("reference_position", "reference_quaternion_wxyz", 0),
        "executed": ("tcp_position", "tcp_quaternion_wxyz", 1),
    }
    if source not in source_names:
        raise ValueError(f"unknown relative action source: {source}")

    position_name, quaternion_name, offset = source_names[source]
    count = episode.action_count

    # 当前 TCP 位姿 (世界系): 取前 count 步
    current_positions = trajectory["tcp_position"][:count]
    current_rotations = np.asarray(
        [quaternion_to_matrix(value) for value in trajectory["tcp_quaternion_wxyz"][:count]]
    )

    # 目标位姿 (世界系): 从 offset 开始取 count 步
    target_positions = trajectory[position_name][offset : offset + count]
    target_rotations = np.asarray(
        [
            quaternion_to_matrix(value)
            for value in trajectory[quaternion_name][offset : offset + count]
        ]
    )

    # R_rel = R_current^T @ R_target — 批量矩阵乘法, nji 是 R_current 的转置索引
    relative_rotations = np.einsum(
        "nji,njk->nik", current_rotations, target_rotations, optimize=True
    )

    # t_rel = R_current^T @ (t_target - t_current) — 将世界系位移旋转到 TCP 局部系
    position_delta = target_positions - current_positions
    relative_positions = np.einsum("nji,nj->ni", current_rotations, position_delta, optimize=True)

    # 提取 6D 旋转: 前两行展平
    rotation_6d = relative_rotations[:, :2, :].reshape(count, 6)

    return np.concatenate((relative_positions, rotation_6d), axis=1).astype(np.float32)
