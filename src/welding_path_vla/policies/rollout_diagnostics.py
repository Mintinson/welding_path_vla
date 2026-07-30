"""策略仿真 rollout 的逐步记录结构与诊断摘要。"""

from __future__ import annotations

from typing import Any

import numpy as np

from welding_path_vla.core.config import AppConfig, DeploymentConfig


def new_rollout_arrays() -> dict[str, list[Any]]:
    """创建逐步记录容器；同一索引描述一次完整的状态转移。"""
    return {
        # 当前动作执行完成的仿真时间，单位为秒。
        "timestamp": [],
        # 策略执行前看到的 TCP 世界系位置，即 state_t。
        "observation_tcp_position": [],
        # 策略执行前看到的 TCP 世界系姿态，四元数顺序为 wxyz。
        "observation_tcp_quaternion_wxyz": [],
        # 策略执行前看到的六关节角，也是 13 维策略状态向量的一部分。
        "observation_joint_position": [],
        # 动作执行后的实际 TCP 世界系位置，即 state_{t+1}。
        "tcp_position": [],
        # 动作执行后的实际 TCP 世界系姿态，四元数顺序为 wxyz。
        "tcp_quaternion_wxyz": [],
        # 动作执行后的实际六关节角。
        "joint_position": [],
        # 动作执行后的实际六关节速度，用于安全与平滑性评价。
        "joint_velocity": [],
        # 策略输出的 9D TCP 局部增量：[平移 3D，旋转 6D]。
        "action": [],
        # 将 action 解码到世界系后，希望 TCP 到达的目标位置。
        "command_tcp_position": [],
        # 将 action 解码到世界系后，希望 TCP 到达的目标姿态。
        "command_tcp_quaternion_wxyz": [],
        # IK 针对目标 TCP 求得并送入安全门检查的六关节命令。
        "joint_command": [],
        # 目标 TCP 的 IK 残差，单位为米；失败步骤可能为 NaN。
        "ik_residual_m": [],
        # 执行后 TCP 在目标焊缝有向中心线上的归一化进度 [0, 1]。
        "seam_progress": [],
        # 执行后 TCP 到目标焊缝有限线段的欧氏距离，单位为米。
        "seam_distance_m": [],
        # 是否进入论文评价使用的焊缝跟踪带。
        "track_mask": [],
        # 当前策略周期是否出现任意不允许的 MuJoCo 接触。
        "collision": [],
        # 碰撞几何体名称对，格式为 first:second，多对使用 | 分隔。
        "collision_pairs": [],
        # IK 关节命令是否越过带安全余量的关节范围。
        "joint_limit": [],
        # 执行后关节速度是否超过配置上限。
        "joint_velocity_limit": [],
        # 相邻策略周期估算的关节加速度是否超过评价上限。
        "joint_acceleration": [],
        # ACT 平移增量或旋转表示是否合法。
        "action_increment": [],
        # 当前步骤的碰撞、安全门或动作解码错误；正常步骤为空字符串。
        "step_error": [],
    }


def rollout_completed(
    progress: float,
    seam_distance_m: float,
    config: DeploymentConfig,
) -> bool:
    """判断策略部署是否已达到自然退出条件。"""
    return (
        progress >= config.completion_progress_min
        and seam_distance_m <= config.completion_distance_m
    )


def finite_max(values: np.ndarray) -> float | None:
    """返回有限值最大值；全部缺失时返回 None。"""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else None


def build_rollout_diagnostics(
    trajectory: dict[str, np.ndarray],
    config: AppConfig,
    seam_start: np.ndarray,
    seam_end: np.ndarray,
    seam_start_normal: np.ndarray,
    seam_end_normal: np.ndarray,
    termination_reason: str,
    video_recorded: bool,
) -> dict[str, Any]:
    """把逐步数组汇总为便于人工排查的结构化信息。

    Args:
        trajectory: rollout 保存的逐步原始数组。
        config: 当前完整应用配置。
        seam_start: 有向焊缝世界坐标起点。
        seam_end: 有向焊缝世界坐标终点。
        seam_start_normal: 起点焊接法向；圆弧中它与终点不同。
        seam_end_normal: 终点焊接法向。
        termination_reason: rollout 的自然退出或失败原因。
        video_recorded: 是否同步写入了双相机视频。

    Returns:
        包含终止、跟踪、控制、安全和视频状态的摘要。
    """
    pairs = sorted(
        {
            pair
            for encoded in trajectory["collision_pairs"].astype(str)
            for pair in encoded.split("|")
            if pair
        }
    )
    errors = sorted({value for value in trajectory["step_error"].astype(str) if value})
    distance = np.asarray(trajectory["seam_distance_m"], dtype=np.float64)
    progress = np.asarray(trajectory["seam_progress"], dtype=np.float64)
    action = np.asarray(trajectory["action"], dtype=np.float64)
    displacement = np.linalg.norm(
        trajectory["tcp_position"] - trajectory["observation_tcp_position"],
        axis=1,
    )
    steps = len(trajectory["timestamp"])
    return {
        "termination": {
            "reason": termination_reason,
            "natural": termination_reason == "completed",
            "timed_out": termination_reason == "timeout",
        },
        "completion_rule": {
            "progress_min": config.deployment.completion_progress_min,
            "seam_distance_max_m": config.deployment.completion_distance_m,
        },
        "reference": {
            "seam_start_m": seam_start.tolist(),
            "seam_end_m": seam_end.tolist(),
            "seam_start_normal": seam_start_normal.tolist(),
            "seam_end_normal": seam_end_normal.tolist(),
        },
        "tracking": {
            "frames": steps,
            "track_frames": int(np.sum(trajectory["track_mask"])),
            "completion_zone_frames": int(
                np.sum(distance <= config.deployment.completion_distance_m)
            ),
            "closest_seam_distance_m": float(np.min(distance)),
            "final_seam_distance_m": float(distance[-1]),
            "max_progress": float(np.max(progress)),
            "final_progress": float(progress[-1]),
        },
        "control": {
            "max_action_translation_m": finite_max(np.linalg.norm(action[:, :3], axis=1)),
            "max_ik_residual_m": finite_max(trajectory["ik_residual_m"]),
            "max_tcp_speed_m_s": finite_max(displacement * config.timing.policy_hz),
            "errors": errors,
        },
        "safety": {
            "collision_frames": int(np.sum(trajectory["collision"])),
            "collision_pairs": pairs,
            "joint_limit_frames": int(np.sum(trajectory["joint_limit"])),
            "joint_velocity_limit_frames": int(np.sum(trajectory["joint_velocity_limit"])),
            "joint_acceleration_limit_frames": int(np.sum(trajectory["joint_acceleration"])),
            "action_increment_frames": int(np.sum(trajectory["action_increment"])),
        },
        "video": {
            "includes_terminal_state": video_recorded,
            "frame_count": steps + 1 if video_recorded else 0,
            "alignment": "frame_0 is initial state; frame_n is state after action_n",
        },
    }


__all__ = [
    "build_rollout_diagnostics",
    "new_rollout_arrays",
    "rollout_completed",
]
