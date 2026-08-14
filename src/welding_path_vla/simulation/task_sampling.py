"""仿真与策略部署共享的轻量任务采样。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import mujoco
import numpy as np

from welding_path_vla.core.config import AppConfig, maximum_seam_length
from welding_path_vla.core.domain import Pose
from welding_path_vla.simulation import ExpertTrajectory, WeldingEnv
from welding_path_vla.simulation.tasks import SeamPath


class StagingPoseError(RuntimeError):
    """IK 求解失败或预置位姿发生碰撞。"""


class TrajectoryPlanningError(RuntimeError):
    """完整参考轨迹不存在连续、可达且无碰撞的关节解。"""


@dataclass(slots=True)
class TrajectorySample:
    """完成录制前可行性检查的随机轨迹样本。

    Attributes:
        seam: 当前随机工件位姿下的焊缝几何。
        expert: 接近、跟踪和退出的 TCP 参考轨迹。
        joint_trajectory: 与参考帧一一对应的连续关节解。
        staging_residual_m: 空中预置位姿的 IK 残差。
        planning_max_ik_residual_m: 完整轨迹预检的最大 IK 残差。
        scene_sampling_attempts: 最终场景内部的工件位姿采样次数。
        motion_sampling_attempts: 完整运动规划的重采样次数。
        initial_joint_offset_deg: 初始关节相对 staging 构型的偏移。
        joint_sampling_attempts: 最终初始关节构型的采样次数。
        initial_tcp_offset_m: 初始 TCP 平移扰动。
        tcp_sampling_attempts: TCP 扰动的采样次数。
        initial_tcp_offset_applied: 是否成功应用 TCP 扰动。
    """

    seam: SeamPath
    expert: ExpertTrajectory
    joint_trajectory: np.ndarray
    staging_residual_m: float
    planning_max_ik_residual_m: float
    scene_sampling_attempts: int
    motion_sampling_attempts: int
    initial_joint_offset_deg: np.ndarray
    joint_sampling_attempts: int
    initial_tcp_offset_m: np.ndarray
    tcp_sampling_attempts: int
    initial_tcp_offset_applied: bool


def sample_task_config(config: AppConfig, rng: np.random.Generator) -> AppConfig:
    """采样一组经过量化的任务参数。

    姿态参数围绕 YAML 标称值均匀变化。圆管反向执行时同时把起点移动到
    同一几何圆弧的另一端，因此方向变化不会意外改变所覆盖的焊缝区域。
    角度取整到度，速度保留三位小数，姿态跟随比例保留两位小数。

    Args:
        config: 只读的实验基准配置。
        rng: episode 专用随机数生成器。

    Returns:
        包含本 episode 实际任务参数的配置副本。
    """
    sampled = deepcopy(config)
    task = sampled.task
    randomization = sampled.randomization

    def sample_value(
        center: float,
        radius: float,
        decimals: int,
        lower: float = -float("inf"),
        upper: float = float("inf"),
    ) -> float:
        """在给定范围均匀采样并按物理量精度取整。"""
        return round(
            float(rng.uniform(max(lower, center - radius), min(upper, center + radius))), decimals
        )

    def sample_angle(
        center: float,
        radius: float,
        lower: float = -float("inf"),
        upper: float = float("inf"),
    ) -> int:
        """在给定范围均匀采样并取整到度。"""
        return round(float(rng.uniform(max(lower, center - radius), min(upper, center + radius))))

    task.work_angle_deg = sample_angle(
        config.task.work_angle_deg, randomization.work_angle_range_deg, 0.0, 90.0
    )
    task.travel_angle_deg = sample_angle(
        config.task.travel_angle_deg, randomization.travel_angle_range_deg
    )
    task.tool_roll_deg = sample_angle(config.task.tool_roll_deg, randomization.tool_roll_range_deg)
    task.orientation_follow_ratio = sample_value(
        config.task.orientation_follow_ratio, randomization.orientation_follow_range, 2, 0.0, 1.0
    )
    task.approach_speed_mps = sample_value(
        config.task.approach_speed_mps,
        randomization.approach_speed_range_mps,
        3,
        0.001,
    )
    task.speed_mps = sample_value(
        config.task.speed_mps,
        randomization.speed_range_mps,
        3,
        0.001,
    )
    task.retreat_speed_mps = sample_value(
        config.task.retreat_speed_mps,
        randomization.retreat_speed_range_mps,
        3,
        0.001,
    )

    if sampled.workpiece.kind == "trihedral_corner":
        size_fields = (
            ("trihedral_floor_size_m", 2),
            ("trihedral_wall_x_size_m", 0),
            ("trihedral_wall_y_size_m", 1),
        )
        for name, thickness_axis in size_fields:
            nominal = getattr(config.workpiece, name)
            dimensions = [
                value
                if axis == thickness_axis
                else sample_value(value, randomization.trihedral_size_range_m, 3, 0.08)
                for axis, value in enumerate(nominal)
            ]
            setattr(sampled.workpiece, name, dimensions)

    seam_limit = maximum_seam_length(sampled.workpiece, task.seam_id)
    if seam_limit is not None and randomization.seam_length_range_m > 0:
        task.seam_length_m = sample_value(
            config.task.seam_length_m,
            randomization.seam_length_range_m,
            3,
            0.03,
            seam_limit,
        )

    if randomization.reverse_probability > 0:
        task.direction = (
            "reverse" if rng.random() < randomization.reverse_probability else "forward"
        )
    if sampled.workpiece.kind == "pipe_on_plate":
        nominal_sweep = abs(config.task.arc_sweep_deg)
        task.arc_sweep_deg = sample_angle(
            nominal_sweep,
            randomization.arc_sweep_range_deg,
            1.0,
            360.0,
        )
        geometric_start = sample_angle(
            config.task.arc_start_deg,
            randomization.arc_start_range_deg,
        )
        task.arc_start_deg = round(
            geometric_start + task.arc_sweep_deg if task.direction == "reverse" else geometric_start
        )
    if sampled.workpiece.kind == "curve_plate":
        task.curve_amplitude_m = sample_value(
            config.task.curve_amplitude_m,
            randomization.curve_amplitude_range_m,
            3,
            0.005,
            0.4 * config.workpiece.curve_plate_size_m[0],
        )
        task.curve_frequency = sample_value(
            config.task.curve_frequency,
            randomization.curve_frequency_range,
            2,
            0.5,
            3.0,
        )
        task.curve_kind = "cosine" if rng.random() < randomization.cosine_probability else "sine"
    return sampled


def sample_episode_task_config(config: AppConfig, episode_index: int) -> AppConfig:
    """按 episode 分组确定性采样任务参数。

    Args:
        config: 只读的实验基准配置。
        episode_index: 数据集中的全局 episode 编号。

    Returns:
        同一组内完全一致、不同组间可复现变化的任务配置。
    """
    group = episode_index // config.randomization.task_group_size
    rng = np.random.default_rng(config.collection.seed + group)
    return sample_task_config(config, rng)


def stage_for_task(
    simulation: WeldingEnv,
    config: AppConfig,
) -> tuple[SeamPath, float]:
    """求解无碰撞预置位姿并返回焊缝几何。"""
    seam = simulation.active_seam()
    draft = ExpertTrajectory(config, simulation.tcp_pose(), seam)
    solution, residual = simulation.solve_ik(Pose(draft.above_pre, draft.welding_quaternion))
    if residual > 0.005:
        raise StagingPoseError(f"cannot solve collision-free staging pose: residual={residual:.6f}")

    simulation.mj_data.qpos[simulation.qpos_ids] = solution
    simulation.mj_data.qvel[:] = 0
    simulation.mj_data.ctrl[simulation.motor_ids] = solution
    mujoco.mj_forward(simulation.mj_model, simulation.mj_data)
    if simulation.collision:
        raise StagingPoseError(f"staging pose collides: {simulation.collision_pairs}")
    return seam, residual


def sample_collision_free_task(
    simulation: WeldingEnv,
    config: AppConfig,
    rng: np.random.Generator,
) -> tuple[SeamPath, float, int]:
    """确定性重采样，直到工件和预置位姿均可行。"""
    last_error: StagingPoseError | None = None
    for attempt in range(1, config.randomization.max_sampling_attempts + 1):
        if attempt > 1:
            simulation.reset()
        simulation.randomize_workpiece(rng)
        try:
            seam, residual = stage_for_task(simulation, config)
        except StagingPoseError as error:
            last_error = error
            continue
        return seam, residual, attempt
    raise RuntimeError(
        "cannot sample a reachable, collision-free task after "
        f"{config.randomization.max_sampling_attempts} attempts: {last_error}"
    ) from last_error


def preflight_trajectory(
    simulation: WeldingEnv,
    expert: ExpertTrajectory,
) -> tuple[np.ndarray, float]:
    """使用上一帧关节解预检完整参考轨迹。

    逐帧连续 IK 可提前发现奇异点、关节跳解、限位和静态碰撞，避免完成
    数千帧渲染后才把 episode 判为失败。

    Args:
        simulation: 已处于随机初始状态的焊接环境。
        expert: 待验证的 TCP 参考轨迹。

    Returns:
        每个参考帧的关节解，以及整条轨迹的最大 IK 残差。

    Raises:
        TrajectoryPlanningError: 任一参考帧不可达、不连续、越限或碰撞。
    """
    seed = simulation.mj_data.qpos[simulation.qpos_ids].copy()
    margin = simulation.config.safety.joint_position_margin_rad
    lower = simulation.joint_ranges[:, 0] + margin
    upper = simulation.joint_ranges[:, 1] - margin
    max_delta = simulation.config.robot.joint_velocity_limit / simulation.config.timing.policy_hz
    solutions: list[np.ndarray] = []
    max_residual = 0.0

    for index, frame in enumerate(expert.frames):
        solution, residual = simulation.solve_ik(frame.pose, seed)
        max_residual = max(max_residual, residual)
        if residual > 0.005:
            raise TrajectoryPlanningError(f"frame {index}: IK residual {residual:.6f} m")
        if np.any(solution < lower) or np.any(solution > upper):
            raise TrajectoryPlanningError(f"frame {index}: joint position margin exceeded")
        if np.max(np.abs(solution - seed)) > max_delta:
            raise TrajectoryPlanningError(f"frame {index}: discontinuous joint solution")
        simulation.ik_data.qvel[:] = 0
        mujoco.mj_forward(simulation.mj_model, simulation.ik_data)
        pairs = simulation.collision_pairs_for(simulation.ik_data)
        if pairs:
            raise TrajectoryPlanningError(f"frame {index}: collision {pairs}")
        solutions.append(solution)
        seed = solution
    return np.asarray(solutions), max_residual


def sample_feasible_trajectory(
    simulation: WeldingEnv,
    config: AppConfig,
    rng: np.random.Generator,
) -> TrajectorySample:
    """重采样场景和初始状态，直到完整运动轨迹通过预检。

    Args:
        simulation: 与任务配置匹配的焊接环境。
        config: 当前 episode 已确定的任务配置。
        rng: episode 专用随机数生成器。

    Returns:
        可直接用于录制的完整轨迹样本。

    Raises:
        RuntimeError: 达到采样上限后仍没有可行运动轨迹。
    """
    last_error: RuntimeError | None = None
    for motion_attempt in range(1, config.randomization.max_sampling_attempts + 1):
        simulation.reset()
        try:
            seam, staging_residual, scene_attempts = sample_collision_free_task(
                simulation, config, rng
            )
            joint_offset, joint_attempts = simulation.randomize_joint_position(
                rng,
                config.randomization.joint_degs,
                config.randomization.max_sampling_attempts,
            )
            tcp_offset, tcp_attempts, tcp_applied = sample_initial_tcp_offset(
                simulation, config, rng
            )
            expert = ExpertTrajectory(config, simulation.tcp_pose(), seam)
            joint_trajectory, planning_residual = preflight_trajectory(simulation, expert)
        except RuntimeError as error:
            last_error = error
            continue
        return TrajectorySample(
            seam,
            expert,
            joint_trajectory,
            staging_residual,
            planning_residual,
            scene_attempts,
            motion_attempt,
            joint_offset,
            joint_attempts,
            tcp_offset,
            tcp_attempts,
            tcp_applied,
        )
    raise RuntimeError(
        "cannot sample a continuous collision-free trajectory after "
        f"{config.randomization.max_sampling_attempts} attempts: {last_error}"
    ) from last_error


def sample_initial_tcp_offset(
    simulation: WeldingEnv,
    config: AppConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, bool]:
    """采样可行的初始 TCP 平移扰动。"""
    for attempt in range(1, config.randomization.max_sampling_attempts + 1):
        offset = rng.uniform(
            -config.randomization.initial_tcp_m,
            config.randomization.initial_tcp_m,
            size=3,
        )
        if simulation.perturb_tcp(offset):
            return offset, attempt, True
    return np.zeros(3), config.randomization.max_sampling_attempts, False


__all__ = [
    "StagingPoseError",
    "TrajectoryPlanningError",
    "TrajectorySample",
    "sample_collision_free_task",
    "sample_episode_task_config",
    "sample_feasible_trajectory",
    "sample_initial_tcp_offset",
    "sample_task_config",
    "stage_for_task",
]
