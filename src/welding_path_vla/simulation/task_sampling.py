"""仿真与策略部署共享的轻量任务采样。"""

from __future__ import annotations

import mujoco
import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.domain import Pose
from welding_path_vla.simulation import ExpertTrajectory, WeldingEnv
from welding_path_vla.simulation.tasks import SeamPath


class StagingPoseError(RuntimeError):
    """IK 求解失败或预置位姿发生碰撞。"""


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
    "sample_collision_free_task",
    "sample_initial_tcp_offset",
    "stage_for_task",
]
