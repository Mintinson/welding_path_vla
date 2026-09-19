"""固定评价场景清单的读取、重放和初态核验。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.simulation.expert_generator import ExpertTrajectory
from welding_path_vla.simulation.task_sampling import (
    TrajectorySample,
    preflight_trajectory,
)

SCENE_BANK_VERSION = 2


def load_scene_bank(path: str | Path) -> dict[str, Any]:
    """加载只包含固定场景状态和任务要求的 scene bank。"""
    bank = json.loads(Path(path).read_text(encoding="utf-8"))
    if bank.get("version") != SCENE_BANK_VERSION or not bank.get("complete"):
        raise ValueError(f"incomplete or unsupported scene bank: {path}")
    if not isinstance(bank.get("scenes"), list) or not bank["scenes"]:
        raise ValueError(f"scene bank has no scenes: {path}")
    episodes: dict[str, set[int]] = {}
    for scene in bank["scenes"]:
        task = scene.get("task", {})
        task_id = task.get("task_id") if isinstance(task, dict) else None
        if not task_id or not all(
            key in scene for key in ("scene", "robot", "workpiece")
        ):
            raise ValueError("each scene must contain scene, robot, workpiece and task state")
        episode = int(scene.get("episode", -1))
        if episode < 0:
            raise ValueError(f"invalid episode for task {task_id}")
        indices = episodes.setdefault(task_id, set())
        if episode in indices:
            raise ValueError(f"duplicate scene: {task_id} episode {episode}")
        indices.add(episode)
    return bank


def config_for_scene(model_config: AppConfig, scene: dict[str, Any]) -> AppConfig:
    """合并固定场景状态与用户配置，deployment 始终来自用户配置。"""
    values = model_config.as_dict()
    fixed_scene = scene["scene"]
    workpiece = scene["workpiece"]
    robot = scene["robot"]
    values["scene"] = fixed_scene["layout"]
    values["scene"]["workpiece_position_m"] = workpiece["pose"]["position_m"]
    values["camera"]["global_fovy_deg"] = fixed_scene["camera"]["global_fovy_deg"]
    values["workpiece"] = workpiece["geometry"]
    values["task"] = scene["task"]
    values["robot"]["initial_joint_deg"] = np.degrees(robot["joint_position_rad"]).tolist()
    return AppConfig.from_dict(values)


def scene_task_ids(config: AppConfig, bank: dict[str, Any]) -> list[str]:
    """返回 scene bank 中实际执行的任务 ID。"""
    available = sorted({scene["task"]["task_id"] for scene in bank["scenes"]})
    task_ids = config.deployment.include_tasks or available
    unknown = sorted(set(task_ids) - set(available))
    if unknown:
        raise ValueError(f"unknown task IDs in scene bank: {unknown}")
    return task_ids


def replay_scene(simulation: Any, scene: dict[str, Any]) -> TrajectorySample:
    """直接应用 JSON 中保存的状态，并为固定场景重建专家轨迹。"""
    simulation.reset()
    simulation.apply_scene_state(scene)
    seam = simulation.active_seam()
    expert = ExpertTrajectory(simulation.config, simulation.tcp_pose(), seam)
    joint_trajectory, planning_residual = preflight_trajectory(simulation, expert)
    return TrajectorySample(
        seam=seam,
        expert=expert,
        joint_trajectory=joint_trajectory,
        staging_residual_m=0.0,
        planning_max_ik_residual_m=planning_residual,
        scene_sampling_attempts=1,
        motion_sampling_attempts=1,
        initial_joint_offset_deg=np.zeros(6),
        joint_sampling_attempts=1,
        initial_tcp_offset_m=np.zeros(3),
        tcp_sampling_attempts=1,
        initial_tcp_offset_applied=True,
    )
