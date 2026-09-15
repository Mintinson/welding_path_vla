"""固定评价场景清单的读取、重放和初态核验。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.simulation.task_sampling import TrajectorySample, sample_feasible_trajectory

SCENE_BANK_VERSION = 1


def scene_fingerprint(simulation: Any) -> str:
    """对初始关节、工件及渲染随机量取指纹，防止跨模型场景漂移。

    Args:
        simulation: 已完成可行性采样的 WeldingEnv。

    Returns:
        与策略无关的 SHA256 场景指纹。
    """
    model = simulation.mj_model
    arrays = (
        model.body_pos[simulation.workpiece_id],
        model.body_quat[simulation.workpiece_id],
        simulation.mj_data.qpos[simulation.qpos_ids],
        model.geom_rgba,
        model.mat_specular,
        model.mat_shininess,
        model.mat_reflectance,
        model.tex_data,
        model.light_diffuse,
        model.light_pos,
        model.light_dir,
        model.light_active,
    )
    digest = hashlib.sha256()
    for values in arrays:
        data = np.ascontiguousarray(values)
        digest.update(str(data.shape).encode())
        digest.update(data.tobytes())
    return digest.hexdigest()


def load_scene_bank(path: str | Path) -> dict[str, Any]:
    """加载完整的固定评价清单，拒绝尚未生成完的场景集。"""
    bank = json.loads(Path(path).read_text(encoding="utf-8"))
    if bank.get("version") != SCENE_BANK_VERSION or not bank.get("complete"):
        raise ValueError(f"incomplete or unsupported scene bank: {path}")
    episodes: dict[str, list[int]] = {}
    for scene in bank["scenes"]:
        episodes.setdefault(scene["task"], []).append(int(scene["episode"]))
    expected = bank["protocol"]["scenes_per_task"]
    if set(episodes) != set(bank["protocol"]["tasks"]) or any(
        sorted(indices) != list(range(expected)) for indices in episodes.values()
    ):
        counts = {task: len(indices) for task, indices in episodes.items()}
        raise ValueError(f"scene bank task counts do not match {expected}: {counts}")
    control = bank["protocol"].get("rollout_control")
    if control is not None:
        for scene in bank["scenes"]:
            deployment = scene["config"]["deployment"]
            if (
                scene.get("rollout_control") != control
                or deployment["max_target_increment_m"] != control["max_target_increment_m"]
                or deployment["max_ik_residual"] != control["max_ik_residual_m"]
                or (
                    "terminal_frame_hold_s" in control
                    and deployment["terminal_frame_hold_s"] != control["terminal_frame_hold_s"]
                )
            ):
                raise ValueError("scene bank rollout control differs between episodes")
    return bank


def config_for_scene(model_config: AppConfig, scene: dict[str, Any]) -> AppConfig:
    """使用清单的环境与评价配置，只从模型配置读取策略和输出位置。"""
    config = AppConfig.from_dict(scene["config"])
    config.policy = model_config.policy
    config.evaluation.report_detailed_metrics = model_config.evaluation.report_detailed_metrics
    config.deployment.output_root = model_config.deployment.output_root
    config.deployment.log_dir = model_config.deployment.log_dir
    config.deployment.auto_log_dir = model_config.deployment.auto_log_dir
    config.deployment.scene_bank_path = model_config.deployment.scene_bank_path
    config.deployment.inference_steps = model_config.deployment.inference_steps
    config.deployment.record_video = model_config.deployment.record_video
    config.deployment.run_all_tasks = False
    return config


def replay_scene(simulation: Any, scene: dict[str, Any]) -> TrajectorySample:
    """确定性重放专家可达场景，并验证状态和外观与预生成时一致。"""
    seed = int(scene["seed"])
    sample = sample_feasible_trajectory(simulation, simulation.config, np.random.default_rng(seed))
    actual = scene_fingerprint(simulation)
    if actual != scene["fingerprint"]:
        raise RuntimeError(
            f"scene fingerprint mismatch for {scene['task']} episode {scene['episode']}: "
            "generation and rollout environments differ"
        )
    return sample
