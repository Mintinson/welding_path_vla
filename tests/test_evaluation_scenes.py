"""场景清单必须完整，并覆盖模型专属的评价配置。"""

import json
import math
from pathlib import Path

import pytest

from welding_path_vla.core.config import AppConfig
from welding_path_vla.simulation.evaluation_scenes import config_for_scene, load_scene_bank


def minimal_scene(task_id: str, episode: int) -> dict:
    """构造只用于 schema 校验的最小固定场景。"""
    return {
        "task": {"task_id": task_id},
        "episode": episode,
        "scene": {},
        "robot": {},
        "workpiece": {},
    }


def test_scene_bank_rejects_duplicate_scenes(tmp_path: Path) -> None:
    path = tmp_path / "bank.json"
    bank = {
        "version": 2,
        "complete": True,
        "scenes": [minimal_scene("l_joint", 0)],
    }
    path.write_text(json.dumps(bank), encoding="utf-8")
    assert len(load_scene_bank(path)["scenes"]) == 1
    bank["scenes"].append(minimal_scene("l_joint", 0))
    path.write_text(json.dumps(bank), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate scene"):
        load_scene_bank(path)


def test_scene_config_keeps_deployment_from_model_and_loads_scene_state() -> None:
    reference = AppConfig()
    reference.task.task_id = "l_joint"
    model = AppConfig()
    model.policy.family = "pi0"
    model.policy.checkpoint = "model-checkpoint"
    model.deployment.max_steps = 20
    model.deployment.output_root = "outputs/compare"
    model.deployment.scene_bank_path = "configs/bank.json"
    model.camera.render_gpu_device_id = 1
    reference_values = reference.as_dict()
    scene = {
        "task": reference_values["task"],
        "episode": 0,
        "scene": {
            "layout": reference_values["scene"],
            "camera": {"global_fovy_deg": 61.0},
            "lighting": {},
            "background": {},
        },
        "robot": {
            "joint_position_rad": [
                math.radians(value) for value in reference.robot.initial_joint_deg
            ],
            "joint_velocity_rad_s": [0.0] * 6,
        },
        "workpiece": {
            "geometry": reference_values["workpiece"],
            "pose": {"position_m": reference_values["scene"]["workpiece_position_m"],
                     "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "surface": {},
        },
    }
    replay = config_for_scene(model, scene)
    assert replay.policy.checkpoint == "model-checkpoint"
    assert replay.deployment.max_steps == 20
    assert replay.deployment.output_root == "outputs/compare"
    assert replay.camera.render_gpu_device_id == 1
    assert replay.camera.global_fovy_deg == 61.0
    assert replay.task.task_id == "l_joint"


def test_scene_bank_rejects_old_embedded_config(tmp_path: Path) -> None:
    bank = {
        "version": 1,
        "complete": True,
        "scenes": [{"task": "l_joint", "episode": 0, "config": {}}],
    }
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or unsupported"):
        load_scene_bank(path)
