"""场景清单必须完整，并覆盖模型专属的评价配置。"""

import json
from pathlib import Path

import pytest

from welding_path_vla.core.config import AppConfig
from welding_path_vla.simulation.evaluation_scenes import config_for_scene, load_scene_bank


def test_scene_bank_rejects_partial_task_counts(tmp_path: Path) -> None:
    path = tmp_path / "bank.json"
    bank = {
        "version": 1,
        "complete": True,
        "protocol": {"tasks": ["l_joint"], "scenes_per_task": 2, "action_steps": 8},
        "scenes": [{"task": "l_joint", "episode": 0}],
    }
    path.write_text(json.dumps(bank), encoding="utf-8")
    with pytest.raises(ValueError, match="task counts"):
        load_scene_bank(path)
    bank["scenes"].append({"task": "l_joint", "episode": 1})
    path.write_text(json.dumps(bank), encoding="utf-8")
    assert len(load_scene_bank(path)["scenes"]) == 2


def test_scene_config_keeps_model_checkpoint_but_freezes_protocol() -> None:
    reference = AppConfig()
    reference.task.task_id = "l_joint"
    reference.deployment.max_steps = 1200
    reference.evaluation.coverage_tolerance_m = 0.005
    model = AppConfig()
    model.policy.family = "pi0"
    model.policy.checkpoint = "model-checkpoint"
    model.deployment.max_steps = 20
    model.deployment.output_root = "outputs/compare"
    model.deployment.scene_bank_path = "configs/bank.json"
    scene = {"task": "l_joint", "episode": 0, "config": reference.as_dict()}
    replay = config_for_scene(model, scene)
    assert replay.policy.checkpoint == "model-checkpoint"
    assert replay.deployment.max_steps == 1200
    assert replay.deployment.output_root == "outputs/compare"
    assert replay.evaluation.coverage_tolerance_m == 0.005


def test_scene_bank_rejects_mixed_rollout_controls(tmp_path: Path) -> None:
    config = AppConfig()
    config.deployment.max_target_increment_m = 0.025
    config.deployment.max_ik_residual = 0.010
    control = {
        "max_target_increment_m": 0.025,
        "max_ik_residual_m": 0.010,
        "joint_position_margin_rad": 0.010,
        "replan_after_clip": True,
    }
    scene = {
        "task": "l_joint", "episode": 0, "config": config.as_dict(),
        "rollout_control": {**control, "max_ik_residual_m": 0.005},
    }
    bank = {
        "version": 1, "complete": True,
        "protocol": {"tasks": ["l_joint"], "scenes_per_task": 1,
                     "rollout_control": control},
        "scenes": [scene],
    }
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank), encoding="utf-8")
    with pytest.raises(ValueError, match="rollout control"):
        load_scene_bank(path)
