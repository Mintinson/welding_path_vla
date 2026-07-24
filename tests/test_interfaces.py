import numpy as np
import pytest

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.domain import Pose, RobotState
from welding_path_vla.evaluation.collision_metrics import collision_report
from welding_path_vla.policies.training import TrainingRequest
from welding_path_vla.robot.safety_monitor import SafetyMonitor, SafetyViolation


def test_safety_gate_rejects_joint_limit_violation() -> None:
    config = AppConfig()
    ranges = np.tile([-1.0, 1.0], (6, 1))
    monitor = SafetyMonitor(config.safety, ranges)
    state = RobotState(np.zeros(6), np.zeros(6), Pose(np.zeros(3), np.array([1, 0, 0, 0])))
    monitor.validate_state(state)
    monitor.validate_joint_command(np.zeros(6))
    with pytest.raises(SafetyViolation, match="safe range"):
        monitor.validate_joint_command(np.ones(6))


def test_collision_report_preserves_contact_pairs() -> None:
    report = collision_report(
        {
            "collision": np.array([False, True, True]),
            "collision_pairs": np.array(["", "torch_tip:plate_vertical", "torch_tip:table"]),
        }
    )
    assert report.collision_frames == 2
    assert report.collision_rate == pytest.approx(2 / 3)
    assert report.pairs == ("torch_tip:plate_vertical", "torch_tip:table")


def test_training_command_uses_unified_config() -> None:
    config = AppConfig()
    config.training.dataset_repo_id = "mintinson/weldpath_sim_v1"
    command = TrainingRequest(config.policy, config.training).command()
    assert "--policy.type=smolvla" in command
    assert "--policy.chunk_size=10" in command
    assert "--dataset.root=datasets/weldpath_lerobot_v1" in command
