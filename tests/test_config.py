from pathlib import Path

import pytest

from welding_path_vla.config import AppConfig, TimingConfig


def test_default_config_is_valid() -> None:
    config = AppConfig.load(Path("configs/default.yaml"))
    assert config.timing.physics_steps_per_control == 5
    assert config.timing.controls_per_policy == 5
    assert config.camera.width == 640
    assert config.camera.offscreen_backend == "egl"
    assert config.scene.robot_base_yaw_deg == -90.0
    assert config.robot.initial_joint_deg[0] == 90.9411
    assert len(config.randomization.joint_degs) == 6
    assert all(value > 0 for value in config.randomization.joint_degs)
    assert config.policy.family == "smolvla"
    assert config.policy.action_source == "safe_command"
    assert not config.real_robot.enabled


def test_frequency_layers_must_be_integral() -> None:
    with pytest.raises(ValueError, match="frequencies"):
        TimingConfig(physics_hz=500, control_hz=120, policy_hz=20).validate()
