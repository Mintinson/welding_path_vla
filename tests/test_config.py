from pathlib import Path

import pytest

from welding_path_vla.core.config import AppConfig, TimingConfig


def test_default_config_is_valid() -> None:
    config = AppConfig.load(Path("configs/default.yaml"))
    assert config.as_dict() == AppConfig().as_dict()
    assert config.timing.physics_steps_per_control == 5
    assert config.timing.controls_per_policy == 4
    assert config.timing.policy_hz >= 30
    assert config.camera.width == 640
    assert config.camera.offscreen_backend == "egl"
    assert config.scene.robot_base_yaw_deg == -90.0
    assert config.robot.initial_joint_deg[0] == 90.9411
    assert config.workpiece.kind == "l_joint"
    assert config.task.seam_id == "straight_fillet"
    assert len(config.randomization.joint_degs) == 6
    assert all(value > 0 for value in config.randomization.joint_degs)
    assert config.policy.family == "smolvla"
    assert config.policy.action_source == "safe_command"
    assert not config.real_robot.enabled


@pytest.mark.parametrize(
    ("path", "seam_id"),
    [
        ("configs/pipe_bottom.yaml", "pipe_bottom"),
        ("configs/pipe_top.yaml", "pipe_top"),
    ],
)
def test_pipe_configs_select_matching_workpiece_and_seam(path: str, seam_id: str) -> None:
    """圆管配置应覆盖必要字段，并继承其余统一默认值。"""
    config = AppConfig.load(path)
    assert config.workpiece.kind == "pipe_on_plate"
    assert config.task.seam_id == seam_id
    assert config.robot.model_asset == "elfin5/elfin5pro_robot.xml"
    assert config.task.approach_speed_mps > config.task.speed_mps
    if seam_id == "pipe_top":
        assert config.task.arc_sweep_deg == 360
        assert config.task.orientation_follow_ratio == 0


def test_frequency_layers_must_be_integral() -> None:
    with pytest.raises(ValueError, match="frequencies"):
        TimingConfig(physics_hz=500, control_hz=120, policy_hz=20).validate()
