from pathlib import Path

import pytest

from welding_path_vla.core.config import AppConfig, TimingConfig
from welding_path_vla.core.config_files import compose_config


def test_default_config_is_valid() -> None:
    config = AppConfig.load(Path("configs/default.yaml"))
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


def test_curve_plate_config_selects_fixed_orientation_curve_task() -> None:
    """曲线平板入口应同时选定工件、焊缝和固定姿态约束。"""
    config = AppConfig.load("configs/curve_plate.yaml")
    assert config.workpiece.kind == "curve_plate"
    assert config.task.seam_id == "curve_seam"
    assert config.task.curve_kind == "sine"
    assert config.task.orientation_follow_ratio == 0
    assert config.collection.dataset_root.endswith("weldpath_curve_plate_raw_v2")


def test_trihedral_config_selects_vertical_corner_task() -> None:
    """三面角入口应默认选择自下而上的竖直内角焊缝。"""
    config = AppConfig.load("configs/trihedral_vertical.yaml")
    assert config.workpiece.kind == "trihedral_corner"
    assert config.task.seam_id == "vertical_corner"
    assert config.task.direction == "forward"
    assert config.randomization.trihedral_size_range_m == 0.008
    assert config.collection.dataset_root.endswith("weldpath_trihedral_vertical_raw_v2")


def test_trihedral_horizontal_config_selects_continuous_pair() -> None:
    """水平任务应一次连续执行两条水平内角焊缝，并保留长度随机化。"""
    config = AppConfig.load("configs/trihedral_horizontal.yaml")
    assert config.workpiece.kind == "trihedral_corner"
    assert config.task.seam_id == "horizontal_pair"
    assert config.task.direction == "forward"
    assert config.randomization.seam_length_range_m == 0.010
    assert config.collection.dataset_root.endswith("weldpath_trihedral_horizontal_raw_v2")


@pytest.mark.parametrize(
    ("path", "seam_id", "max_steps"),
    [
        ("configs/deploy/smolvla_l_joint.yaml", "straight_fillet", 1000),
        ("configs/deploy/smolvla_pipe_bottom.yaml", "pipe_bottom", 1200),
        ("configs/deploy/smolvla_pipe_top.yaml", "pipe_top", 3300),
        ("configs/deploy/smolvla_curve_plate.yaml", "curve_seam", 1500),
        ("configs/deploy/smolvla_trihedral_horizontal.yaml", "horizontal_pair", 1500),
        ("configs/deploy/smolvla_trihedral_vertical.yaml", "vertical_corner", 1500),
    ],
)
def test_deployment_profiles_select_complete_task(
    path: str,
    seam_id: str,
    max_steps: int,
) -> None:
    """单个部署入口应同时选定模型、工件、焊缝和运行参数。"""
    config = AppConfig.load(path)
    assert config.policy.family == "smolvla"
    assert config.policy.checkpoint
    assert config.task.seam_id == seam_id
    assert config.deployment.max_steps == max_steps


def test_config_includes_merge_nested_sections_in_order(tmp_path: Path) -> None:
    """入口文件和后置模块应只覆盖自己声明的嵌套字段。"""
    (tmp_path / "base.yaml").write_text(
        "deployment:\n  episodes: 1\n  max_steps: 1000\n",
        encoding="utf-8",
    )
    (tmp_path / "task.yaml").write_text(
        "deployment:\n  max_steps: 3300\n",
        encoding="utf-8",
    )
    entry = tmp_path / "entry.yaml"
    entry.write_text(
        "includes: [base.yaml, task.yaml]\ndeployment:\n  episodes: 5\n",
        encoding="utf-8",
    )

    assert compose_config(entry)["deployment"] == {"episodes": 5, "max_steps": 3300}


def test_frequency_layers_must_be_integral() -> None:
    with pytest.raises(ValueError, match="frequencies"):
        TimingConfig(physics_hz=500, control_hz=120, policy_hz=20).validate()


@pytest.mark.parametrize(
    ("path", "steps"),
    [
        ("configs/act.yaml", 1_837_012),
        ("configs/smolvla.yaml", 229_627),
        ("configs/trajectory_vla.yaml", 229_627),
        ("configs/traj_vla_qwen.yaml", 918_506),
        ("configs/pi0.yaml", 3_674_023),
        ("configs/pi0_5.yaml", 3_674_023),
        ("configs/pi0_a100.yaml", 459_253),
        ("configs/pi0_5_a100.yaml", 459_253),
        ("configs/act_a100.yaml", 57_407),
        ("configs/smolvla_a100.yaml", 57_407),
        ("configs/trajectory_vla_a100.yaml", 57_407),
        ("configs/traj_vla_qwen_a100.yaml", 114_814),
    ],
)
def test_training_steps_cover_current_large_dataset_once(path: str, steps: int) -> None:
    """正式配置应近似覆盖 DataDiskD 当前训练帧一次。"""
    config = AppConfig.load(path)
    assert config.training.steps == steps
    assert config.training.wandb
    assert config.training.wandb_mode == "offline"


@pytest.mark.parametrize(
    ("path", "batch_size", "compiled"),
    [
        ("configs/act_a100.yaml", 32, False),
        ("configs/smolvla_a100.yaml", 32, True),
        ("configs/trajectory_vla_a100.yaml", 32, True),
        ("configs/traj_vla_qwen_a100.yaml", 16, True),
        ("configs/pi0_a100.yaml", 4, True),
        ("configs/pi0_5_a100.yaml", 4, True),
    ],
)
def test_a100_profiles_use_large_per_gpu_batches(
    path: str,
    batch_size: int,
    compiled: bool,
) -> None:
    """双 A100 配置应提高每卡利用率，并只启用模型支持的编译优化。"""
    config = AppConfig.load(path)
    assert config.training.batch_size == batch_size
    assert config.policy.parameters.get("compile_model", False) is compiled
