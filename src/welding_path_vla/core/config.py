"""项目各运行环境共享的类型化配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class TimingConfig:
    physics_hz: int = 500
    control_hz: int = 100
    policy_hz: int = 20

    def validate(self) -> None:
        if self.physics_hz % self.control_hz or self.control_hz % self.policy_hz:
            raise ValueError("frequencies must divide exactly from physics to control to policy")

    @property
    def physics_steps_per_control(self) -> int:
        return self.physics_hz // self.control_hz

    @property
    def controls_per_policy(self) -> int:
        return self.control_hz // self.policy_hz


@dataclass(slots=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    global_name: str = "global"
    wrist_name: str = "wrist"
    global_fovy_deg: float = 55.0
    wrist_fovy_deg: float = 85.0
    offscreen_backend: str = "egl"
    wrist_position_link6_m: list[float] = field(default_factory=lambda: [0.0, -0.080, 0.134])
    wrist_target_link6_m: list[float] = field(
        default_factory=lambda: [0.057557, -0.025778, 0.389520]
    )
    wrist_up_link6: list[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])


@dataclass(slots=True)
class SceneConfig:
    table_center_m: list[float] = field(default_factory=lambda: [0.35, 0.0, 0.27])
    table_half_size_m: list[float] = field(default_factory=lambda: [0.55, 0.45, 0.02])
    robot_base_position_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.29])
    robot_base_yaw_deg: float = -90.0
    workpiece_position_m: list[float] = field(default_factory=lambda: [0.45, 0.0, 0.2925])
    global_camera_position_table_m: list[float] = field(default_factory=lambda: [0.90, 0.0, 0.78])
    global_camera_target_table_m: list[float] = field(default_factory=lambda: [0.10, 0.0, 0.0225])
    global_camera_up_table: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])


@dataclass(slots=True)
class RobotConfig:
    model_id: str = "elfin5pro_photo_layout_v6"
    model_asset: str = "elfin5/elfin5_welding.xml"
    initial_joint_deg: list[float] = field(
        default_factory=lambda: [90.9411, -72.1133, 22.0613, 45.5546, 128.4704, -51.3849]
    )
    joint_velocity_limit: float = 1.57
    ik_damping: float = 0.01
    ik_max_step: float = 0.08
    ik_tolerance: float = 0.0005
    ik_iterations: int = 80


@dataclass(slots=True)
class TaskConfig:
    instruction: str = "沿 L 形工件的直线角焊缝完成焊接轨迹。"
    speed_mps: float = 0.02
    work_angle_deg: float = 45.0
    travel_angle_deg: float = 10.0
    tool_roll_deg: float = -20.0
    approach_distance_m: float = 0.04
    staging_clearance_m: float = 0.18
    retreat_distance_m: float = 0.04
    seam_length_m: float = 0.20
    tcp_clearance_m: float = 0.0015


@dataclass(slots=True)
class RandomizationConfig:
    xy_m: float = 0.1
    z_m: float = 0.0
    yaw_deg: float = 30.0
    joint_degs: list[float] = field(default_factory=lambda: [60.0, 20.0, 20.0, 30.0, 60.0, 60.0])
    max_sampling_attempts: int = 10
    initial_tcp_m: float = 0.1
    recovery_probability: float = 0.25
    recovery_position_m: float = 0.005
    recovery_rotation_deg: float = 3.0


@dataclass(slots=True)
class CollectionConfig:
    dataset_root: str = "datasets/weldpath_raw_v1"
    episodes: int = 50
    max_attempt_multiplier: int = 3
    seed: int = 20260721
    video_codec: str = "avc1"
    headless: bool = True


@dataclass(slots=True)
class QualityConfig:
    minimum_progress: float = 0.98
    cross_track_mean_m: float = 0.001
    cross_track_p95_m: float = 0.002
    cross_track_max_m: float = 0.005
    orientation_p95_deg: float = 2.0
    orientation_max_deg: float = 5.0


@dataclass(slots=True)
class EvaluationConfig:
    pcr_min: float = 0.95
    direction_ratio_min: float = 0.90
    cte_rmse_m: float = 0.0015
    cte_p95_m: float = 0.002
    cte_max_m: float = 0.005
    orientation_p95_deg: float = 2.0
    speed_mape_max: float = 0.20
    jerk_ratio_max: float = 2.0
    jerk_min_sample_rate_hz: float = 80.0
    jerk_reference_floor_m_s3: float = 0.001
    joint_acceleration_limit_rad_s2: float = 10.0
    require_smoothness_for_success: bool = False


@dataclass(slots=True)
class RealRobotConfig:
    enabled: bool = False
    host: str | None = None
    port: int = 10003
    connect_timeout_s: float = 3.0
    command_timeout_s: float = 0.1
    control_mode: str = "servo_j"


@dataclass(slots=True)
class SafetyConfig:
    enabled: bool = True
    joint_position_margin_rad: float = 0.02
    joint_velocity_limit_rad_s: float = 1.57
    tcp_speed_limit_m_s: float = 0.10
    command_timeout_s: float = 0.10


@dataclass(slots=True)
class PolicyConfig:
    family: str = "smolvla"
    checkpoint: str | None = None
    device: str = "cuda"
    action_horizon: int = 10
    action_stride: int = 1
    action_source: str = "safe_command"
    include_current: bool = False


@dataclass(slots=True)
class TrainingConfig:
    dataset_repo_id: str | None = None
    dataset_root: str | None = "datasets/weldpath_lerobot_v1"
    output_dir: str = "outputs/train"
    batch_size: int = 16
    steps: int = 100_000


@dataclass(slots=True)
class DeploymentConfig:
    dry_run: bool = True
    log_dir: str = "outputs/deploy"


@dataclass(slots=True)
class AppConfig:
    timing: TimingConfig = field(default_factory=TimingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    real_robot: RealRobotConfig = field(default_factory=RealRobotConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        with Path(path).open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        if not isinstance(raw, dict):
            raise ValueError("configuration root must be a YAML mapping")
        config = cls(
            timing=TimingConfig(**raw.get("timing", {})),
            camera=CameraConfig(**raw.get("camera", {})),
            scene=SceneConfig(**raw.get("scene", {})),
            robot=RobotConfig(**raw.get("robot", {})),
            task=TaskConfig(**raw.get("task", {})),
            randomization=RandomizationConfig(**raw.get("randomization", {})),
            collection=CollectionConfig(**raw.get("collection", {})),
            quality=QualityConfig(**raw.get("quality", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            real_robot=RealRobotConfig(**raw.get("real_robot", {})),
            safety=SafetyConfig(**raw.get("safety", {})),
            policy=PolicyConfig(**raw.get("policy", {})),
            training=TrainingConfig(**raw.get("training", {})),
            deployment=DeploymentConfig(**raw.get("deployment", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        self.timing.validate()
        if len(self.robot.initial_joint_deg) != 6:
            raise ValueError("robot.initial_joint_deg must contain six values")
        if not 0 < self.camera.global_fovy_deg < 180 or not 0 < self.camera.wrist_fovy_deg < 180:
            raise ValueError("camera field of view must be in (0, 180)")
        if self.camera.offscreen_backend not in {"egl", "glfw", "osmesa"}:
            raise ValueError("camera.offscreen_backend must be egl, glfw, or osmesa")
        if any(len(value) != 3 for value in asdict(self.scene).values() if isinstance(value, list)):
            raise ValueError("scene vectors must contain three values")
        if not 0 <= self.randomization.recovery_probability <= 1:
            raise ValueError("recovery_probability must be in [0, 1]")
        if len(self.randomization.joint_degs) != 6 or any(
            value < 0 for value in self.randomization.joint_degs
        ):
            raise ValueError("randomization.joint_degs must contain six non-negative values")
        if self.randomization.max_sampling_attempts < 1:
            raise ValueError("randomization.max_sampling_attempts must be positive")
        if self.collection.episodes < 1 or self.collection.max_attempt_multiplier < 1:
            raise ValueError("collection counts must be positive")
        if not 0 <= self.evaluation.pcr_min <= 1:
            raise ValueError("evaluation.pcr_min must be in [0, 1]")
        if not 0 <= self.evaluation.direction_ratio_min <= 1:
            raise ValueError("evaluation.direction_ratio_min must be in [0, 1]")
        if (
            self.evaluation.jerk_min_sample_rate_hz <= 0
            or self.evaluation.jerk_reference_floor_m_s3 < 0
        ):
            raise ValueError("evaluation jerk sampling rate/floor is invalid")
        if self.real_robot.enabled and not self.real_robot.host:
            raise ValueError("real_robot.host is required when real_robot.enabled is true")
        if self.policy.action_horizon < 1 or self.policy.action_stride < 1:
            raise ValueError("policy action horizon and stride must be positive")
        if self.policy.action_source not in {"safe_command", "reference", "executed"}:
            raise ValueError("policy.action_source is not supported")
        if self.training.steps < 1:
            raise ValueError("policy horizon and training steps must be positive")
        minimum_clearance = 0.00025 + self.robot.ik_tolerance
        if self.task.tcp_clearance_m < minimum_clearance:
            raise ValueError("task.tcp_clearance_m must cover the wire radius and IK tolerance")
        if self.task.staging_clearance_m <= self.task.approach_distance_m:
            raise ValueError("task.staging_clearance_m must exceed approach_distance_m")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = Path("configs/default.yaml")
