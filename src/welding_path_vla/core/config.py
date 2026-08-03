"""项目各运行环境共享的类型化配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from draccus.argparsing import parse

from welding_path_vla.core.config_files import materialized_config


@dataclass(slots=True)
class TimingConfig:
    physics_hz: int = 600
    control_hz: int = 120
    policy_hz: int = 30

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
    wrist_position_link6_m: list[float] = field(
        default_factory=lambda: [0.008607, -0.071891, 0.172212]
    )
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
    model_asset: str = "elfin5/elfin5pro_robot.xml"
    initial_joint_deg: list[float] = field(
        default_factory=lambda: [90.9411, -72.1133, 22.0613, 45.5546, 128.4704, -51.3849]
    )
    joint_velocity_limit: float = 1.57
    ik_damping: float = 0.01
    ik_max_step: float = 0.08
    ik_tolerance: float = 0.0005
    ik_iterations: int = 80


@dataclass(slots=True)
class WorkpieceConfig:
    """可替换工件的几何参数。

    Attributes:
        kind: 工件类型，当前支持 ``l_joint`` 和 ``pipe_on_plate``。
        l_joint_length_m: L 形工件沿焊缝方向的总长度。
        l_joint_width_m: L 形工件水平板宽度及竖板高度。
        l_joint_thickness_m: L 形工件两块钢板的厚度。
        pipe_plate_size_m: 圆管工件底板的长、宽、厚。
        pipe_outer_radius_m: 圆管外半径。
        pipe_wall_thickness_m: 圆管壁厚。
        pipe_height_m: 圆管从底板上表面起算的高度。
        pipe_segments: 用于近似空心圆管的环向分段数。
    """

    kind: str = "l_joint"
    l_joint_length_m: float = 0.30
    l_joint_width_m: float = 0.10
    l_joint_thickness_m: float = 0.005
    pipe_plate_size_m: list[float] = field(default_factory=lambda: [0.18, 0.18, 0.005])
    pipe_outer_radius_m: float = 0.05
    pipe_wall_thickness_m: float = 0.004
    pipe_height_m: float = 0.12
    pipe_segments: int = 32


@dataclass(slots=True)
class TaskConfig:
    """焊缝任务、分阶段速度和焊枪姿态参数。

    Attributes:
        instruction: 当前任务的自然语言描述。
        seam_id: 工件提供的焊缝标识。
        direction: 沿焊缝正向或反向执行。
        arc_start_deg: 圆弧起始角；直线任务忽略该字段。
        arc_sweep_deg: 圆弧扫描角；正负号表示方向。
        approach_speed_mps: 空中接近工件阶段的 TCP 参考速度。
        speed_mps: 沿焊缝执行阶段的目标速度，也是评价使用的期望速度。
        retreat_speed_mps: 完成焊接后远离工件的 TCP 参考速度。
        orientation_follow_ratio: 姿态跟随局部焊缝标架的比例，范围为 ``[0, 1]``。
        work_angle_deg: 焊枪相对工件法向的工作角。
        travel_angle_deg: 焊枪相对焊缝切向的行走角。
        tool_roll_deg: 焊枪绕自身轴线的滚转角。
        approach_distance_m: 焊缝起点外的预接近距离。
        retreat_distance_m: 焊缝终点外的退出距离。
        staging_clearance_m: 空中转移点高于焊缝的最小距离。
        tcp_clearance_m: TCP 相对理论焊缝中心的安全净空。
        seam_length_m: 直线焊缝长度。
    """

    instruction: str = "沿 L 形工件的直线角焊缝完成焊接轨迹。"
    seam_id: str = "straight_fillet"
    direction: str = "forward"
    arc_start_deg: float = -175.0
    arc_sweep_deg: float = 350.0
    approach_speed_mps: float = 0.06
    speed_mps: float = 0.02
    retreat_speed_mps: float = 0.04
    orientation_follow_ratio: float = 1.0
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
    """场景、机器人状态和焊接任务的随机化范围。

    Attributes:
        xy_m: 工件在桌面 XY 方向的最大平移。
        z_m: 工件沿 Z 方向的最大平移。
        yaw_deg: 工件偏航角的最大变化。
        joint_degs: 六个关节相对 staging 构型的最大变化。
        max_sampling_attempts: 不可达或碰撞时的最大重采样次数。
        initial_tcp_m: 初始 TCP 各轴最大平移扰动。
        recovery_probability: episode 中插入恢复扰动的概率。
        recovery_position_m: 恢复扰动的最大位置幅度。
        recovery_rotation_deg: 恢复扰动的最大姿态幅度。
        work_angle_range_deg: 工作角相对任务标称值的采样半径。
        travel_angle_range_deg: 行走角相对任务标称值的采样半径。
        tool_roll_range_deg: 工具滚转角相对任务标称值的采样半径。
        orientation_follow_range: 姿态跟随比例相对标称值的采样半径。
        arc_start_range_deg: 圆弧几何起点相对标称值的采样半径。
        arc_sweep_range_deg: 圆弧扫掠角相对标称值的采样半径。
        approach_speed_range_mps: 接近速度相对标称值的采样半径。
        speed_range_mps: 焊接速度相对标称值的采样半径。
        retreat_speed_range_mps: 退出速度相对标称值的采样半径。
        reverse_probability: 将任务采样为反向执行的概率。
        task_group_size: 连续多少个 episode 编号共享一组任务参数。
    """

    xy_m: float = 0.05
    z_m: float = 0.0
    yaw_deg: float = 15.0
    joint_degs: list[float] = field(default_factory=lambda: [30.0, 10.0, 10.0, 15.0, 25.0, 25.0])
    max_sampling_attempts: int = 10
    initial_tcp_m: float = 0.03
    recovery_probability: float = 0.25
    recovery_position_m: float = 0.003
    recovery_rotation_deg: float = 2.0
    work_angle_range_deg: float = 3.0
    travel_angle_range_deg: float = 3.0
    tool_roll_range_deg: float = 5.0
    orientation_follow_range: float = 0.05
    arc_start_range_deg: float = 0.0
    arc_sweep_range_deg: float = 0.0
    approach_speed_range_mps: float = 0.005
    speed_range_mps: float = 0.002
    retreat_speed_range_mps: float = 0.005
    reverse_probability: float = 0.5
    task_group_size: int = 10


@dataclass(slots=True)
class CollectionConfig:
    dataset_root: str = "datasets/weldpath_raw_v2"
    episodes: int = 50
    max_attempt_multiplier: int = 3
    seed: int = 20260721
    headless: bool = True
    workers: int = 1


@dataclass(slots=True)
class QualityConfig:
    minimum_progress: float = 0.98
    cross_track_mean_m: float = 0.001
    cross_track_p95_m: float = 0.002
    cross_track_max_m: float = 0.01
    orientation_p95_deg: float = 2.0
    orientation_max_deg: float = 5.0


@dataclass(slots=True)
class EvaluationConfig:
    pcr_min: float = 0.95
    direction_ratio_min: float = 0.90
    tracking_band_m: float = 0.01
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
    """机器人运行与焊接接触安全参数。

    Attributes:
        enabled: 是否启用实机安全监控。
        joint_position_margin_rad: 关节位置相对软限位的安全余量。
        joint_velocity_limit_rad_s: 允许的最大关节速度。
        tcp_speed_limit_m_s: 允许的最大 TCP 平移速度。
        command_timeout_s: 实机控制命令的超时时间。
        tip_contact_penetration_limit_m: 焊丝尖端与目标工件之间允许的瞬时
            数值穿透深度；位置伺服产生的接触力不用于判断这种浅接触。
        tip_contact_force_limit_n: 焊丝尖端接触工件时允许的合力上限；超过
            该值仍按碰撞处理，其他几何体之间的接触不使用此容差。
    """

    enabled: bool = True
    joint_position_margin_rad: float = 0.02
    joint_velocity_limit_rad_s: float = 1.57
    tcp_speed_limit_m_s: float = 0.20
    command_timeout_s: float = 0.10
    tip_contact_penetration_limit_m: float = 0.0005
    tip_contact_force_limit_n: float = 3.0


@dataclass(slots=True)
class PolicyConfig:
    """策略结构与动作轨迹配置。

    Attributes:
        action_representation: 统一动作定义；当前仅支持 `relative_action`。
        action_horizon: 每次预测的 future target 数量。
        action_steps: 一次预测后实际执行的动作数量。
        action_stride: chunk 中相邻 target 的数据帧间隔。
        action_source: 原始数据中用于监督的绝对目标来源。
    """

    family: str = "smolvla"
    checkpoint: str | None = None
    device: str = "cuda"
    action_horizon: int = 30
    action_steps: int = 8
    action_stride: int = 1
    action_representation: str = "relative_action"
    action_source: str = "safe_command"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingConfig:
    dataset_repo_id: str | None = None
    dataset_root: str | None = "datasets/weldpath_lerobot_relative_v1"
    output_dir: str = "outputs/train"
    batch_size: int = 16
    lr: float | None = None
    steps: int = 100_000
    num_workers: int = 4
    video_backend: str = "torchcodec"
    eval_split: float = 0.1
    eval_steps: int = 2_000
    max_eval_samples: int = 1_000
    log_freq: int = 100
    save_freq: int = 10_000
    seed: int = 20260724
    wandb: bool = False
    amp_dtype: str = "bfloat16"
    resume: bool = False
    peft: dict[str, Any] | None = None


@dataclass(slots=True)
class PolicyEvaluationConfig:
    """策略离线测试参数。"""

    batch_size: int = 4
    num_workers: int = 2
    max_batches: int = 50
    held_out_episodes: int = 9
    output_dir: str = "outputs/evaluation/policies"


@dataclass(slots=True)
class LeRobotExportConfig:
    """LeRobot 转换的选择、增量与编码参数。"""

    incremental: bool = False
    start_episode: int | None = None
    end_episode: int | None = None
    save_images: bool = False
    streaming_encoding: bool = True
    parallel_video_encoding: bool = True
    image_writer_processes: int = 0
    image_writer_threads: int = 8
    encoder_queue_maxsize: int = 64
    encoder_threads: int | None = None
    video_codec: str = "libsvtav1"
    video_quality: int = 30
    video_preset: str | int | None = "12"


@dataclass(slots=True)
class DeploymentConfig:
    dry_run: bool = True
    log_dir: str = "outputs/deploy"
    episodes: int = 5
    max_steps: int = 1_000
    seed: int = 20260724
    record_video: bool = True
    completion_progress_min: float = 0.95
    completion_distance_m: float = 0.01


@dataclass(slots=True)
class AppConfig:
    timing: TimingConfig = field(default_factory=TimingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    workpiece: WorkpieceConfig = field(default_factory=WorkpieceConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    real_robot: RealRobotConfig = field(default_factory=RealRobotConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    policy_evaluation: PolicyEvaluationConfig = field(default_factory=PolicyEvaluationConfig)
    lerobot_export: LeRobotExportConfig = field(default_factory=LeRobotExportConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    def __post_init__(self) -> None:
        """构造后立即检查跨模块配置约束。"""
        self.validate()

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        """组合 YAML 模块后通过 Draccus 完成类型解析。"""
        with materialized_config(path) as config_path:
            return parse(cls, config_path=config_path, args=[])

    def validate(self) -> None:
        self.timing.validate()
        if len(self.robot.initial_joint_deg) != 6:
            raise ValueError("robot.initial_joint_deg must contain six values")
        if self.workpiece.kind not in {"l_joint", "pipe_on_plate"}:
            raise ValueError("workpiece.kind must be l_joint or pipe_on_plate")
        if len(self.workpiece.pipe_plate_size_m) != 3:
            raise ValueError("workpiece.pipe_plate_size_m must contain three values")
        if self.workpiece.pipe_segments < 12:
            raise ValueError("workpiece.pipe_segments must be at least 12")
        if not 0 < self.workpiece.pipe_wall_thickness_m < self.workpiece.pipe_outer_radius_m:
            raise ValueError("pipe wall thickness must be smaller than the outer radius")
        if (
            self.workpiece.kind == "l_joint"
            and self.task.seam_length_m > self.workpiece.l_joint_length_m
        ):
            raise ValueError("task.seam_length_m cannot exceed the L-joint length")
        allowed_seams = {
            "l_joint": {"straight_fillet"},
            "pipe_on_plate": {"pipe_bottom", "pipe_top"},
        }
        if self.task.seam_id not in allowed_seams[self.workpiece.kind]:
            raise ValueError(
                f"task.seam_id={self.task.seam_id!r} is invalid for {self.workpiece.kind}"
            )
        if self.task.direction not in {"forward", "reverse"}:
            raise ValueError("task.direction must be forward or reverse")
        if not 0 < abs(self.task.arc_sweep_deg) <= 360:
            raise ValueError("task.arc_sweep_deg must be in [-360, 360] and non-zero")
        if (
            min(
                self.task.approach_speed_mps,
                self.task.speed_mps,
                self.task.retreat_speed_mps,
            )
            <= 0
        ):
            raise ValueError("task phase speeds must be positive")
        if not 0 <= self.task.orientation_follow_ratio <= 1:
            raise ValueError("task.orientation_follow_ratio must be in [0, 1]")
        if not 0 < self.camera.global_fovy_deg < 180 or not 0 < self.camera.wrist_fovy_deg < 180:
            raise ValueError("camera field of view must be in (0, 180)")
        if self.camera.offscreen_backend not in {"egl", "glfw", "osmesa"}:
            raise ValueError("camera.offscreen_backend must be egl, glfw, or osmesa")
        if any(len(value) != 3 for value in asdict(self.scene).values() if isinstance(value, list)):
            raise ValueError("scene vectors must contain three values")
        if not 0 <= self.randomization.recovery_probability <= 1:
            raise ValueError("recovery_probability must be in [0, 1]")
        if not 0 <= self.randomization.reverse_probability <= 1:
            raise ValueError("reverse_probability must be in [0, 1]")
        task_randomization_ranges = (
            self.randomization.work_angle_range_deg,
            self.randomization.travel_angle_range_deg,
            self.randomization.tool_roll_range_deg,
            self.randomization.orientation_follow_range,
            self.randomization.arc_start_range_deg,
            self.randomization.arc_sweep_range_deg,
            self.randomization.approach_speed_range_mps,
            self.randomization.speed_range_mps,
            self.randomization.retreat_speed_range_mps,
        )
        if any(value < 0 for value in task_randomization_ranges):
            raise ValueError("task randomization ranges must be non-negative")
        if len(self.randomization.joint_degs) != 6 or any(
            value < 0 for value in self.randomization.joint_degs
        ):
            raise ValueError("randomization.joint_degs must contain six non-negative values")
        if self.randomization.max_sampling_attempts < 1 or self.randomization.task_group_size < 1:
            raise ValueError("randomization sampling counts must be positive")
        if (
            self.collection.episodes < 1
            or self.collection.max_attempt_multiplier < 1
            or self.collection.workers < 1
        ):
            raise ValueError("collection counts must be positive")
        if not 0 <= self.evaluation.pcr_min <= 1:
            raise ValueError("evaluation.pcr_min must be in [0, 1]")
        if not 0 <= self.evaluation.direction_ratio_min <= 1:
            raise ValueError("evaluation.direction_ratio_min must be in [0, 1]")
        if self.evaluation.tracking_band_m <= 0:
            raise ValueError("evaluation.tracking_band_m must be positive")
        if (
            self.evaluation.jerk_min_sample_rate_hz <= 0
            or self.evaluation.jerk_reference_floor_m_s3 < 0
        ):
            raise ValueError("evaluation jerk sampling rate/floor is invalid")
        if self.real_robot.enabled and not self.real_robot.host:
            raise ValueError("real_robot.host is required when real_robot.enabled is true")
        if (
            min(
                self.safety.tip_contact_penetration_limit_m,
                self.safety.tip_contact_force_limit_n,
            )
            < 0
        ):
            raise ValueError("tip contact tolerances must be non-negative")
        if (
            self.policy.action_horizon < 1
            or self.policy.action_steps < 1
            or self.policy.action_stride < 1
            or self.policy.action_steps > self.policy.action_horizon
        ):
            raise ValueError("policy action horizon and stride must be positive")
        if self.policy.action_source not in {"safe_command", "reference", "executed"}:
            raise ValueError("policy.action_source is not supported")
        if self.policy.action_representation != "relative_action":
            raise ValueError("policy.action_representation currently only supports relative_action")
        if (
            self.training.steps < 1
            or self.training.batch_size < 1
            or self.training.num_workers < 0
            or not 0 <= self.training.eval_split < 1
        ):
            raise ValueError("policy horizon and training steps must be positive")
        if self.training.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("training.amp_dtype must be float16 or bfloat16")
        if (
            self.policy_evaluation.batch_size < 1
            or self.policy_evaluation.num_workers < 0
            or self.policy_evaluation.max_batches < 1
            or self.policy_evaluation.held_out_episodes < 1
        ):
            raise ValueError("policy evaluation counts must be positive")
        if self.deployment.episodes < 1 or self.deployment.max_steps < 1:
            raise ValueError("deployment counts must be positive")
        if not 0 <= self.deployment.completion_progress_min <= 1:
            raise ValueError("deployment.completion_progress_min must be in [0, 1]")
        if self.deployment.completion_distance_m <= 0:
            raise ValueError("deployment.completion_distance_m must be positive")
        export = self.lerobot_export
        if (
            export.start_episode is not None
            and export.end_episode is not None
            and export.start_episode > export.end_episode
        ):
            raise ValueError("lerobot_export episode range is invalid")
        if (
            min(
                export.image_writer_processes,
                export.image_writer_threads,
                export.encoder_queue_maxsize,
            )
            < 0
        ):
            raise ValueError("lerobot_export worker counts must be non-negative")
        if export.encoder_threads is not None and export.encoder_threads < 1:
            raise ValueError("lerobot_export.encoder_threads must be positive")
        minimum_clearance = 0.00025 + self.robot.ik_tolerance
        if self.task.tcp_clearance_m < minimum_clearance:
            raise ValueError("task.tcp_clearance_m must cover the wire radius and IK tolerance")
        if self.task.staging_clearance_m <= self.task.approach_distance_m:
            raise ValueError("task.staging_clearance_m must exceed approach_distance_m")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
