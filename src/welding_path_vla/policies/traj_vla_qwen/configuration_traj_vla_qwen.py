"""Prismatic-Qwen Trajectory-VLA 的模型与训练配置。"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("traj_vla_qwen")
@dataclass
class TrajVLAQwenConfig(PreTrainedConfig):
    """逐层交织式 Prismatic-Qwen 策略配置。

    版本相关参数集中在 ``language_model_family`` 与 ``language_model_name``；
    动作专家只依赖统一 decoder adapter，后续增加 Qwen3 时不需要修改
    Flow Matching、LeRobot Policy 或数据处理代码。
    """

    n_obs_steps: int = 1
    chunk_size: int = 30
    n_action_steps: int = 8
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    max_state_dim: int = 32
    max_action_dim: int = 32
    resize_imgs_with_padding: tuple[int, int] = (224, 224)
    empty_cameras: int = 0
    tokenizer_max_length: int = 160
    pad_language_to: str = "max_length"

    num_steps: int = 10
    use_cache: bool = True
    flow_beta_alpha: float = 1.5
    flow_beta_beta: float = 1.0
    flow_time_scale: float = 0.999
    flow_time_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    language_model_family: str = "qwen2_5"
    language_model_name: str = "Qwen/Qwen2.5-0.5B"
    prismatic_repo_id: str = "Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b"
    prismatic_checkpoint_file: str = "checkpoints/step-020792-epoch-01-loss=0.5268.pt"
    load_prismatic_weights: bool = True
    load_base_weights: bool = True
    num_extra_tokens: int = 256

    dino_model_name: str = "vit_large_patch14_reg4_dinov2.lvd142m"
    siglip_model_name: str = "vit_so400m_patch14_siglip_224"
    vision_patch_grid: int = 16
    token_merge_factor: int = 2
    add_image_special_tokens: bool = False
    prefix_length: int = 0

    num_vlm_layers: int = 16
    num_expert_layers: int = 16
    expert_width_multiplier: float = 0.75
    # self_attn 保留第一版逐层联合注意力；cross_attn 使用周期性 SA / CA。
    attention_mode: str = "self_attn"
    self_attn_every_n_layers: int = 2
    use_geometry_branch: bool = False
    geometry_num_queries: int = 16
    geometry_num_heads: int = 8
    use_geometry_grounding: bool = False
    geometry_corridor_radius_px: float = 8.0
    geometry_aux_loss_weights: tuple[float, float, float] = (0.1, 0.05, 0.05)
    geometry_camera_keys: tuple[str, str] = (
        "observation.images.global",
        "observation.images.wrist",
    )
    geometry_camera_fovy_deg: tuple[float, float] = (55.0, 85.0)
    # 当前仿真标定的 world_from_global_camera 与 tcp_from_wrist_camera 位姿。
    geometry_global_camera_pose_world: tuple[float, ...] = (
        1.25,
        0.0,
        1.05,
        0.64952979,
        0.27948354,
        0.27948354,
        0.64952979,
    )
    geometry_wrist_camera_pose_tcp: tuple[float, ...] = (
        -0.12672863,
        -0.07579556,
        0.17303227,
        0.87218524,
        0.05562066,
        -0.34143130,
        0.34586690,
    )

    use_motion_latent: bool = False
    motion_latent_dim: int = 16
    motion_kl_weight: float = 1e-3
    training_stage: str = "standard"

    train_vision_encoder: bool = False
    train_token_merger: bool = True
    train_projector: bool = True
    train_geometry_resampler: bool = True
    train_language_model: bool = False
    train_language_last_n_layers: int = 0
    train_expert: bool = True
    train_state_proj: bool = True

    frozen_vision_dtype: str = "float32"
    lora_target: str = "expert"
    gradient_checkpointing_qwen: bool = False
    gradient_checkpointing_expert: bool = False

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    compile_model: bool = False
    compile_mode: str = "max-autotune"

    def __post_init__(self) -> None:
        """验证层配对、视觉网格和训练范围。"""
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        if self.language_model_family != "qwen2_5":
            raise ValueError(
                "only qwen2_5 is implemented; Qwen3 uses the reserved adapter interface"
            )
        if not 0 < self.expert_width_multiplier <= 1:
            raise ValueError("expert_width_multiplier must be in (0, 1]")
        if self.num_vlm_layers < 1 or self.num_expert_layers < 1:
            raise ValueError("Qwen and expert layer counts must be positive")
        if self.num_vlm_layers != self.num_expert_layers:
            raise ValueError("the first paired-layer version requires equal Qwen and expert depths")
        if self.attention_mode not in {"self_attn", "cross_attn"}:
            raise ValueError("attention_mode must be self_attn or cross_attn")
        if self.frozen_vision_dtype not in {"float32", "bfloat16"}:
            raise ValueError("frozen_vision_dtype must be float32 or bfloat16")
        if self.lora_target not in {"expert", "qwen", "all"}:
            raise ValueError("lora_target must be expert, qwen or all")
        if self.self_attn_every_n_layers < 1:
            raise ValueError("self_attn_every_n_layers must be positive")
        if self.use_geometry_branch and self.attention_mode != "cross_attn":
            raise ValueError("the geometry branch requires attention_mode=cross_attn")
        if self.use_geometry_branch and self.self_attn_every_n_layers == 1:
            raise ValueError("the geometry branch requires at least one cross-attention layer")
        if self.use_geometry_branch and self.num_vlm_layers < 2:
            raise ValueError("the geometry branch requires at least two paired layers")
        if self.geometry_num_queries < 1 or self.geometry_num_heads < 1:
            raise ValueError("geometry query and head counts must be positive")
        if self.use_geometry_grounding and not self.use_geometry_branch:
            raise ValueError("geometry grounding requires use_geometry_branch=true")
        if self.use_motion_latent and not self.use_geometry_branch:
            raise ValueError("motion latent requires use_geometry_branch=true")
        if self.training_stage not in {"standard", "grounding_warmup", "policy_joint"}:
            raise ValueError("training_stage must be standard, grounding_warmup or policy_joint")
        if self.training_stage == "grounding_warmup" and not self.use_geometry_grounding:
            raise ValueError("grounding_warmup requires use_geometry_grounding=true")
        if self.training_stage == "grounding_warmup" and self.use_motion_latent:
            raise ValueError("grounding_warmup does not train motion latent")
        if self.training_stage == "grounding_warmup" and not self.train_geometry_resampler:
            raise ValueError("grounding_warmup requires train_geometry_resampler=true")
        if self.training_stage == "policy_joint" and not self.use_geometry_grounding:
            raise ValueError("policy_joint requires use_geometry_grounding=true")
        if self.geometry_corridor_radius_px <= 0 or self.motion_latent_dim < 1:
            raise ValueError("grounding radius and motion latent dimension must be positive")
        if len(self.geometry_aux_loss_weights) != 3 or min(self.geometry_aux_loss_weights) < 0:
            raise ValueError("geometry_aux_loss_weights must contain three non-negative values")
        if len(self.geometry_camera_keys) != 2 or len(self.geometry_camera_fovy_deg) != 2:
            raise ValueError("geometry grounding currently requires global and wrist cameras")
        if any(not 0 < fovy < 180 for fovy in self.geometry_camera_fovy_deg):
            raise ValueError("geometry camera vertical FOV must be in (0, 180)")
        if len(self.geometry_global_camera_pose_world) != 7:
            raise ValueError("global camera pose must contain xyz and wxyz")
        if len(self.geometry_wrist_camera_pose_tcp) != 7:
            raise ValueError("wrist camera pose must contain xyz and wxyz")
        if self.motion_kl_weight < 0:
            raise ValueError("motion_kl_weight must be non-negative")
        if self.vision_patch_grid % self.token_merge_factor:
            raise ValueError("token_merge_factor must divide vision_patch_grid")
        if not 0 <= self.train_language_last_n_layers <= self.num_vlm_layers:
            raise ValueError("train_language_last_n_layers is outside the retained Qwen depth")
        if self.train_vision_encoder and self.frozen_vision_dtype != "float32":
            raise ValueError("frozen_vision_dtype only applies to a frozen vision encoder")

    def validate_features(self) -> None:
        """补齐空相机并确认视觉与动作 feature 存在。"""
        if self.input_features is None:
            self.input_features = {}
        for index in range(self.empty_cameras):
            self.input_features[f"{OBS_IMAGES}.empty_camera_{index}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
        if not self.image_features:
            raise ValueError("TrajVLA-Qwen requires at least one image feature")
        if self.action_feature is None:
            raise ValueError("TrajVLA-Qwen requires an action feature")

    def get_optimizer_preset(self) -> AdamWConfig:
        """返回动作专家训练使用的 AdamW 配置。"""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """返回 warmup 与余弦衰减调度器。"""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        """只读取当前观测。"""
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        """读取完整未来动作块。"""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """行为克隆策略不读取 reward。"""
        return None


__all__ = ["TrajVLAQwenConfig"]
