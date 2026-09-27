"""VLA-Adapter 的模型与训练配置。"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("vla_adapter")
@dataclass
class VLAAdapterConfig(PreTrainedConfig):
    """Prismatic-Qwen 与逐层 Bridge Policy 的统一配置。"""

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

    action_head: str = "l1"
    num_action_queries: int = 64
    num_bridge_layers: int = 24
    bridge_num_heads: int = 8

    resize_imgs_with_padding: tuple[int, int] = (224, 224)
    empty_cameras: int = 0
    tokenizer_max_length: int = 160
    pad_language_to: str = "max_length"

    language_model_name: str = "Qwen/Qwen2.5-0.5B"
    prismatic_repo_id: str = "Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b"
    prismatic_checkpoint_file: str = "checkpoints/step-020792-epoch-01-loss=0.5268.pt"
    load_prismatic_weights: bool = True
    load_base_weights: bool = True
    num_extra_tokens: int = 256
    dino_model_name: str = "vit_large_patch14_reg4_dinov2.lvd142m"
    siglip_model_name: str = "vit_so400m_patch14_siglip_224"

    train_vision_encoder: bool = False
    use_vision_lora: bool = True
    train_projector: bool = True
    train_language_model: bool = False
    train_language_last_n_layers: int = 0
    frozen_vision_dtype: str = "float32"
    gradient_checkpointing_qwen: bool = False
    # 仅为读取旧 checkpoint 保留；VLA-Adapter 不再执行 torch.compile。
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    use_amp: bool = True

    num_steps: int = 10
    flow_beta_alpha: float = 1.5
    flow_beta_beta: float = 1.0
    flow_time_scale: float = 0.999
    flow_time_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        """验证动作头、Bridge 深度和 Flow 参数。"""
        super().__post_init__()
        if self.action_head not in {"l1", "flow_matching"}:
            raise ValueError("action_head must be l1 or flow_matching")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        if min(self.num_action_queries, self.num_bridge_layers, self.bridge_num_heads) < 1:
            raise ValueError("query, bridge layer and head counts must be positive")
        if self.num_bridge_layers > 24:
            raise ValueError("Qwen2.5-0.5B provides at most 24 bridge-aligned layers")
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if self.frozen_vision_dtype not in {"float32", "bfloat16"}:
            raise ValueError("frozen_vision_dtype must be float32 or bfloat16")
        if not 0 <= self.train_language_last_n_layers <= self.num_bridge_layers:
            raise ValueError("train_language_last_n_layers is outside the retained Qwen depth")
        if self.train_vision_encoder and self.frozen_vision_dtype != "float32":
            raise ValueError("frozen_vision_dtype only applies to a frozen vision encoder")

    def validate_features(self) -> None:
        """补齐空相机，并确认图像、状态与动作 feature 均存在。"""
        if self.input_features is None:
            self.input_features = {}
        for index in range(self.empty_cameras):
            self.input_features[f"{OBS_IMAGES}.empty_camera_{index}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
        if not self.image_features:
            raise ValueError("VLA-Adapter requires at least one image feature")
        if self.robot_state_feature is None:
            raise ValueError("VLA-Adapter requires a robot state feature")
        if self.action_feature is None:
            raise ValueError("VLA-Adapter requires an action feature")

    def get_optimizer_preset(self) -> AdamWConfig:
        """返回动作 Bridge 微调使用的 AdamW 配置。"""
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
        """只读取当前时刻观测。"""
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        """读取完整未来动作块。"""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """行为克隆策略不读取 reward。"""
        return None


__all__ = ["VLAAdapterConfig"]
