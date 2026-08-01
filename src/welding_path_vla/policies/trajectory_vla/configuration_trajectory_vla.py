# Copyright 2025 Hugging Face Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trajectory-VLA 的模型与训练配置。"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("trajectory_vla")
@dataclass
class TrajectoryVLAConfig(PreTrainedConfig):
    """本地 Trajectory-VLA 配置。

    默认值与 LeRobot 0.6 官方 SmolVLA 对齐，使官方 ``smolvla_base`` 权重可以作为
    初始化来源。所有结构参数均保留为公开字段，便于后续替换轨迹表示或动作专家。
    """

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    max_state_dim: int = 32
    max_action_dim: int = 32
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    empty_cameras: int = 0
    tokenizer_max_length: int = 48
    pad_language_to: str = "max_length"

    num_steps: int = 10
    use_cache: bool = True
    flow_beta_alpha: float = 1.5
    flow_beta_beta: float = 1.0
    flow_time_scale: float = 0.999
    flow_time_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = True
    add_image_special_tokens: bool = False
    attention_mode: str = "cross_attn"
    prefix_length: int = 0
    num_expert_layers: int = 0
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75

    compile_model: bool = False
    compile_mode: str = "max-autotune"

    def __post_init__(self) -> None:
        """检查动作块、网络宽度和 attention 配置。"""
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        if not 0 < self.expert_width_multiplier <= 1:
            raise ValueError("expert_width_multiplier must be in (0, 1]")
        if self.attention_mode not in {"self_attn", "cross_attn"}:
            raise ValueError("attention_mode must be self_attn or cross_attn")
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")

    def validate_features(self) -> None:
        """补齐可选空相机，并确认当前任务具有视觉和动作 feature。"""
        if self.input_features is None:
            self.input_features = {}
        for index in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{index}"
            self.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
        if not self.image_features:
            raise ValueError("Trajectory-VLA requires at least one image feature")
        if self.action_feature is None:
            raise ValueError("Trajectory-VLA requires an action feature")

    def get_optimizer_preset(self) -> AdamWConfig:
        """返回与官方 SmolVLA 对齐的 AdamW 配置。"""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """返回 warmup + cosine decay 调度器。"""
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
        """读取完整 future trajectory chunk。"""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """该行为克隆策略不读取 reward。"""
        return None


__all__ = ["TrajectoryVLAConfig"]
