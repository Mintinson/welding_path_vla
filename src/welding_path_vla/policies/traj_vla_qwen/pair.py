"""PAIR 的动作教师与感知 Bridge；教师仅用于训练期表征监督。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


@dataclass
class ActionAEConfig:
    """轻量动作自编码器参数；horizon 与有效动作维数由数据契约决定。"""

    horizon: int = 30
    action_dim: int = 9
    hidden_dim: int = 256
    num_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 1
    mask_probability: float = 0.5
    noise_std: float = 0.05

    def __post_init__(self) -> None:
        """拒绝无法构造注意力或没有有效动作的配置。"""
        if (
            min(
                self.horizon,
                self.action_dim,
                self.hidden_dim,
                self.num_heads,
                self.encoder_layers,
                self.decoder_layers,
            )
            < 1
        ):
            raise ValueError("Action AE dimensions and layer counts must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("Action AE hidden_dim must be divisible by num_heads")
        if not 0 <= self.mask_probability < 1 or self.noise_std < 0:
            raise ValueError("Action AE requires mask_probability in [0,1) and noise_std >= 0")


def masked_time_mean(values: Tensor, padding: Tensor) -> Tensor:
    """按样本归约有效时间步；padding=True 表示 episode 末尾补齐。"""
    valid = ~padding
    return values.masked_fill(padding, 0).sum(-1) / valid.sum(-1).clamp_min(1)


class ActionEncoder(nn.Module):
    """把有效动作序列编码为一一对应的时间 latent，禁止 padding 参与注意力。"""

    def __init__(self, config: ActionAEConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(config.action_dim, config.hidden_dim)
        self.positions = nn.Parameter(torch.randn(config.horizon, config.hidden_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(config.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            config.hidden_dim,
            config.num_heads,
            config.hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(
            layer,
            config.encoder_layers,
            norm=nn.LayerNorm(config.hidden_dim),
            enable_nested_tensor=False,
        )

    def forward(self, actions: Tensor, padding: Tensor, masked: Tensor | None = None) -> Tensor:
        """编码 clean 或受扰动动作；mask token 与真实 padding 使用不同语义。"""
        hidden = self.input_projection(actions.to(self.input_projection.weight.dtype))
        if masked is not None:
            hidden = torch.where(masked[..., None], self.mask_token.to(hidden.dtype), hidden)
        hidden = hidden + self.positions.to(hidden.dtype)
        # 全 padding 样本临时开放一个零 token，最终仍屏蔽所有输出和损失。
        safe_padding = padding.clone()
        safe_padding[:, 0] &= ~padding.all(dim=1)
        hidden = hidden.masked_fill(padding[..., None], 0)
        return self.layers(hidden, src_key_padding_mask=safe_padding).masked_fill(
            padding[..., None], 0
        )


class ActionAutoencoder(nn.Module):
    """阶段一学习动作时序，阶段二冻结 encoder 并仅编码干净动作。"""

    def __init__(self, config: ActionAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ActionEncoder(config)
        layer = nn.TransformerEncoderLayer(
            config.hidden_dim,
            config.num_heads,
            config.hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer,
            config.decoder_layers,
            norm=nn.LayerNorm(config.hidden_dim),
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(self, actions: Tensor, padding: Tensor, corrupt: bool = True) -> Tensor:
        """重建归一化动作，遮蔽仅发生于有效时间步，不改变监督目标。"""
        masked = None
        inputs = actions
        if corrupt:
            masked = (
                torch.rand_like(padding, dtype=torch.float32) < self.config.mask_probability
            ) & ~padding
            inputs = actions + torch.randn_like(actions) * self.config.noise_std
        latent = self.encoder(inputs, padding, masked)
        safe_padding = padding.clone()
        safe_padding[:, 0] &= ~padding.all(dim=1)
        decoded = self.decoder(latent, src_key_padding_mask=safe_padding)
        return self.output_projection(decoded).masked_fill(padding[..., None], 0)

    def save(self, directory: Path, metadata: dict) -> None:
        """保存完整 AE、可移植教师和数据/归一化契约，不保存任何视觉权重。"""
        directory.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
            str(directory / "model.safetensors"),
        )
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.encoder.state_dict().items()},
            str(directory / "encoder.safetensors"),
        )
        (directory / "config.json").write_text(
            json.dumps({"model": asdict(self.config), **metadata}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class ActionTeacher(nn.Module):
    """冻结且可随 PEFT 保存的动作编码器；forward 始终停止梯度。"""

    def __init__(self, config: ActionAEConfig) -> None:
        super().__init__()
        self.encoder = ActionEncoder(config)
        self.requires_grad_(False)

    def load_encoder(self, directory: str) -> None:
        """只读取阶段一 encoder，阶段二不需要动作重建 decoder。"""
        self.encoder.load_state_dict(load_file(str(Path(directory) / "encoder.safetensors")))

    @torch.no_grad()
    def forward(self, actions: Tensor, padding: Tensor) -> Tensor:
        """提供干净动作的固定监督目标；不接触策略的带噪动作或推理输入。"""
        self.eval()
        # 与阶段一 FP32 编码保持一致，不让主策略的 BF16 autocast 改变教师目标。
        with torch.autocast(device_type=actions.device.type, enabled=False):
            return self.encoder(actions.float(), padding).detach()


class ActionBridge(nn.Module):
    """由当前 Context 产生 horizon 个动作条件，保留主策略的所有视觉路径。"""

    def __init__(
        self,
        context_dim: int,
        expert_dim: int,
        horizon: int,
        hidden_dim: int,
        num_heads: int,
        target_dim: int,
    ) -> None:
        super().__init__()
        self.context_norm = nn.LayerNorm(context_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(horizon, hidden_dim) * 0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.self_attention = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.alignment = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, target_dim)
        )
        self.injection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, expert_dim)
        )
        self.gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        # 先由独立对齐损失训练 Bridge；零门控避免初始化时改变动作输入。
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self, context: Tensor, valid: Tensor, align: bool = True
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """每次观测计算一次注入残差；padding 和缺失相机不参与注意力。"""
        context = self.context_norm(context.to(self.context_norm.weight.dtype))
        memory = self.context_projection(context)
        queries = self.queries[None].expand(context.shape[0], -1, -1).to(memory.dtype)
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            memory,
            memory,
            key_padding_mask=~valid,
            need_weights=False,
        )
        bridge = queries + attended
        bridge = self.self_attention(bridge + self.cross_mlp(bridge))
        gate = self.gate(bridge).tanh()
        auxiliary = {"pair.prediction": self.alignment(bridge), "pair.gate": gate} if align else {}
        return gate * self.injection(bridge), auxiliary


def pair_alignment_loss(prediction: Tensor, target: Tensor, padding: Tensor) -> Tensor:
    """在 FP32 中计算逐时间步余弦距离，返回每个样本的有效步均值。"""
    distance = 1 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)
    return masked_time_mean(distance, padding)


def validate_teacher_dataset(config, train_dataset, eval_dataset) -> None:
    """核对两阶段数据划分及动作归一化，避免固定教师收到另一种动作空间。

    Args:
        config: 即将训练的 TrajVLAQwenConfig，元数据会随模型配置保存。
        train_dataset: 已按训练 episode 重算统计量的 LeRobot 数据集。
        eval_dataset: 同一工厂划分的留出集，可为 None。
    """
    metadata = config.pair_teacher_metadata
    if not metadata:
        if not config.pair_teacher_checkpoint:
            raise ValueError("PAIR Stage 2 requires a Stage 1 teacher checkpoint")
        metadata = json.loads((Path(config.pair_teacher_checkpoint) / "config.json").read_text())
    train_episodes = sorted(train_dataset.episodes or range(train_dataset.num_episodes))
    eval_episodes = sorted(eval_dataset.episodes) if eval_dataset is not None else []
    if train_episodes != metadata["train_episodes"] or eval_episodes != metadata["eval_episodes"]:
        raise ValueError("PAIR stages must use identical train/eval episode splits")
    if config.chunk_size != metadata["model"]["horizon"] or (
        list(config.action_delta_indices) != metadata["action_delta_indices"]
    ):
        raise ValueError("PAIR stages must use identical action horizon and stride")
    if train_dataset.meta.repo_id != metadata["dataset_repo_id"]:
        raise ValueError("PAIR stages must use the same dataset repo_id")
    for name in ("mean", "std"):
        actual = np.asarray(train_dataset.meta.stats["action"][name])
        expected = np.asarray(metadata["action_stats"][name])
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=1e-5, atol=1e-8
        ):
            raise ValueError(f"PAIR Stage 1/2 action {name} mismatch; retrain the teacher")
    config.pair_teacher_metadata = metadata
