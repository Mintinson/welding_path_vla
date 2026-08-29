"""
将动作模式与稠密几何解耦的 Conditional Motion Latent。

该模块实现了一个条件变分自编码器（CVAE）风格的潜在变量模型，用于从
任务上下文（语言、状态、几何信息）中生成动作模式（motion token）。
训练时使用后验编码器（基于真实动作）学习潜在分布，推理时仅使用先验网络。
潜在变量解耦了高层动作模式与稠密几何特征，便于模型学习多样化的运动策略。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from welding_path_vla.policies.traj_vla_qwen.prismatic import DenseGeometryContext, SeamQueryOutput


@dataclass(slots=True)
class MotionLatentOutput:
    """一次 conditional prior/posterior 推断的 latent、token 与诊断量。

    Attributes:
        token: 映射到专家维度（如 Qwen 隐藏层维度）的运动 token，形状 (B, expert_size)
        auxiliary_outputs: 包含 latent 统计量、KL 散度等辅助信息的字典
    """

    token: Tensor
    auxiliary_outputs: dict[str, Tensor]


def diagonal_gaussian_kl(
    posterior_mean: Tensor,
    posterior_log_variance: Tensor,
    prior_mean: Tensor,
    prior_log_variance: Tensor,
) -> Tensor:
    """
    计算两个对角高斯分布之间的 KL 散度（逐样本解析解）。

    参数：
        posterior_mean: 后验均值，形状 (B, L)
        posterior_log_variance: 后验对数方差，形状 (B, L)
        prior_mean: 先验均值，形状 (B, L)
        prior_log_variance: 先验对数方差，形状 (B, L)

    返回：
        每个样本的 KL 散度，形状 (B,)
    """
    variance_ratio = torch.exp(posterior_log_variance - prior_log_variance)
    mean_error = (posterior_mean - prior_mean).square() * torch.exp(-prior_log_variance)
    return 0.5 * (
        prior_log_variance - posterior_log_variance + variance_ratio + mean_error - 1
    ).sum(dim=-1)


class MotionPosteriorEncoder(nn.Module):
    """
    后验编码器：根据干净的动作块和状态编码运动策略分布。

    输入为动作序列（通常是未来动作块）和当前状态，通过 Transformer 编码后池化，
    输出潜在变量的均值和对数方差。
    """

    def __init__(self, action_size: int, context_size: int, latent_size: int) -> None:
        super().__init__()
        hidden_size = 128
        self.action_projection = nn.Linear(action_size, hidden_size)
        self.state_projection = nn.Linear(context_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            dropout=0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.distribution = nn.Linear(hidden_size, latent_size * 2)

    def forward(
        self,
        actions: Tensor,
        action_padding_mask: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """返回 posterior 的均值与对数方差，并忽略 episode 尾部 padding。"""
        hidden = self.action_projection(actions.to(self.action_projection.weight.dtype))
        state_hidden = self.state_projection(state.to(self.state_projection.weight.dtype))
        hidden = hidden + state_hidden[:, None]
        padding = action_padding_mask.bool()
        hidden = self.transformer(hidden, src_key_padding_mask=padding)
        weights = (~padding).to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        mean, log_variance = self.distribution(pooled).chunk(2, dim=-1)
        return mean, log_variance.clamp(-10, 10)


class ConditionalMotionLatent(nn.Module):
    """
    由 Qwen/geometry prior 和训练期 action posterior 生成 motion token 的模块。

    该模块是条件变分自编码器的核心：它包含一个先验网络（基于任务、状态和几何上下文）
    和一个后验编码器（基于真实动作）。训练时从后验分布采样潜在变量，并计算 KL 散度；
    推理时直接使用先验均值。最终将潜在变量映射为运动 token 注入到主模型中。
    """

    def __init__(
        self,
        action_size: int,
        context_size: int,
        expert_size: int,
        latent_size: int,
    ) -> None:
        super().__init__()
        prior_input_size = context_size * 2 + expert_size
        self.prior = nn.Sequential(
            nn.Linear(prior_input_size, 128),
            nn.GELU(),
            nn.Linear(128, latent_size * 2),
        )
        self.posterior: MotionPosteriorEncoder | None = MotionPosteriorEncoder(
            action_size,
            context_size,
            latent_size,
        )
        self.latent_projection = nn.Linear(latent_size, expert_size)
        self.slot = nn.Parameter(torch.empty(expert_size))
        nn.init.normal_(self.slot, std=0.02)

    def masked_pool(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """
        对输入的 token 序列进行有效位置的均值池化。

        参数：
            hidden: 形状 (B, T, D) 的张量
            mask: 形状 (B, T) 的布尔张量，True 表示有效

        返回：
            形状 (B, D) 的池化结果
        """
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    def prior_distribution(
        self,
        context_hidden: Tensor,
        geometry_context: DenseGeometryContext,
        geometry: SeamQueryOutput,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        根据任务、状态和几何 latent 预测条件先验分布。

        参数：
            context_hidden: Qwen 模型输出的上下文 token，形状 (B, T, context_size)
            geometry_context: 包含语言掩码和状态掩码的几何上下文
            geometry: 几何查询输出，包含 latent_tokens

        返回：
            (prior_mean, prior_log_variance, state_pooled) 元组
            prior_mean, prior_log_variance 形状 (B, latent_size)
            state_pooled 是池化后的状态表示，用于后续后验编码器
        """
        task = self.masked_pool(context_hidden, geometry_context.language_mask)
        state = self.masked_pool(context_hidden, geometry_context.state_mask)
        geometry_pool = geometry.latent_tokens.mean(dim=1)
        values = torch.cat((task, state, geometry_pool), dim=-1)
        prior_input = cast(nn.Linear, self.prior[0])
        mean, log_variance = self.prior(values.to(prior_input.weight.dtype)).chunk(2, dim=-1)
        return mean, log_variance.clamp(-10, 10), state

    def forward(
        self,
        context_hidden: Tensor,
        geometry_context: DenseGeometryContext,
        geometry: SeamQueryOutput,
    ) -> MotionLatentOutput:
        """训练采 posterior，推理取 prior mean，并返回解析 KL。"""
        prior_mean, prior_log_variance, state = self.prior_distribution(
            context_hidden,
            geometry_context,
            geometry,
        )
        posterior_mean: Tensor | None = None
        posterior_log_variance: Tensor | None = None
        actions = geometry_context.clean_actions
        padding = geometry_context.action_padding_mask
        posterior = self.posterior
        if self.training and posterior is not None and actions is not None and padding is not None:
            posterior_mean_value, posterior_log_variance_value = posterior(actions, padding, state)
            posterior_mean = posterior_mean_value
            posterior_log_variance = posterior_log_variance_value
            standard_deviation = torch.exp(0.5 * posterior_log_variance_value)
            latent = posterior_mean_value + standard_deviation * torch.randn_like(
                standard_deviation
            )
            kl = diagonal_gaussian_kl(
                posterior_mean_value,
                posterior_log_variance_value,
                prior_mean,
                prior_log_variance,
            )
        else:
            latent = prior_mean
            kl = torch.zeros(prior_mean.shape[0], dtype=prior_mean.dtype, device=prior_mean.device)

        token = self.latent_projection(latent.to(self.latent_projection.weight.dtype))
        auxiliary = {
            "motion.prior_mean": prior_mean,
            "motion.prior_logvar": prior_log_variance,
            "motion.latent": latent,
            "motion.kl": kl,
        }
        if posterior_mean is not None and posterior_log_variance is not None:
            auxiliary.update(
                {
                    "motion.posterior_mean": posterior_mean,
                    "motion.posterior_logvar": posterior_log_variance,
                }
            )
        return MotionLatentOutput(token, auxiliary)

    def initial_slot(self, batch_size: int, device: torch.device) -> Tensor:
        """返回第 0 层使用的 learned motion slot。"""
        return self.slot.to(device)[None, None].expand(batch_size, 1, -1)

    def discard_posterior(self) -> None:
        """部署时删除不会参与 prior 推理的 posterior encoder。"""
        self.posterior = None


__all__ = [
    "ConditionalMotionLatent",
    "MotionLatentOutput",
    "MotionPosteriorEncoder",
    "diagonal_gaussian_kl",
]
