"""VLA-Adapter Pro 的逐层 Bridge Attention。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def rotate_pairs(value: Tensor) -> Tensor:
    """把偶数/奇数通道组成的二维向量旋转九十度。"""
    even, odd = value[..., ::2], value[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class RotaryPositionEmbedding(nn.Module):
    """为三路 attention 独立生成标准 RoPE 正余弦表。"""

    def __init__(self, dimension: int, base: float = 10_000) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("RoPE head dimension must be even")
        frequency = 1 / base ** (torch.arange(0, dimension, 2).float() / dimension)
        self.register_buffer("frequency", frequency, persistent=False)

    def forward(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """返回可广播到 ``[B,H,L,D]`` 的正余弦张量。"""
        position = torch.arange(length, device=device, dtype=self.frequency.dtype)
        phase = torch.outer(position, self.frequency).repeat_interleave(2, dim=-1)  # (T, dim)
        return phase.cos().to(dtype)[None, None], phase.sin().to(dtype)[None, None]


def apply_rope(value: Tensor, rope: RotaryPositionEmbedding) -> Tensor:
    """沿 token 维对 Q 或 K 应用 RoPE。"""
    cosine, sine = rope(value.shape[2], value.device, value.dtype)
    return value * cosine + rotate_pairs(value) * sine


class ProBridgeBlock(nn.Module):
    """融合动作自身、ActionQuery+状态和 raw patch 的 Pro Bridge 层。

    三路 K/V 使用独立投影。raw patch 的 attention logits 乘以零初始化的
    ``tanh(g)``，与官方当前 Pro 实现保持一致；三路 logits 拼接后只做一次
    softmax，避免原仓库重复拼接 adapter logits 的明显错误。
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("bridge hidden size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.self_key = nn.Linear(hidden_size, hidden_size)
        self.self_value = nn.Linear(hidden_size, hidden_size)
        self.adapter_key = nn.Linear(hidden_size, hidden_size)
        self.adapter_value = nn.Linear(hidden_size, hidden_size)
        self.raw_key = nn.Linear(hidden_size, hidden_size)
        self.raw_value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.raw_gate = nn.Parameter(torch.zeros(()))
        self.rope = RotaryPositionEmbedding(self.head_size)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def split_heads(self, value: Tensor) -> Tensor:
        """把 ``[B,L,D]`` 改排为 ``[B,H,L,D/H]``。"""
        batch, length, _ = value.shape
        return value.view(batch, length, self.num_heads, self.head_size).transpose(1, 2)

    def forward(
        self,
        action: Tensor,
        action_query: Tensor,
        raw: Tensor,
        state: Tensor,
        raw_mask: Tensor | None = None,
    ) -> Tensor:
        """执行一层三路联合 attention 与残差 FFN。"""
        action_query = action_query.to(action.dtype)
        raw = raw.to(action.dtype)
        state = state.to(action.dtype)
        adapter = torch.cat((action_query, state), dim=1)
        query = apply_rope(self.split_heads(self.query(action)), self.rope)
        keys = [
            apply_rope(self.split_heads(self.self_key(action)), self.rope),
            apply_rope(self.split_heads(self.adapter_key(adapter)), self.rope),
            apply_rope(self.split_heads(self.raw_key(raw)), self.rope),
        ]
        values = [
            self.split_heads(self.self_value(action)),
            self.split_heads(self.adapter_value(adapter)),
            self.split_heads(self.raw_value(raw)),
        ]
        scores = [torch.matmul(query.float(), key.float().transpose(-2, -1)) for key in keys]
        scores[-1] = scores[-1] * torch.tanh(self.raw_gate.float())
        attention_scores = torch.cat(scores, dim=-1) / math.sqrt(self.head_size)
        if raw_mask is not None:
            always = torch.ones(
                raw.shape[0],
                action.shape[1] + adapter.shape[1],
                dtype=torch.bool,
                device=raw.device,
            )
            valid = torch.cat((always, raw_mask), dim=1)
            attention_scores = attention_scores.masked_fill(
                ~valid[:, None, None],
                torch.finfo(attention_scores.dtype).min,
            )
        probabilities = torch.softmax(attention_scores, dim=-1).to(values[0].dtype)
        attended = torch.matmul(probabilities, torch.cat(values, dim=2))
        attended = attended.transpose(1, 2).contiguous().flatten(2)
        return self.feed_forward(action + self.output(attended))


class ProBridgePolicy(nn.Module):
    """把 Qwen 各层条件与同深度 Bridge 层一一对齐。"""

    def __init__(self, hidden_size: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            ProBridgeBlock(hidden_size, num_heads) for _ in range(num_layers)
        )

    def forward(
        self,
        action: Tensor,
        raw_layers: tuple[Tensor, ...],
        query_layers: tuple[Tensor, ...],
        state: Tensor,
        raw_mask: Tensor,
    ) -> Tensor:
        """逐层注入对应深度的 raw 与 ActionQuery hidden states。"""
        if len(raw_layers) != len(self.layers) or len(query_layers) != len(self.layers):
            raise ValueError("VLM and Bridge layer counts must match")
        for layer, raw, action_query in zip(
            self.layers,
            raw_layers,
            query_layers,
            strict=True,
        ):
            action = layer(action, action_query, raw, state, raw_mask)
        return action


__all__ = ["ProBridgeBlock", "ProBridgePolicy", "RotaryPositionEmbedding", "apply_rope"]
