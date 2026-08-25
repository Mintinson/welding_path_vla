"""Prismatic 的双视觉编码器、Token Merger 与投影器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

GEOMETRY_READOUT_NAMES = ("seam", "tangent", "posture")


@dataclass(slots=True)
class DenseGeometryContext:
    """Resampler 所需的稠密视觉和 Context 位置掩码。

    Attributes:
        patch_tokens: ``[B,C,P,D]``，按相机保留的 DINOv2 patch。
        patch_mask: ``[B,C,P]``，标记真实相机对应的 patch。
        language_mask: ``[B,L]``，标记 Qwen Context 中的语言 token。
        state_mask: ``[B,L]``，标记 Qwen Context 中的状态 token。
    """

    patch_tokens: Tensor
    patch_mask: Tensor
    language_mask: Tensor
    state_mask: Tensor


@dataclass(slots=True)
class SeamQueryOutput:
    """稠密几何重采样结果及后续辅助监督接口。

    Attributes:
        patch_tokens: ``[B,C,P,D_expert]``，已投影并加入相机编码的 patch。
        patch_mask: ``[B,C,P]``，有效 patch mask。
        latent_tokens: ``[B,K,D_expert]``，供 Action Expert 使用的几何 token。
        readout_tokens: ``[B,3,D_expert]``，依次对应焊缝、切向和姿态。
        attention_weights: ``[B,K+3,C,P]``，query 对稠密 patch 的注意力。
    """

    patch_tokens: Tensor
    patch_mask: Tensor
    latent_tokens: Tensor
    readout_tokens: Tensor
    attention_weights: Tensor

    def auxiliary_outputs(self) -> dict[str, Tensor]:
        """返回供 ``FlowMatchingOutput`` 暴露的稳定命名张量。"""
        return {
            "geometry.patch_tokens": self.patch_tokens,
            "geometry.patch_mask": self.patch_mask,
            "geometry.latent_tokens": self.latent_tokens,
            "geometry.readout_tokens": self.readout_tokens,
            "geometry.attention_weights": self.attention_weights,
        }


class SeamQueryResampler(nn.Module):
    """用任务与状态条件 query 从稠密 DINO patch 中提取焊缝几何。"""

    def __init__(
        self,
        patch_size: int,
        context_size: int,
        output_size: int,
        num_cameras: int,
        num_queries: int = 16,
        num_heads: int = 8,
    ) -> None:
        """构造单层 task-conditioned cross-attention resampler。

        Args:
            patch_size: 原始 DINOv2 patch 宽度。
            context_size: Qwen Context hidden 宽度。
            output_size: Action Expert hidden 宽度。
            num_cameras: 模型配置中的相机数量。
            num_queries: 供动作专家使用的 latent query 数量。
            num_heads: Resampler Cross-Attention 头数。
        """
        super().__init__()
        if min(num_cameras, num_queries, num_heads) < 1:
            raise ValueError("geometry camera, query and head counts must be positive")
        if output_size % num_heads:
            raise ValueError("geometry output size must be divisible by num_heads")
        self.num_queries = num_queries
        self.patch_projection = nn.Linear(patch_size, output_size)
        self.task_projection = nn.Linear(context_size, output_size)
        self.state_projection = nn.Linear(context_size, output_size)
        self.camera_embedding = nn.Embedding(num_cameras, output_size)
        self.latent_queries = nn.Parameter(torch.empty(num_queries, output_size))
        self.readout_queries = nn.Parameter(torch.empty(len(GEOMETRY_READOUT_NAMES), output_size))
        self.query_norm = nn.LayerNorm(output_size)
        self.patch_norm = nn.LayerNorm(output_size)
        self.cross_attention = nn.MultiheadAttention(
            output_size,
            num_heads,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(output_size)
        self.output_norm = nn.LayerNorm(output_size)
        self.mlp = nn.Sequential(
            nn.Linear(output_size, output_size * 2),
            nn.GELU(),
            nn.Linear(output_size * 2, output_size),
        )
        nn.init.normal_(self.latent_queries, std=0.02)
        nn.init.normal_(self.readout_queries, std=0.02)

    def masked_pool(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """按 Context mask 对语言 token 或者状态 token 分别做均值池化，得到两个总结向量。"""
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    def forward(self, context_hidden: Tensor, geometry: DenseGeometryContext) -> SeamQueryOutput:
        """把多相机稠密 patch 重采样成固定数量的几何 query。

        Args:
            context_hidden: 第一个 paired layer 后的 Qwen hidden，形状 ``[B,L,D]``。
            geometry: DINO patch、相机 mask 与语言/状态位置 mask。

        Returns:
            动作 latent、三个预留 readout 和 patch 注意力。
        """
        batch, cameras, patches, _ = geometry.patch_tokens.shape

        dense = self.patch_projection(
            geometry.patch_tokens.to(self.patch_projection.weight.dtype)
        )  # [B, C, P, D_out]

        camera_ids = torch.arange(cameras, device=dense.device)
        dense = dense + self.camera_embedding(camera_ids)[None, :, None]  # 加入相机嵌入

        dense_flat = dense.flatten(1, 2)  # [B, C*P, D_out]
        patch_mask = geometry.patch_mask.flatten(1, 2)  # [B, C*P]

        # 构建条件向量
        task = self.masked_pool(context_hidden, geometry.language_mask)  # [B, D_context]
        state = self.masked_pool(context_hidden, geometry.state_mask)  # [B, D_context]
        condition = self.task_projection(task.to(self.task_projection.weight.dtype))
        # [B, D_out]
        condition = condition + self.state_projection(state.to(self.state_projection.weight.dtype))

        # 生成 queries
        learned = torch.cat([self.latent_queries, self.readout_queries], dim=0)
        queries = learned[None].expand(batch, -1, -1) + condition[:, None]

        # attention
        normalized_patches = self.patch_norm(dense_flat)
        attended, attention = self.cross_attention(
            self.query_norm(queries),
            normalized_patches,
            normalized_patches,
            key_padding_mask=~patch_mask,
            need_weights=True,
        )
        output = queries + attended
        output = self.output_norm(output + self.mlp(self.mlp_norm(output)))
        attention = attention.view(batch, output.shape[1], cameras, patches)
        return SeamQueryOutput(
            patch_tokens=dense,
            patch_mask=geometry.patch_mask,
            latent_tokens=output[:, : self.num_queries],
            readout_tokens=output[:, self.num_queries :],
            attention_weights=attention,
        )


class PrismaticVisionEncoder(nn.Module):
    """复现 MiniVLA 的 DINOv2 + SigLIP 融合视觉编码器。

    两个编码器都读取同一 RGB 图像，但使用各自预训练时的归一化方式；
    输出取倒数第二个 Transformer block 的 patch token，并沿特征维拼接。
    """

    dino_mean: Tensor
    dino_std: Tensor

    def __init__(
        self,
        dino_model_name: str,
        siglip_model_name: str,
        image_size: int,
        load_base_weights: bool,
    ) -> None:
        super().__init__()
        import timm

        self.dino_featurizer = timm.create_model(
            dino_model_name,
            pretrained=load_base_weights,
            num_classes=0,
            img_size=image_size,
        )
        self.siglip_featurizer = timm.create_model(
            siglip_model_name,
            pretrained=load_base_weights,
            num_classes=0,
            img_size=image_size,
        )
        self.dino_layer = len(self.dino_featurizer.blocks) - 2
        self.siglip_layer = len(self.siglip_featurizer.blocks) - 2
        self.register_buffer(
            "dino_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "dino_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    @property
    def hidden_size(self) -> int:
        """返回两个视觉编码器拼接后的 patch 宽度。"""
        return int(self.dino_featurizer.embed_dim + self.siglip_featurizer.embed_dim)

    @property
    def dino_hidden_size(self) -> int:
        """返回独立稠密几何支路使用的 DINOv2 patch 宽度。"""
        return int(self.dino_featurizer.embed_dim)

    def patch_tokens(self, model: Any, image: Tensor, layer: int) -> Tensor:
        """读取指定视觉模型倒数第二层的纯 patch token。"""
        outputs = model.get_intermediate_layers(image, n={layer})
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs

    def forward_features(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """一次前向返回 DINO 稠密 patch 与原 Prismatic 融合 token。"""
        dtype = next(self.dino_featurizer.parameters()).dtype
        image = image.to(dtype)
        dino_image = (image - self.dino_mean.to(dtype)) / self.dino_std.to(dtype)
        siglip_image = (image - 0.5) / 0.5
        dino = self.patch_tokens(self.dino_featurizer, dino_image, self.dino_layer)
        siglip = self.patch_tokens(self.siglip_featurizer, siglip_image, self.siglip_layer)
        return dino, torch.cat([dino, siglip], dim=-1)

    def forward(self, image: Tensor) -> Tensor:
        """保持原接口，仅返回 DINOv2 与 SigLIP 融合 patch token。"""
        return self.forward_features(image)[1]


class SpatialTokenMerger(nn.Module):
    """用可学习的局部拼接把视觉 token 数缩小 ``factor²`` 倍。

    初始权重等价于局部平均池化，因此接入预训练 Prismatic Projector 时不会
    突然改变特征量纲；训练后可为焊缝等细长结构学习非均匀的局部组合。
    """

    def __init__(self, hidden_size: int, grid_size: int = 16, factor: int = 2) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.factor = factor
        self.projection = nn.Linear(hidden_size * factor**2, hidden_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """将投影初始化为各局部 patch 的逐通道平均。"""
        hidden_size = self.projection.out_features
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.bias.zero_()
            identity = torch.eye(hidden_size) / self.factor**2
            for index in range(self.factor**2):
                start = index * hidden_size
                self.projection.weight[:, start : start + hidden_size] = identity

    def forward(self, tokens: Tensor) -> Tensor:
        """合并 ``[B, grid², D]`` 中每个相邻局部窗口。"""
        batch, length, width = tokens.shape
        if length != self.grid_size**2:
            raise ValueError(f"expected {self.grid_size**2} visual tokens, got {length}")
        output_grid = self.grid_size // self.factor
        tokens = tokens.view(batch, self.grid_size, self.grid_size, width)
        tokens = tokens.view(batch, output_grid, self.factor, output_grid, self.factor, width)
        tokens = tokens.permute(0, 1, 3, 2, 4, 5).reshape(
            batch,
            output_grid**2,
            self.factor**2 * width,
        )
        return self.projection(tokens)


class PrismaticProjector(nn.Module):
    """复现官方 fused-gelu-mlp，将视觉特征映射到 Qwen 宽度。"""

    def __init__(self, vision_size: int, language_size: int) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_size, 4 * vision_size),
            nn.GELU(),
            nn.Linear(4 * vision_size, language_size),
            nn.GELU(),
            nn.Linear(language_size, language_size),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """投影视觉 token，保持 batch 和序列维不变。"""
        return self.projector(tokens)


__all__ = [
    "GEOMETRY_READOUT_NAMES",
    "DenseGeometryContext",
    "PrismaticProjector",
    "PrismaticVisionEncoder",
    "SeamQueryOutput",
    "SeamQueryResampler",
    "SpatialTokenMerger",
]
