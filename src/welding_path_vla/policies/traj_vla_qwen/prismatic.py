"""Prismatic 的双视觉编码器、Token Merger 与投影器。"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


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

    def patch_tokens(self, model: Any, image: Tensor, layer: int) -> Tensor:
        """读取指定视觉模型倒数第二层的纯 patch token。"""
        outputs = model.get_intermediate_layers(image, n={layer})
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs

    def forward(self, image: Tensor) -> Tensor:
        """把 ``[0,1]`` RGB 图像编码为融合 patch token。"""
        dtype = next(self.dino_featurizer.parameters()).dtype
        image = image.to(dtype)
        dino_image = (image - self.dino_mean.to(dtype)) / self.dino_std.to(dtype)
        siglip_image = (image - 0.5) / 0.5
        dino = self.patch_tokens(self.dino_featurizer, dino_image, self.dino_layer)
        siglip = self.patch_tokens(self.siglip_featurizer, siglip_image, self.siglip_layer)
        return torch.cat([dino, siglip], dim=-1)


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


__all__ = ["PrismaticProjector", "PrismaticVisionEncoder", "SpatialTokenMerger"]
