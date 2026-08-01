# Copyright 2025 Hugging Face Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trajectory-VLA 的 flow-matching 轨迹生成器。

设计概述
--------
Flow matching 把"从噪声生成动作轨迹"建模为一条连续插值路径上的速度场回归：

    训练：对每条动作轨迹 action 采样高斯终点 noise 与插值时间 t，
        构造中间轨迹 noisy = t * noise + (1 - t) * action，
        训练动作专家预测速度场 v(noisy, t)，监督信号为解析速度
        v* = d(noisy)/dt = noise - action。
    推理：从纯噪声 x = noise（t = 1）出发，用 Euler 法沿时间反向积分
        ODE dx/dt = v(x, t)，逐步还原出真实轨迹（t = 0）。

与自回归逐 token 生成不同，flow matching 一次前向即可得到整段轨迹的
速度场，推理步数（num_steps）可以按质量/速度权衡自由调整。

本模块同时提供 flow matching 的配套工具：连续时间正弦编码、
block-wise 因果 attention mask、图像与序列的补齐函数。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from lerobot.utils.device_utils import get_safe_dtype
from torch import Tensor, nn

from welding_path_vla.policies.trajectory_vla.configuration_trajectory_vla import (
    TrajectoryVLAConfig,
)


@dataclass(slots=True)
class TokenSequence:
    """一组 Transformer token 及其两种 mask。

    Attributes:
        embeddings: ``[B, L, D]`` token embedding。
        padding_mask: ``[B, L]``，标记真实 token（True）与补零 token（False）。
        attention_ar: ``[B, L]``，标记新的自回归 attention block：取值为
            True 的位置是一个新 block 的起点（如机器人状态 token 的起始处）。
            make_attention_masks 用 cumsum 把它展开为"分块因果"mask——
            每个 token 只能看到自己所在 block 及更早的 block，block 内部双向。

    该结构贯穿整个模型：encode_context（VLM prefix）与 encode_action_tokens
    （动作 token）都返回它，predict_velocity / denoise_step 再把两者的 mask
    沿序列维拼接后交给双流 Transformer。
    """

    embeddings: Tensor
    padding_mask: Tensor
    attention_ar: Tensor


@dataclass(slots=True)
class FlowMatchingOutput:
    """一次 flow-matching 训练前向的完整中间量。

    保留全部中间量而非只保留 loss 的目的：训练循环可以直接记录/可视化
    插值轨迹、速度场的分布（诊断模型行为），后续实现更复杂的 flow
    matching 变体（如最优传输路径、logit 加权）时也能直接复用。

    Attributes:
        noise: 采样的高斯终点，形状与 ``actions`` 相同。
        time: 每个样本的插值时间，``[B]``，范围
            [flow_time_offset, flow_time_scale + flow_time_offset]。
        noisy_actions: 时间 ``t`` 上的动作轨迹：``t * noise + (1 - t) * actions``。
        target_velocity: 解析目标速度 ``noise - actions``，即 noisy_actions 对 t 的导数。
        predicted_velocity: 动作专家预测的速度场，形状与 target_velocity 一致。
        losses: 未约简的逐时间、逐动作维 MSE，由调用方决定如何约简/加权。
    """

    noise: Tensor
    time: Tensor
    noisy_actions: Tensor
    target_velocity: Tensor
    predicted_velocity: Tensor
    losses: Tensor


@dataclass(slots=True)
class DenoisingState:
    """推理期一个 Euler 去噪步骤，可交给回调记录或修改。

    只是纯数据快照，不含计算逻辑。sample_trajectory 的 ``on_step`` 回调
    收到它后可以记录每一步的去噪中间结果（如导出成视频/图像），也可以
    根据诊断信息提前中止、替换轨迹，实现"观察与干预"而不侵入主循环。
    """

    step: int
    time: Tensor
    actions: Tensor
    velocity: Tensor


def sinusoidal_time_embedding(
    time: Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
) -> Tensor:
    """把标量 flow time 编码为正弦余弦向量。

    与 Transformer position embedding 同源：取 ``dimension / 2`` 个波长，
    从 ``min_period`` 到 ``max_period`` 按对数刻度均匀分布（几何级数），
    每个维度对时间取 sin / cos。flow time 是连续标量而非离散位置，因此
    直接计算相位 ``t * 2π / period``。波长跨度越大，模型既能分辨微小的时间
    差异（高频分量），也能感知整体时间进度（低频分量）。

    Args:
        time: ``[B]`` 每个样本的 flow time。
        dimension: 输出维度，必须为偶数。
        min_period: 最短波长（高频分量）。
        max_period: 最长波长（低频分量）。

    Returns:
        ``[B, dimension]``，前一半是 sin、后一半是 cos。
    """
    if dimension % 2:
        raise ValueError("time embedding dimension must be even")
    if time.ndim != 1:
        raise ValueError("time must have shape [batch]")
    dtype = get_safe_dtype(torch.float64, time.device.type)
    # 在 [min_period, max_period] 上按对数刻度取 dimension/2 个波长
    fraction = torch.linspace(0, 1, dimension // 2, dtype=dtype, device=time.device)
    period = min_period * (max_period / min_period) ** fraction
    # time[:, None] 广播为 [B, D/2]，得到每个样本每个波长的相位角
    phase = time[:, None] * (2 * math.pi / period)[None, :]
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)


def make_attention_masks(padding_mask: Tensor, attention_ar: Tensor) -> Tensor:
    """把一维 block 标记转换为 ``[B, query, key]`` attention mask。

    核心技巧是用 cumsum 把一维标记展开成二维的"分块因果"约束：
    - ``blocks = cumsum(attention_ar)``：每个 token 获得一个 block 编号，
      每次 attention_ar=1 都会开启一个新编号（也即新 block）；
    - ``blocks[q] <= blocks[k]``：query 只能注意自己所在 block 及更早
      block 的 key——同 block 内双向可见，跨 block 单向可见；
    - ``padding_mask`` 同时屏蔽掉两侧的补零 token（不能作为 query 或 key）。

    Args:
        padding_mask: ``[B, L]`` 有效 token 标记。
        attention_ar: ``[B, L]`` 新 block 起点标记（True=新 block）。

    Returns:
        ``[B, L, L]`` 布尔 mask：True 表示允许 query 注意 key。
    """
    if padding_mask.ndim != 2 or attention_ar.ndim != 2:
        raise ValueError("padding and attention masks must have shape [B, L]")
    # 每个 token 的 block 编号：attention_ar=1 的位置编号加一
    blocks = torch.cumsum(attention_ar, dim=1)
    # query 只能注意 block 编号不大于自己的 key（同 block 双向、跨 block 单向）
    causal_blocks = blocks[:, None, :] <= blocks[:, :, None]
    # 补零 token 既不能作为 query 也不能作为 key
    valid_tokens = padding_mask[:, None, :] * padding_mask[:, :, None]
    return causal_blocks & valid_tokens


def resize_with_pad(image: Tensor, width: int, height: int, value: float = 0) -> Tensor:
    """保持宽高比缩放 ``[B,C,H,W]`` 图像并在左上侧补齐。

    处理流程：先按最大收缩比例缩放（宽高均不超过目标尺寸，不拉伸变形），
    再补齐到目标尺寸。注意 F.pad 的 pad 元组顺序为 (左, 右, 上, 下)，
    这里只补左、上两侧，因此缩放后的图像位于画布右下角，其余区域填充
    ``value``。用于把不同分辨率/宽高比的相机图像统一到模型输入尺寸
    （config.resize_imgs_with_padding，默认 512x512）。

    Args:
        image: ``[B, C, H, W]`` 输入图像。
        width, height: 目标画布尺寸。
        value: 补齐区域填充值。

    Returns:
        ``[B, C, height, width]``。
    """
    if image.ndim != 4:
        raise ValueError(f"expected [B,C,H,W], got {tuple(image.shape)}")
    current_height, current_width = image.shape[2:]
    # 取最大收缩比例：保证任一维都不超过目标尺寸
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    # 双线性插值缩放（缩放比例未必是整数）
    resized = F.interpolate(
        image,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    # 在左侧与上侧补齐（pad 顺序: 左, 右, 上, 下）
    return F.pad(
        resized,
        (width - resized_width, 0, height - resized_height, 0),
        value=value,
    )


def pad_vector(vector: Tensor, dimension: int) -> Tensor:
    """只在最后一维补零，不修改 batch 或时间维。

    用于把不同来源的向量统一到相同维度（如动作维不足 max_action_dim 时），
    使整个 batch 能在同一张量上并行运算。
    """
    if vector.shape[-1] > dimension:
        raise ValueError(f"cannot pad dimension {vector.shape[-1]} to {dimension}")
    return F.pad(vector, (0, dimension - vector.shape[-1]))


def pad_sequence(tensor: Tensor, length: int, value: float = 0) -> Tensor:
    """把序列维补到固定长度。

    用于把拼接后的 prefix（图像 + 语言 + 状态 token）统一补齐到
    ``prefix_length``，使同 batch 长度不同的样本可以并行推理。
    若序列已经足够长则原样返回（不做裁剪）。
    """
    if tensor.shape[1] >= length:
        return tensor
    padded = torch.full(
        (tensor.shape[0], length, *tensor.shape[2:]),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:, : tensor.shape[1]] = tensor
    return padded


class TrajectoryFlowModel(nn.Module):
    """使用 SmolVLM context 和动作专家速度场生成短时轨迹。

    核心思想（双流 + flow matching）：
    - VLM 流：把多相机图像、语言指令、机器人状态编码成"上下文"prefix；
    - 动作专家流：输入是"带噪声的动作轨迹 + 时间条件"，在 VLM 上下文
      的引导下预测速度场；
    - 训练：回归解析速度 ``noise - actions``；推理：Euler 积分从噪声还原轨迹。

    研究接口按处理顺序公开为：
    ``encode_context`` → ``encode_action_tokens`` → ``flow_matching_output``；
    推理则公开为 ``denoise_step`` 和 ``sample_trajectory``。
    """

    def __init__(self, config: TrajectoryVLAConfig) -> None:
        super().__init__()
        from welding_path_vla.policies.trajectory_vla.smolvlm_action_expert import (
            SmolVLMActionExpert,
        )

        # ---- 基础组件：VLM + 动作专家（见 smolvlm_action_expert.py）----
        # 动作专家共享 VLM 的视觉/语言理解，自身维护一套独立宽度的 token 流
        self.config = config
        self.vlm_with_expert = SmolVLMActionExpert(
            model_id=config.vlm_model_name,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            load_vlm_weights=config.load_vlm_weights,
            attention_mode=config.attention_mode,
            num_expert_layers=config.num_expert_layers,
            num_vlm_layers=config.num_vlm_layers,
            self_attn_every_n_layers=config.self_attn_every_n_layers,
            expert_width_multiplier=config.expert_width_multiplier,
        )
        vlm_width = self.vlm_with_expert.config.text_config.hidden_size
        expert_width = self.vlm_with_expert.expert_hidden_size
        # ---- 输入输出投影 ----
        # 状态投影：机器人状态 [B, max_state_dim] → VLM 宽度的"状态 token"
        self.state_proj = nn.Linear(config.max_state_dim, vlm_width)
        # 动作输入投影：噪声轨迹 [B, T, max_action_dim] → 专家宽度
        self.action_in_proj = nn.Linear(config.max_action_dim, expert_width)
        # 动作输出投影：专家隐状态 → 动作空间速度场 [B, T, max_action_dim]
        self.action_out_proj = nn.Linear(expert_width, config.max_action_dim)
        # 时间融合 MLP：拼接 [动作 embedding, 时间 embedding] 后两层融合
        self.action_time_mlp_in = nn.Linear(expert_width * 2, expert_width)
        self.action_time_mlp_out = nn.Linear(expert_width, expert_width)
        self.set_requires_grad()

        # ---- SmolVLM 图像边界特殊 token ----
        # fake_image_token 标记"图像开始/结束"，global_image_token 标记
        # 全局/局部视角切换；它们作为普通 token 参与 attention，使输入
        # 格式与 SmolVLM 预训练数据一致（add_image_special_tokens 控制开关）
        tokenizer = self.vlm_with_expert.processor.tokenizer
        self.fake_image_token = tokenizer.fake_image_token_id
        self.global_image_token = tokenizer.global_image_token_id
        # 起始 token 序列：[fake_image, global_image]
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token],
            dtype=torch.long,
        )
        # 结束 token 序列：[fake_image]
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)

    def set_requires_grad(self) -> None:
        """单独控制 state projection 是否参与训练。

        ``train_state_proj`` 关闭时冻结 state_proj（如状态维度语义尚不稳定、
        或需要与预训练权重行为严格对齐的实验）；其余模块不受影响。
        """
        for parameter in self.state_proj.parameters():
            parameter.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """采样 flow matching 的高斯终点；可在子类中替换。

        训练与推理共用此方法：替换它即可整体切换噪声分布（如截断高斯、
        低方差噪声），训练目标速度与推理起点会保持一致。
        """
        return torch.randn(shape, dtype=torch.float32, device=device)

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        """采样带偏置的训练时间；参数全部暴露在配置中。

        t ~ Beta(flow_beta_alpha, flow_beta_beta) 后经线性变换
        ``t = t * flow_time_scale + flow_time_offset`` 映射到目标区间
        （默认 [0.001, 1.0]）。默认 Beta(1.5, 1.0) 的密度随 t 递增，
        在靠近 1（噪声端）采样更多，让模型更充分学习"噪声 → 轨迹"的
        过渡阶段；两个参数可配置以调节采样侧重。scale/offset 整体平移
        缩放采样区间，offset 略大于 0 可避开 t=0 处数值上易出问题的端点。
        """
        distribution = torch.distributions.Beta(
            self.config.flow_beta_alpha,
            self.config.flow_beta_beta,
        )
        time = distribution.sample((batch_size,)).to(device=device, dtype=torch.float32)
        return time * self.config.flow_time_scale + self.config.flow_time_offset

    def image_special_tokens(self, batch_size: int, start: bool, device: torch.device) -> Tensor:
        """返回可选的 SmolVLM 图像边界 token embedding。

        start=True 返回 ``[fake_image, global_image]`` 的 embedding（图像起始），
        start=False 返回 ``[fake_image]`` 的 embedding（图像结束）。
        与语言 token 一样走 embedding 查表，作为 prefix 的一部分参与
        attention；不引入可学习参数，仅用于维持 SmolVLM 的输入格式。
        """
        token = self.global_image_start_token if start else self.image_end_token
        embedding = self.vlm_with_expert.embed_language_tokens(token.to(device))
        # [1, L, D] → [B, L, D]：同一 batch 的每张图使用相同的边界 token
        return embedding.unsqueeze(0).expand(batch_size, -1, -1)

    def encode_context(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
        state: Tensor,
    ) -> TokenSequence:
        """编码图像、语言和机器人状态，形成 VLM prefix。

        拼接顺序：对每张相机图像（可选起始边界 token）→ 图像 patch 序列
        （可选结束边界 token），随后是语言 token，最后是状态 token（作为
        新的自回归 block 起点，attention_ar=True）。

        返回的 TokenSequence：
        - padding_mask：标记真实 token；补零位置在 attention 中被屏蔽；
        - attention_ar：仅在状态 token 起始处为 True，使状态成为新 block；
        - 总长度不足 prefix_length 时末尾补零（被 padding_mask 屏蔽）。
        """
        embeddings: list[Tensor] = []
        padding_masks: list[Tensor] = []
        attention_blocks: list[int] = []

        # 逐张处理相机图像：与语言/状态同属 prefix 首块（block 编号 0）
        for image, image_mask in zip(images, image_masks, strict=True):
            if self.config.add_image_special_tokens:
                # 图像起始边界 token，提示 VLM"接下来是一张图像"
                special = self.image_special_tokens(image.shape[0], True, image.device)
                embeddings.append(special)
                padding_masks.append(torch.ones_like(special[:, :, 0], dtype=torch.bool))
                attention_blocks.extend([0] * special.shape[1])

            # 视觉塔 + connector 输出 patch 级 token；开方缩放对齐语言 token 量纲
            image_embedding = self.vlm_with_expert.embed_image(image)
            image_embedding = image_embedding * math.sqrt(image_embedding.shape[-1])
            # image_mask: [B] 该相机是否有效 → 沿 patch 维展开成 [B, num_patches]
            mask = image_mask[:, None].expand(image.shape[0], image_embedding.shape[1])
            embeddings.append(image_embedding)
            padding_masks.append(mask)
            attention_blocks.extend([0] * image_embedding.shape[1])

            if self.config.add_image_special_tokens:
                # 图像结束边界 token
                special = self.image_special_tokens(image.shape[0], False, image.device)
                embeddings.append(special)
                padding_masks.append(torch.ones_like(special[:, :, 0], dtype=torch.bool))
                attention_blocks.extend([0] * special.shape[1])

        # 语言指令 token 化后的 embedding，同样开方缩放保持量纲一致
        language_embedding = self.vlm_with_expert.embed_language_tokens(language_tokens)
        language_embedding = language_embedding * math.sqrt(language_embedding.shape[-1])
        embeddings.append(language_embedding)
        padding_masks.append(language_mask)
        attention_blocks.extend([0] * language_embedding.shape[1])

        # 机器人状态经投影后作为 prefix 的最后一个部分
        state_embedding = self.state_proj(state)
        if state_embedding.ndim == 2:
            # 无图像/语言时可能没有时间维，补成长度 1 的序列
            state_embedding = state_embedding[:, None, :]
        embeddings.append(state_embedding)
        padding_masks.append(
            torch.ones(
                state_embedding.shape[:2],
                dtype=torch.bool,
                device=state_embedding.device,
            )
        )
        # 状态 token 开启新 block（attention_ar=1）：后面的动作 token 以此
        # 为界，形成"prefix 整体 + 因果动作序列"的分块结构
        attention_blocks.extend([1] * state_embedding.shape[1])

        # 各段沿序列维拼接为完整 prefix
        merged = torch.cat(embeddings, dim=1)
        padding = torch.cat(padding_masks, dim=1)
        # 一维 block 标记 → [1, L]（后面按 batch 广播）
        attention = torch.tensor(
            attention_blocks,
            dtype=torch.bool,
            device=padding.device,
        )[None, :]
        if merged.shape[1] < self.config.prefix_length:
            # 统一补到 prefix_length，保证同 batch 内序列长度一致
            merged = pad_sequence(merged, self.config.prefix_length)
            padding = pad_sequence(padding, self.config.prefix_length)
            attention = pad_sequence(attention, self.config.prefix_length)
        return TokenSequence(
            merged,
            padding,
            attention.expand(merged.shape[0], -1),
        )

    def encode_action_tokens(self, noisy_actions: Tensor, time: Tensor) -> TokenSequence:
        """融合 noisy trajectory 与 flow time，形成动作专家 token。

        时间信息作为"条件"注入：动作 embedding 投影到专家宽度后，与时间
        正弦编码按特征维拼接，再经两层带 SiLU 门控的 MLP 融合。这样速度场
        预测器能同时看到"当前轨迹的形状"与"插值进行到哪个时刻"。
        """
        action_embedding = self.action_in_proj(noisy_actions)
        # 时间编码维度取专家宽度，保证与动作 embedding 可拼接
        time_embedding = sinusoidal_time_embedding(
            time,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
        ).to(action_embedding.dtype)
        # [B, 1, D] → [B, T, D]，每个动作 token 携带同一个时间条件
        time_embedding = time_embedding[:, None, :].expand_as(action_embedding)
        # 特征维拼接 + SwiGLU 风格的两层 MLP 融合
        embedding = torch.cat([action_embedding, time_embedding], dim=-1)
        embedding = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(embedding)))
        # 动作 token 全部有效；attention_ar 全 True → 每个 token 自成一块，
        # 使动作序列内部严格因果（只能看更早的动作 token），而整个 prefix
        # （block 0）对任何动作 token 都可见
        mask = torch.ones(embedding.shape[:2], dtype=torch.bool, device=embedding.device)
        attention = torch.ones_like(mask)
        return TokenSequence(embedding, mask, attention)

    def predict_velocity(
        self,
        context: TokenSequence,
        action_tokens: TokenSequence,
    ) -> Tensor:
        """联合执行 VLM 与动作专家，输出动作空间速度场。

        把 context 与动作 token 拼接成一条序列送入双流 Transformer：
        - VLM 流处理 prefix，为动作专家提供理解后的上下文；
        - 动作专家流处理动作 token，每个动作 token 都能注意整个 prefix
          以及更早的动作 token（分块因果）。
        最终取专家流输出的最后 ``chunk_size`` 个位置（即动作轨迹段）
        投影回动作空间，得到速度场。
        """
        # 两个流的 mask 沿序列维拼接；attention_ar 决定分块因果结构
        padding = torch.cat([context.padding_mask, action_tokens.padding_mask], dim=1)
        attention_ar = torch.cat([context.attention_ar, action_tokens.attention_ar], dim=1)
        attention = make_attention_masks(padding, attention_ar)
        # 位置编号 = 有效 token 的累计数 - 1（只给真实 token 分配位置）
        positions = torch.cumsum(padding, dim=1) - 1
        outputs, _ = self.vlm_with_expert(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[context.embeddings, action_tokens.embeddings],
            use_cache=False,
            fill_kv_cache=False,
        )
        expert_output = outputs[1]
        if expert_output is None:
            raise ValueError("action expert did not produce hidden states")
        # 只取动作轨迹段（最后 chunk_size 个位置）投影为速度场
        return self.action_out_proj(expert_output[:, -self.config.chunk_size :].float())

    def flow_matching_output(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
        state: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> FlowMatchingOutput:
        """返回未约简 loss 及其所有构造中间量。

        训练目标（最优传输 flow matching 的最简形式——直线插值路径 +
        高斯边界分布）：
            x_t    = t * noise + (1 - t) * actions   # 线性插值轨迹
            v*     = noise - actions                 # 解析速度场，即 dx_t/dt
            loss   = MSE(v*, v_pred)                 # 速度场回归

        噪声与时间默认自动采样；外部显式传入可固定它们（复现实验、
        课程学习等）。
        """
        # 自动采样噪声与时间（可显式传入固定值）
        noise = (
            noise if noise is not None else self.sample_noise(tuple(actions.shape), actions.device)
        )
        time = time if time is not None else self.sample_time(actions.shape[0], actions.device)
        # t 沿 [B, T, D] 广播：t=0 是真实轨迹，t=1 是纯噪声
        interpolation = time[:, None, None]
        noisy_actions = interpolation * noise + (1 - interpolation) * actions
        # 直线插值路径的解析速度就是两个端点之差，与 t 无关
        target_velocity = noise - actions
        context = self.encode_context(
            images,
            image_masks,
            language_tokens,
            language_mask,
            state,
        )
        action_tokens = self.encode_action_tokens(noisy_actions, time)
        predicted_velocity = self.predict_velocity(context, action_tokens)
        losses = F.mse_loss(target_velocity, predicted_velocity, reduction="none")
        return FlowMatchingOutput(
            noise,
            time,
            noisy_actions,
            target_velocity,
            predicted_velocity,
            losses,
        )

    def forward(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
        state: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """兼容 LeRobot policy 的标准前向，只返回逐维 loss。

        训练循环通常调用 flow_matching_output 获取全部中间量用于记录/诊断；
        需要直接对接 LeRobot 既有训练代码时才走本方法。
        """
        return self.flow_matching_output(
            images,
            image_masks,
            language_tokens,
            language_mask,
            state,
            actions,
            noise,
            time,
        ).losses

    def cache_context(self, context: TokenSequence) -> dict[int, dict[str, Tensor]] | None:
        """计算一次可供整个 Euler 去噪循环复用的 VLM KV cache。

        去噪的 num_steps 步中 prefix（图像 + 语言 + 状态）完全不变，
        每步重算它的前向是浪费。这里只前向一次 VLM 流（动作流为 None），
        把各层 key/value 缓存下来；后续 denoise_step 每步只计算动作 token，
        推理开销由"num_steps 次全序列前向"降为"1 次 prefix + num_steps 次后缀"。
        返回 None 表示配置未启用 KV cache（use_cache=False）。
        """
        attention = make_attention_masks(context.padding_mask, context.attention_ar)
        positions = torch.cumsum(context.padding_mask, dim=1) - 1
        # 只输入 VLM 流（inputs_embeds[1] = None），专家流不参与
        _, cache = self.vlm_with_expert(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[context.embeddings, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        return cache

    def denoise_step(
        self,
        prefix_padding_mask: Tensor,
        cache: dict[int, dict[str, Tensor]] | None,
        actions: Tensor,
        time: Tensor,
    ) -> Tensor:
        """在一个时间点计算 Euler 更新所需的速度场。

        与 predict_velocity 的区别：prefix 的 KV 从 cache 直接复用，前向
        只包含当前时刻的动作 token。attention mask 需要手工拼接：
        - prefix 区：每个动作 token 都能看到整个 prefix（block 0）；
        - 动作区：由 make_attention_masks 生成的分块因果结构（严格因果）。
        """
        tokens = self.encode_action_tokens(actions, time)
        suffix_length = tokens.padding_mask.shape[1]
        prefix_length = prefix_padding_mask.shape[1]
        # 每个动作 token 对 prefix 的全部 token 可见（块编号 0 最小）
        prefix_mask = prefix_padding_mask[:, None, :].expand(
            actions.shape[0],
            suffix_length,
            prefix_length,
        )
        # 动作段内部的分块因果 mask
        suffix_mask = make_attention_masks(tokens.padding_mask, tokens.attention_ar)
        # 拼接成 [B, L_suffix, L_prefix + L_suffix] 的完整可见性矩阵
        attention = torch.cat([prefix_mask, suffix_mask], dim=2)
        # 位置编号接在 prefix 之后：offset = prefix 有效 token 数
        offsets = prefix_padding_mask.sum(dim=-1)[:, None]
        positions = offsets + torch.cumsum(tokens.padding_mask, dim=1) - 1
        outputs, _ = self.vlm_with_expert(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=cache,
            inputs_embeds=[None, tokens.embeddings],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        expert_output = outputs[1]
        if expert_output is None:
            raise ValueError("action expert did not produce hidden states")
        # 只取动作轨迹段（最后 chunk_size 个位置）投影为速度场
        return self.action_out_proj(expert_output[:, -self.config.chunk_size :].float())

    @torch.no_grad()
    def sample_trajectory(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
        state: Tensor,
        noise: Tensor | None = None,
        on_step: Callable[[DenoisingState], None] | None = None,
    ) -> Tensor:
        """用 Euler 法从高斯噪声积分得到完整动作轨迹。

        数值格式（时间从 t=1 走到 t=0，共 num_steps 步，默认 10 步）：
            v        = 速度场(x_t, t)          # 专家一次前向
            x_{t-Δt} = x_t + Δt * v,  Δt = -1 / num_steps
        每一步只需一次专家前向（prefix KV 已缓存），因此 num_steps 通常取
        4~10 即可在质量与速度间取得平衡；结果轨迹形状
        ``[B, chunk_size, max_action_dim]``，由调用方裁剪多余的动作维。

        Args:
            images: 预处理后的多相机图像。
            image_masks: 每个相机是否有效。
            language_tokens: tokenizer 输出。
            language_mask: 语言 padding mask。
            state: 补齐后的机器人状态。
            noise: 可选固定噪声，用于可复现实验。
            on_step: 每一步完成后调用的诊断或干预回调。

        Returns:
            ``[B, chunk_size, max_action_dim]`` 动作轨迹。
        """
        batch_size = state.shape[0]
        if noise is None:
            # 默认从标准高斯出发；显式传入 noise 可复现同一条轨迹
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim),
                state.device,
            )
        # 先编码上下文并缓存 prefix KV，整个去噪循环只算一次
        context = self.encode_context(
            images,
            image_masks,
            language_tokens,
            language_mask,
            state,
        )
        cache = self.cache_context(context)
        # 时间从 1 反向走到 0，每步步长为 -1/num_steps
        step_size = -1.0 / self.config.num_steps
        actions = noise
        for step in range(self.config.num_steps):
            # 当前积分时刻：1.0, 1-1/N, ..., 1/N（最后一次更新后到达 0）
            time = torch.full(
                (batch_size,),
                1.0 + step * step_size,
                dtype=torch.float32,
                device=state.device,
            )
            velocity = self.denoise_step(context.padding_mask, cache, actions, time)
            # 显式 Euler 更新：沿速度场方向走一步
            actions = actions + step_size * velocity
            if on_step is not None:
                # 回调可记录/干预每一步的去噪状态
                on_step(DenoisingState(step, time, actions, velocity))
        return actions

    sample_actions = sample_trajectory


__all__ = [
    "DenoisingState",
    "FlowMatchingOutput",
    "TokenSequence",
    "TrajectoryFlowModel",
    "make_attention_masks",
    "pad_sequence",
    "pad_vector",
    "resize_with_pad",
    "sinusoidal_time_embedding",
]
