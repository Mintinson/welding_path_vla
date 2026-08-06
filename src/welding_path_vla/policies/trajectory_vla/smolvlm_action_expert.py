# Copyright 2025 Hugging Face Inc.
# SPDX-License-Identifier: Apache-2.0
"""SmolVLM 与动作专家的本地双流 Transformer 实现。

该文件依据 LeRobot 0.6 官方 ``smolvlm_with_expert.py`` 重写。类、注意力计算和
中间 embedding 均位于项目内，不依赖 ``lerobot.policies.smolvla``。

双流结构说明
------------
一条输入序列由两类 token 组成：
- VLM 流：图像 patch、语言指令、机器人状态等"上下文"token，
  宽度 = SmolVLM text hidden size；
- 动作专家流：带噪声的动作轨迹 token，宽度 = expert_width
  （默认按 expert_width_multiplier=0.75 缩放，见配置）。

两种 attention 模式（由 ``attention_mode`` 控制，默认 cross_attn）：
- ``self_attn``（共享注意力）：两流 token 拼接为一条序列做统一的
  multi-head attention，可见性由分块因果 mask 控制（专家 token 可注意
  全部 VLM 上下文与更早的动作 token）；
- ``cross_attn``（交叉注意力）：每层先由 VLM 流做自己的自注意力，
  再用其 key/value 驱动专家流 query 的注意力，上下文信息单向从 VLM
  流向专家。

推理期利用 prefix KV cache：VLM 流只前向一次，去噪每一步只重算专家流。
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
    SmolVLMModel,
)


def apply_rope(x: Tensor, positions: Tensor, max_wavelength: int = 10_000) -> Tensor:
    """对 ``[B, L, H, D]`` 的 query/key 应用旋转位置编码。

    RoPE 把位置信息编码成特征维上的旋转：第 i 个分量的旋转角为
    ``theta_i = p / max_wavelength^(2i/D)``（p 为 token 位置，D 为 head 维）。
    实现上把最后一维切为前后两半，用二维旋转的标准公式混合两半：
        x1' = x1 * cos - x2 * sin
        x2' = x2 * cos + x1 * sin
    该编码不含可学习参数、数值稳定，且配合 KV cache 时只需按绝对位置
    计算，位置可自然外推（训练序列长度之外的位置依然有定义）。

    Args:
        x: ``[B, L, H, D]`` 的 query 或 key，D 必须为偶数。
        positions: ``[B, L]`` 每个 token 的绝对位置。
        max_wavelength: 最长波长，控制最低频旋转分量的周期。

    Returns:
        旋转后的张量，形状与 dtype 与输入一致。
    """
    half = x.shape[-1] // 2
    dtype = x.dtype
    # 旋转运算在 fp32 下进行，避免 bf16 精度损失，最后再转回原 dtype
    values = x.to(torch.float32)
    # 指数序列 0, 2/D, 4/D, ..., 1-2/D：波长按 10k^(2i/D) 几何分布，
    # 前一半分量高频（短波长）、后一半低频（长波长）
    exponents = (
        2.0
        / values.shape[-1]
        * torch.arange(
            half,
            dtype=torch.float32,
            device=values.device,
        )
    )
    timescale = max_wavelength**exponents
    # 每个 (batch, token, 分量) 的旋转角：位置 / 波长
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :]
    sine, cosine = torch.sin(radians[..., None, :]), torch.cos(radians[..., None, :])
    # 二维旋转：前后两半互相混合（分组旋转式，非逐元素交错式）
    first, second = values.split(half, dim=-1)
    result = torch.empty_like(values)
    result[..., :half] = first * cosine - second * sine
    result[..., half:] = second * cosine + first * sine
    return result.to(dtype)


def expert_intermediate_size(
    hidden_dim: int,
    multiplier: float = 4,
    multiple_of: int = 256,
) -> int:
    """按 SmolVLM 的 SwiGLU 规则计算动作专家 FFN 宽度。

    SwiGLU 的中间宽度约为 ``2/3 * hidden * 4``（2/3 来自门控激活的形状
    系数，4 为默认扩展倍数），再向上取整到 256 的倍数以保证张量形状对齐、
    利用硬件对齐加速。动作专家层数独立设置、宽度按 multiplier 缩放，
    参数量级与 VLM 单层相当，用于承载轨迹预测的专用容量。
    """
    width = int(multiplier * int(2 * hidden_dim / 3))
    return multiple_of * ((width + multiple_of - 1) // multiple_of)


class SmolVLMActionExpert(nn.Module):
    """共享 SmolVLM 上下文、独立动作 token 隐状态的双流网络。

    结构组成：
    - ``self.vlm``：完整 SmolVLM（视觉塔 + connector + 文本层），提供
      embed_image / embed_language_tokens，并承载 VLM 流的前向；
    - ``self.lm_expert``：由 text config 深拷贝改造的专家模型，hidden
      size 按 expert_width_multiplier 缩放（默认 0.75 倍）、层数独立设置，
      且不保留词表 embedding（输入直接是 VLM 提供的隐状态，不经查表）。

    公开方法 ``embed_image``、``embed_language_tokens`` 和 ``forward`` 可直接
    替换，是后续修改视觉融合、语言融合和动作专家 attention 的主要接口。
    """

    def __init__(
        self,
        model_id: str,
        load_vlm_weights: bool,
        train_expert_only: bool,
        freeze_vision_encoder: bool,
        attention_mode: str,
        num_expert_layers: int,
        num_vlm_layers: int,
        self_attn_every_n_layers: int,
        expert_width_multiplier: float,
    ) -> None:
        super().__init__()
        # ---- 加载 VLM 与 processor ----
        if load_vlm_weights:
            # 加载预训练权重（bf16 + 低内存模式），继承其视觉/语言理解能力
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            # 只按配置随机初始化（从头训练，需要更多数据与更长训练时间）
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
        # tokenizer / image processor，用于文本 token 化与图像预处理
        self.processor = AutoProcessor.from_pretrained(model_id)

        if num_vlm_layers > 0:
            # 可选：只保留前 num_vlm_layers 层文本层（截断深层），
            # 用于控制参数量与推理开销
            self.vlm_model().text_model.layers = self.vlm_model().text_model.layers[:num_vlm_layers]
        self.num_vlm_layers = len(self.vlm_model().text_model.layers)
        self.config = config

        # ---- 构建动作专家 ----
        # 从 VLM 文本配置深拷贝一份再改造，保留 attention 头数等结构参数
        expert_config = copy.deepcopy(config.text_config)
        # 专家隐藏宽度按 multiplier 缩放（默认 0.75×，控制参数总量）
        expert_config.hidden_size = int(expert_config.hidden_size * expert_width_multiplier)
        expert_config.intermediate_size = expert_intermediate_size(expert_config.hidden_size)
        # 专家层数：显式指定，否则与 VLM 层数一致
        expert_config.num_hidden_layers = (
            num_expert_layers if num_expert_layers > 0 else self.num_vlm_layers
        )
        # 专家层数必须整除 VLM 层数，保证两者能逐层均匀对齐
        if self.num_vlm_layers % expert_config.num_hidden_layers:
            raise ValueError("num_vlm_layers must be divisible by num_expert_layers")
        self.lm_expert = AutoModel.from_config(expert_config)
        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = self_attn_every_n_layers

        if attention_mode == "cross_attn":
            # 交叉注意力模式：专家 K/V 投影的输入宽度要匹配 VLM 隐状态
            self.configure_cross_attention(expert_config)
        # 专家流输入来自 VLM 隐状态而非 token id，词表 embedding 置空省显存
        self.lm_expert.embed_tokens = None

        # 记下 VLM 的注意力头数，eager_attention 中按 GQA 分组展开
        self.num_attention_heads = config.text_config.num_attention_heads
        self.num_key_value_heads = config.text_config.num_key_value_heads
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_mode = attention_mode
        self.expert_hidden_size = expert_config.hidden_size
        # 按配置冻结相应模块（见 set_requires_grad）
        self.set_requires_grad()

    def vlm_model(self) -> SmolVLMModel:
        """返回 Transformers 中实际承载视觉与文本层的模型。

        SmolVLM 自动类结构为 ``vlm → model``：视觉塔、connector 与
        text_model 均挂在 ``vlm.model`` 下。统一从这里取数，避免各处
        写死访问路径。
        """
        return self.vlm.model

    def configure_cross_attention(self, expert_config: Any) -> None:
        """让动作专家的 key/value 投影接收 VLM hidden width。

        交叉注意力模式下，专家层的 query 用自己的投影，但 key/value 来自
        VLM 流的隐状态（宽度为 VLM hidden），所以必须把专家层 k_proj /
        v_proj 的输入宽度改为 VLM 的 ``num_key_value_heads * head_dim``。
        每 ``self_attn_every_n_layers`` 层保留一个专家自注意力层（跳过
        替换），保证专家流内部也有信息流通。
        """
        vlm_config = self.config.text_config
        for layer_index, layer in enumerate(self.lm_expert.layers):
            # 每隔 N 层保留专家自注意力：不替换投影，输入输出仍是专家宽度
            if (
                self.self_attn_every_n_layers > 0
                and layer_index % self.self_attn_every_n_layers == 0
            ):
                continue
            layer.self_attn.k_proj = nn.Linear(
                vlm_config.num_key_value_heads * vlm_config.head_dim,
                expert_config.num_key_value_heads * expert_config.head_dim,
                bias=expert_config.attention_bias,
            )
            layer.self_attn.v_proj = nn.Linear(
                vlm_config.num_key_value_heads * vlm_config.head_dim,
                expert_config.num_key_value_heads * expert_config.head_dim,
                bias=expert_config.attention_bias,
            )

    def set_requires_grad(self) -> None:
        """按照配置冻结视觉编码器或整个 VLM。

        三种冻结策略：
        1. freeze_vision_encoder：只冻结视觉塔（图像编码器），文本层与
           专家正常训练；
        2. train_expert_only：冻结整个 VLM，仅训练动作专家（小数据场景
           的默认选择）；
        3. 其余情况（VLM 微调）：冻结 lm_head、最后 1~2 层文本层与最终
           norm，与官方 checkpoint 的推理行为保持一致，其余 VLM 层微调。
        动作专家自己的 lm_head 始终冻结（专家流不产出词表概率）。
        """
        if self.freeze_vision_encoder:
            # eval() 防止 dropout/BatchNorm 改变冻结前向；requires_grad=False 省内存
            self.vlm_model().vision_model.eval()
            for parameter in self.vlm_model().vision_model.parameters():
                parameter.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for parameter in self.vlm.parameters():
                parameter.requires_grad = False
        else:
            # 冻结最后 1~2 层文本层：与官方实现保持一致，便于复用其 checkpoint
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)
            frozen = ["lm_head", "text_model.model.norm.weight"]
            frozen.extend(f"text_model.model.layers.{index}." for index in last_layers)
            for name, parameter in self.vlm.named_parameters():
                if any(fragment in name for fragment in frozen):
                    parameter.requires_grad = False
        for name, parameter in self.lm_expert.named_parameters():
            if "lm_head" in name:
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> SmolVLMActionExpert:
        """切换训练模式，同时保持冻结模块处于 eval。

        nn.Module.train() 会把所有子模块切到训练态；被冻结的模块必须
        始终保持 eval，否则其中的 dropout/BatchNorm 会改变前向结果
        （冻结权重也会引入随机性）。因此切换后把冻结部分强制拉回 eval。
        """
        super().train(mode)
        if self.freeze_vision_encoder:
            self.vlm_model().vision_model.eval()
        if self.train_expert_only:
            self.vlm.eval()
        return self

    def embed_image(self, image: Tensor) -> Tensor:
        """把一张相机图像编码为视觉 token 序列。

        流程：视觉塔（ViT 风格，输出 patch 级特征）→ connector（MLP，
        把视觉特征宽度映射到文本 hidden size）。返回
        ``[B, num_patches, text_hidden]`` 的 token 序列，之后与语言/状态
        token 拼接。patch_attention_mask 传 None 表示所有 patch 有效
        （图像已预处理为正方形网格，无需掩码）。
        """
        hidden = (
            self.vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.vlm_model().vision_model.dtype),
                patch_attention_mask=None,
            )
            .last_hidden_state
        )
        return self.vlm_model().connector(hidden)

    def embed_language_tokens(self, tokens: Tensor) -> Tensor:
        """把 tokenizer id 映射为 VLM token embedding。

        即词表 embedding 查表：输入 ``[B, L]`` 的 token id，输出
        ``[B, L, hidden]``。图像边界特殊 token（fake_image / global_image）
        也通过这里获取 embedding。
        """
        return self.vlm_model().text_model.get_input_embeddings()(tokens)

    def aligned_layers(self) -> tuple[list[Any], list[Any]]:
        """把较少的专家层均匀对齐到 VLM 层。

        返回两个等长列表（长度 = num_vlm_layers）：
        - vlm_layers[index]：第 index 个 VLM 文本层；
        - expert_layers[index]：与它配对的专家层；没有配对则为 None
          （该层只处理 VLM 流）。
        配对规则：专家层 i 对齐到 VLM 层 ``i * multiple``（multiple =
        num_vlm_layers // num_expert_layers），使专家层在 VLM 深度上均匀
        分布；两层之间不设专家的 VLM 层只做纯上下文编码。
        """
        vlm_layers: list[Any] = []
        expert_layers: list[Any] = []
        multiple = self.num_vlm_layers // self.num_expert_layers
        for index in range(self.num_vlm_layers):
            expert_layer = None
            # 仅当 index 是 multiple 的整数倍时该层才有专家层
            if not (multiple > 0 and index > 0 and index % multiple != 0):
                expert_layer = self.lm_expert.layers[index // multiple if multiple else index]
            vlm_layers.append(self.vlm_model().text_model.layers[index])
            expert_layers.append(expert_layer)
        return vlm_layers, expert_layers

    def project_qkv(
        self,
        layers: tuple[list[Any], list[Any]],
        embeddings: list[Tensor | None],
        layer_index: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """投影并连接 VLM 与动作专家的 query/key/value。

        共享注意力模式下两个流的 token 视作同一条序列：每个流用本层自己
        的投影矩阵生成 Q/K/V（先 layernorm 再投影，再 reshape 成多头形状），
        最后沿序列维拼接成覆盖全部 token 的 Q/K/V。
        """
        queries, keys, values = [], [], []
        # 逐流处理：VLM 流与专家流各用本层投影（宽度可能不同）
        for stream, hidden in enumerate(embeddings):
            layer = layers[stream][layer_index]
            # 该流无输入（如缓存阶段只有专家流）或无对应层（无专家层配对）
            if hidden is None or layer is None:
                continue
            normalized = layer.input_layernorm(hidden)
            # reshape 成多头形状 [..., num_heads, head_dim]
            shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
            normalized = normalized.to(layer.self_attn.q_proj.weight.dtype)
            queries.append(layer.self_attn.q_proj(normalized).view(shape))
            keys.append(layer.self_attn.k_proj(normalized).view(shape))
            values.append(layer.self_attn.v_proj(normalized).view(shape))
        return (
            torch.cat(queries, dim=1),
            torch.cat(keys, dim=1),
            torch.cat(values, dim=1),
        )

    def self_attention_layer(
        self,
        layers: tuple[list[Any], list[Any]],
        embeddings: list[Tensor | None],
        layer_index: int,
        position_ids: Tensor,
        attention_mask: Tensor,
        use_cache: bool,
        fill_kv_cache: bool,
        cache: dict[int, dict[str, Tensor]] | None,
    ) -> tuple[list[Tensor], dict[int, dict[str, Tensor]] | None]:
        """执行一次连接双流 token 的 self-attention。

        把 VLM 与专家 token 视为同一条序列做多头注意力：两流的 Q/K/V
        由 project_qkv 拼接，RoPE 按绝对位置施加，再做统一 attention。
        KV cache 语义（配合 use_cache）：
        - fill_kv_cache=True：把本次 K/V 写入 cache（前缀填充阶段）；
        - fill_kv_cache=False：把本次 K/V 追加到 cache 已有内容之后
          （去噪阶段，只计算新增 token 的 query）。
        """
        query, key, value = self.project_qkv(layers, embeddings, layer_index)
        sequence_length = query.shape[1]
        if sequence_length < position_ids.shape[1]:
            # 使用 KV cache 时只有新增 token 参与前向，位置与 mask 相应截断
            positions = position_ids[:, :sequence_length]
            mask = attention_mask[:, :sequence_length, :sequence_length]
        else:
            positions = position_ids
            mask = attention_mask
        # query/key 施加旋转位置编码（value 不需要位置信息）
        query, key = apply_rope(query, positions), apply_rope(key, positions)

        if use_cache and cache is None:
            cache = {}
        if use_cache and cache is not None:
            if fill_kv_cache:
                # 填充阶段：整段序列的 K/V 全部入缓存
                cache[layer_index] = {"key_states": key, "value_states": value}
            else:
                # 增量阶段：新 token 的 K/V 追加到历史之后
                key = torch.cat([cache[layer_index]["key_states"], key], dim=1)
                value = torch.cat([cache[layer_index]["value_states"], value], dim=1)

        output = self.eager_attention(mask, query, key, value)
        return [output], cache

    def cross_attention_layer(
        self,
        layers: tuple[list[Any], list[Any]],
        embeddings: list[Tensor | None],
        layer_index: int,
        position_ids: Tensor,
        attention_mask: Tensor,
        use_cache: bool,
        fill_kv_cache: bool,
        cache: dict[int, dict[str, Tensor]] | None,
    ) -> tuple[list[Tensor | None], dict[int, dict[str, Tensor]] | None]:
        """VLM 自注意力后，让动作专家 query 读取 VLM key/value。

        每层分两段执行：
        1. VLM 流（context）：与普通自注意力相同，用自己的 Q/K/V 投影，
           RoPE 位置从序列开头计算；
        2. 专家流（action）：query 用专家自己的 Q 投影，但 key/value 是
           VLM 隐状态经专家 K/V 投影映射而来（输入宽度由
           configure_cross_attention 匹配），上下文信息单向流入专家流；
           专家 token 的 RoPE 位置从 0 开始（相对 prefix）。
        若该层是"专家自注意力层"（self_attn_every_n_layers 整除层号），
        不会被调到这里，而是走 self_attention_layer。
        """
        outputs: list[Tensor | None] = []
        context = embeddings[0]
        action = embeddings[1]
        if context is not None and cache is None:
            # 前缀填充阶段：完整计算 VLM 流的自注意力
            context_length = context.shape[1]
            context_positions = position_ids[:, :context_length]
            context_mask = attention_mask[:, :context_length, :context_length]
            layer = layers[0][layer_index]
            normalized = layer.input_layernorm(context)
            shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
            normalized = normalized.to(layer.self_attn.q_proj.weight.dtype)
            query = apply_rope(layer.self_attn.q_proj(normalized).view(shape), context_positions)
            key = apply_rope(layer.self_attn.k_proj(normalized).view(shape), context_positions)
            value = layer.self_attn.v_proj(normalized).view(shape)
            outputs.append(self.eager_attention(context_mask, query, key, value))
        else:
            # 去噪阶段：VLM 流不重新前向，K/V 从 cache 直接取出
            key = value = None

        if use_cache and cache is None:
            cache = {}
        if use_cache and cache is not None:
            if fill_kv_cache:
                # 前缀填充：VLM 的 K/V 写入缓存（此时 key/value 必非空）
                if key is None or value is None:
                    raise ValueError("cannot fill an empty VLM key/value cache")
                cache[layer_index] = {"key_states": key, "value_states": value}
            else:
                # 去噪阶段：直接复用缓存中的 VLM K/V
                key = cache[layer_index]["key_states"]
                value = cache[layer_index]["value_states"]

        expert_layer = layers[1][layer_index]
        if expert_layer is None or action is None:
            # 该层无专家配对或无动作输入：专家流本层不更新
            outputs.append(None)
            return outputs, cache
        if key is None or value is None:
            raise ValueError("cross attention requires VLM key/value states")

        # 专家 query：本层投影 + 多头 reshape
        normalized_action = expert_layer.input_layernorm(action)
        shape = (
            *normalized_action.shape[:-1],
            -1,
            expert_layer.self_attn.head_dim,
        )
        normalized_action = normalized_action.to(expert_layer.self_attn.q_proj.weight.dtype)
        query = expert_layer.self_attn.q_proj(normalized_action).view(shape)
        # 专家 key/value：把 VLM 的 K/V 展平成 [B, L, kv_width] 后经专家
        # K/V 投影映射到专家宽度，再切回多头形状
        flat_key = key.to(expert_layer.self_attn.k_proj.weight.dtype).flatten(2)
        flat_value = value.to(expert_layer.self_attn.v_proj.weight.dtype).flatten(2)
        expert_key = expert_layer.self_attn.k_proj(flat_key).view(
            *flat_key.shape[:-1],
            -1,
            expert_layer.self_attn.head_dim,
        )
        expert_value = expert_layer.self_attn.v_proj(flat_value).view(
            *flat_value.shape[:-1],
            -1,
            expert_layer.self_attn.head_dim,
        )
        # 专家 token 的 RoPE 位置从 0 起算（相对 prefix），保证位置连续紧凑
        action_positions = position_ids[:, -action.shape[1] :]
        action_positions = action_positions - action_positions.min(dim=1, keepdim=True).values
        # mask 只保留动作 query × [前缀 + 动作] key 的可见性
        expert_mask = attention_mask[:, -action.shape[1] :, : expert_key.shape[1]]
        query = apply_rope(query, action_positions)
        outputs.append(self.eager_attention(expert_mask, query, expert_key, expert_value))
        return outputs, cache

    def forward(
        self,
        attention_mask: Tensor,
        position_ids: Tensor,
        past_key_values: dict[int, dict[str, Tensor]] | None,
        inputs_embeds: list[Tensor | None],
        use_cache: bool,
        fill_kv_cache: bool,
    ) -> tuple[list[Tensor | None], dict[int, dict[str, Tensor]] | None]:
        """执行完整双流 Transformer，并返回每个流的最终 hidden state。

        逐层循环（共 num_vlm_layers 层）：
        1. 决定本层用共享自注意力还是交叉注意力；
        2. 执行注意力（可能更新 KV cache）；
        3. 对每个流分别做输出投影 + 残差 + MLP（含 post-attention LN）。
        最后对两个流各自做 final layer norm，返回 ``[VLM 输出, 专家输出]``
        与 KV cache（未启用时返回 None）。
        """
        models = (self.vlm_model().text_model, self.lm_expert)
        layers = self.aligned_layers()
        embeddings = inputs_embeds
        for layer_index in range(self.num_vlm_layers):
            # 共享自注意力的三种情形：缓存填充阶段；非交叉注意力模式；
            # 每隔 self_attn_every_n_layers 层的"专家自注意力层"
            shared_attention = (
                fill_kv_cache
                or self.attention_mode != "cross_attn"
                or (
                    self.self_attn_every_n_layers > 0
                    and layer_index % self.self_attn_every_n_layers == 0
                )
            )
            if shared_attention:
                attention_outputs, past_key_values = self.self_attention_layer(
                    layers,
                    embeddings,
                    layer_index,
                    position_ids,
                    attention_mask,
                    use_cache,
                    fill_kv_cache,
                    past_key_values,
                )
            else:
                attention_outputs, past_key_values = self.cross_attention_layer(
                    layers,
                    embeddings,
                    layer_index,
                    position_ids,
                    attention_mask,
                    use_cache,
                    fill_kv_cache,
                    past_key_values,
                )

            # ---- 逐流更新：输出投影 + 残差 + MLP ----
            updated: list[Tensor | None] = []
            start = 0
            for stream, hidden in enumerate(embeddings):
                layer = layers[stream][layer_index]
                # 共享注意力只有一个拼接输出，按流长度切片；
                # 交叉注意力则每个流有自己的输出
                output = (
                    attention_outputs[stream]
                    if stream < len(attention_outputs)
                    else attention_outputs[0]
                )
                if hidden is None:
                    updated.append(None)
                    continue
                if layer is None:
                    updated.append(hidden)
                    continue
                if output is None:
                    raise ValueError("active transformer layer has no attention output")
                end = start + hidden.shape[1]
                # 从拼接输出中切出本流对应的片段，经 o_proj 映射回流宽度
                projected = layer.self_attn.o_proj(
                    output[:, start:end].to(layer.self_attn.o_proj.weight.dtype)
                )
                # 保留官方 bf16 原位加法的舍入顺序，确保可复用其 checkpoint。
                projected += hidden
                residual = projected.clone()
                # 标准的 Pre-Norm Transformer 块：LN → MLP → 残差
                projected = layer.mlp(layer.post_attention_layernorm(projected))
                projected += residual
                updated.append(projected)
                # 共享注意力按流长度连续切片；交叉注意力无需累计 offset
                start = end if len(attention_outputs) == 1 else 0
            embeddings = updated

        # 两个流各自经过 final layer norm（与 Transformers 默认输出一致）
        normalized: list[Tensor | None] = []
        for stream, hidden in enumerate(embeddings):
            normalized.append(models[stream].norm(hidden) if hidden is not None else None)
        return normalized, past_key_values

    def eager_attention(
        self,
        attention_mask: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """显式计算多头 attention，便于后续插入稀疏或几何 attention。

        手写实现替代 transformers 的 FlashAttention 路径，方便按需替换成
        自定义注意力（稀疏窗口、几何先验等），是本项目研究工作的主要
        插入点。支持 GQA：K/V 按 num_key_value_heads 存储，先复制扩展成
        num_attention_heads 份，再与 query 逐头做 attention。
        """
        batch_size, sequence_length = key.shape[:2]
        head_dim = query.shape[-1]
        # GQA：每个 KV head 复制 groups 份，供多个 query head 共享
        groups = self.num_attention_heads // self.num_key_value_heads
        key = key[:, :, :, None, :].expand(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            groups,
            head_dim,
        )
        value = value[:, :, :, None, :].expand_as(key)
        # 展开为与 query 相同的头数
        key = key.reshape(batch_size, sequence_length, self.num_attention_heads, head_dim)
        value = value.reshape(batch_size, sequence_length, self.num_attention_heads, head_dim)
        # Q K^T：attention score（fp32 计算避免精度损失）
        weights = torch.matmul(
            query.float().transpose(1, 2),
            key.float().transpose(1, 2).transpose(2, 3),
        )
        # 缩放因子 1 / sqrt(d)，稳定 softmax 数值
        weights *= head_dim**-0.5
        # 不允许注意到的位置填最小浮点数，softmax 后权重为 0
        negative = torch.finfo(weights.dtype).min
        weights = torch.where(attention_mask[:, None, :, :], weights, negative)
        probabilities = nn.functional.softmax(weights, dim=-1).to(value.dtype)
        # 概率加权求和得到注意力输出
        output = torch.matmul(probabilities, value.permute(0, 2, 1, 3))
        # 头维度合并还原为 [B, L, n_heads * head_dim]
        return output.permute(0, 2, 1, 3).reshape(
            batch_size,
            -1,
            self.num_attention_heads * head_dim,
        )


__all__ = ["SmolVLMActionExpert", "apply_rope", "expert_intermediate_size"]
