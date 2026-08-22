"""Qwen Context Stream 与 Action Expert Stream 的成对层联合注意力。"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from welding_path_vla.policies.transformer import expert_intermediate_size


class DecoderAdapter(ABC):
    """隔离不同 Qwen 版本 decoder 内部差异的最小接口。

    Qwen3 可通过新 adapter 处理 Q/K Norm，Qwen3.5 可进一步处理混合层；
    视觉、Flow Matching 和 LeRobot Policy 不依赖具体 Transformers 类名。
    """

    family: str

    @abstractmethod
    def apply_rotary(
        self,
        query: Tensor,
        key: Tensor,
        rotary_embedding: Any,
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """应用当前模型版本的位置编码。"""

    def project_qkv(self, layer: Any, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """执行层归一化与 Q/K/V 投影，返回 ``[B,H,L,D]``。"""
        normalized = layer.input_layernorm(hidden)
        attention = layer.self_attn
        shape = (*normalized.shape[:-1], -1, attention.head_dim)
        normalized = normalized.to(attention.q_proj.weight.dtype)
        query = attention.q_proj(normalized).view(shape).transpose(1, 2)
        key = attention.k_proj(normalized).view(shape).transpose(1, 2)
        value = attention.v_proj(normalized).view(shape).transpose(1, 2)
        query_norm = getattr(attention, "q_norm", None)
        key_norm = getattr(attention, "k_norm", None)
        return (
            query_norm(query) if query_norm is not None else query,
            key_norm(key) if key_norm is not None else key,
            value,
        )

    def project_query(self, layer: Any, hidden: Tensor) -> Tensor:
        """只投影 Expert query，供单向 Cross-Attention 使用。"""
        normalized = layer.input_layernorm(hidden)
        attention = layer.self_attn
        shape = (*normalized.shape[:-1], -1, attention.head_dim)
        query = attention.q_proj(normalized.to(attention.q_proj.weight.dtype))
        query = query.view(shape).transpose(1, 2)
        query_norm = getattr(attention, "q_norm", None)
        return query_norm(query) if query_norm is not None else query

    def project_cross_kv(
        self,
        layer: Any,
        context_key: Tensor,
        context_value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """把 Context K/V 映射到 Expert 的 KV 头空间。"""
        attention = layer.self_attn
        key_input = context_key.transpose(1, 2).flatten(2)
        value_input = context_value.transpose(1, 2).flatten(2)
        key = attention.k_proj(key_input.to(attention.k_proj.weight.dtype))
        value = attention.v_proj(value_input.to(attention.v_proj.weight.dtype))
        shape = (*key.shape[:-1], -1, attention.head_dim)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        key_norm = getattr(attention, "k_norm", None)
        return (key_norm(key) if key_norm is not None else key), value

    def apply_rotary_single(
        self,
        value: Tensor,
        rotary_embedding: Any,
        position_ids: Tensor,
    ) -> Tensor:
        """对单个 Q 或 K 应用与当前 Qwen 版本一致的 RoPE。"""
        return self.apply_rotary(value, value, rotary_embedding, position_ids)[0]

    def complete_layer(self, layer: Any, hidden: Tensor, attention: Tensor) -> Tensor:
        """完成每条流自己的输出投影、残差和 MLP。"""
        projected = layer.self_attn.o_proj(attention.to(layer.self_attn.o_proj.weight.dtype))
        hidden = hidden + projected
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


class Qwen25DecoderAdapter(DecoderAdapter):
    """Qwen2.5 decoder 的 RoPE 与层访问实现。"""

    family = "qwen2_5"

    def apply_rotary(
        self,
        query: Tensor,
        key: Tensor,
        rotary_embedding: Any,
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """调用 Transformers 官方 Qwen2 RoPE。"""
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        cosine, sine = rotary_embedding(key, position_ids)
        return apply_rotary_pos_emb(query, key, cosine, sine)


def make_decoder_adapter(family: str) -> DecoderAdapter:
    """按配置选择 Qwen decoder adapter。"""
    if family == "qwen2_5":
        return Qwen25DecoderAdapter()
    raise ValueError(f"decoder adapter is not implemented: {family}")


class PairedLayerDecoder(nn.Module):
    """在每一层联合计算 Context Stream 与 Action Expert Stream。

    两条流各自保留 LayerNorm、Q/K/V、输出投影和 MLP；只有注意力矩阵在
    token 维连接。非对称 mask 保证上下文不读取带噪动作，而动作可以读取
    上下文和更早的动作。推理时缓存 Context Stream 的逐层 K/V。
    """

    def __init__(
        self,
        language_model: Any,
        expert_model: Any,
        adapter: DecoderAdapter,
        checkpoint_qwen: bool = False,
        checkpoint_expert: bool = False,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.expert_model = expert_model
        self.adapter = adapter
        self.checkpoint_qwen = checkpoint_qwen
        self.checkpoint_expert = checkpoint_expert
        self.num_attention_heads = language_model.config.num_attention_heads
        self.num_key_value_heads = language_model.config.num_key_value_heads
        if len(language_model.layers) != len(expert_model.layers):
            raise ValueError("paired Qwen and expert streams must have equal depths")

    def run_checkpointed(
        self,
        function: Any,
        *inputs: Tensor,
        enabled: bool,
    ) -> Any:
        """训练时重算指定子图，推理和默认配置保持直接执行。"""
        if self.training and enabled and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(function, *inputs, use_reentrant=False)
        return function(*inputs)

    def attention(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:
        """计算支持 GQA 的联合注意力并返回 ``[B,L,H*D]``。"""
        groups = self.num_attention_heads // self.num_key_value_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        weights = torch.matmul(query.float(), key.float().transpose(2, 3))
        weights *= query.shape[-1] ** -0.5
        weights = weights.masked_fill(~mask[:, None], torch.finfo(weights.dtype).min)
        probabilities = torch.softmax(weights, dim=-1).to(value.dtype)
        output = torch.matmul(probabilities, value)
        return output.transpose(1, 2).flatten(2)

    def paired_layer(
        self,
        layer_index: int,
        context: Tensor | None,
        action: Tensor | None,
        attention_mask: Tensor,
        position_ids: Tensor,
        cache: dict[int, dict[str, Tensor]] | None,
        fill_kv_cache: bool,
    ) -> tuple[list[Tensor | None], dict[int, dict[str, Tensor]] | None]:
        """执行一层双流联合 Self-Attention。"""
        layers = (self.language_model.layers[layer_index], self.expert_model.layers[layer_index])
        streams = ((context, layers[0]), (action, layers[1]))
        checkpointing = (self.checkpoint_qwen, self.checkpoint_expert)
        projected = [
            self.run_checkpointed(
                partial(self.adapter.project_qkv, layer),
                hidden,
                enabled=enabled,
            )
            for (hidden, layer), enabled in zip(streams, checkpointing, strict=True)
            if hidden is not None
        ]
        queries = torch.cat([values[0] for values in projected], dim=2)
        keys = torch.cat([values[1] for values in projected], dim=2)
        values = torch.cat([values[2] for values in projected], dim=2)
        queries, keys = self.adapter.apply_rotary(
            queries,
            keys,
            self.language_model.rotary_emb,
            position_ids,
        )

        if cache is not None:
            if fill_kv_cache:
                cache[layer_index] = {"key_states": keys, "value_states": values}
            else:
                prefix = cache[layer_index]
                keys = torch.cat([prefix["key_states"], keys], dim=2)
                values = torch.cat([prefix["value_states"], values], dim=2)

        attended = self.run_checkpointed(
            self.attention,
            queries,
            keys,
            values,
            attention_mask,
            enabled=any(checkpointing),
        )
        updated: list[Tensor | None] = []
        offset = 0
        for (hidden, layer), enabled in zip(streams, checkpointing, strict=True):
            if hidden is None:
                updated.append(None)
                continue
            end = offset + hidden.shape[1]
            updated.append(
                self.run_checkpointed(
                    partial(self.adapter.complete_layer, layer),
                    hidden,
                    attended[:, offset:end],
                    enabled=enabled,
                )
            )
            offset = end
        return updated, cache

    def prepare_inputs(
        self,
        inputs_embeds: list[Tensor | None],
    ) -> tuple[Tensor | None, Tensor | None]:
        """把双流输入转换到各自主干的计算精度。"""
        context, action = inputs_embeds
        if context is not None:
            context = context.to(self.language_model.layers[0].self_attn.q_proj.weight.dtype)
        if action is not None:
            action = action.to(self.expert_model.layers[0].self_attn.q_proj.weight.dtype)
        return context, action

    def normalize_outputs(
        self,
        context: Tensor | None,
        action: Tensor | None,
    ) -> list[Tensor | None]:
        """分别应用 Context 与 Expert 的最终 LayerNorm。"""
        return [
            self.language_model.norm(context) if context is not None else None,
            self.expert_model.norm(action) if action is not None else None,
        ]

    def forward(
        self,
        attention_mask: Tensor,
        position_ids: Tensor,
        past_key_values: dict[int, dict[str, Tensor]] | None,
        inputs_embeds: list[Tensor | None],
        use_cache: bool,
        fill_kv_cache: bool,
    ) -> tuple[list[Tensor | None], dict[int, dict[str, Tensor]] | None]:
        """执行成对层前向并可填充或读取 Context Stream KV cache。"""
        context, action = self.prepare_inputs(inputs_embeds)
        cache = {} if use_cache and past_key_values is None else past_key_values
        for layer_index in range(len(self.language_model.layers)):
            (context, action), cache = self.paired_layer(
                layer_index,
                context,
                action,
                attention_mask,
                position_ids,
                cache,
                fill_kv_cache,
            )
        return self.normalize_outputs(context, action), cache


class InterleavedSACADecoder(PairedLayerDecoder):
    """交替执行联合 Self-Attention 与单向 Cross-Attention。

    每 ``self_attn_every_n_layers`` 层使用一次双流联合 SA，使动作 token
    彼此通信；其余层分别更新 Qwen Context，并让 Expert query 只读取
    Context K/V。该结构与 ``trajectory_vla`` 的 ``cross_attn`` 模式一致。
    """

    def __init__(
        self,
        language_model: Any,
        expert_model: Any,
        adapter: DecoderAdapter,
        self_attn_every_n_layers: int,
        checkpoint_qwen: bool = False,
        checkpoint_expert: bool = False,
    ) -> None:
        super().__init__(
            language_model,
            expert_model,
            adapter,
            checkpoint_qwen,
            checkpoint_expert,
        )
        self.self_attn_every_n_layers = self_attn_every_n_layers
        self.configure_cross_attention()

    def uses_self_attention(self, layer_index: int) -> bool:
        """返回本层是否承担双流联合 SA。"""
        return layer_index % self.self_attn_every_n_layers == 0

    def configure_cross_attention(self) -> None:
        """让 CA 层的 Expert K/V 接收 Qwen Context K/V。"""
        for index, (context_layer, expert_layer) in enumerate(
            zip(self.language_model.layers, self.expert_model.layers, strict=True)
        ):
            if self.uses_self_attention(index):
                continue
            context_width = context_layer.self_attn.k_proj.out_features
            for name in ("k_proj", "v_proj"):
                original = getattr(expert_layer.self_attn, name)
                projection = nn.Linear(
                    context_width,
                    original.out_features,
                    bias=original.bias is not None,
                ).to(device=original.weight.device, dtype=original.weight.dtype)
                setattr(expert_layer.self_attn, name, projection)

    def cross_attention_layer(
        self,
        layer_index: int,
        context: Tensor | None,
        action: Tensor | None,
        attention_mask: Tensor,
        position_ids: Tensor,
        cache: dict[int, dict[str, Tensor]] | None,
        fill_kv_cache: bool,
    ) -> tuple[list[Tensor | None], dict[int, dict[str, Tensor]] | None]:
        """更新 Context 自注意力，并让 Expert 单向读取 Context。"""
        context_layer = self.language_model.layers[layer_index]
        expert_layer = self.expert_model.layers[layer_index]
        context_key: Tensor | None = None
        context_value: Tensor | None = None
        updated_context: Tensor | None = None

        if context is not None:
            context_length = context.shape[1]
            projected_context = self.run_checkpointed(
                partial(self.adapter.project_qkv, context_layer),
                context,
                enabled=self.checkpoint_qwen,
            )
            query, key, value = projected_context
            query, context_key = self.adapter.apply_rotary(
                query,
                key,
                self.language_model.rotary_emb,
                position_ids[:, :context_length],
            )
            context_value = value
            attended = self.run_checkpointed(
                self.attention,
                query,
                context_key,
                value,
                attention_mask[:, :context_length, :context_length],
                enabled=self.checkpoint_qwen,
            )
            updated_context = self.run_checkpointed(
                partial(self.adapter.complete_layer, context_layer),
                context,
                attended,
                enabled=self.checkpoint_qwen,
            )

        if cache is not None:
            if fill_kv_cache:
                if context_key is None or context_value is None:
                    raise ValueError("cannot cache an empty Context stream")
                cache[layer_index] = {
                    "key_states": context_key,
                    "value_states": context_value,
                }
            else:
                context_key = cache[layer_index]["key_states"]
                context_value = cache[layer_index]["value_states"]

        if action is None:
            return [updated_context, None], cache
        if context_key is None or context_value is None:
            raise ValueError("Expert Cross-Attention requires Context key/value states")

        query = self.run_checkpointed(
            partial(self.adapter.project_query, expert_layer),
            action,
            enabled=self.checkpoint_expert,
        )
        action_positions = position_ids[:, -action.shape[1] :]
        action_positions = action_positions - action_positions.min(dim=1, keepdim=True).values
        query = self.adapter.apply_rotary_single(
            query,
            self.language_model.rotary_emb,
            action_positions,
        )
        expert_key, expert_value = self.run_checkpointed(
            partial(self.adapter.project_cross_kv, expert_layer),
            context_key,
            context_value,
            enabled=self.checkpoint_expert,
        )
        expert_mask = attention_mask[:, -action.shape[1] :, : expert_key.shape[2]]
        attended = self.run_checkpointed(
            self.attention,
            query,
            expert_key,
            expert_value,
            expert_mask,
            enabled=self.checkpoint_expert,
        )
        updated_action = self.run_checkpointed(
            partial(self.adapter.complete_layer, expert_layer),
            action,
            attended,
            enabled=self.checkpoint_expert,
        )
        return [updated_context, updated_action], cache

    def forward(
        self,
        attention_mask: Tensor,
        position_ids: Tensor,
        past_key_values: dict[int, dict[str, Tensor]] | None,
        inputs_embeds: list[Tensor | None],
        use_cache: bool,
        fill_kv_cache: bool,
    ) -> tuple[list[Tensor | None], dict[int, dict[str, Tensor]] | None]:
        """按配置的周期执行 Interleaved SA/CA Layers。"""
        context, action = self.prepare_inputs(inputs_embeds)
        cache = {} if use_cache and past_key_values is None else past_key_values
        for layer_index in range(len(self.language_model.layers)):
            if self.uses_self_attention(layer_index):
                (context, action), cache = self.paired_layer(
                    layer_index,
                    context,
                    action,
                    attention_mask,
                    position_ids,
                    cache,
                    fill_kv_cache,
                )
            else:
                (context, action), cache = self.cross_attention_layer(
                    layer_index,
                    context,
                    action,
                    attention_mask,
                    position_ids,
                    cache,
                    fill_kv_cache,
                )
        return self.normalize_outputs(context, action), cache


def make_expert_decoder(
    config: Any,
    language_model: Any,
    expert_model: Any,
    adapter: DecoderAdapter,
) -> PairedLayerDecoder:
    """按统一 attention 配置构造 Qwen Action Expert decoder。"""
    common = {
        "language_model": language_model,
        "expert_model": expert_model,
        "adapter": adapter,
        "checkpoint_qwen": config.gradient_checkpointing_qwen,
        "checkpoint_expert": config.gradient_checkpointing_expert,
    }
    if config.attention_mode == "cross_attn":
        return InterleavedSACADecoder(
            **common,
            self_attn_every_n_layers=config.self_attn_every_n_layers,
        )
    return PairedLayerDecoder(**common)


class PrismaticQwenWithExpert(nn.Module):
    """预对齐 Prismatic-Qwen2.5 与轻量 Action Expert 的完整双流主干。"""

    def __init__(self, config: Any) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        from welding_path_vla.policies.traj_vla_qwen.prismatic import (
            PrismaticProjector,
            PrismaticVisionEncoder,
            SpatialTokenMerger,
        )

        self.policy_config = config
        self.adapter = make_decoder_adapter(config.language_model_family)
        tokenizer = AutoTokenizer.from_pretrained(config.language_model_name, padding_side="right")
        tokenizer.add_tokens([f"<|extra_{index}|>" for index in range(config.num_extra_tokens)])
        self.tokenizer = tokenizer

        if config.load_base_weights and not config.load_prismatic_weights:
            self.qwen = AutoModelForCausalLM.from_pretrained(
                config.language_model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            qwen_config = AutoConfig.from_pretrained(config.language_model_name)
            if not config.load_prismatic_weights:
                qwen_config.num_hidden_layers = config.num_vlm_layers
            self.qwen = AutoModelForCausalLM.from_config(qwen_config)
        self.qwen.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

        self.vision_encoder = PrismaticVisionEncoder(
            config.dino_model_name,
            config.siglip_model_name,
            config.resize_imgs_with_padding[0],
            config.load_base_weights,
        )
        self.token_merger = SpatialTokenMerger(
            self.vision_encoder.hidden_size,
            config.vision_patch_grid,
            config.token_merge_factor,
        )
        self.projector = PrismaticProjector(
            self.vision_encoder.hidden_size,
            self.qwen.config.hidden_size,
        )
        if config.load_prismatic_weights:
            self.load_prismatic_checkpoint(
                config.prismatic_repo_id,
                config.prismatic_checkpoint_file,
            )

        self.qwen.model.layers = self.qwen.model.layers[: config.num_vlm_layers]
        self.qwen.config.num_hidden_layers = config.num_vlm_layers
        self.qwen.model.config.num_hidden_layers = config.num_vlm_layers
        expert_config = copy.deepcopy(self.qwen.config)
        expert_config.hidden_size = int(
            self.qwen.config.hidden_size * config.expert_width_multiplier
        )
        expert_config.intermediate_size = expert_intermediate_size(expert_config.hidden_size)
        expert_config.num_hidden_layers = config.num_expert_layers
        expert_config.head_dim = self.qwen.model.layers[0].self_attn.head_dim
        from transformers import AutoModel

        self.expert = AutoModel.from_config(expert_config)
        self.expert.embed_tokens = None
        self.qwen.to(torch.bfloat16)
        self.expert.to(torch.bfloat16)
        if not config.train_vision_encoder:
            vision_dtype = {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }[config.frozen_vision_dtype]
            self.vision_encoder.to(vision_dtype)
        self.expert_hidden_size = expert_config.hidden_size
        self.language_hidden_size = self.qwen.config.hidden_size
        self.decoder = make_expert_decoder(
            config,
            self.qwen.model,
            self.expert,
            self.adapter,
        )
        self.set_trainable_modules()

    def load_prismatic_checkpoint(self, repo_id: str, filename: str) -> None:
        """下载并加载 MiniVLA 原始 Prismatic checkpoint。"""
        from huggingface_hub import hf_hub_download

        checkpoint = Path(hf_hub_download(repo_id, filename))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        self.load_prismatic_state(state)

    def load_prismatic_state(self, state: dict[str, dict[str, Tensor]]) -> None:
        """把 Prismatic 模块化 state dict 映射到本地实现。"""
        if "vision_backbone" in state:
            self.vision_encoder.load_state_dict(state["vision_backbone"])
        self.projector.load_state_dict(state["projector"])
        qwen_state = {
            name.removeprefix("llm."): value for name, value in state["llm_backbone"].items()
        }
        self.qwen.load_state_dict(qwen_state)

    def set_trainable_modules(self) -> None:
        """按配置冻结或解冻视觉、投影器、Qwen 和动作专家。"""
        config = self.policy_config
        self.vision_encoder.requires_grad_(config.train_vision_encoder)
        self.token_merger.requires_grad_(config.train_token_merger)
        self.projector.requires_grad_(config.train_projector)
        self.qwen.requires_grad_(config.train_language_model)
        if not config.train_language_model and config.train_language_last_n_layers:
            layers = self.qwen.model.layers[-config.train_language_last_n_layers :]
            for layer in layers:
                layer.requires_grad_(True)
        self.expert.requires_grad_(config.train_expert)

    def train(self, mode: bool = True) -> PrismaticQwenWithExpert:
        """保持完全冻结的预训练主干处于评估模式。"""
        super().train(mode)
        if not self.policy_config.train_vision_encoder:
            self.vision_encoder.eval()
        if not self.policy_config.train_language_model:
            self.qwen.eval()
            if self.policy_config.train_language_last_n_layers:
                for layer in self.qwen.model.layers[
                    -self.policy_config.train_language_last_n_layers :
                ]:
                    layer.train(mode)
        return self

    def embed_image(self, image: Tensor) -> Tensor:
        """编码、压缩并投影单路相机图像。"""
        tokens = self.vision_encoder(image)
        tokens = self.token_merger(tokens.to(self.token_merger.projection.weight.dtype))
        tokens = self.projector(tokens.to(self.projector.projector[0].weight.dtype))
        return tokens.to(self.qwen.get_input_embeddings().weight.dtype)

    def embed_language_tokens(self, tokens: Tensor) -> Tensor:
        """将 Qwen token id 映射为 Context Stream embedding。"""
        return self.qwen.get_input_embeddings()(tokens)

    def forward(
        self,
        attention_mask: Tensor,
        position_ids: Tensor,
        past_key_values: dict[int, dict[str, Tensor]] | None,
        inputs_embeds: list[Tensor | None],
        use_cache: bool,
        fill_kv_cache: bool,
    ) -> tuple[list[Tensor | None], dict[int, dict[str, Tensor]] | None]:
        """执行配置选择的双流 decoder。"""
        return self.decoder(
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            fill_kv_cache,
        )


__all__ = [
    "DecoderAdapter",
    "InterleavedSACADecoder",
    "PairedLayerDecoder",
    "PrismaticQwenWithExpert",
    "Qwen25DecoderAdapter",
    "make_decoder_adapter",
    "make_expert_decoder",
]
