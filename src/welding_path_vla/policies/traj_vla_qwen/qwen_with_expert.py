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
        context, action = inputs_embeds
        if context is not None:
            context = context.to(self.language_model.layers[0].self_attn.q_proj.weight.dtype)
        if action is not None:
            action = action.to(self.expert_model.layers[0].self_attn.q_proj.weight.dtype)
        cache = {} if use_cache and past_key_values is None else past_key_values

        for layer_index, (context_layer, expert_layer) in enumerate(
            zip(self.language_model.layers, self.expert_model.layers, strict=True)
        ):
            streams = ((context, context_layer), (action, expert_layer))
            checkpointing = (self.checkpoint_qwen, self.checkpoint_expert)
            projected = []
            for (hidden, layer), enabled in zip(streams, checkpointing, strict=True):
                if hidden is not None:
                    projected.append(
                        self.run_checkpointed(
                            partial(self.adapter.project_qkv, layer),
                            hidden,
                            enabled=enabled,
                        )
                    )
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
            context, action = updated

        outputs = [
            self.language_model.norm(context) if context is not None else None,
            self.expert_model.norm(action) if action is not None else None,
        ]
        return outputs, cache


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
        self.decoder = PairedLayerDecoder(
            self.qwen.model,
            self.expert,
            self.adapter,
            checkpoint_qwen=config.gradient_checkpointing_qwen,
            checkpoint_expert=config.gradient_checkpointing_expert,
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
        """执行逐层交织的双流 decoder。"""
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
    "PairedLayerDecoder",
    "PrismaticQwenWithExpert",
    "Qwen25DecoderAdapter",
    "make_decoder_adapter",
]
