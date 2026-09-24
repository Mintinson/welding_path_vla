"""VLA-Adapter 的 LeRobot Policy、Prismatic 主干与双动作头。"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from torch import Tensor, nn

from welding_path_vla.policies.traj_vla_qwen.prismatic import (
    PrismaticProjector,
    PrismaticVisionEncoder,
)
from welding_path_vla.policies.trajectory_vla.flow_matching import (
    DenoisingState,
    FlowMatchingOutput,
    sinusoidal_time_embedding,
)
from welding_path_vla.policies.vla_adapter.bridge import ProBridgePolicy
from welding_path_vla.policies.vla_adapter.configuration_vla_adapter import (
    VLAAdapterConfig,
)


@dataclass(frozen=True, slots=True)
class BridgeContext:
    """VLM 全层条件及机器人状态投影。"""

    raw_layers: tuple[Tensor, ...]
    query_layers: tuple[Tensor, ...]
    raw_mask: Tensor
    state: Tensor


class VLAAdapterBackbone(nn.Module):
    """DINOv2、SigLIP 与 Qwen2.5-0.5B 组成的 Prismatic VLM。"""

    def __init__(self, config: VLAAdapterConfig) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.language_model_name, padding_side="right")
        tokenizer.add_tokens([f"<|extra_{index}|>" for index in range(config.num_extra_tokens)])

        if config.load_base_weights and not config.load_prismatic_weights:
            self.qwen = AutoModelForCausalLM.from_pretrained(
                config.language_model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            qwen_config = AutoConfig.from_pretrained(config.language_model_name)
            self.qwen = AutoModelForCausalLM.from_config(qwen_config)
        self.qwen.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

        self.vision_encoder = PrismaticVisionEncoder(
            config.dino_model_name,
            config.siglip_model_name,
            config.resize_imgs_with_padding[0],
            config.load_base_weights,
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

        self.qwen.model.layers = self.qwen.model.layers[: config.num_bridge_layers]
        self.qwen.config.num_hidden_layers = config.num_bridge_layers
        self.qwen.model.config.num_hidden_layers = config.num_bridge_layers
        self.qwen.to(torch.bfloat16)
        if config.gradient_checkpointing_qwen:
            self.qwen.gradient_checkpointing_enable()
            self.qwen.config.use_cache = False
        if not config.train_vision_encoder:
            dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[
                config.frozen_vision_dtype
            ]
            self.vision_encoder.to(dtype)
        self.hidden_size = int(self.qwen.config.hidden_size)
        self.config = config
        self.set_trainable_modules()

    def load_prismatic_checkpoint(self, repo_id: str, filename: str) -> None:
        """下载并加载与上游相同的 MiniVLA Prismatic checkpoint。"""
        from huggingface_hub import hf_hub_download

        checkpoint = Path(hf_hub_download(repo_id, filename))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        if "vision_backbone" in state:
            self.vision_encoder.load_state_dict(state["vision_backbone"])
        self.projector.load_state_dict(state["projector"])
        qwen_state = {
            name.removeprefix("llm."): value for name, value in state["llm_backbone"].items()
        }
        self.qwen.load_state_dict(qwen_state)

    def set_trainable_modules(self) -> None:
        """按配置冻结视觉主干、投影器和 Qwen。"""
        config = self.config
        self.vision_encoder.requires_grad_(config.train_vision_encoder)
        self.projector.requires_grad_(config.train_projector)
        self.qwen.requires_grad_(config.train_language_model)
        if not config.train_language_model and config.train_language_last_n_layers:
            for layer in self.qwen.model.layers[-config.train_language_last_n_layers :]:
                layer.requires_grad_(True)

    def train(self, mode: bool = True) -> VLAAdapterBackbone:
        """保持完全冻结的预训练模块处于评估模式。"""
        super().train(mode)
        if not self.config.train_vision_encoder:
            self.vision_encoder.eval()
        if not self.config.train_language_model:
            self.qwen.eval()
            if self.config.train_language_last_n_layers:
                for layer in self.qwen.model.layers[-self.config.train_language_last_n_layers :]:
                    layer.train(mode)
        return self

    def embed_image(self, image: Tensor) -> Tensor:
        """保留全部 256 个视觉 patch 并投影到 Qwen hidden size。"""
        tokens = self.vision_encoder(image)
        projector_dtype = next(self.projector.parameters()).dtype
        return self.projector(tokens.to(projector_dtype)).to(
            self.qwen.get_input_embeddings().weight
        )

    def forward(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
        action_queries: Tensor,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], Tensor]:
        """返回每个 Qwen 层的 raw patch 与 ActionQuery hidden states。"""
        image_embeddings = [self.embed_image(image) for image in images]
        raw_length = sum(embedding.shape[1] for embedding in image_embeddings)
        raw_mask = torch.cat(
            [
                mask[:, None].expand(-1, embedding.shape[1])
                for mask, embedding in zip(
                    image_masks,
                    image_embeddings,
                    strict=True,
                )
            ],
            dim=1,
        )
        language = self.qwen.get_input_embeddings()(language_tokens)
        queries = action_queries[:, None].expand(-1, language.shape[0], -1).transpose(0, 1)
        embeddings = torch.cat((*image_embeddings, language, queries.to(language.dtype)), dim=1)
        query_mask = torch.ones(
            language.shape[0],
            action_queries.shape[0],
            dtype=language_mask.dtype,
            device=language_mask.device,
        )
        attention_mask = torch.cat((raw_mask.to(language_mask.dtype), language_mask, query_mask), 1)
        outputs = self.qwen.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        layers = tuple(outputs.hidden_states[1 : self.config.num_bridge_layers + 1])
        raw_layers = tuple(hidden[:, :raw_length] for hidden in layers)
        query_layers = tuple(hidden[:, -action_queries.shape[0] :] for hidden in layers)
        return raw_layers, query_layers, raw_mask


class VLAAdapterModel(nn.Module):
    """共享 VLA 条件 Bridge 的 L1 与 Flow Matching 动作模型。"""

    def __init__(self, config: VLAAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = VLAAdapterBackbone(config)
        hidden_size = self.backbone.hidden_size
        state_feature = config.robot_state_feature
        action_feature = config.action_feature
        if state_feature is None or action_feature is None:
            raise ValueError("VLA-Adapter requires inferred state and action features")
        self.action_dim = action_feature.shape[0]
        self.action_queries = nn.Embedding(config.num_action_queries, hidden_size)
        nn.init.zeros_(self.action_queries.weight)
        self.state_projector = nn.Sequential(
            nn.Linear(state_feature.shape[0], hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.bridge = ProBridgePolicy(
            hidden_size,
            config.num_bridge_layers,
            config.bridge_num_heads,
        )
        if config.action_head == "l1":
            self.l1_input = nn.Sequential(
                nn.LayerNorm(hidden_size * self.action_dim),
                nn.Linear(hidden_size * self.action_dim, hidden_size),
                nn.ReLU(),
            )
            self.l1_output = nn.Sequential(
                nn.LayerNorm(hidden_size), nn.Linear(hidden_size, self.action_dim)
            )
        else:
            self.flow_action_input = nn.Linear(self.action_dim, hidden_size)
            self.flow_time_input = nn.Linear(hidden_size * 2, hidden_size)
            self.flow_time_output = nn.Linear(hidden_size, hidden_size)
            self.flow_output = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, self.action_dim),
            )

    def encode_context(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
        state: Tensor,
    ) -> BridgeContext:
        """只运行一次 VLM，缓存 Flow 全部 Euler 步共享的逐层条件。"""
        raw, queries, raw_mask = self.backbone(
            images,
            image_masks,
            language_tokens,
            language_mask,
            self.action_queries.weight,
        )
        projector_dtype = next(self.state_projector.parameters()).dtype
        state_token = self.state_projector(state.to(projector_dtype))[:, None]
        return BridgeContext(raw, queries, raw_mask, state_token)

    def bridge_forward(self, action: Tensor, context: BridgeContext) -> Tensor:
        """把动作 latent 送入逐层 Pro Bridge。"""
        return self.bridge(
            action,
            context.raw_layers,
            context.query_layers,
            context.state,
            context.raw_mask,
        )

    def predict_l1(self, context: BridgeContext) -> Tensor:
        """从全零 latent 一次性回归完整动作块。"""
        batch = context.state.shape[0]
        width = self.backbone.hidden_size * self.action_dim
        action = torch.zeros(
            batch,
            self.config.chunk_size,
            width,
            dtype=context.state.dtype,
            device=context.state.device,
        )
        return self.l1_output(self.bridge_forward(self.l1_input(action), context).float())

    def encode_flow_action(self, actions: Tensor, time: Tensor) -> Tensor:
        """融合 noisy action 与连续时间编码。"""
        action = self.flow_action_input(actions)
        time_embedding = sinusoidal_time_embedding(
            time,
            action.shape[-1],
            self.config.min_period,
            self.config.max_period,
        ).to(action.dtype)
        time_embedding = time_embedding[:, None].expand_as(action)
        combined = torch.cat((action, time_embedding), dim=-1)
        return self.flow_time_output(F.silu(self.flow_time_input(combined)))

    def predict_velocity(self, context: BridgeContext, actions: Tensor, time: Tensor) -> Tensor:
        """预测 Flow Matching 速度场。"""
        hidden = self.bridge_forward(self.encode_flow_action(actions, time), context)
        return self.flow_output(hidden.float())

    def sample_noise(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """采样训练和推理共用的标准高斯终点。"""
        return torch.randn(shape, dtype=torch.float32, device=device)

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        """按现有 Trajectory-VLA 的 Beta 分布采样 Flow 时间。"""
        distribution = torch.distributions.Beta(
            self.config.flow_beta_alpha,
            self.config.flow_beta_beta,
        )
        time = distribution.sample((batch_size,)).to(device=device, dtype=torch.float32)
        return time * self.config.flow_time_scale + self.config.flow_time_offset

    def flow_matching_output(
        self,
        context: BridgeContext,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> FlowMatchingOutput:
        """构造直线 Flow 路径，并返回逐维速度场 MSE。"""
        noise = (
            noise if noise is not None else self.sample_noise(tuple(actions.shape), actions.device)
        )
        time = time if time is not None else self.sample_time(actions.shape[0], actions.device)
        interpolation = time[:, None, None]
        noisy_actions = interpolation * noise + (1 - interpolation) * actions
        target_velocity = noise - actions
        predicted_velocity = self.predict_velocity(context, noisy_actions, time)
        losses = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
        return FlowMatchingOutput(
            noise,
            time,
            noisy_actions,
            target_velocity,
            predicted_velocity,
            losses,
        )

    @torch.no_grad()
    def sample_flow(
        self,
        context: BridgeContext,
        noise: Tensor | None = None,
        on_step: Callable[[DenoisingState], None] | None = None,
    ) -> Tensor:
        """用 Euler 法从高斯噪声反向积分至动作轨迹。"""
        batch = context.state.shape[0]
        actions = (
            noise
            if noise is not None
            else self.sample_noise(
                (batch, self.config.chunk_size, self.action_dim),
                context.state.device,
            )
        )
        step_size = -1 / self.config.num_steps
        for step in range(self.config.num_steps):
            time = torch.full(
                (batch,),
                1 + step * step_size,
                dtype=torch.float32,
                device=actions.device,
            )
            velocity = self.predict_velocity(context, actions, time)
            actions = actions + step_size * velocity
            if on_step is not None:
                on_step(DenoisingState(step, time, actions, velocity))
        return actions


class VLAAdapterPolicy(PreTrainedPolicy):
    """可训练、保存、离线评估和仿真部署的 VLA-Adapter。"""

    config_class: Any = VLAAdapterConfig
    name: Any = "vla_adapter"

    def __init__(self, config: VLAAdapterConfig, **kwargs: Any) -> None:
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = VLAAdapterModel(config)
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.forward = torch.compile(self.forward, mode=config.compile_mode)
            if config.action_head == "flow_matching":
                self.model.sample_flow = torch.compile(
                    self.model.sample_flow,
                    mode=config.compile_mode,
                )
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: Any,
        *,
        config: Any | None = None,
        **kwargs: Any,
    ) -> VLAAdapterPolicy:
        """加载本地权重时跳过 Prismatic 基座的重复下载与初始化。"""
        if config is None:
            from lerobot.configs import PreTrainedConfig

            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
        if not isinstance(config, VLAAdapterConfig):
            raise ValueError("checkpoint config is not VLA-Adapter")
        if str(pretrained_name_or_path) == config.prismatic_repo_id:
            return cls.make_peft_base(config)
        config.load_prismatic_weights = False
        config.load_base_weights = False
        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)

    @classmethod
    def make_peft_base(cls, config: VLAAdapterConfig) -> VLAAdapterPolicy:
        """从 Prismatic 基座重建没有独立 policy 路径的 LoRA 基模型。"""
        config.load_prismatic_weights = True
        config.load_base_weights = True
        return cls(config)

    def wrap_with_peft(
        self,
        peft_config: Any | None = None,
        peft_cli_overrides: dict[str, Any] | None = None,
    ) -> PreTrainedPolicy:
        """让 PEFT checkpoint 记录可重建内部加载基座的 Prismatic 来源。"""
        if not self.config.pretrained_path and self.config.load_prismatic_weights:
            # VLA-Adapter 从 scratch 初始化，但基座权重实际来自 Prismatic。
            self.name_or_path = self.config.prismatic_repo_id
        return super().wrap_with_peft(peft_config, peft_cli_overrides)

    def reset(self) -> None:
        """清空当前 episode 尚未执行的动作。"""
        self.action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> Any:
        """只返回显式解冻的模型参数。"""
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def prepare_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """缩放多相机 RGB，并保持 Prismatic 编码器需要的 ``[0,1]``。"""
        images: list[Tensor] = []
        masks: list[Tensor] = []
        present = [key for key in self.config.image_features if key in batch]
        missing = [key for key in self.config.image_features if key not in batch]
        if not present:
            raise ValueError("all configured image features are missing")
        for key in present:
            image = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            image = F.interpolate(
                image,
                size=tuple(self.config.resize_imgs_with_padding),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp(0, 1)
            images.append(image)
            masks.append(
                batch.get(
                    f"{key}_padding_mask",
                    torch.ones(image.shape[0], dtype=torch.bool, device=image.device),
                ).bool()
            )
        for _ in range(min(len(missing), self.config.empty_cameras)):
            images.append(torch.zeros_like(images[0]))
            masks.append(torch.zeros_like(masks[0]))
        return images, masks

    def model_inputs(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[list[Tensor], list[Tensor], Tensor, Tensor, Tensor]:
        """集中读取图像、语言 token 和最近机器人状态。"""
        images, masks = self.prepare_images(batch)
        state = batch[OBS_STATE][:, -1] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        return (
            images,
            masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
        )

    def encode_context(self, batch: dict[str, Tensor]) -> BridgeContext:
        """把当前 LeRobot batch 编码为可复用的逐层 Bridge 条件。"""
        return self.model.encode_context(*self.model_inputs(batch))

    def reduce_losses(
        self,
        losses: Tensor,
        action_is_pad: Tensor | None,
        reduction: str,
    ) -> Tensor:
        """只对有效 horizon 位置约简逐维动作损失。"""
        if action_is_pad is not None:
            losses = losses * (~action_is_pad).unsqueeze(-1)
        if reduction == "none":
            if action_is_pad is None:
                return losses.mean(dim=(1, 2))
            valid = ((~action_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
            return losses.sum(dim=(1, 2)) / valid
        if action_is_pad is None:
            return losses.mean()
        valid = ((~action_is_pad).sum() * losses.shape[-1]).clamp_min(1)
        return losses.sum() / valid

    def masked_mae(self, errors: Tensor, action_is_pad: Tensor | None, start: int) -> Tensor:
        """汇总当前步或未来步的 padding-aware MAE。"""
        selected = errors[:, start:]
        if selected.shape[1] == 0:
            return errors.new_zeros(())
        if action_is_pad is None:
            return selected.mean()
        valid = ~action_is_pad[:, start:]
        return (selected * valid.unsqueeze(-1)).sum() / (
            valid.sum() * selected.shape[-1]
        ).clamp_min(1)

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Any]]:
        """按配置执行 L1 或 Flow Matching 训练前向。"""
        context = self.encode_context(batch)
        actions = batch[ACTION]
        padding = batch.get("action_is_pad")
        if self.config.action_head == "l1":
            predicted = self.model.predict_l1(context)
            errors = (predicted - actions).abs()
            loss = self.reduce_losses(errors, padding, reduction)
            logged = loss.mean() if loss.ndim else loss
            return loss, {
                "loss": float(logged.detach()),
                "l1_loss": float(logged.detach()),
                "current_action_mae": float(
                    self.masked_mae(
                        errors[:, :1],
                        padding[:, :1] if padding is not None else None,
                        0,
                    ).detach()
                ),
                "future_action_mae": float(self.masked_mae(errors, padding, 1).detach()),
            }
        output = self.model.flow_matching_output(context, actions, noise, time)
        loss = self.reduce_losses(output.losses, padding, reduction)
        logged = loss.mean() if loss.ndim else loss
        return loss, {
            "loss": float(logged.detach()),
            "flow_loss": float(logged.detach()),
            "flow_time_mean": float(output.time.mean().detach()),
            "target_velocity_norm": float(output.target_velocity.norm(dim=-1).mean().detach()),
            "predicted_velocity_norm": float(
                output.predicted_velocity.norm(dim=-1).mean().detach()
            ),
        }

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        on_step: Callable[[DenoisingState], None] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """预测完整动作块；L1 一次前向，Flow 使用配置步数积分。"""
        self.eval()
        context = self.encode_context(batch)
        if self.config.action_head == "l1":
            return self.model.predict_l1(context)
        return self.model.sample_flow(context, noise, on_step)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        """按配置的执行长度从动作块队列逐步返回动作。"""
        self.eval()
        if not self.action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)
            self.action_queue.extend(chunk.transpose(0, 1)[: self.config.n_action_steps])
        return self.action_queue.popleft()

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """对 Qwen 与视觉主干的线性层使用 LoRA，并保存新增模块。"""
        qwen_linear_names = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        target_modules = [
            name
            for name, module in self.named_modules()
            if isinstance(module, nn.Linear)
            and (
                (
                    name.startswith("model.backbone.qwen.")
                    and name.rsplit(".", 1)[-1] in qwen_linear_names
                )
                or (
                    self.config.use_vision_lora
                    and name.startswith("model.backbone.vision_encoder.")
                )
            )
        ]
        modules_to_save = [
            "model.action_queries",
            "model.state_projector",
            "model.bridge",
        ]
        modules_to_save.extend(
            ["model.l1_input", "model.l1_output"]
            if self.config.action_head == "l1"
            else [
                "model.flow_action_input",
                "model.flow_time_input",
                "model.flow_time_output",
                "model.flow_output",
            ]
        )
        if self.config.train_projector:
            modules_to_save.append("model.backbone.projector")
        return {
            "target_modules": target_modules,
            "modules_to_save": modules_to_save,
        }

    def _validate_peft_config(self, peft_config: Any) -> None:
        """允许把外部 Prismatic checkpoint 作为 LoRA 预训练来源。"""
        if self.config.load_prismatic_weights:
            return
        super()._validate_peft_config(peft_config)


__all__ = [
    "BridgeContext",
    "VLAAdapterBackbone",
    "VLAAdapterModel",
    "VLAAdapterPolicy",
]
