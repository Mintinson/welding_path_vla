"""Prismatic-Qwen Trajectory-VLA 的 LeRobot Policy。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from welding_path_vla.policies.traj_vla_qwen.configuration_traj_vla_qwen import (
    TrajVLAQwenConfig,
)
from welding_path_vla.policies.traj_vla_qwen.geometry_grounding import (
    GeometryGroundingHeads,
    geometry_grounding_losses,
)
from welding_path_vla.policies.traj_vla_qwen.prismatic import DenseGeometryContext
from welding_path_vla.policies.traj_vla_qwen.processor_traj_vla_qwen import (
    QwenPromptProcessorStep,
)
from welding_path_vla.policies.traj_vla_qwen.qwen_with_expert import (
    PrismaticQwenWithExpert,
)
from welding_path_vla.policies.trajectory_vla.flow_matching import (
    FlowMatchingOutput,
    TokenSequence,
    TrajectoryFlowModel,
    VelocityPrediction,
    make_attention_masks,
)
from welding_path_vla.policies.trajectory_vla.modeling_trajectory_vla import (
    TrajectoryVLAPolicy,
)


class QwenTrajectoryFlowModel(TrajectoryFlowModel):
    """复用公共 Flow Matching，只替换双流多模态主干。"""

    def __init__(self, config: TrajVLAQwenConfig) -> None:
        backbone = PrismaticQwenWithExpert(config)
        super().__init__(config, vlm_with_expert=backbone)
        self.geometry_grounding_heads = (
            GeometryGroundingHeads(backbone.expert_hidden_size)
            if config.use_geometry_grounding
            else None
        )
        if config.training_stage == "grounding_warmup":
            self.requires_grad_(False)
            resampler = backbone.decoder.geometry_resampler
            if resampler is None or self.geometry_grounding_heads is None:
                raise ValueError("grounding warm-up requires Resampler and grounding heads")
            resampler.requires_grad_(True)
            self.geometry_grounding_heads.requires_grad_(True)

    def embed_context_image(self, image: Tensor) -> tuple[Tensor, Tensor | None]:
        """启用几何支路时同时取得压缩前 DINOv2 patch。"""
        if not self.config.use_geometry_branch:
            return super().embed_context_image(image)
        semantic, dense = self.vlm_with_expert.embed_image_with_geometry(image)
        return semantic, dense

    def attach_expert_context(
        self,
        context: TokenSequence,
        image_features: list[Any | None],
        image_masks: list[Tensor],
        language_slice: slice,
        state_slice: slice,
    ) -> TokenSequence:
        """把双相机 DINO patch 和 Qwen 片段 mask 附加给 Action Expert。"""
        if not self.config.use_geometry_branch:
            return context
        dense_features = [feature for feature in image_features if isinstance(feature, Tensor)]
        if len(dense_features) != len(image_features):
            raise ValueError("the geometry branch requires dense features for every camera")
        patch_tokens = torch.stack(dense_features, dim=1)
        patch_mask = torch.stack(
            [mask[:, None].expand(-1, patch_tokens.shape[2]) for mask in image_masks],
            dim=1,
        )
        language_mask = torch.zeros_like(context.padding_mask)
        state_mask = torch.zeros_like(context.padding_mask)
        language_mask[:, language_slice] = context.padding_mask[:, language_slice]
        state_mask[:, state_slice] = context.padding_mask[:, state_slice]
        context.expert_context = DenseGeometryContext(
            patch_tokens,
            patch_mask,
            language_mask,
            state_mask,
        )
        return context

    def attach_action_context(
        self,
        context: TokenSequence,
        actions: Tensor,
        action_padding_mask: Tensor | None,
    ) -> TokenSequence:
        """把 clean normalized action chunk 附加给训练期 motion posterior。"""
        if not self.config.use_motion_latent:
            return context
        if not isinstance(context.expert_context, DenseGeometryContext):
            raise ValueError("motion latent requires dense geometry context")
        context.expert_context.clean_actions = actions
        context.expert_context.action_padding_mask = (
            action_padding_mask
            if action_padding_mask is not None
            else torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)
        )
        return context

    def encode_action_tokens(self, noisy_actions: Tensor, time: Tensor) -> TokenSequence:
        """启用 motion latent 时在动作块前加入 learned motion slot。"""
        tokens = super().encode_action_tokens(noisy_actions, time)
        if not self.config.use_motion_latent:
            return tokens
        motion = self.vlm_with_expert.decoder.motion_latent
        if motion is None:
            raise ValueError("motion latent module was not constructed")
        slot = motion.initial_slot(noisy_actions.shape[0], noisy_actions.device).to(
            tokens.embeddings.dtype
        )
        mask = torch.ones(slot.shape[:2], dtype=torch.bool, device=slot.device)
        return TokenSequence(
            torch.cat((slot, tokens.embeddings), dim=1),
            torch.cat((mask, tokens.padding_mask), dim=1),
            torch.cat((mask, tokens.attention_ar), dim=1),
        )

    def add_grounding_predictions(self, auxiliary: dict[str, Tensor]) -> dict[str, Tensor]:
        """启用监督头时把三个 readout prediction 加入研究输出。"""
        if self.geometry_grounding_heads is None or not auxiliary:
            return auxiliary
        return {**auxiliary, **self.geometry_grounding_heads(auxiliary)}

    def predict_velocity_output(
        self,
        context: TokenSequence,
        action_tokens: TokenSequence,
    ) -> VelocityPrediction:
        """复用 Flow Matching 前向，并在训练期附加 grounding predictions。"""
        prediction = super().predict_velocity_output(context, action_tokens)
        return VelocityPrediction(
            prediction.velocity,
            self.add_grounding_predictions(prediction.auxiliary_outputs),
        )

    def grounding_intermediates(self, context: TokenSequence) -> dict[str, Tensor]:
        """执行 Stage 1 的第 0 个 Qwen layer、Resampler 和三个监督头。"""
        if not isinstance(context.expert_context, DenseGeometryContext):
            raise ValueError("geometry grounding requires dense DINO context")
        attention = make_attention_masks(context.padding_mask, context.attention_ar)
        positions = torch.cumsum(context.padding_mask, dim=1) - 1
        auxiliary = self.vlm_with_expert.forward_geometry(
            attention,
            positions,
            context.embeddings,
            context.expert_context,
        )
        return self.add_grounding_predictions(auxiliary)


class TrajVLAQwenPolicy(TrajectoryVLAPolicy):
    """可训练、保存、评估和仿真部署的 Prismatic-Qwen 策略。"""

    config_class: Any = TrajVLAQwenConfig
    name: Any = "traj_vla_qwen"
    flow_model_class: Any = QwenTrajectoryFlowModel
    prompt_processor_class: Any = QwenPromptProcessorStep

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: Any,
        *,
        config: Any | None = None,
        **kwargs: Any,
    ) -> TrajVLAQwenPolicy:
        """加载本地 Policy 权重时跳过 2.6 GB Prismatic 重复初始化。"""
        if config is None:
            from lerobot.configs import PreTrainedConfig

            loaded = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
            if not isinstance(loaded, TrajVLAQwenConfig):
                raise ValueError("checkpoint is not a TrajVLA-Qwen policy")
            config = loaded
        if not isinstance(config, TrajVLAQwenConfig):
            raise ValueError("checkpoint config is not TrajVLA-Qwen")
        config.load_prismatic_weights = False
        config.load_base_weights = False
        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            **kwargs,
        )

    def prepare_images(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """把多相机 RGB 直接缩放到 Prismatic 的 224×224 输入。"""
        images: list[Tensor] = []
        masks: list[Tensor] = []
        present = [key for key in self.config.image_features if key in batch]
        missing = [key for key in self.config.image_features if key not in batch]
        if not present:
            raise ValueError("all configured image features are missing")

        target = tuple(self.config.resize_imgs_with_padding)
        for key in present:
            image = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            image = F.interpolate(
                image,
                size=target,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            images.append(image.clamp(0, 1))
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

    def get_optim_params(self) -> Any:
        """只返回显式解冻的模块参数。"""
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def prepare_for_inference(self) -> None:
        """删除只在训练期使用的 grounding heads 与 motion posterior。"""
        self.model.geometry_grounding_heads = None
        motion = self.model.vlm_with_expert.decoder.motion_latent
        if motion is not None:
            motion.discard_posterior()

    def forward_grounding_intermediates(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """返回 Stage 1 的几何 readout、预测和 patch 诊断张量。"""
        images, masks, language, language_mask, state = self.model_inputs(batch)
        context = self.model.encode_context(images, masks, language, language_mask, state)
        return self.model.grounding_intermediates(context)

    def reduce_auxiliary_loss(self, loss: Tensor, reduction: str) -> Tensor:
        """保持与 Flow Matching 相同的 mean/none 训练约定。"""
        return loss if reduction == "none" else loss.mean()

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Any]]:
        """按训练阶段组合 Flow Matching、几何辅助损失和诊断日志。"""
        output: FlowMatchingOutput | None = None
        if self.config.training_stage == "grounding_warmup":
            auxiliary = self.forward_grounding_intermediates(batch)
            flow_loss: Tensor | None = None
        else:
            output = self.forward_intermediates(batch, noise, time)
            auxiliary = output.auxiliary_outputs
            flow_loss = self.reduce_losses(output.losses, batch.get("action_is_pad"), reduction)

        auxiliary_losses = (
            geometry_grounding_losses(auxiliary, batch)
            if self.config.use_geometry_grounding
            else {}
        )
        weights = self.config.geometry_aux_loss_weights
        if auxiliary_losses:
            first_loss = next(iter(auxiliary_losses.values()))
            weighted = torch.zeros_like(self.reduce_auxiliary_loss(first_loss, reduction))
            for name, weight in zip(("seam", "tangent", "orientation"), weights, strict=True):
                weighted = weighted + weight * self.reduce_auxiliary_loss(
                    auxiliary_losses[name],
                    reduction,
                )
        elif flow_loss is not None:
            weighted = torch.zeros_like(flow_loss)
        else:
            raise ValueError("grounding warm-up did not produce auxiliary losses")
        motion_kl = auxiliary.get("motion.kl")
        if motion_kl is not None:
            weighted = weighted + self.config.motion_kl_weight * self.reduce_auxiliary_loss(
                motion_kl,
                reduction,
            )
        loss = weighted if flow_loss is None else flow_loss + weighted
        logged = loss.mean() if loss.ndim else loss
        info: dict[str, Any] = {"loss": float(logged.detach())}
        if flow_loss is not None and output is not None:
            info["flow_loss"] = float((flow_loss.mean() if flow_loss.ndim else flow_loss).detach())
            info["flow_time_mean"] = float(output.time.mean().detach())
            info["target_velocity_norm"] = float(
                output.target_velocity.norm(dim=-1).mean().detach()
            )
            info["predicted_velocity_norm"] = float(
                output.predicted_velocity.norm(dim=-1).mean().detach()
            )
        info.update(
            {
                f"geometry_{name}_loss": float(value.mean().detach())
                for name, value in auxiliary_losses.items()
            }
        )
        if motion_kl is not None:
            info["motion_kl_loss"] = float(motion_kl.mean().detach())
        return loss, info

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """按配置选择 Expert、Qwen 或两者的 LoRA attention 投影。"""
        targets = {
            "expert": r"model\.vlm_with_expert\.expert\.layers\.\d+\.self_attn\.(q|v)_proj",
            "qwen": (
                r"model\.vlm_with_expert\.qwen\.model\.layers\.\d+"
                r"\.self_attn\.(q|v)_proj"
            ),
        }
        selected = (
            [targets["expert"], targets["qwen"]]
            if self.config.lora_target == "all"
            else [targets[self.config.lora_target]]
        )
        modules_to_save = [
            "model.action_in_proj",
            "model.action_out_proj",
            "model.action_time_mlp_in",
            "model.action_time_mlp_out",
        ]
        if self.config.train_state_proj:
            modules_to_save.append("model.state_proj")
        if self.config.train_token_merger:
            modules_to_save.append("model.vlm_with_expert.token_merger")
        if self.config.train_projector:
            modules_to_save.append("model.vlm_with_expert.projector")
        if self.config.use_geometry_branch and self.config.train_geometry_resampler:
            modules_to_save.append("model.vlm_with_expert.decoder.geometry_resampler")
        if self.config.use_geometry_grounding:
            modules_to_save.append("model.geometry_grounding_heads")
        if self.config.use_motion_latent:
            modules_to_save.append("model.vlm_with_expert.decoder.motion_latent")
        if self.config.lora_target == "qwen" and self.config.train_expert:
            modules_to_save.append("model.vlm_with_expert.expert")
        return {
            "target_modules": f"({'|'.join(selected)})",  # 需要 LoRA 的模块
            "modules_to_save": modules_to_save,  # 需要全量训练的模块
        }

    def _validate_peft_config(self, peft_config: Any) -> None:
        """把外部 Prismatic 权重视为有效的 PEFT 预训练来源。"""
        if self.config.load_prismatic_weights:
            return
        # 在标准 PEFT / Transformers 库中，父类 super()._validate_peft_config(peft_config)
        # 会检查预训练权重的结构与传入的 peft_config 是否严格匹配（例如检查配置文件中的 peft_type、
        # 适配器层名是否能在已加载的权重文件中找到等）。

        # 当从外部来源（例如 Prismatic VLM 架构）加载自定义的预训练权重时，
        # 权重的命名空间或保存格式可能与标准 PEFT 规范略有不同，这会导致父类的 _validate_peft_config
        # 抛出 ValueError 或警告，误判为“非法/不兼容的 PEFT 适配器配置文件”。
        super()._validate_peft_config(peft_config)


__all__ = ["QwenTrajectoryFlowModel", "TrajVLAQwenPolicy"]
