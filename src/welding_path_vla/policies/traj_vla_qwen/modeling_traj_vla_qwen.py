"""Prismatic-Qwen Trajectory-VLA 的 LeRobot Policy。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from welding_path_vla.policies.traj_vla_qwen.configuration_traj_vla_qwen import (
    TrajVLAQwenConfig,
)
from welding_path_vla.policies.traj_vla_qwen.prismatic import DenseGeometryContext
from welding_path_vla.policies.traj_vla_qwen.processor_traj_vla_qwen import (
    QwenPromptProcessorStep,
)
from welding_path_vla.policies.traj_vla_qwen.qwen_with_expert import (
    PrismaticQwenWithExpert,
)
from welding_path_vla.policies.trajectory_vla.flow_matching import (
    TokenSequence,
    TrajectoryFlowModel,
)
from welding_path_vla.policies.trajectory_vla.modeling_trajectory_vla import (
    TrajectoryVLAPolicy,
)


class QwenTrajectoryFlowModel(TrajectoryFlowModel):
    """复用公共 Flow Matching，只替换双流多模态主干。"""

    def __init__(self, config: TrajVLAQwenConfig) -> None:
        backbone = PrismaticQwenWithExpert(config)
        super().__init__(config, vlm_with_expert=backbone)

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
        ) # TODO: dangerous?
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
