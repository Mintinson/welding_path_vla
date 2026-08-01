# Copyright 2025 Hugging Face Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trajectory-VLA 的 LeRobot policy 实现。"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from torch import Tensor

from welding_path_vla.policies.trajectory_vla.configuration_trajectory_vla import (
    TrajectoryVLAConfig,
)
from welding_path_vla.policies.trajectory_vla.flow_matching import (
    DenoisingState,
    FlowMatchingOutput,
    TrajectoryFlowModel,
    pad_vector,
    resize_with_pad,
)


class TrajectoryVLAPolicy(PreTrainedPolicy):
    """可训练、保存并部署的本地 SmolVLA 派生策略。

    ``model`` 是项目内实现的 :class:`TrajectoryFlowModel`，不会调用 LeRobot 的
    ``SmolVLAPolicy``。``forward_intermediates`` 额外暴露 flow matching 全部中间量。
    """

    config_class: Any = TrajectoryVLAConfig
    name: Any = "trajectory_vla"

    def __init__(self, config: TrajectoryVLAConfig, **kwargs: Any) -> None:
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = TrajectoryFlowModel(config)
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.model.forward = torch.compile(
                self.model.forward,
                mode=config.compile_mode,
            )
            self.model.sample_trajectory = torch.compile(
                self.model.sample_trajectory,
                mode=config.compile_mode,
            )
        self.reset()

    def reset(self) -> None:
        """清空当前 episode 尚未执行的动作。"""
        self.action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> Any:
        """返回所有未被冻结的 policy 参数。"""
        return self.parameters()

    def prepare_images(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """缩放多相机 RGB，并从 ``[0,1]`` 映射到 SigLIP 的 ``[-1,1]``。"""
        images: list[Tensor] = []
        masks: list[Tensor] = []
        present = [key for key in self.config.image_features if key in batch]
        missing = [key for key in self.config.image_features if key not in batch]
        if not present:
            raise ValueError("all configured image features are missing")

        for key in present:
            image = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            image = resize_with_pad(
                image,
                *self.config.resize_imgs_with_padding,
                value=0,
            )
            image = image * 2 - 1
            mask = batch.get(
                f"{key}_padding_mask",
                torch.ones(image.shape[0], dtype=torch.bool, device=image.device),
            ).bool()
            images.append(image)
            masks.append(mask)

        for _ in range(min(len(missing), self.config.empty_cameras)):
            images.append(torch.full_like(images[0], -1))
            masks.append(torch.zeros_like(masks[0]))
        return images, masks

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        """读取最近状态并补到动作专家固定维数。"""
        state = batch[OBS_STATE][:, -1] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        return pad_vector(state, self.config.max_state_dim)

    def prepare_action(self, batch: dict[str, Tensor]) -> Tensor:
        """把监督动作轨迹补到动作专家固定维数。"""
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    def model_inputs(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[list[Tensor], list[Tensor], Tensor, Tensor, Tensor]:
        """集中构造模型输入，方便后续增加深度图或几何 token。"""
        images, masks = self.prepare_images(batch)
        return (
            images,
            masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            self.prepare_state(batch),
        )

    def forward_intermediates(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> FlowMatchingOutput:
        """返回 flow 的 noisy actions、目标速度、预测速度和逐维 loss。"""
        images, masks, language, language_mask, state = self.model_inputs(batch)
        return self.model.flow_matching_output(
            images,
            masks,
            language,
            language_mask,
            state,
            self.prepare_action(batch),
            noise,
            time,
        )

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        on_step: Callable[[DenoisingState], None] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """预测完整短时轨迹，并裁掉内部 padding 维。"""
        self.eval()
        images, masks, language, language_mask, state = self.model_inputs(batch)
        actions = self.model.sample_trajectory(
            images,
            masks,
            language,
            language_mask,
            state,
            noise=noise,
            on_step=on_step,
        )
        action_feature = self.config.action_feature
        if action_feature is None:
            raise ValueError("Trajectory-VLA requires an action feature")
        return actions[:, :, : action_feature.shape[0]]

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """按配置的执行长度从动作块队列逐步返回动作。"""
        self.eval()
        if not self.action_queue:
            chunk = self.predict_action_chunk(batch, noise=noise, **kwargs)
            self.action_queue.extend(chunk.transpose(0, 1)[: self.config.n_action_steps])
        return self.action_queue.popleft()

    def reduce_losses(
        self,
        losses: Tensor,
        action_is_pad: Tensor | None,
        reduction: str,
    ) -> Tensor:
        """对有效动作维和有效时间步约简 flow matching loss。"""
        action_feature = self.config.action_feature
        if action_feature is None:
            raise ValueError("Trajectory-VLA requires an action feature")
        action_dimension = action_feature.shape[0]
        losses = losses[:, :, :action_dimension]
        if action_is_pad is not None:
            losses = losses * (~action_is_pad).unsqueeze(-1)
        if reduction == "none":
            if action_is_pad is None:
                return losses.mean(dim=(1, 2))
            valid = ((~action_is_pad).sum(dim=1) * action_dimension).clamp_min(1)
            return losses.sum(dim=(1, 2)) / valid
        if action_is_pad is None:
            return losses.mean()
        valid = ((~action_is_pad).sum() * action_dimension).clamp_min(1)
        return losses.sum() / valid

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Any]]:
        """执行训练前向，并返回 LeRobot 约定的 loss 与日志字典。"""
        output = self.forward_intermediates(batch, noise, time)
        loss = self.reduce_losses(output.losses, batch.get("action_is_pad"), reduction)
        logged_loss = loss.mean() if loss.ndim else loss
        return loss, {
            "loss": float(logged_loss.detach()),
            "flow_time_mean": float(output.time.mean().detach()),
            "target_velocity_norm": float(output.target_velocity.norm(dim=-1).mean().detach()),
            "predicted_velocity_norm": float(
                output.predicted_velocity.norm(dim=-1).mean().detach()
            ),
        }

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """仅对动作专家 attention 和轨迹投影层应用 LoRA。"""
        projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        return {
            "target_modules": (
                rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|"
                rf"model\.({projections}))"
            ),
            "modules_to_save": [],
        }


__all__ = ["TrajectoryVLAPolicy"]
