"""Trajectory-VLA 的 tokenizer、归一化与在线动作运行时。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available

from welding_path_vla.policies.base import Observation
from welding_path_vla.policies.checkpoint import resolve_checkpoint
from welding_path_vla.policies.trajectory_vla.configuration_trajectory_vla import (
    TrajectoryVLAConfig,
)
from welding_path_vla.policies.trajectory_vla.modeling_trajectory_vla import (
    TrajectoryVLAPolicy,
)


@dataclass(slots=True)
class TrajectoryVLARuntime:
    """实现项目 Policy 协议，并保留语言与双相机输入。"""

    policy: TrajectoryVLAPolicy
    preprocessor: Any
    postprocessor: Any
    device: str

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        device: str,
    ) -> TrajectoryVLARuntime:
        """加载本地模型实现及 checkpoint 保存的 processor。"""
        path = resolve_checkpoint(checkpoint)
        selected = device if is_torch_device_available(device) else str(auto_select_torch_device())
        config = PreTrainedConfig.from_pretrained(path)
        if not isinstance(config, TrajectoryVLAConfig):
            raise ValueError(f"checkpoint is not Trajectory-VLA: {path}")
        config.device = selected
        policy = TrajectoryVLAPolicy.from_pretrained(path, config=config).to(selected).eval()
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": selected}},
        )
        return cls(policy, preprocessor, postprocessor, selected)

    def reset(self) -> None:
        """清空尚未执行的动作块。"""
        self.policy.reset()

    def select_action(self, observation: Observation) -> np.ndarray:
        """把项目观测转换成一个已反归一化动作。"""
        import torch

        batch: dict[str, Any] = {
            "observation.state": torch.as_tensor(
                observation.state,
                dtype=torch.float32,
            ).unsqueeze(0),
            "task": [observation.instruction],
        }
        for name, image in observation.images.items():
            batch[f"observation.images.{name}"] = (
                torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32)
                .permute(2, 0, 1)
                .div(255)
                .unsqueeze(0)
            )
        processed = self.preprocessor(batch)
        with torch.inference_mode():
            action = self.postprocessor(self.policy.select_action(processed))
        return action.squeeze(0).detach().cpu().numpy()


__all__ = ["TrajectoryVLARuntime"]
