"""带 LeRobot tokenizer、归一化器和动作反归一化器的 SmolVLA 运行时。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available

from welding_path_vla.policies.base import Observation
from welding_path_vla.policies.checkpoint import resolve_checkpoint


@dataclass(slots=True)
class SmolVLARuntime:
    """实现项目 Policy 协议，并保留自然语言任务输入。"""

    policy: SmolVLAPolicy
    preprocessor: Any
    postprocessor: Any
    device: str

    @classmethod
    def from_pretrained(cls, checkpoint: str | Path, device: str) -> SmolVLARuntime:
        """加载模型、tokenizer 和训练数据归一化统计。"""
        path = resolve_checkpoint(checkpoint)
        selected_device = (
            device if is_torch_device_available(device) else str(auto_select_torch_device())
        )
        config = PreTrainedConfig.from_pretrained(path)
        config.device = selected_device
        policy = SmolVLAPolicy.from_pretrained(path, config=config).to(selected_device).eval()
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": selected_device}},
        )
        return cls(policy, preprocessor, postprocessor, selected_device)

    def reset(self) -> None:
        """清空 SmolVLA 内部动作队列。"""
        self.policy.reset()

    def select_action(self, observation: Observation) -> np.ndarray:
        """把双相机、13D 状态和任务指令转换为一个物理量动作。"""
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
            action = self.policy.select_action(processed)
            action = self.postprocessor(action)
        return action.squeeze(0).detach().cpu().numpy()


__all__ = ["SmolVLARuntime"]
