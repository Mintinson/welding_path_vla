"""π0 系列共享的 LeRobot 运行时。"""

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
from welding_path_vla.policies.pi_family.spec import PIFamilySpec


@dataclass(slots=True)
class PIRuntime:
    """把项目观测转换为 π0 系列动作块。

    Attributes:
        policy: LeRobot 官方 π0 或 π0.5 模型。
        preprocessor: checkpoint 保存的 tokenizer、归一化和设备处理器。
        postprocessor: checkpoint 保存的动作反归一化处理器。
        device: 实际执行推理的 torch 设备。
        family: 当前运行时对应的模型规格。
    """

    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: str
    family: PIFamilySpec

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        device: str,
        family: PIFamilySpec,
    ) -> PIRuntime:
        """加载模型、PaliGemma tokenizer 和数据集归一化统计。"""
        config_class = family.config_class()
        policy_class = family.policy_class()
        path = resolve_checkpoint(checkpoint)
        selected_device = (
            device if is_torch_device_available(device) else str(auto_select_torch_device())
        )
        config = PreTrainedConfig.from_pretrained(path)
        if not isinstance(config, config_class):
            raise ValueError(f"checkpoint policy is not {family.display_name}: {path}")
        config.device = selected_device
        if config.use_peft:
            from peft import PeftConfig, PeftModel

            adapter_config = PeftConfig.from_pretrained(path)
            base_model = adapter_config.base_model_name_or_path
            if not base_model:
                raise ValueError(f"PEFT checkpoint does not identify its base model: {path}")
            policy = policy_class.from_pretrained(base_model, config=config)
            policy = PeftModel.from_pretrained(policy, path, config=adapter_config)
        else:
            policy = policy_class.from_pretrained(path, config=config)
        policy = policy.to(selected_device).eval()
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": selected_device}},
        )
        return cls(policy, preprocessor, postprocessor, selected_device, family)

    def reset(self) -> None:
        """清空策略内部尚未执行的动作队列。"""
        self.policy.reset()

    def select_action(self, observation: Observation) -> np.ndarray:
        """用双相机、13D 状态和语言指令预测一个物理量动作。"""
        import torch

        sample: dict[str, Any] = {
            "observation.state": torch.as_tensor(observation.state, dtype=torch.float32),
            "task": observation.instruction,
        }
        for name, image in observation.images.items():
            sample[f"observation.images.{name}"] = (
                torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32)
                .permute(2, 0, 1)
                .div(255)
            )
        processed = self.preprocessor(sample)
        with torch.inference_mode():
            action = self.postprocessor(self.policy.select_action(processed))
        return action.squeeze(0).detach().cpu().numpy()


__all__ = ["PIRuntime"]
