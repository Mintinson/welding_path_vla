"""不同 LeRobot 模型共享的在线运行时。"""

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
from welding_path_vla.policies.spec import LeRobotPolicySpec


@dataclass(slots=True)
class LeRobotRuntime:
    """把项目统一观测送入任意已注册 LeRobot 策略。

    Attributes:
        policy: 已加载并切到 eval 模式的模型。
        preprocessor: checkpoint 保存的输入处理器。
        postprocessor: checkpoint 保存的动作反归一化处理器。
        device: 实际推理设备。
        spec: 当前策略的输入和加载差异。
    """

    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: str
    spec: LeRobotPolicySpec

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        device: str,
        spec: LeRobotPolicySpec,
    ) -> LeRobotRuntime:
        """加载模型、processor 和可选 PEFT adapter。"""
        path = resolve_checkpoint(checkpoint)
        selected = device if is_torch_device_available(device) else str(auto_select_torch_device())
        config_class = spec.config_class()
        policy_class = spec.policy_class()
        config = PreTrainedConfig.from_pretrained(path)
        if not isinstance(config, config_class):
            raise ValueError(f"checkpoint policy is not {spec.display_name}: {path}")
        config.device = selected

        if config.use_peft:
            from peft import PeftConfig, PeftModel

            adapter = PeftConfig.from_pretrained(path)
            if not adapter.base_model_name_or_path:
                raise ValueError(f"PEFT checkpoint does not identify its base model: {path}")
            policy = policy_class.from_pretrained(adapter.base_model_name_or_path, config=config)
            policy = PeftModel.from_pretrained(policy, path, config=adapter)
        else:
            policy = policy_class.from_pretrained(path, config=config)
        policy = policy.to(selected).eval()
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": selected}},
        )
        return cls(policy, preprocessor, postprocessor, selected, spec)

    def reset(self) -> None:
        """清空策略内部尚未执行的动作队列。"""
        self.policy.reset()

    def observation_sample(self, observation: Observation) -> dict[str, Any]:
        """构造 processor 需要的状态、双相机和可选语言输入。"""
        import torch

        state = torch.as_tensor(observation.state, dtype=torch.float32)
        sample: dict[str, Any] = {"observation.state": state}
        for name, image in observation.images.items():
            sample[f"observation.images.{name}"] = (
                torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32)
                .permute(2, 0, 1)
                .div(255)
            )
        if self.spec.language:
            sample["task"] = observation.instruction
        if self.spec.processor_adds_batch:
            return sample
        batched = {
            key: value.unsqueeze(0) if hasattr(value, "unsqueeze") else [value]
            for key, value in sample.items()
        }
        return batched

    def select_action(self, observation: Observation) -> np.ndarray:
        """返回一个经过反归一化的物理量动作。"""
        import torch

        processed = self.preprocessor(self.observation_sample(observation))
        with torch.inference_mode():
            action = self.postprocessor(self.policy.select_action(processed))
        return action.squeeze(0).detach().cpu().numpy()


__all__ = ["LeRobotRuntime"]
