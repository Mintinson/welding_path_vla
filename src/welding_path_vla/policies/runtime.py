"""不同 LeRobot 模型共享的在线运行时。"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from draccus.argparsing import parse
from draccus.options import config_type
from lerobot.configs import PreTrainedConfig
from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available

from welding_path_vla.policies.action_processors import make_relative_pre_post_processors
from welding_path_vla.policies.base import Observation
from welding_path_vla.policies.checkpoint import resolve_checkpoint
from welding_path_vla.policies.spec import LeRobotPolicySpec


def load_policy_config(
    checkpoint: Path,
    config_class: type[PreTrainedConfig],
) -> PreTrainedConfig:
    """加载 checkpoint 配置，并恢复嵌套的 LeRobot 强类型字段。

    Args:
        checkpoint: 包含 ``config.json`` 的模型目录。
        config_class: 当前策略注册的具体配置类型。

    Returns:
        完整反序列化的策略配置。
    """
    config_path = checkpoint / "config.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    if "type" in config_data:
        return PreTrainedConfig.from_pretrained(checkpoint)
    with config_type("json"):
        return parse(config_class, config_path, args=[])


@dataclass(slots=True)
class LeRobotRuntime:
    """把项目统一观测送入任意已注册 LeRobot 策略。

    Attributes:
        policy: 已加载并切到 eval 模式的模型。
        preprocessor: checkpoint 保存的输入处理器。
        postprocessor: checkpoint 保存的动作反归一化处理器。
        device: 实际推理设备。
        spec: 当前策略的输入和加载差异。
        action_queue: 同一 TCP 锚点解码后的待执行世界系目标。
    """

    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: str
    spec: LeRobotPolicySpec
    action_queue: deque[np.ndarray] = field(default_factory=deque, init=False)

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
        config = load_policy_config(path, config_class)
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
        preprocessor, postprocessor = make_relative_pre_post_processors(
            config,
            pretrained_path=str(path),
            require_saved=True,
            preprocessor_overrides={"device_processor": {"device": selected}},
        )
        return cls(policy, preprocessor, postprocessor, selected, spec)

    def reset(self) -> None:
        """清空策略和 processor 的跨步状态。"""
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.action_queue.clear()

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
        """返回一个世界系绝对 EE 目标，并按 `n_action_steps` 复用动作块。"""
        import torch

        if not self.action_queue:
            processed = self.preprocessor(self.observation_sample(observation))
            with torch.inference_mode():
                relative_chunk = self.policy.predict_action_chunk(processed)
                absolute_chunk = self.postprocessor(relative_chunk)
            count = min(self.policy.config.n_action_steps, absolute_chunk.shape[1])
            self.action_queue.extend(
                absolute_chunk[0, :count].detach().cpu().numpy().astype(np.float64)
            )
        return self.action_queue.popleft()


__all__ = ["LeRobotRuntime", "load_policy_config"]
