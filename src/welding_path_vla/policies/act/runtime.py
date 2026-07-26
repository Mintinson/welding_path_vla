"""带 LeRobot 前后处理器的 ACT 推理运行时。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available

from welding_path_vla.policies.base import Observation


def resolve_checkpoint(path: str | Path) -> Path:
    """接受 run、step 或 pretrained_model 目录并定位模型。"""
    root = Path(path)
    candidates = (
        root,
        root / "pretrained_model",
        root / "checkpoints" / "last" / "pretrained_model",
    )
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot find ACT pretrained_model under: {root}")


@dataclass(slots=True)
class ACTRuntime:
    """实现项目 Policy 协议，并忽略自然语言字段。"""

    policy: ACTPolicy
    preprocessor: Any
    postprocessor: Any
    device: str

    @classmethod
    def from_pretrained(cls, checkpoint: str | Path, device: str) -> ACTRuntime:
        """加载权重以及训练时保存的归一化处理器。"""

        path = resolve_checkpoint(checkpoint)
        selected_device = (
            device if is_torch_device_available(device) else str(auto_select_torch_device())
        )
        config = PreTrainedConfig.from_pretrained(path)
        config.device = selected_device
        policy = ACTPolicy.from_pretrained(path, config=config).to(selected_device).eval()
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": selected_device}},
        )
        return cls(policy, preprocessor, postprocessor, selected_device)

    def reset(self) -> None:
        """清空 ACT 内部动作队列。"""
        self.policy.reset()

    def select_action(self, observation: Observation) -> np.ndarray:
        """把双相机 RGB 和 13D 状态送入 ACT，返回一个物理量 action。"""
        import torch

        batch: dict[str, Any] = {
            "observation.state": torch.as_tensor(observation.state, dtype=torch.float32).unsqueeze(
                0
            )
        }
        for name, image in observation.images.items():
            key = f"observation.images.{name}"
            tensor = torch.as_tensor(
                np.ascontiguousarray(image),
                dtype=torch.float32,
            ).permute(2, 0, 1)
            batch[key] = tensor.div(255).unsqueeze(0)
        processed = self.preprocessor(batch)
        with torch.inference_mode():
            action = self.policy.select_action(processed)
            action = self.postprocessor(action)
        return action.squeeze(0).detach().cpu().numpy()


__all__ = ["ACTRuntime", "resolve_checkpoint"]
