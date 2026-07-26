"""SmolVLA 在统一策略接口中的注册实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from welding_path_vla.core.config import AppConfig, PolicyConfig
from welding_path_vla.policies.process import run_logged_command


@dataclass(frozen=True, slots=True)
class SmolVLAPipeline:
    """保留 SmolVLA 基线的训练接口。"""

    def training_overrides(self, config: PolicyConfig) -> dict[str, Any]:
        return {
            "policy.type": "smolvla",
            "policy.device": config.device,
            "policy.push_to_hub": False,
            "policy.chunk_size": config.action_horizon,
            "policy.n_action_steps": config.action_steps,
            **{f"policy.{name}": value for name, value in config.parameters.items()},
        }

    def train(self, policy: PolicyConfig, training: Any) -> Any:
        """SmolVLA 继续使用 LeRobot 官方训练入口。"""
        from pathlib import Path

        from welding_path_vla.policies.training import TrainingRequest

        command = TrainingRequest(policy, training).command()
        return run_logged_command(command, Path(training.output_dir))

    def training_plan(self, policy: PolicyConfig, training: Any) -> dict[str, Any]:
        return {
            "backend": "lerobot-train",
            "policy": policy.family,
            "dataset": training.dataset_root,
            "steps": training.steps,
            "output_dir": training.output_dir,
        }

    def load(self, checkpoint: str, device: str) -> Any:
        raise NotImplementedError(
            "SmolVLA runtime will be implemented with its deployment pipeline"
        )

    def evaluate(self, config: AppConfig, checkpoint: str) -> Any:
        raise NotImplementedError("SmolVLA evaluation will be implemented with its policy pipeline")

    def deploy_simulation(self, config: AppConfig, checkpoint: str) -> Any:
        raise NotImplementedError("SmolVLA simulation deployment is not implemented")


PIPELINE = SmolVLAPipeline()

__all__ = ["PIPELINE", "SmolVLAPipeline"]
