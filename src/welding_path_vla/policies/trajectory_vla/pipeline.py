"""Trajectory-VLA 在项目统一策略接口中的注册实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from welding_path_vla.core.config import AppConfig, PolicyConfig
from welding_path_vla.policies.trajectory_vla.training import DEFAULT_PRETRAINED_MODEL


@dataclass(frozen=True, slots=True)
class TrajectoryVLAPipeline:
    """提供训练、加载、离线评估和仿真部署入口。"""

    def training_overrides(self, config: PolicyConfig) -> dict[str, Any]:
        """把项目配置转换为等价的 LeRobot CLI 参数。"""
        parameters = dict(config.parameters)
        pretrained = config.checkpoint or parameters.pop(
            "pretrained_model",
            DEFAULT_PRETRAINED_MODEL,
        )
        return {
            "policy.type": "trajectory_vla",
            "policy.pretrained_path": pretrained,
            "policy.device": config.device,
            "policy.push_to_hub": False,
            "policy.input_features": None,
            "policy.chunk_size": config.action_horizon,
            "policy.n_action_steps": config.action_steps,
            **{f"policy.{name}": value for name, value in parameters.items()},
        }

    def train(self, policy: PolicyConfig, training: Any) -> Any:
        """运行本地 Trajectory-VLA 的 LeRobot 训练流水线。"""
        from welding_path_vla.policies.trajectory_vla.training import train

        return train(policy, training)

    def training_plan(self, policy: PolicyConfig, training: Any) -> dict[str, Any]:
        """返回可记录、可复查的训练计划。"""
        from welding_path_vla.policies.trajectory_vla.training import training_plan

        return training_plan(policy, training)

    def load(self, checkpoint: str, device: str) -> Any:
        """加载本地模型及其 processor。"""
        from welding_path_vla.policies.trajectory_vla.runtime import TrajectoryVLARuntime

        return TrajectoryVLARuntime.from_pretrained(checkpoint, device)

    def evaluate(self, config: AppConfig, checkpoint: str) -> Any:
        """运行留出 episode 离线评估。"""
        from welding_path_vla.policies.trajectory_vla.evaluation import evaluate_checkpoint

        return evaluate_checkpoint(config, checkpoint)

    def deploy_simulation(self, config: AppConfig, checkpoint: str) -> Any:
        """在项目 robosuite 环境运行闭环 rollout。"""
        from welding_path_vla.policies.trajectory_vla.rollout import deploy_simulation

        return deploy_simulation(config, checkpoint)


PIPELINE = TrajectoryVLAPipeline()

__all__ = ["PIPELINE", "TrajectoryVLAPipeline"]
