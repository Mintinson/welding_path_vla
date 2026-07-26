"""ACT 在统一策略接口中的注册实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from welding_path_vla.core.config import AppConfig, PolicyConfig


@dataclass(frozen=True, slots=True)
class ACTPipeline:
    """提供 ACT 的训练参数与运行时加载入口。"""

    def training_overrides(self, config: PolicyConfig) -> dict[str, Any]:
        """把项目配置转换为 LeRobot ACT 参数。"""
        return {
            "policy.type": "act",
            "policy.device": config.device,
            "policy.push_to_hub": False,
            "policy.chunk_size": config.action_horizon,
            "policy.n_action_steps": config.action_steps,
            **{f"policy.{name}": value for name, value in config.parameters.items()},
        }

    def train(self, policy: PolicyConfig, training: Any) -> Any:
        """运行 LeRobot 官方 ACT 训练流水线。"""
        from welding_path_vla.policies.act.training import train

        return train(policy, training)

    def training_plan(self, policy: PolicyConfig, training: Any) -> dict[str, Any]:
        """返回 ACT 训练计划。"""
        from welding_path_vla.policies.act.training import training_plan

        return training_plan(policy, training)

    def load(self, checkpoint: str, device: str) -> Any:
        """加载带 LeRobot 前后处理器的 ACT runtime。"""
        from welding_path_vla.policies.act.runtime import ACTRuntime

        return ACTRuntime.from_pretrained(checkpoint, device)

    def evaluate(self, config: AppConfig, checkpoint: str) -> Any:
        """运行 ACT 留出 episode 的离线评估。"""
        from welding_path_vla.policies.act.evaluation import evaluate_checkpoint

        return evaluate_checkpoint(config, checkpoint)

    def deploy_simulation(self, config: AppConfig, checkpoint: str) -> Any:
        """在项目 MuJoCo 环境运行 ACT 闭环 rollout。"""
        from welding_path_vla.policies.act.rollout import deploy_simulation

        return deploy_simulation(config, checkpoint)


PIPELINE = ACTPipeline()

__all__ = ["PIPELINE", "ACTPipeline"]
