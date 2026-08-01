"""声明式策略规格对应的统一项目 pipeline。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from welding_path_vla.core.config import AppConfig, PolicyConfig
from welding_path_vla.policies.spec import LeRobotPolicySpec


@dataclass(frozen=True, slots=True)
class LeRobotPipeline:
    """提供训练、加载、离线评估和仿真部署公共入口。"""

    spec: LeRobotPolicySpec

    def training_overrides(self, config: PolicyConfig) -> dict[str, Any]:
        """生成与程序化训练配置等价的 LeRobot CLI 参数。"""
        from welding_path_vla.policies.lerobot_training import pretrained_source

        source, parameters = pretrained_source(config, self.spec)
        if self.spec.config_mode == "pretrained" or (self.spec.config_mode == "scratch" and source):
            selector = {"policy.path": source}
        else:
            selector = {"policy.type": self.spec.policy_type}
            if source:
                selector["policy.pretrained_path"] = source
        return {
            **selector,
            "policy.device": config.device,
            "policy.push_to_hub": False,
            "policy.input_features": None if source else {},
            "policy.chunk_size": config.action_horizon,
            "policy.n_action_steps": config.action_steps,
            **{f"policy.{name}": value for name, value in parameters.items()},
        }

    def train(self, policy: PolicyConfig, training: Any) -> Any:
        """运行共享 LeRobot 训练流程。"""
        from welding_path_vla.policies.lerobot_training import train

        return train(policy, training, self.spec)

    def training_plan(self, policy: PolicyConfig, training: Any) -> dict[str, Any]:
        """返回统一训练计划。"""
        from welding_path_vla.policies.lerobot_training import training_plan

        return training_plan(policy, training, self.spec)

    def load(self, checkpoint: str, device: str) -> Any:
        """加载模型和 checkpoint processor。"""
        from welding_path_vla.policies.runtime import LeRobotRuntime

        return LeRobotRuntime.from_pretrained(checkpoint, device, self.spec)

    def evaluate(self, config: AppConfig, checkpoint: str) -> Any:
        """运行任务均衡的离线评估。"""
        from welding_path_vla.policies.offline_evaluation import evaluate_checkpoint

        return evaluate_checkpoint(config, checkpoint, self.spec)

    def deploy_simulation(self, config: AppConfig, checkpoint: str) -> Any:
        """在 robosuite 环境运行闭环 rollout。"""
        from welding_path_vla.policies.runtime import LeRobotRuntime
        from welding_path_vla.policies.simulation_rollout import deploy_episodes

        runtime = LeRobotRuntime.from_pretrained(checkpoint, config.policy.device, self.spec)
        return deploy_episodes(config, runtime)


__all__ = ["LeRobotPipeline"]
