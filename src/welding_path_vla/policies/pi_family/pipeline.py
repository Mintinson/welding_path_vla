"""π0 系列在项目统一策略接口中的实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from welding_path_vla.core.config import AppConfig, PolicyConfig
from welding_path_vla.policies.pi_family.spec import PIFamilySpec


@dataclass(frozen=True, slots=True)
class PIPipeline:
    """为一个 π0 系列模型提供训练、评估和部署入口。"""

    family: PIFamilySpec

    def training_overrides(self, config: PolicyConfig) -> dict[str, Any]:
        """把项目配置转换为等价的 LeRobot CLI 参数。"""
        parameters = dict(config.parameters)
        configured_source = parameters.pop("pretrained_model", self.family.pretrained_model)
        return {
            "policy.path": config.checkpoint or configured_source,
            "policy.device": config.device,
            "policy.push_to_hub": False,
            "policy.input_features": None,
            "policy.chunk_size": config.action_horizon,
            "policy.n_action_steps": config.action_steps,
            **{f"policy.{name}": value for name, value in parameters.items()},
        }

    def train(self, policy: PolicyConfig, training: Any) -> Any:
        """运行 LeRobot 官方 π0 系列训练流水线。"""
        from welding_path_vla.policies.pi_family.training import train

        return train(policy, training, self.family)

    def training_plan(self, policy: PolicyConfig, training: Any) -> dict[str, Any]:
        """返回可记录、可复查的训练计划。"""
        from welding_path_vla.policies.pi_family.training import training_plan

        return training_plan(policy, training, self.family)

    def load(self, checkpoint: str, device: str) -> Any:
        """加载带 tokenizer 和归一化器的运行时。"""
        from welding_path_vla.policies.pi_family.runtime import PIRuntime

        return PIRuntime.from_pretrained(checkpoint, device, self.family)

    def evaluate(self, config: AppConfig, checkpoint: str) -> Any:
        """运行留出 episode 的离线评估。"""
        from welding_path_vla.policies.pi_family.evaluation import evaluate_checkpoint

        return evaluate_checkpoint(config, checkpoint, self.family)

    def deploy_simulation(self, config: AppConfig, checkpoint: str) -> Any:
        """在 robosuite 环境运行闭环 rollout。"""
        from welding_path_vla.policies.pi_family.runtime import PIRuntime
        from welding_path_vla.policies.simulation_rollout import deploy_episodes

        runtime = PIRuntime.from_pretrained(checkpoint, config.policy.device, self.family)
        return deploy_episodes(config, runtime)


__all__ = ["PIPipeline"]
