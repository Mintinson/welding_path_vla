from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from welding_path_vla.core.config import AppConfig, DeploymentConfig, PolicyConfig
from welding_path_vla.policies.factory import get_policy_pipeline


@dataclass(frozen=True, slots=True)
class DeploymentRequest:
    policy: PolicyConfig
    deployment: DeploymentConfig

    def validate(self) -> None:
        if not self.policy.checkpoint:
            raise ValueError("policy.checkpoint is required for deployment")

    @property
    def log_dir(self) -> Path:
        return Path(self.deployment.log_dir)


def deploy_simulation(config: AppConfig):
    """通过策略注册表启动对应的 MuJoCo rollout。"""
    request = DeploymentRequest(config.policy, config.deployment)
    request.validate()
    return get_policy_pipeline(config.policy.family).deploy_simulation(
        config,
        config.policy.checkpoint or "",
    )


__all__ = ["DeploymentRequest", "deploy_simulation"]
