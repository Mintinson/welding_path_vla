from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from welding_path_vla.core.config import DeploymentConfig, PolicyConfig


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


__all__ = ["DeploymentRequest"]
