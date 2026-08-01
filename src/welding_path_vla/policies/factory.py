"""策略 pipeline 的惰性注册入口。"""

from __future__ import annotations

from typing import Any, Protocol

from welding_path_vla.core.config import AppConfig, PolicyConfig
from welding_path_vla.policies.lerobot_pipeline import LeRobotPipeline
from welding_path_vla.policies.spec import POLICY_SPECS, LeRobotPolicySpec


class PolicyPipeline(Protocol):
    """不同策略需要实现的项目级接口。"""

    @property
    def spec(self) -> LeRobotPolicySpec:
        """返回驱动公共流程的策略规格。"""
        ...

    def training_overrides(self, config: PolicyConfig) -> dict[str, Any]: ...

    def train(self, policy: PolicyConfig, training: Any) -> Any: ...

    def training_plan(self, policy: PolicyConfig, training: Any) -> dict[str, Any]: ...

    def load(self, checkpoint: str, device: str) -> Any: ...

    def evaluate(self, config: AppConfig, checkpoint: str) -> Any: ...

    def deploy_simulation(self, config: AppConfig, checkpoint: str) -> Any: ...


PIPELINES = {name: LeRobotPipeline(spec) for name, spec in POLICY_SPECS.items()}


def get_policy_pipeline(family: str) -> PolicyPipeline:
    """按项目配置中的策略名称返回统一 pipeline。"""
    pipeline = PIPELINES.get(family)
    if pipeline is None:
        raise ValueError(f"policy pipeline is not implemented: {family}")
    return pipeline


__all__ = ["PIPELINES", "PolicyPipeline", "get_policy_pipeline"]
