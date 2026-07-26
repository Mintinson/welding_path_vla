"""策略 pipeline 的惰性注册入口。"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol

from welding_path_vla.core.config import AppConfig, PolicyConfig


class PolicyPipeline(Protocol):
    """不同策略需要实现的项目级接口。"""

    def training_overrides(self, config: PolicyConfig) -> dict[str, Any]: ...

    def train(self, policy: PolicyConfig, training: Any) -> Any: ...

    def training_plan(self, policy: PolicyConfig, training: Any) -> dict[str, Any]: ...

    def load(self, checkpoint: str, device: str) -> Any: ...

    def evaluate(self, config: AppConfig, checkpoint: str) -> Any: ...

    def deploy_simulation(self, config: AppConfig, checkpoint: str) -> Any: ...


PIPELINE_MODULES = {
    "act": "welding_path_vla.policies.act.pipeline",
    "smolvla": "welding_path_vla.policies.smolvla.pipeline",
}


def get_policy_pipeline(family: str) -> PolicyPipeline:
    """按策略名称惰性加载实现，避免无关环境导入重依赖。"""
    module_name = PIPELINE_MODULES.get(family)
    if module_name is None:
        raise ValueError(f"policy pipeline is not implemented: {family}")
    return import_module(module_name).PIPELINE


__all__ = ["PolicyPipeline", "get_policy_pipeline"]
