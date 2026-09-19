"""部署仿真入口：根据策略配置启动对应的 robosuite rollout。

本模块提供两个核心组件：
  1. DeploymentRequest：封装策略配置和部署配置，并在部署前校验必要字段。
  2. deploy_simulation：顶层部署函数，校验用户提供的部署配置，并通过策略工厂
     获取对应的策略管道来执行仿真部署。

设计意图：
  - 将部署前的校验逻辑集中到 DeploymentRequest，避免在多个策略实现中重复。
  - 通过策略注册表（get_policy_pipeline）解耦具体策略与部署流程。
  - scene bank 只提供固定场景状态和任务要求，rollout 控制项始终来自用户配置。
"""

from __future__ import annotations

from dataclasses import dataclass

from welding_path_vla.core.config import AppConfig, DeploymentConfig, PolicyConfig
from welding_path_vla.policies.factory import get_policy_pipeline


@dataclass(frozen=True, slots=True)
class DeploymentRequest:
    policy: PolicyConfig
    deployment: DeploymentConfig

    def validate(self) -> None:
        """校验部署请求的必要字段。

        目前仅要求 policy.checkpoint 非空，因为大多数策略管道在部署时需要
        加载训练好的权重。若未提供 checkpoint，则抛出 ValueError。

        Raises:
            ValueError: 当 policy.checkpoint 为空字符串或 None 时。
        """
        if not self.policy.checkpoint:
            raise ValueError("policy.checkpoint is required for deployment")


def deploy_simulation(config: AppConfig):
    """通过策略注册表启动对应的 robosuite rollout。

    执行流程：
      1. 从全局配置中提取策略和部署配置，构建 DeploymentRequest 并校验。
      2. 通过策略工厂获取对应策略家族的管道，调用其 deploy_simulation 方法，
         传入完整配置和 checkpoint 路径，返回部署结果。

    Args:
        config: 全局应用配置，包含 policy、deployment 等子配置。

    Returns:
        策略管道 deploy_simulation 的返回值，通常为 rollout 报告列表或类似结构。
    """
    # 构建并校验部署请求，确保 checkpoint 存在。
    request = DeploymentRequest(config.policy, config.deployment)
    request.validate()

    return get_policy_pipeline(config.policy.family).deploy_simulation(
        config,
        config.policy.checkpoint or "",
    )


__all__ = ["DeploymentRequest", "deploy_simulation"]
