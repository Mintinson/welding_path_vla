"""π0 训练入口。"""

from typing import Any

from welding_path_vla.core.config import PolicyConfig, TrainingConfig
from welding_path_vla.policies.pi_family.spec import PI0
from welding_path_vla.policies.pi_family.training import (
    lerobot_train_config as family_train_config,
)
from welding_path_vla.policies.pi_family.training import pi_config as family_policy_config
from welding_path_vla.policies.pi_family.training import train as family_train
from welding_path_vla.policies.pi_family.training import training_plan as family_training_plan


def pi0_config(policy: PolicyConfig) -> Any:
    """构造 LeRobot 官方 π0 配置。"""
    return family_policy_config(policy, PI0)


def lerobot_train_config(policy: PolicyConfig, training: TrainingConfig) -> Any:
    """构造 π0 的 LeRobot 训练配置。"""
    return family_train_config(policy, training, PI0)


def training_plan(policy: PolicyConfig, training: TrainingConfig) -> dict[str, Any]:
    """返回 π0 训练计划。"""
    return family_training_plan(policy, training, PI0)


def train(policy: PolicyConfig, training: TrainingConfig):
    """启动 π0 训练。"""
    return family_train(policy, training, PI0)


__all__ = ["lerobot_train_config", "pi0_config", "train", "training_plan"]
