#!/usr/bin/env python3
"""从统一 YAML 训练 PAIR 动作教师；不加载 VLA 或解码视频。"""

from dataclasses import dataclass, field

from common import cli

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.traj_vla_qwen.pair import ActionAEConfig
from welding_path_vla.policies.traj_vla_qwen.pair_training import train_action_autoencoder


@dataclass
class ActionAEArguments(AppConfig):
    """公共数据/训练配置与阶段一独立网络参数。"""

    action_ae: ActionAEConfig = field(default_factory=ActionAEConfig)


@cli
def main(config: ActionAEArguments) -> None:
    """只启动轻量动作自编码器，阶段二仍使用 train_policy.py。"""
    train_action_autoencoder(config, config.action_ae)


if __name__ == "__main__":
    main()
