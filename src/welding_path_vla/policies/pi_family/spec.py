"""π0 与 π0.5 的稳定名称和 LeRobot 映射。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PIFamilySpec:
    """一个 π0 系列模型的项目级标识。

    Attributes:
        family: 项目 YAML 和策略注册表使用的名称。
        lerobot_type: LeRobot 内部注册的策略类型。
        display_name: 报告和日志使用的可读名称。
        pretrained_model: Hugging Face 官方基础 checkpoint。
    """

    family: str
    lerobot_type: str
    display_name: str
    pretrained_model: str

    def config_class(self) -> type[Any]:
        """惰性返回官方配置类。"""
        if self.lerobot_type == "pi0":
            from lerobot.policies.pi0.configuration_pi0 import PI0Config

            return PI0Config
        from lerobot.policies.pi05.configuration_pi05 import PI05Config

        return PI05Config

    def policy_class(self) -> type[Any]:
        """惰性返回官方模型类，避免配置检查提前加载 Transformers。"""
        if self.lerobot_type == "pi0":
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy

            return PI0Policy
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        return PI05Policy


PI0 = PIFamilySpec("pi0", "pi0", "π0", "lerobot/pi0_base")
PI05 = PIFamilySpec("pi0_5", "pi05", "π0.5", "lerobot/pi05_base")

__all__ = ["PI0", "PI05", "PIFamilySpec"]
