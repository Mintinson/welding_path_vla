from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SmolVlaPolicyConfig:
    """SmolVLA 在 30 Hz 数据上的最小策略配置。"""

    action_horizon: int = 15
    precision: str = "bfloat16"


__all__ = ["SmolVlaPolicyConfig"]
