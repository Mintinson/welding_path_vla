from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SmolVlaPolicyConfig:
    action_horizon: int = 10
    precision: str = "bfloat16"


__all__ = ["SmolVlaPolicyConfig"]
