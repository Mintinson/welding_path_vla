from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DiffusionPolicyConfig:
    horizon: int = 16
    inference_steps: int = 10


__all__ = ["DiffusionPolicyConfig"]
