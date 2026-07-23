from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActPolicyConfig:
    chunk_size: int = 100
    hidden_dim: int = 512


__all__ = ["ActPolicyConfig"]
