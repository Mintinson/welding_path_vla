"""ACT 数据、训练、评估和部署适配器。"""

from dataclasses import dataclass

from .data_adapter import ACTDataReport, validate_dataset


@dataclass(frozen=True, slots=True)
class ActPolicyConfig:
    """保留用于轻量实验覆盖的 ACT 模型参数。"""

    chunk_size: int = 100
    hidden_dim: int = 512


__all__ = [
    "ACTDataReport",
    "ActPolicyConfig",
    "validate_dataset",
]
