import numpy as np

rng = np.random.default_rng(104)


def sample_value(
    center: float,
    radius: float,
    decimals: int,
    lower: float = -float("inf"),
    upper: float = float("inf"),
) -> float:
    """在给定范围均匀采样并按物理量精度取整。"""
    return round(
        float(rng.uniform(max(lower, center - radius), min(upper, center + radius))), decimals
    )


speed_mps = sample_value(
    0.02,
    0.003,
    3,
    0.001,
)

print(speed_mps)