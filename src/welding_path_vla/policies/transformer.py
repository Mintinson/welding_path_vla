"""不同 VLA Transformer 共享的轻量结构计算。"""


def expert_intermediate_size(
    hidden_dim: int,
    multiplier: float = 4,
    multiple_of: int = 256,
) -> int:
    """按 SwiGLU 规则计算并对齐动作专家的 FFN 宽度。"""
    width = int(multiplier * int(2 * hidden_dim / 3))
    return multiple_of * ((width + multiple_of - 1) // multiple_of)


__all__ = ["expert_intermediate_size"]
