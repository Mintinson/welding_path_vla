"""LeRobot 策略共享的 checkpoint 路径解析。"""

from pathlib import Path


def resolve_checkpoint(path: str | Path) -> Path:
    """接受 run、step 或 ``pretrained_model`` 目录并定位模型。

    Args:
        path: LeRobot 训练输出目录、单个 checkpoint 目录或模型目录。

    Returns:
        含 ``config.json`` 的绝对模型目录。
    """
    root = Path(path)
    candidates = (
        root,
        root / "pretrained_model",
        root / "checkpoints" / "last" / "pretrained_model",
    )
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot find LeRobot pretrained_model under: {root}")


__all__ = ["resolve_checkpoint"]
