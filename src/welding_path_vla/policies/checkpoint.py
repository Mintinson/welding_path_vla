"""LeRobot 策略共享的 checkpoint 路径解析。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ResumeCheckpoint:
    """一次可恢复的 LeRobot checkpoint。

    Attributes:
        root: 同时包含模型和训练状态的 step checkpoint 目录。
        model: 包含权重、processor 和训练配置的 ``pretrained_model`` 目录。
        config: LeRobot 保存的 ``train_config.json``。
        step: 已完成的 optimizer update 数量。
    """

    root: Path
    model: Path
    config: Path
    step: int


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


def find_resume_checkpoint(output_dir: str | Path) -> ResumeCheckpoint:
    """从训练输出目录定位 LeRobot 的 ``checkpoints/last``。

    Args:
        output_dir: 原训练使用的输出目录。

    Returns:
        带模型配置和训练 step 的恢复点。
    """
    from lerobot.common.train_utils import load_training_step

    model = resolve_checkpoint(output_dir)
    root = model.parent
    config = model / "train_config.json"
    if not config.is_file():
        raise FileNotFoundError(f"missing LeRobot train config: {config}")
    return ResumeCheckpoint(root, model, config, load_training_step(root / "training_state"))


__all__ = ["ResumeCheckpoint", "find_resume_checkpoint", "resolve_checkpoint"]
