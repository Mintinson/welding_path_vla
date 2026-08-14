"""LeRobot 训练进程的轻量运行辅助。"""

from __future__ import annotations

import sys
from collections.abc import Generator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any


@contextmanager
def lerobot_config_argument(config_path: Path | None) -> Generator[None, None, None]:
    """让 LeRobot 在恢复训练时读取 checkpoint 配置。

    项目 CLI 的 ``config_path`` 指向统一 YAML，而 LeRobot 恢复逻辑要求它指向
    checkpoint 内的 ``train_config.json``。此上下文只在调用官方训练器期间切换
    参数，退出后立即恢复项目原始命令行。

    Args:
        config_path: LeRobot checkpoint 配置；``None`` 表示普通训练。
    """
    if config_path is None:
        yield
        return

    arguments = sys.argv
    sys.argv = [arguments[0], f"--config_path={config_path}"]
    try:
        yield
    finally:
        sys.argv = arguments


@contextmanager
def lerobot_training_log(log_path: Path) -> Generator[None, None, None]:
    """把 LeRobot 控制台指标同步保存到本地日志。

    LeRobot 自带的 ``init_logging`` 已支持文件 handler，但其训练入口没有暴露
    ``log_file`` 参数。这里仅为该现成功能补上传参，不复制训练日志实现。

    Args:
        log_path: 本地日志路径；恢复训练时继续追加同一文件。
    """
    train_module = import_module("lerobot.scripts.lerobot_train")
    init_logging = train_module.init_logging

    def init_logging_with_file(*args: Any, **kwargs: Any) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("file_level", "INFO")
        init_logging(*args, log_file=log_path, **kwargs)

    train_module.init_logging = init_logging_with_file
    try:
        yield
    finally:
        train_module.init_logging = init_logging


__all__ = ["lerobot_config_argument", "lerobot_training_log"]
