#!/usr/bin/env python3
"""将原始焊接 episode 导出为 LeRobot 数据集。"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

from common import cli, output_json

from datasets import disable_progress_bars
from welding_path_vla.core.config import AppConfig
from welding_path_vla.dataset.export_lerobot import export_lerobot_many

os.environ.setdefault("SVT_LOG_FILE", os.devnull)
# DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/base.yaml"


@dataclass
class ExportArguments(AppConfig):
    """LeRobot 导出的路径和仓库标识。"""

    dataset: Path = Path("datasets/weldpath_raw_v2")
    datasets: list[Path] = field(default_factory=list)
    dataset_glob: str | None = None
    output: Path = Path("datasets/weldpath_lerobot_relative_v1")
    repo_id: str = "huayan/weldpath_relative_v1"


def resolve_sources(config: ExportArguments) -> list[Path]:
    """按显式列表、glob 或兼容的单路径选择原始数据集。

    Args:
        config: 包含三种源路径表达的导出参数。

    Returns:
        去重前、保持用户指定顺序的原始数据集路径。
    """
    if config.datasets:
        return config.datasets
    if config.dataset_glob:
        return [Path(path) for path in sorted(glob(config.dataset_glob))]
    return [config.dataset]


# def add_default_config_path(arguments: list[str]) -> None:
#     """未显式选择 YAML 时，让导出命令使用项目基础配置。

#     Args:
#         arguments: 待传给 Draccus 的进程参数列表。
#     """
#     if not any(argument.startswith("--config_path=") for argument in arguments[1:]):
#         arguments.insert(1, f"--config_path={DEFAULT_CONFIG_PATH}")


@contextmanager
def suppress_encoder_banners():
    """阻止 LeRobot 编码线程恢复 FFmpeg 原生 stderr 输出。"""
    from av import logging as av_logging

    restore_callback = av_logging.restore_default_callback

    def keep_python_callback() -> None:
        """保留 PyAV 的 Python 日志回调。"""

    av_logging.restore_default_callback = keep_python_callback
    try:
        yield
    finally:
        av_logging.restore_default_callback = restore_callback
        restore_callback()


@cli
def main(config: ExportArguments) -> None:
    """按配置选择 episode 并执行增量导出。"""
    disable_progress_bars()
    with suppress_encoder_banners():
        report = export_lerobot_many(
            resolve_sources(config),
            config.output,
            config.repo_id,
            config.lerobot_export,
            config.policy.action_horizon,
            config.policy.action_stride,
        )
    output_json(report.as_dict())


if __name__ == "__main__":
    # add_default_config_path(sys.argv)
    main()
