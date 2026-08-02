#!/usr/bin/env python3
"""将原始焊接 episode 导出为 LeRobot 数据集。"""

from dataclasses import dataclass
from pathlib import Path

from common import cli, output_json

from welding_path_vla.core.config import AppConfig
from welding_path_vla.dataset.export_lerobot import export_lerobot


@dataclass
class ExportArguments(AppConfig):
    """LeRobot 导出的路径和仓库标识。"""

    dataset: Path = Path("datasets/weldpath_raw_v2")
    output: Path = Path("datasets/weldpath_lerobot_relative_v1")
    repo_id: str = "huayan/weldpath_relative_v1"


@cli
def main(config: ExportArguments) -> None:
    """按配置选择 episode 并执行增量导出。"""
    report = export_lerobot(
        config.dataset,
        config.output,
        config.repo_id,
        config.lerobot_export,
        config.policy.action_horizon,
        config.policy.action_stride,
    )
    output_json(report.as_dict())


if __name__ == "__main__":
    main()
