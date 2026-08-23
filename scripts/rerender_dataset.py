#!/usr/bin/env python3
"""按旧状态原地重渲染 raw 或 LeRobot 数据集的视频。"""

import os
from dataclasses import dataclass
from pathlib import Path

from common import cli, output_json

from datasets import disable_progress_bars
from welding_path_vla.dataset.rerender import rerender_dataset

os.environ.setdefault("SVT_LOG_FILE", os.devnull)


@dataclass
class RerenderArguments:
    """数据集重渲染参数。"""

    dataset: Path
    raw_dataset_glob: str = "datasets/*_raw_v2"
    keep_backup: bool = False


@cli
def main(config: RerenderArguments) -> None:
    """自动识别数据格式，并安全执行原地重渲染。"""
    disable_progress_bars()
    report = rerender_dataset(
        config.dataset,
        config.raw_dataset_glob,
        config.keep_backup,
    )
    output_json(report.as_dict())


if __name__ == "__main__":
    main()
