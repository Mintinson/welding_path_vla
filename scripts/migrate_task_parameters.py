#!/usr/bin/env python3
"""给已有 LeRobot 数据集原地补充 raw 焊接任务参数。"""

from dataclasses import dataclass
from glob import glob
from pathlib import Path

from common import cli, output_json

from welding_path_vla.dataset.migrate_task_parameters import migrate_task_parameters


@dataclass
class MigrationArguments:
    """任务参数迁移参数。

    Attributes:
        dataset: 待原地更新的 LeRobot 数据集根目录。
        raw_dataset_glob: 与导出清单对应的 raw 数据集路径表达式。
        verify_only: 只校验映射、字段和值，不修改任何文件。
    """

    dataset: Path
    raw_dataset_glob: str = "datasets/*_raw_v2"
    verify_only: bool = False


@cli
def main(config: MigrationArguments) -> None:
    """执行可中断后重跑的原地迁移，并输出 JSON 报告。"""
    roots = [Path(path) for path in sorted(glob(config.raw_dataset_glob))]
    report = migrate_task_parameters(
        config.dataset,
        roots,
        verify_only=config.verify_only,
    )
    output_json(report.as_dict())


if __name__ == "__main__":
    main()
