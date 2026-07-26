#!/usr/bin/env python3
"""校验原始数据集中的 episode 质量。"""

from collections import Counter
from pathlib import Path

from common import cli, output_json

from welding_path_vla.core.config import AppConfig
from welding_path_vla.evaluation.trajectory_metrics import validate_episode


@cli
def main(config: AppConfig) -> None:
    """输出有效数量和各质量状态计数。"""
    paths = sorted((Path(config.collection.dataset_root) / "episodes").glob("episode_*"))
    reports = [validate_episode(path, config) for path in paths]
    output_json(
        {
            "episodes": len(reports),
            "valid": sum(report.valid for report in reports),
            "status": Counter(report.status.value for report in reports),
        }
    )


if __name__ == "__main__":
    main()
