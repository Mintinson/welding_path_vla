#!/usr/bin/env python3
"""将原始焊接 episode 导出为 LeRobot 数据集。"""

from __future__ import annotations

import argparse

from welding_path_vla.dataset.export_lerobot import export_lerobot


def main() -> None:
    """解析数据路径和仓库标识后执行导出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", default="huayan/weldpath_sim_v1")
    arguments = parser.parse_args()
    print(export_lerobot(arguments.dataset, arguments.output, arguments.repo_id))


if __name__ == "__main__":
    main()
