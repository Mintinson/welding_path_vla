#!/usr/bin/env python3
"""为旧版原始数据集回填视频统计 sidecar。"""

from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

workspace = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(workspace / "src"))

from common import output_json
from welding_path_vla.dataset.backfill_raw_video_stats import backfill_raw_video_stats


def arguments() -> argparse.Namespace:
    """解析回填范围、并行度和明确写入开关。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-glob", default="datasets/raw_v3/*", help="原始数据集根目录 glob"
    )
    parser.add_argument("--workers", type=int, default=1, help="视频解码进程数量")
    parser.add_argument(
        "--apply", action="store_true", help="实际写入 sidecar；默认仅预演"
    )
    return parser.parse_args()


def main() -> None:
    """执行回填并以 JSON 输出汇总结果。"""
    config = arguments()
    roots = tuple(Path(path) for path in sorted(glob(config.raw_glob)))
    output_json(backfill_raw_video_stats(roots, config.workers, config.apply).as_dict())


if __name__ == "__main__":
    main()
