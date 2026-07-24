#!/usr/bin/env python3
"""采集 MuJoCo 专家轨迹原始数据。"""

from __future__ import annotations

import argparse
import os
from collections import Counter

from common import load_config, output_json

from welding_path_vla.core.config import DEFAULT_CONFIG


def main() -> None:
    """解析参数并采集指定数量的有效 episode。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset")
    parser.add_argument("--episodes", type=int)
    arguments = parser.parse_args()
    config = load_config(arguments.config, arguments.dataset)
    if config.collection.headless:
        os.environ.setdefault("MUJOCO_GL", config.camera.offscreen_backend)

    from welding_path_vla.evaluation.trajectory_metrics import validate_episode
    from welding_path_vla.simulation.collector import collect_dataset

    paths = collect_dataset(config, arguments.episodes)
    status = Counter(validate_episode(path, config).status.value for path in paths)
    output_json({"episodes": len(paths), "status": status})


if __name__ == "__main__":
    main()
