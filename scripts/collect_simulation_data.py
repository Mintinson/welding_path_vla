#!/usr/bin/env python3
"""采集 MuJoCo 专家轨迹原始数据。"""

import os
from collections import Counter

from common import cli, output_json

from welding_path_vla.core.config import AppConfig


@cli
def main(config: AppConfig) -> None:
    """按照统一配置采集有效 episode。"""
    if config.collection.headless:
        os.environ.setdefault("MUJOCO_GL", config.camera.offscreen_backend)

    from welding_path_vla.evaluation.trajectory_metrics import validate_episode
    from welding_path_vla.simulation.collector import collect_dataset

    paths = collect_dataset(config)
    status = Counter(validate_episode(path, config).status.value for path in paths)
    output_json({"episodes": len(paths), "status": status})


if __name__ == "__main__":
    main()
