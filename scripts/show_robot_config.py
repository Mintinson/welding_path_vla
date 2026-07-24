#!/usr/bin/env python3
"""显示机器人、安装和安全配置。"""

from __future__ import annotations

import argparse

from common import load_config, output_json

from welding_path_vla.core.config import DEFAULT_CONFIG


def main() -> None:
    """加载并输出机器人相关配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    resolved = config.as_dict()
    output_json(
        {
            "robot": resolved["robot"],
            "robot_mount": {
                "position_m": resolved["scene"]["robot_base_position_m"],
                "yaw_deg": resolved["scene"]["robot_base_yaw_deg"],
            },
            "real_robot": resolved["real_robot"],
            "safety": resolved["safety"],
        }
    )


if __name__ == "__main__":
    main()
