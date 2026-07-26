#!/usr/bin/env python3
"""显示机器人、安装和安全配置。"""

from common import cli, output_json

from welding_path_vla.core.config import AppConfig


@cli
def main(config: AppConfig) -> None:
    """输出机器人、安装外参和安全配置。"""
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
