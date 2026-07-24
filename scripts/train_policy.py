#!/usr/bin/env python3
"""根据统一 YAML 启动 LeRobot 策略训练。"""

from __future__ import annotations

import argparse
import shlex
import subprocess

from common import load_config

from welding_path_vla.core.config import DEFAULT_CONFIG
from welding_path_vla.policies.training import TrainingRequest


def main() -> None:
    """生成训练命令，并按需只执行预览。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    command = TrainingRequest(config.policy, config.training).command()
    if arguments.dry_run:
        print(shlex.join(command))
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
