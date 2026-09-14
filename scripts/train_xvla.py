#!/usr/bin/env python3
"""X-VLA 训练薄入口；默认使用通用配置并复用统一 LeRobot pipeline。"""

from __future__ import annotations

import sys

from train_policy import main


if __name__ == "__main__":
    if not any(argument.startswith("--config_path=") for argument in sys.argv[1:]):
        sys.argv.append("--config_path=configs/xvla.yaml")
    main()
