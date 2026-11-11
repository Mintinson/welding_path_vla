#!/usr/bin/env python3
"""为原生 lerobot-record 增加任务条件与连续观测配对边界。"""

from common import cli
from lerobot.utils.import_utils import register_third_party_plugins

from welding_path_vla.dataset.record_conditions import (
    ConditionedRecordConfig,
    record_with_conditions,
)


@cli
def main(config: ConditionedRecordConfig) -> None:
    """复用 LeRobot 录制流程，附加任务标签及每段循环的配对初始化和清理。"""
    record_with_conditions(config)


if __name__ == "__main__":
    register_third_party_plugins()
    main()
