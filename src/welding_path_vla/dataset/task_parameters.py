"""Raw 与 LeRobot 共用的焊接任务参数定义。"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

TASK_DIRECTION = "task.direction"
TASK_PARAMETERS = "task.parameters"
TASK_PARAMETER_NAMES = (
    "welding_speed_mps",
    "work_angle_deg",
    "travel_angle_deg",
    "tool_roll_deg",
)


class TaskFeatureDefinition(TypedDict):
    """LeRobot 数值任务特征的静态 schema。"""

    dtype: str
    shape: tuple[int, ...]
    names: list[str]


TASK_FEATURES: dict[str, TaskFeatureDefinition] = {
    TASK_DIRECTION: {
        "dtype": "int64",
        "shape": (1,),
        "names": ["direction"],
    },
    TASK_PARAMETERS: {
        "dtype": "float32",
        "shape": (4,),
        "names": list(TASK_PARAMETER_NAMES),
    },
}
DIRECTION_CODES = {"forward": 0, "reverse": 1}


def task_feature_values(metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    """从 raw episode 元数据提取固定长度的数值任务特征。

    Args:
        metadata: ``metadata.json`` 的完整内容。

    Returns:
        可直接传给 ``LeRobotDataset.add_frame`` 的两个数值数组。方向编码为
        ``forward=0``、``reverse=1``，不会创建新的语言 ``task`` 字段。
    """
    parameters = metadata["task_parameters"]
    direction = parameters.get("direction", metadata.get("direction"))
    if direction not in DIRECTION_CODES:
        raise ValueError(f"unknown welding direction: {direction!r}")
    return {
        TASK_DIRECTION: np.asarray([DIRECTION_CODES[direction]], dtype=np.int64),
        TASK_PARAMETERS: np.asarray(
            [
                parameters["speed_mps"],
                parameters["work_angle_deg"],
                parameters["travel_angle_deg"],
                parameters["tool_roll_deg"],
            ],
            dtype=np.float32,
        ),
    }
