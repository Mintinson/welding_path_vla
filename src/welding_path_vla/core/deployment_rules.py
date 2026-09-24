"""策略仿真部署使用的固定完成判定与视频保留规则。"""

from typing import Final

COMPLETION_PROGRESS_MIN: Final = 0.85
COMPLETION_DISTANCE_M: Final = 0.015
COMPLETION_EXIT_DISTANCE_M: Final = 0.01
COLLISION_FRAME_HOLD_S: Final = 1.0


def deployment_rules_metadata() -> dict[str, float]:
    """返回本次部署实际使用的固定规则，供结果文件记录。"""
    return {
        "completion_progress_min": COMPLETION_PROGRESS_MIN,
        "completion_distance_m": COMPLETION_DISTANCE_M,
        "completion_exit_distance_m": COMPLETION_EXIT_DISTANCE_M,
        "collision_frame_hold_s": COLLISION_FRAME_HOLD_S,
    }


__all__ = [
    "COLLISION_FRAME_HOLD_S",
    "COMPLETION_DISTANCE_M",
    "COMPLETION_EXIT_DISTANCE_M",
    "COMPLETION_PROGRESS_MIN",
    "deployment_rules_metadata",
]
