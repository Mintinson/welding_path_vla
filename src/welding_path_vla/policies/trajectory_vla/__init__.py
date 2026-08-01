"""基于官方 SmolVLA 算法、本地可修改的轨迹 VLA。"""

from welding_path_vla.policies.trajectory_vla.configuration_trajectory_vla import (
    TrajectoryVLAConfig,
)
from welding_path_vla.policies.trajectory_vla.modeling_trajectory_vla import (
    TrajectoryVLAPolicy,
)
from welding_path_vla.policies.trajectory_vla.processor_trajectory_vla import (
    make_trajectory_vla_pre_post_processors,
)

__all__ = [
    "TrajectoryVLAConfig",
    "TrajectoryVLAPolicy",
    "make_trajectory_vla_pre_post_processors",
]
