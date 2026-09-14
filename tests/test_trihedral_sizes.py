"""紧凑三面角尺寸与随机焊缝边界回归测试。"""

import numpy as np
import pytest

from welding_path_vla.core.config import AppConfig, maximum_seam_length
from welding_path_vla.simulation.task_sampling import sample_task_config


@pytest.mark.parametrize("task", ["trihedral_horizontal", "trihedral_vertical"])
def test_compact_trihedral_keeps_both_seams_inside_plates(task: str) -> None:
    """尺寸随机化后，活动焊缝及同时显示的另一条焊缝均不能超出板件。"""
    base = AppConfig.load(f"configs/{task}.yaml")
    assert base.workpiece.trihedral_floor_size_m == [0.18, 0.17, 0.005]
    assert base.workpiece.trihedral_wall_x_size_m == [0.005, 0.17, 0.145]
    assert base.workpiece.trihedral_wall_y_size_m == [0.18, 0.005, 0.145]
    assert base.task.seam_length_m == 0.11
    assert base.workpiece.trihedral_turn_radius_m == 0.03
    for seed in range(64):
        config = sample_task_config(base, np.random.default_rng(seed))
        for seam_id in ("horizontal_pair", "vertical_corner"):
            limit = maximum_seam_length(config.workpiece, seam_id)
            assert limit is not None
            assert config.task.seam_length_m <= limit
