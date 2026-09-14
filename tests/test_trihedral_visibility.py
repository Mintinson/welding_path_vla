"""三面角相机朝向约束及双向专家轨迹回归测试。"""

import numpy as np
import pytest

from welding_path_vla.core.config import AppConfig
from welding_path_vla.simulation import WeldingEnv
from welding_path_vla.simulation.task_sampling import (
    sample_feasible_trajectory,
    sample_task_config,
)


@pytest.mark.parametrize("task", ["trihedral_horizontal", "trihedral_vertical"])
def test_configured_trihedral_yaw_uses_visible_range(task: str) -> None:
    """默认采样既保留竖缝可见性，也应覆盖明显的左右旋转姿态。"""
    config = AppConfig.load(f"configs/{task}.yaml")
    simulation = WeldingEnv(config, camera_observations=False)
    bearings: list[float] = []
    try:
        camera = simulation.mj_model.camera(config.camera.global_name).id
        for seed in range(64):
            simulation.randomize_workpiece(np.random.default_rng(seed))
            rotation = simulation.body_rotation("workpiece")
            position = simulation.mj_model.body_pos[simulation.workpiece_id]
            camera_position = simulation.mj_data.cam_xpos[camera]
            local_camera = rotation.T @ (camera_position - position)
            bearings.append(float(np.degrees(np.arctan2(local_camera[1], local_camera[0]))))
    finally:
        simulation.close()

    margin = config.randomization.trihedral_camera_margin_deg
    assert min(bearings) >= margin
    assert max(bearings) <= 90 - margin
    assert min(bearings) < 27
    assert max(bearings) > 63
    assert np.ptp(bearings) > 36


@pytest.mark.parametrize("task", ["trihedral_horizontal", "trihedral_vertical"])
@pytest.mark.parametrize("yaw_deg", [6.0, 12.0, 90.0])
def test_trihedral_opening_faces_camera(task: str, yaw_deg: float) -> None:
    """平移和偏航变化后，竖缝仍在相机画面内且不被自身立板遮住。"""
    config = AppConfig.load(f"configs/{task}.yaml")
    config.randomization.yaw_deg = yaw_deg
    simulation = WeldingEnv(config, camera_observations=False)
    try:
        camera = simulation.mj_model.camera(config.camera.global_name).id
        seam_ids = [
            index for index in simulation.weld_visual_geom_ids
            if "vertical_corner" in simulation.mj_model.geom(int(index)).name
        ]
        assert seam_ids
        for seed in range(32):
            simulation.randomize_workpiece(np.random.default_rng(seed))
            rotation = simulation.body_rotation("workpiece")
            position = simulation.mj_model.body_pos[simulation.workpiece_id]
            camera_position = simulation.mj_data.cam_xpos[camera]
            local_camera = rotation.T @ (camera_position - position)
            bearing = np.degrees(np.arctan2(local_camera[1], local_camera[0]))
            margin = config.randomization.trihedral_camera_margin_deg
            assert margin - 1e-8 <= bearing <= 90 - margin + 1e-8

            # 从 +X/+Y 开放侧到竖缝的整条视线必须位于两块立板内表面的外侧。
            points = simulation.mj_data.geom_xpos[seam_ids]
            local_points = (points - position) @ rotation
            wall_x = config.workpiece.trihedral_wall_x_size_m[0] / 2
            wall_y = config.workpiece.trihedral_wall_y_size_m[1] / 2
            assert np.all(local_points[:, 0] >= wall_x - 1e-8)
            assert np.all(local_points[:, 1] >= wall_y - 1e-8)
            assert local_camera[0] > wall_x and local_camera[1] > wall_y

            # MuJoCo 相机沿局部 -Z 观察；检查整条竖缝均落在全局画面内。
            camera_rotation = simulation.mj_data.cam_xmat[camera].reshape(3, 3)
            camera_points = (points - camera_position) @ camera_rotation
            depth = -camera_points[:, 2]
            half_height = np.tan(np.radians(simulation.mj_model.cam_fovy[camera]) / 2)
            half_width = half_height * config.camera.width / config.camera.height
            assert np.all(depth > 0)
            assert np.all(np.abs(camera_points[:, 0]) < depth * half_width)
            assert np.all(np.abs(camera_points[:, 1]) < depth * half_height)
    finally:
        simulation.close()


@pytest.mark.parametrize("task", ["trihedral_horizontal", "trihedral_vertical"])
@pytest.mark.parametrize("reverse", [False, True])
def test_camera_facing_trihedral_has_feasible_expert(task: str, reverse: bool) -> None:
    """面向相机的工件仍应支持正反向完整连续 IK 和碰撞预检。"""
    base = AppConfig.load(f"configs/{task}.yaml")
    base.randomization.reverse_probability = float(reverse)
    config = sample_task_config(base, np.random.default_rng(9100))
    simulation = WeldingEnv(config, camera_observations=False, ignore_done=True)
    try:
        sample = sample_feasible_trajectory(simulation, config, np.random.default_rng(9300))
        assert sample.planning_max_ik_residual_m <= 0.005
        assert len(sample.expert.frames) == len(sample.joint_trajectory)
    finally:
        simulation.close()
