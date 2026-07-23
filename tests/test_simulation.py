import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from welding_path_vla.config import AppConfig
from welding_path_vla.geometry import quaternion_to_matrix
from welding_path_vla.simulation import ExpertTrajectory, WeldingSimulation
from welding_path_vla.simulation.collector import sample_collision_free_task, stage_for_task


def test_mujoco_model_and_cameras_load() -> None:
    config = AppConfig.load("configs/default.yaml")
    config.camera.width = 64
    config.camera.height = 48
    simulation = WeldingSimulation(config)
    try:
        assert simulation.model.njnt == 6
        assert simulation.model.nu == 6
        assert simulation.model.nmesh == 7
        assert simulation.model.body("tool_payload").mass[0] == 0.216
        assert (
            mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_GEOM, "elfin_link6_visual") >= 0
        )
        assert mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_GEOM, "torch_nozzle") >= 0
        ring = simulation.model.geom("elfin_joint6_ring")
        np.testing.assert_allclose(ring.rgba, [0.02, 0.02, 0.025, 1.0])
        np.testing.assert_allclose(ring.pos, [0, -0.097, 0])
        np.testing.assert_allclose(ring.size[:2], [0.043, 0.008])
        assert simulation.model.geom_contype[ring.id] == 0
        flange = simulation.model.geom("torch_flange")
        tip = simulation.model.geom("torch_tip")
        assert 2 * flange.size[1] == pytest.approx(0.020)
        assert 2 * tip.size[0] == pytest.approx(0.0005)
        np.testing.assert_allclose(flange.pos, [0, 0, 0.1665])
        np.testing.assert_allclose(simulation.model.geom("torch_tube_1").pos[:2], [0, 0])
        tool_segments = [
            *(f"torch_tube_{index}" for index in range(1, 7)),
            "torch_collar",
            "torch_nozzle",
            "torch_tip",
        ]
        tool_length = 2 * flange.size[1] + sum(
            2 * simulation.model.geom(name).size[1] for name in tool_segments
        )
        assert tool_length == pytest.approx(0.270, abs=0.015)
        tcp_id = simulation.name_id(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        np.testing.assert_allclose(
            simulation.model.site_pos[tcp_id] - [0, 0, 0.1565],
            [0.057557, -0.025778, 0.233020],
        )
        mount_id = simulation.model.body("robot_mount").id
        assert simulation.model.geom("elfin_base_visual").bodyid[0] == mount_id
        np.testing.assert_allclose(simulation.model.body("robot_mount").pos, [0, 0, 0.29])
        expected_base_rotation = Rotation.from_euler("z", -90, degrees=True).as_matrix()
        np.testing.assert_allclose(simulation.base_rotation(), expected_base_rotation, atol=1e-7)
        np.testing.assert_allclose(
            quaternion_to_matrix(simulation.model.body("robot_mount").quat),
            expected_base_rotation,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            quaternion_to_matrix(simulation.model.body("elfin_link1").quat),
            np.eye(3),
            atol=1e-7,
        )
        np.testing.assert_allclose(simulation.model.body("elfin_link1").pos, [0, 0, 0.22])
        np.testing.assert_allclose(simulation.data.xmat[mount_id].reshape(3, 3)[:, 2], [0, 0, 1])
        wrist_id = simulation.name_id(mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        np.testing.assert_allclose(
            simulation.model.cam_pos[wrist_id], config.camera.wrist_position_link6_m
        )
        assert not simulation.collision
        assert (
            simulation.site_position("seam_start")[0]
            > simulation.model.body_pos[simulation.workpiece_id][0]
        )
        assert simulation.site_position("seam_end")[1] > simulation.site_position("seam_start")[1]
        images = simulation.render()
        assert images["global"].shape == (48, 64, 3)
        assert images["wrist"].dtype == np.uint8
        assert images["global"].std() > 5
        assert images["wrist"].std() > 5
    finally:
        simulation.close()


def test_torch_body_and_fine_tip_collisions_are_detected() -> None:
    simulation = WeldingSimulation(AppConfig.load("configs/default.yaml"))
    try:
        tube_id = simulation.name_id(mujoco.mjtObj.mjOBJ_GEOM, "torch_tube_3")
        tip_id = simulation.name_id(mujoco.mjtObj.mjOBJ_GEOM, "torch_tip")
        assert simulation.model.geom_contype[tube_id] != 0
        assert simulation.model.geom_contype[tip_id] != 0
        for geom_id in range(simulation.model.ngeom):
            if geom_id != tube_id and simulation.model.geom_conaffinity[geom_id] == 2:
                simulation.model.geom_contype[geom_id] = 0
                simulation.model.geom_conaffinity[geom_id] = 0
        simulation.model.body_pos[simulation.workpiece_id] = simulation.data.geom_xpos[
            tube_id
        ] - np.array([0.05, 0, 0])
        mujoco.mj_forward(simulation.model, simulation.data)
        contact_geoms = {
            mujoco.mj_id2name(simulation.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for contact in simulation.data.contact
            for geom_id in (contact.geom1, contact.geom2)
        }
        assert simulation.collision
        assert "torch_tube_3" in contact_geoms
        simulation.execute_pose(simulation.tcp_pose())
        assert any("torch_tube_3" in pair for pair in simulation.last_collision_pairs)
        for geom_id in range(simulation.model.ngeom):
            if simulation.model.geom_conaffinity[geom_id] == 2:
                simulation.model.geom_contype[geom_id] = 0
                simulation.model.geom_conaffinity[geom_id] = 0
        simulation.model.geom_contype[tip_id] = 1
        simulation.model.geom_conaffinity[tip_id] = 2
        tip_position = simulation.data.geom_xpos[tip_id].copy()
        simulation.model.body_pos[simulation.workpiece_id] = tip_position - np.array(
            [0.05, 0, 0.0025]
        )
        mujoco.mj_forward(simulation.model, simulation.data)
        tip_contact_geoms = {
            mujoco.mj_id2name(simulation.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for contact in simulation.data.contact
            for geom_id in (contact.geom1, contact.geom2)
        }
        assert "torch_tip" in tip_contact_geoms
    finally:
        simulation.close()


def test_reference_welding_pose_clears_workpiece() -> None:
    config = AppConfig.load("configs/default.yaml")
    simulation = WeldingSimulation(config)
    try:
        seam_start = simulation.site_position("seam_start")
        seam_end = simulation.site_position("seam_end")
        work_angle = np.radians(config.task.work_angle_deg)
        normal = simulation.body_rotation("workpiece") @ np.array(
            [np.sin(work_angle), 0, np.cos(work_angle)]
        )
        expert = ExpertTrajectory(config, simulation.tcp_pose(), seam_start, seam_end, normal)
        track = [frame for frame in expert.frames if frame.phase.value == "track"]
        for frame in track:
            joint_position, residual = simulation.solve_ik(frame.pose)
            simulation.data.qpos[simulation.qpos_ids] = joint_position
            mujoco.mj_forward(simulation.model, simulation.data)
            assert residual < 0.005
            assert not simulation.collision
    finally:
        simulation.close()


def test_complete_expert_trajectory_is_collision_free() -> None:
    config = AppConfig.load("configs/default.yaml")
    config.task.speed_mps = 0.04
    simulation = WeldingSimulation(config)
    try:
        seam_start, seam_end, normal, residual = stage_for_task(simulation, config)
        assert residual < 0.005
        expert = ExpertTrajectory(config, simulation.tcp_pose(), seam_start, seam_end, normal)
        for frame in expert.frames:
            simulation.execute_pose(frame.pose)
            assert not simulation.last_collision_pairs
    finally:
        simulation.close()


def test_randomized_task_rejects_unreachable_staging_pose() -> None:
    config = AppConfig.load("configs/default.yaml")
    simulation = WeldingSimulation(config)
    try:
        rng = np.random.default_rng(config.collection.seed + 14)
        *_, residual, attempts = sample_collision_free_task(simulation, config, rng)
        assert attempts > 1
        assert residual < 0.005
        assert not simulation.collision
    finally:
        simulation.close()


def test_initial_joint_randomization_is_bounded_and_collision_free() -> None:
    config = AppConfig.load("configs/default.yaml")
    simulation = WeldingSimulation(config)
    try:
        rng = np.random.default_rng(11)
        sample_collision_free_task(simulation, config, rng)
        offset, attempts = simulation.randomize_joint_position(
            rng, config.randomization.joint_degs, config.randomization.max_sampling_attempts
        )
        assert attempts <= config.randomization.max_sampling_attempts
        assert np.any(np.abs(offset) > 5)
        assert np.all(np.abs(offset) <= config.randomization.joint_degs)
        assert not simulation.collision
    finally:
        simulation.close()
