from itertools import pairwise

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.geometry import quaternion_to_matrix
from welding_path_vla.simulation import ExpertTrajectory, WeldingEnv
from welding_path_vla.simulation.models import (
    Elfin5ProRobotModel,
    WeldingArena,
    WorkpieceObject,
)
from welding_path_vla.simulation.robosuite_compat import MujocoEnv, make
from welding_path_vla.simulation.task_sampling import (
    sample_collision_free_task,
    sample_episode_task_config,
    sample_feasible_trajectory,
    sample_task_config,
    stage_for_task,
)
from welding_path_vla.simulation.tasks import CircularSeamPath


def test_robosuite_reset_step_and_observations() -> None:
    """环境应遵循 robosuite 的 reset/step/observation 契约。"""
    config = AppConfig.load("configs/default.yaml")
    config.camera.width = 64
    config.camera.height = 48
    simulation = make("WeldingEnv", config=config, seed=7)
    try:
        assert isinstance(simulation, MujocoEnv)
        observation = simulation.reset(seed=7)
        expected = {
            "joint_position",
            "joint_velocity",
            "tcp_position",
            "tcp_quaternion_wxyz",
            "global_image",
            "wrist_image",
        }
        assert expected <= observation.keys()

        pose = simulation.tcp_pose()
        action = np.concatenate([pose.position, pose.quaternion_wxyz])
        next_observation, reward, done, info = simulation.step(action)

        assert expected <= next_observation.keys()
        assert simulation.mj_data.time == pytest.approx(1 / config.timing.policy_hz)
        assert reward == 0
        assert not done
        assert info["collision"] is False
        assert info["joint_command"].shape == (6,)
    finally:
        simulation.close()


def test_mujoco_model_and_cameras_load() -> None:
    config = AppConfig.load("configs/default.yaml")
    config.camera.width = 64
    config.camera.height = 48
    simulation = WeldingEnv(config)
    try:
        assert simulation.mj_model.njnt == 6
        assert simulation.mj_model.nu == 6
        assert simulation.mj_model.nmesh == 7
        assert isinstance(simulation.robot_model, Elfin5ProRobotModel)
        assert isinstance(simulation.arena, WeldingArena)
        assert isinstance(simulation.workpiece, WorkpieceObject)
        assert simulation.mj_model.body("tool_payload").mass[0] == 0.216
        assert (
            mujoco.mj_name2id(
                simulation.mj_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "elfin_link6_visual",
            )
            >= 0
        )
        assert mujoco.mj_name2id(simulation.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "torch_nozzle") >= 0
        ring = simulation.mj_model.geom("elfin_joint6_ring")
        np.testing.assert_allclose(ring.rgba, [0.02, 0.02, 0.025, 1.0])
        np.testing.assert_allclose(ring.pos, [0, -0.097, 0])
        np.testing.assert_allclose(ring.size[:2], [0.043, 0.008])
        assert simulation.mj_model.geom_contype[ring.id] == 0
        flange = simulation.mj_model.geom("torch_flange")
        tip = simulation.mj_model.geom("torch_tip")
        assert 2 * flange.size[1] == pytest.approx(0.020)
        assert 2 * tip.size[0] == pytest.approx(0.0005)
        np.testing.assert_allclose(flange.pos, [0, 0, 0.1665])
        np.testing.assert_allclose(simulation.mj_model.geom("torch_tube_1").pos[:2], [0, 0])
        tool_segments = [
            *(f"torch_tube_{index}" for index in range(1, 7)),
            "torch_collar",
            "torch_nozzle",
            "torch_tip",
        ]
        tool_length = 2 * flange.size[1] + sum(
            2 * simulation.mj_model.geom(name).size[1] for name in tool_segments
        )
        assert tool_length == pytest.approx(0.270, abs=0.015)
        tcp_id = simulation.name_id(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        np.testing.assert_allclose(
            simulation.mj_model.site_pos[tcp_id] - [0, 0, 0.1565],
            [0.057557, -0.025778, 0.233020],
        )
        mount_id = simulation.mj_model.body("robot_mount").id
        assert simulation.mj_model.geom("elfin_base_visual").bodyid[0] == mount_id
        np.testing.assert_allclose(simulation.mj_model.body("robot_mount").pos, [0, 0, 0.29])
        expected_base_rotation = Rotation.from_euler("z", -90, degrees=True).as_matrix()
        np.testing.assert_allclose(simulation.base_rotation(), expected_base_rotation, atol=1e-7)
        np.testing.assert_allclose(
            quaternion_to_matrix(simulation.mj_model.body("robot_mount").quat),
            expected_base_rotation,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            quaternion_to_matrix(simulation.mj_model.body("elfin_link1").quat),
            np.eye(3),
            atol=1e-7,
        )
        table_frame = simulation.mj_model.body("table_frame")
        assert simulation.mj_model.camera("global").bodyid[0] == table_frame.id
        assert simulation.mj_model.geom("table").bodyid[0] == table_frame.id
        np.testing.assert_allclose(simulation.mj_model.body("elfin_link1").pos, [0, 0, 0.22])
        np.testing.assert_allclose(simulation.mj_data.xmat[mount_id].reshape(3, 3)[:, 2], [0, 0, 1])
        wrist_id = simulation.name_id(mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        np.testing.assert_allclose(
            simulation.mj_model.cam_pos[wrist_id], config.camera.wrist_position_link6_m
        )
        assert not simulation.collision
        assert (
            simulation.site_position("seam_start")[0]
            > simulation.mj_model.body_pos[simulation.workpiece_id][0]
        )
        assert simulation.site_position("seam_end")[1] > simulation.site_position("seam_start")[1]
        images = simulation.images_from_observation(simulation.observe())
        assert images["global"].shape == (48, 64, 3)
        assert images["wrist"].dtype == np.uint8
        assert images["global"].std() > 5
        assert images["wrist"].std() > 5
    finally:
        simulation.close()


def test_global_camera_is_fixed_across_episodes() -> None:
    """工件随机化不能改变全局相机外参。"""
    config = AppConfig.load("configs/default.yaml")
    simulation = WeldingEnv(config)
    try:
        camera_id = simulation.name_id(mujoco.mjtObj.mjOBJ_CAMERA, config.camera.global_name)
        position = simulation.mj_data.cam_xpos[camera_id].copy()
        rotation = simulation.mj_data.cam_xmat[camera_id].copy()
        for seed in range(6):
            simulation.randomize_workpiece(np.random.default_rng(seed))
            np.testing.assert_allclose(simulation.mj_data.cam_xpos[camera_id], position)
            np.testing.assert_allclose(simulation.mj_data.cam_xmat[camera_id], rotation)
    finally:
        simulation.close()


def test_global_camera_image_is_not_underexposed() -> None:
    """固定圆管场景的全局图像平均亮度不应低于可读范围。"""
    config = AppConfig.load("configs/pipe_top.yaml")
    config.camera.width = 64
    config.camera.height = 48
    simulation = WeldingEnv(config, seed=0)
    try:
        image = simulation.reset(seed=0)["global_image"]
        luminance = image @ np.array([0.2126, 0.7152, 0.0722])
        assert float(luminance.mean()) >= 80
    finally:
        simulation.close()


def test_wrist_camera_mount_does_not_occlude_image() -> None:
    """腕部相机自身的安装杆和外壳不应进入图像中心。"""
    config = AppConfig.load("configs/pipe_top.yaml")
    config.camera.width = 160
    config.camera.height = 120
    simulation = WeldingEnv(config, seed=0)
    try:
        segmentation = simulation.robosuite_sim.render(
            camera_name=config.camera.wrist_name,
            width=config.camera.width,
            height=config.camera.height,
            segmentation=True,
        )[::-1]
        mount_ids = {
            simulation.name_id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "wrist_camera_standoff",
                "wrist_camera_bracket",
                "wrist_camera_body",
            )
        }
        center = segmentation[20:100, 20:140]
        mount_mask = (center[..., 0] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(
            center[..., 1], list(mount_ids)
        )
        assert float(mount_mask.mean()) < 0.01
    finally:
        simulation.close()


def test_wrist_camera_table_brightness_remains_readable() -> None:
    """腕部视角中的桌面不应欠曝或接近饱和。"""
    config = AppConfig.load("configs/pipe_top.yaml")
    config.camera.width = 160
    config.camera.height = 120
    simulation = WeldingEnv(config, seed=0)
    try:
        reference_joint_position = np.array(
            [1.914891, 0.429071, 2.304289, -0.651702, 0.621532, 1.219920]
        )
        simulation.set_joint_position(reference_joint_position)
        image = simulation.robosuite_sim.render(
            camera_name=config.camera.wrist_name,
            width=config.camera.width,
            height=config.camera.height,
        )[::-1]
        segmentation = simulation.robosuite_sim.render(
            camera_name=config.camera.wrist_name,
            width=config.camera.width,
            height=config.camera.height,
            segmentation=True,
        )[::-1]
        table_id = simulation.name_id(mujoco.mjtObj.mjOBJ_GEOM, "table")
        table_mask = (segmentation[..., 0] == mujoco.mjtObj.mjOBJ_GEOM) & (
            segmentation[..., 1] == table_id
        )
        luminance = image @ np.array([0.2126, 0.7152, 0.0722])
        assert 90 <= float(luminance[table_mask].mean()) <= 150
    finally:
        simulation.close()


def test_global_camera_moves_with_table_frame() -> None:
    """桌面安装位姿变化时，全局相机应保持相同桌面外参。"""
    config = AppConfig.load("configs/default.yaml")
    simulation = WeldingEnv(config)
    try:
        camera_id = simulation.name_id(mujoco.mjtObj.mjOBJ_CAMERA, config.camera.global_name)
        table_id = simulation.name_id(mujoco.mjtObj.mjOBJ_BODY, "table_frame")
        position = simulation.mj_data.cam_xpos[camera_id].copy()
        offset = np.array([0.12, -0.08, 0.05])
        simulation.mj_model.body_pos[table_id] += offset
        mujoco.mj_forward(simulation.mj_model, simulation.mj_data)
        np.testing.assert_allclose(simulation.mj_data.cam_xpos[camera_id], position + offset)
    finally:
        simulation.close()


def test_torch_body_and_fine_tip_collisions_are_detected() -> None:
    simulation = WeldingEnv(AppConfig.load("configs/default.yaml"))
    try:
        tube_id = simulation.name_id(mujoco.mjtObj.mjOBJ_GEOM, "torch_tube_3")
        tip_id = simulation.name_id(mujoco.mjtObj.mjOBJ_GEOM, "torch_tip")
        assert simulation.mj_model.geom_contype[tube_id] != 0
        assert simulation.mj_model.geom_contype[tip_id] != 0
        for geom_id in range(simulation.mj_model.ngeom):
            if geom_id != tube_id and simulation.mj_model.geom_conaffinity[geom_id] == 2:
                simulation.mj_model.geom_contype[geom_id] = 0
                simulation.mj_model.geom_conaffinity[geom_id] = 0
        simulation.mj_model.body_pos[simulation.workpiece_id] = simulation.mj_data.geom_xpos[
            tube_id
        ] - np.array([0.05, 0, 0])
        mujoco.mj_forward(simulation.mj_model, simulation.mj_data)
        contact_geoms = {
            mujoco.mj_id2name(simulation.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for contact in simulation.mj_data.contact
            for geom_id in (contact.geom1, contact.geom2)
        }
        assert simulation.collision
        assert "torch_tube_3" in contact_geoms
        pose = simulation.tcp_pose()
        simulation.step(np.concatenate([pose.position, pose.quaternion_wxyz]))
        assert any("torch_tube_3" in pair for pair in simulation.last_collision_pairs)
        for geom_id in range(simulation.mj_model.ngeom):
            if simulation.mj_model.geom_conaffinity[geom_id] == 2:
                simulation.mj_model.geom_contype[geom_id] = 0
                simulation.mj_model.geom_conaffinity[geom_id] = 0
        simulation.mj_model.geom_contype[tip_id] = 1
        simulation.mj_model.geom_conaffinity[tip_id] = 2
        tip_position = simulation.mj_data.geom_xpos[tip_id].copy()
        simulation.mj_model.body_pos[simulation.workpiece_id] = tip_position - np.array(
            [0.05, 0, 0.0025]
        )
        mujoco.mj_forward(simulation.mj_model, simulation.mj_data)
        tip_contact_geoms = {
            mujoco.mj_id2name(simulation.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for contact in simulation.mj_data.contact
            for geom_id in (contact.geom1, contact.geom2)
        }
        assert "torch_tip" in tip_contact_geoms
        assert simulation.collision
        assert any("torch_tip" in pair for pair in simulation.collision_pairs)
    finally:
        simulation.close()


def test_numerical_tip_grazing_does_not_fail_welding_episode() -> None:
    """低于配置力阈值的焊丝尖端接触不应判为碰撞失败。"""
    config = AppConfig.load("configs/default.yaml")
    simulation = WeldingEnv(config, seed=0, camera_observations=False)
    try:
        simulation.mj_model.body_pos[simulation.workpiece_id] = [
            0.361922935,
            0.072469151,
            0.2925,
        ]
        simulation.mj_model.body_quat[simulation.workpiece_id] = [
            0.992917289,
            0,
            0,
            -0.118807650,
        ]
        simulation.set_joint_position(
            np.array(
                [
                    0.961549498,
                    -0.638291614,
                    1.704152222,
                    0.726005863,
                    1.649208937,
                    -1.279462066,
                ]
            )
        )
        contacts = [
            contact
            for contact in simulation.mj_data.contact
            if {
                mujoco.mj_id2name(
                    simulation.mj_model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    contact.geom1,
                ),
                mujoco.mj_id2name(
                    simulation.mj_model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    contact.geom2,
                ),
            }
            == {"torch_tip", "plate_vertical"}
        ]
        assert len(contacts) == 1
        assert -0.0002 < contacts[0].dist < 0
        force = np.zeros(6)
        contact_index = list(simulation.mj_data.contact).index(contacts[0])
        mujoco.mj_contactForce(simulation.mj_model, simulation.mj_data, contact_index, force)
        assert np.linalg.norm(force[:3]) < config.safety.tip_contact_force_limit_n
        assert ("torch_tip", "plate_vertical") not in simulation.collision_pairs
    finally:
        simulation.close()


def test_shallow_tip_penetration_does_not_trigger_stiff_contact_false_positive() -> None:
    """焊丝尖端的亚毫米瞬时穿透不应因刚性接触力被误判为碰撞。"""
    config = AppConfig.load("configs/default.yaml")
    simulation = WeldingEnv(config, camera_observations=False)
    try:
        # 该状态来自一次真实失败 episode：视频中没有可见碰撞，但位置伺服
        # 对 0.47 mm 的瞬时穿透产生了 200 N 以上的数值接触力。
        simulation.mj_model.body_pos[simulation.workpiece_id] = [
            0.503268160,
            0.053294313,
            0.2925,
        ]
        simulation.mj_model.body_quat[simulation.workpiece_id] = [
            0.999350242,
            0.0,
            0.0,
            0.036042949,
        ]
        simulation.set_joint_position(
            np.array(
                [
                    1.120044391,
                    -1.912930020,
                    -0.644004063,
                    1.211661987,
                    2.457059563,
                    -0.450719372,
                ]
            )
        )
        contacts = [
            contact
            for contact in simulation.mj_data.contact
            if simulation.torch_tip_id in (contact.geom1, contact.geom2)
        ]

        assert len(contacts) == 1
        assert -0.0005 < contacts[0].dist < 0
        assert not simulation.collision
    finally:
        simulation.close()


def test_reference_welding_pose_clears_workpiece() -> None:
    config = AppConfig.load("configs/default.yaml")
    simulation = WeldingEnv(config)
    try:
        expert = ExpertTrajectory(config, simulation.tcp_pose(), simulation.active_seam())
        track = [frame for frame in expert.frames if frame.phase.value == "track"]
        for frame in track:
            joint_position, residual = simulation.solve_ik(frame.pose)
            simulation.mj_data.qpos[simulation.qpos_ids] = joint_position
            mujoco.mj_forward(simulation.mj_model, simulation.mj_data)
            assert residual < 0.005
            assert not simulation.collision
    finally:
        simulation.close()


def test_complete_expert_trajectory_is_collision_free() -> None:
    config = AppConfig.load("configs/default.yaml")
    config.task.speed_mps = 0.04
    simulation = WeldingEnv(config)
    try:
        seam, residual = stage_for_task(simulation, config)
        assert residual < 0.005
        expert = ExpertTrajectory(config, simulation.tcp_pose(), seam)
        for frame in expert.frames:
            action = np.concatenate([frame.pose.position, frame.pose.quaternion_wxyz])
            simulation.step(action)
            assert not simulation.last_collision_pairs
    finally:
        simulation.close()


def test_randomized_task_rejects_unreachable_staging_pose() -> None:
    config = AppConfig.load("configs/default.yaml")
    simulation = WeldingEnv(config)
    try:
        rng = np.random.default_rng(config.collection.seed + 14)
        _, residual, attempts = sample_collision_free_task(simulation, config, rng)
        assert attempts > 1
        assert residual < 0.005
        assert not simulation.collision
    finally:
        simulation.close()


def test_pipe_task_randomization_is_bounded_and_reproducible() -> None:
    """圆管 episode 应改变姿态、起点、圆弧长度和方向，但不修改基准配置。"""
    config = AppConfig.load("configs/pipe_bottom.yaml")
    first = sample_task_config(config, np.random.default_rng(7))
    repeated = sample_task_config(config, np.random.default_rng(7))
    assert first.as_dict() == repeated.as_dict()
    assert config.task.direction == "forward"
    assert config.task.arc_start_deg == -90
    assert config.task.arc_sweep_deg == 90
    assert first.task.instruction == config.task.instruction

    samples = [sample_task_config(config, np.random.default_rng(seed)) for seed in range(32)]
    directions = {sample.task.direction for sample in samples}
    geometric_starts = [
        sample.task.arc_start_deg
        - (sample.task.arc_sweep_deg if sample.task.direction == "reverse" else 0)
        for sample in samples
    ]
    assert directions == {"forward", "reverse"}
    assert all(80 <= sample.task.arc_sweep_deg <= 100 for sample in samples)
    assert all(-100 <= start <= -80 for start in geometric_starts)
    assert all(42 <= sample.task.work_angle_deg <= 48 for sample in samples)
    assert all(7 <= sample.task.travel_angle_deg <= 13 for sample in samples)
    assert all(-25 <= sample.task.tool_roll_deg <= -15 for sample in samples)
    assert all(isinstance(sample.task.work_angle_deg, int) for sample in samples)
    assert all(isinstance(sample.task.travel_angle_deg, int) for sample in samples)
    assert all(isinstance(sample.task.tool_roll_deg, int) for sample in samples)
    assert all(isinstance(sample.task.arc_start_deg, int) for sample in samples)
    assert all(isinstance(sample.task.arc_sweep_deg, int) for sample in samples)
    assert all(0.003 <= sample.task.speed_mps <= 0.005 for sample in samples)
    assert all(round(sample.task.speed_mps, 3) == sample.task.speed_mps for sample in samples)


def test_task_parameters_change_once_per_ten_episode_indices() -> None:
    """同组任务参数应完全相同，跨组则重新确定性采样。"""
    config = AppConfig.load("configs/pipe_bottom.yaml")
    first_group = [sample_episode_task_config(config, index) for index in range(10)]
    second_group = sample_episode_task_config(config, 10)
    first_parameters = first_group[0].task

    assert all(sample.task == first_parameters for sample in first_group)
    assert second_group.task != first_parameters
    assert all(sample.task.instruction == config.task.instruction for sample in first_group)


@pytest.mark.parametrize(
    "config_path",
    ["configs/default.yaml", "configs/pipe_bottom.yaml", "configs/pipe_top.yaml"],
)
def test_randomized_task_and_workpiece_are_jointly_rejected_until_reachable(
    config_path: str,
) -> None:
    """每类任务都应能在采样上限内获得无碰撞 staging 姿态。"""
    config = sample_episode_task_config(AppConfig.load(config_path), 0)
    rng = np.random.default_rng(config.collection.seed)
    simulation = WeldingEnv(config, camera_observations=False, ignore_done=True)
    try:
        _, residual, attempts = sample_collision_free_task(simulation, config, rng)
        assert attempts <= config.randomization.max_sampling_attempts
        assert residual < 0.005
        assert not simulation.collision
    finally:
        simulation.close()


@pytest.mark.parametrize(
    "config_path",
    ["configs/default.yaml", "configs/pipe_bottom.yaml", "configs/pipe_top.yaml"],
)
def test_randomized_episode_has_continuous_collision_free_joint_plan(config_path: str) -> None:
    """三个任务都应在录制前获得连续、无碰撞且可达的完整关节轨迹。"""
    config = sample_episode_task_config(AppConfig.load(config_path), 0)
    simulation = WeldingEnv(config, camera_observations=False, ignore_done=True)
    try:
        sample = sample_feasible_trajectory(
            simulation,
            config,
            np.random.default_rng(config.collection.seed),
        )
        assert sample.planning_max_ik_residual_m <= 0.005
        assert sample.motion_sampling_attempts <= config.randomization.max_sampling_attempts
        assert len(sample.expert.frames) == len(sample.joint_trajectory)
    finally:
        simulation.close()


@pytest.mark.parametrize("config_path", ["configs/pipe_bottom.yaml", "configs/pipe_top.yaml"])
def test_pipe_workpiece_exposes_reachable_circular_seam(config_path: str) -> None:
    """上下圆弧应共用接口，并按配置生成连续焊枪姿态。"""
    config = AppConfig.load(config_path)
    simulation = WeldingEnv(config, camera_observations=False, ignore_done=True)
    try:
        seam, residual = stage_for_task(simulation, config)
        assert isinstance(seam, CircularSeamPath)
        assert residual < 0.005
        assert seam.length_m == pytest.approx(
            np.radians(config.task.arc_sweep_deg) * seam.effective_radius_m
        )
        expert = ExpertTrajectory(config, simulation.tcp_pose(), seam)
        track = [frame for frame in expert.frames if frame.phase.value == "track"]
        start_rotation = quaternion_to_matrix(expert.welding_quaternion)
        end_rotation = quaternion_to_matrix(track[-1].pose.quaternion_wxyz)
        if config.task.orientation_follow_ratio == 0:
            np.testing.assert_allclose(start_rotation, end_rotation, atol=1e-7)
        elif abs(config.task.arc_sweep_deg) < 360:
            assert not np.allclose(start_rotation, end_rotation)
        if config.task.seam_id == "pipe_top":
            assert config.task.arc_sweep_deg == 360
            np.testing.assert_allclose(seam.start.position, seam.end.position, atol=1e-7)
        for frame in expert.frames:
            action = np.concatenate([frame.pose.position, frame.pose.quaternion_wxyz])
            *_, info = simulation.step(action)
            assert info["ik_residual_m"] < 0.005
            assert not info["collision_pairs"]
        wall_positions = [
            simulation.mj_model.geom(f"pipe_wall_{index:02d}").pos[:2]
            for index in range(config.workpiece.pipe_segments)
        ]
        radii = np.linalg.norm(wall_positions, axis=1)
        assert np.min(radii) > config.workpiece.pipe_wall_thickness_m
    finally:
        simulation.close()


def test_expert_uses_independent_phase_speeds() -> None:
    """接近阶段的参考位移应明显大于低速焊接阶段。"""
    config = AppConfig.load("configs/pipe_bottom.yaml")
    simulation = WeldingEnv(config, camera_observations=False)
    try:
        seam, _ = stage_for_task(simulation, config)
        frames = ExpertTrajectory(config, simulation.tcp_pose(), seam).frames
        displacements: dict[str, list[float]] = {"approach": [], "track": [], "retreat": []}
        for previous, current in pairwise(frames):
            if previous.phase != current.phase:
                continue
            displacement = float(np.linalg.norm(current.pose.position - previous.pose.position))
            if displacement > 1e-9:
                displacements[current.phase.value].append(displacement)
        approach_step = float(np.median(displacements["approach"]))
        track_step = float(np.median(displacements["track"]))
        assert approach_step > 10 * track_step
        assert approach_step == pytest.approx(
            config.task.approach_speed_mps / config.timing.policy_hz,
            rel=0.05,
        )
        assert track_step == pytest.approx(
            config.task.speed_mps / config.timing.policy_hz,
            rel=0.05,
        )
    finally:
        simulation.close()


def test_initial_joint_randomization_is_bounded_and_collision_free() -> None:
    config = AppConfig.load("configs/default.yaml")
    simulation = WeldingEnv(config)
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
