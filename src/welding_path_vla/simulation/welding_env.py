"""基于 robosuite 的 Elfin5-Pro 焊接环境。"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.domain import Pose, RobotState
from welding_path_vla.core.geometry import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_error,
    yaw_degrees_to_matrix,
)
from welding_path_vla.simulation.models import (
    SEAM_PENDING_RGBA,
    SEAM_WELDED_RGBA,
    Elfin5ProRobotModel,
    WeldingArena,
    WorkpieceObject,
)
from welding_path_vla.simulation.robosuite_compat import (
    MujocoEnv,
    Observable,
    Task,
    register_env,
    sensor,
)
from welding_path_vla.simulation.tasks import SeamPath


class WeldingEnv(MujocoEnv):
    """Elfin5-Pro 焊接任务的 robosuite 环境。

    robosuite 负责 ``reset``、``step``、观测更新和 episode 时钟；项目继续
    复用已经验证的完整 MJCF、阻尼最小二乘 IK、TCP 与碰撞几何。环境动作是
    世界坐标系下的绝对 TCP 位姿 ``[x, y, z, qw, qx, qy, qz]``。

    Attributes:
        config: 项目统一配置。
        last_collision_pairs: 最近一个策略周期内出现过的碰撞几何体名称对。
        last_joint_command: 最近一次送入 MuJoCo 位置执行器的六关节目标。
        last_ik_residual: 最近一次 IK 的六维位姿误差范数。
    """

    joint_names = tuple(f"elfin_joint{index}" for index in range(1, 7))
    motor_names = tuple(f"motor{index}" for index in range(1, 7))

    def __init__(
        self,
        config: AppConfig,
        seed: int | None = None,
        camera_observations: bool = True,
        ignore_done: bool = False,
    ) -> None:
        """创建环境并初始化 robosuite 生命周期。

        Args:
            config: 全局应用配置。
            seed: 环境随机数种子；任务采样仍可显式传入独立生成器。
            camera_observations: 是否创建离屏上下文并返回双相机观测。
            ignore_done: 是否忽略 horizon；专家采集使用该模式保存完整轨迹。
        """
        self.config = config
        self.camera_observations = camera_observations
        self.last_collision_pairs: tuple[tuple[str, str], ...] = ()
        self.last_joint_command = np.radians(config.robot.initial_joint_deg)
        self.last_ik_residual = float("nan")
        self.target_pose: Pose | None = None
        self.policy_physics_step = 0
        self.control_index = 0
        self.seam_progress_hint = 0.0
        self.observed_contacts: set[tuple[str, str]] = set()
        super().__init__(
            has_renderer=False,
            has_offscreen_renderer=camera_observations,
            render_camera=config.camera.global_name,
            render_collision_mesh=False,
            render_visual_mesh=True,
            control_freq=config.timing.policy_hz,
            horizon=config.deployment.max_steps,
            ignore_done=ignore_done,
            hard_reset=False,
            renderer="mujoco",
            seed=seed,
        )

    @property
    def robosuite_sim(self) -> Any:
        """返回 robosuite 的 MuJoCo 运行时包装器。"""
        return self.sim

    @property
    def mj_model(self) -> mujoco.MjModel:
        """返回 robosuite 持有的底层 MuJoCo 模型。"""
        return self.robosuite_sim.model._model

    @property
    def mj_data(self) -> mujoco.MjData:
        """返回 robosuite 持有的底层 MuJoCo 运行时数据。"""
        return self.robosuite_sim.data._data

    @property
    def action_dim(self) -> int:
        """返回绝对 TCP 位姿动作维数。"""
        return 7

    @property
    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``[位置, wxyz 四元数]`` 动作的数值范围。"""
        low = np.array([-np.inf, -np.inf, -np.inf, -1, -1, -1, -1], dtype=np.float64)
        high = np.array([np.inf, np.inf, np.inf, 1, 1, 1, 1], dtype=np.float64)
        return low, high

    def initialize_time(self, control_freq: float) -> None:
        """按项目配置建立 600/120/30 Hz 多速率时钟。

        robosuite 默认从全局宏读取物理步长。这里使用实例配置，避免修改进程
        级全局状态，也允许测试用配置安全地改变频率。

        Args:
            control_freq: robosuite 调用 ``step`` 的策略频率。
        """
        self.cur_time = 0
        self.model_timestep = 1.0 / self.config.timing.physics_hz
        self.control_freq = control_freq
        self.control_timestep = 1.0 / control_freq

    def _load_model(self) -> None:
        """分别构建机器人、Arena 和可替换工件，再组合为 robosuite Task。"""
        self.robot_model = Elfin5ProRobotModel(self.config)
        self.arena = WeldingArena(self.config)
        self.workpiece = WorkpieceObject(self.config)
        self.model = Task(self.arena, self.robot_model, self.workpiece)
        self.arena.apply_headlight(self.model.root)

    def _setup_references(self) -> None:
        """缓存关节、执行器、TCP 和工件的底层索引。"""
        super()._setup_references()
        self.mj_model.opt.timestep = 1.0 / self.config.timing.physics_hz
        self.tcp_id = self.name_id(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.workpiece_id = self.name_id(mujoco.mjtObj.mjOBJ_BODY, "workpiece")
        self.torch_tip_id = self.name_id(mujoco.mjtObj.mjOBJ_GEOM, "torch_tip")
        self.workpiece_geom_ids = {
            geom_id
            for geom_id in range(self.mj_model.ngeom)
            if self.mj_model.geom_bodyid[geom_id] == self.workpiece_id
        }
        self.weld_visual_geom_ids = np.asarray(
            [
                self.name_id(mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in self.workpiece.weld_visual_names
            ],
            dtype=np.int32,
        )
        self.joint_ids = [
            self.name_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names
        ]
        self.qpos_ids = [int(self.mj_model.jnt_qposadr[joint_id]) for joint_id in self.joint_ids]
        self.dof_ids = [int(self.mj_model.jnt_dofadr[joint_id]) for joint_id in self.joint_ids]
        self.joint_ranges = self.mj_model.jnt_range[self.joint_ids].copy()
        self.motor_ids = [
            self.name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.motor_names
        ]
        self.ik_data = mujoco.MjData(self.mj_model)

    def _reset_internal(self) -> None:
        """重置 robosuite 状态、关节初值和渲染可见组。"""
        super()._reset_internal()
        if self.camera_observations:
            context = self.robosuite_sim._render_context_offscreen
            context.vopt.geomgroup[0] = 0
            context.vopt.sitegroup[5] = 0
        self.set_joint_position(np.radians(self.config.robot.initial_joint_deg))
        self.reset_weld_visuals()
        self.last_collision_pairs = ()
        self.target_pose = None
        self.seam_progress_hint = 0.0

    def _setup_observables(self) -> OrderedDict[str, Observable]:
        """建立双相机、关节和 TCP 的统一 observation dictionary。"""
        observables: OrderedDict[str, Observable] = OrderedDict()

        @sensor(modality="proprio")
        def joint_position(obs_cache: dict[str, Any]) -> np.ndarray:
            """读取六轴关节角，单位为弧度。"""
            return self.mj_data.qpos[self.qpos_ids].copy()

        @sensor(modality="proprio")
        def joint_velocity(obs_cache: dict[str, Any]) -> np.ndarray:
            """读取六轴关节速度，单位为弧度每秒。"""
            return self.mj_data.qvel[self.dof_ids].copy()

        @sensor(modality="proprio")
        def tcp_position(obs_cache: dict[str, Any]) -> np.ndarray:
            """读取 TCP 世界坐标，单位为米。"""
            return self.mj_data.site_xpos[self.tcp_id].copy()

        @sensor(modality="proprio")
        def tcp_quaternion_wxyz(obs_cache: dict[str, Any]) -> np.ndarray:
            """读取 TCP 世界姿态，四元数顺序为 wxyz。"""
            matrix = self.mj_data.site_xmat[self.tcp_id].reshape(3, 3)
            return matrix_to_quaternion(matrix)

        proprio_sensors = (
            joint_position,
            joint_velocity,
            tcp_position,
            tcp_quaternion_wxyz,
        )
        for proprio_sensor in proprio_sensors:
            name = proprio_sensor.__name__
            observables[name] = Observable(
                name=name,
                sensor=proprio_sensor,
                sampling_rate=self.config.timing.control_hz,
            )

        camera_names = (
            (self.config.camera.global_name, self.config.camera.wrist_name)
            if self.camera_observations
            else ()
        )
        for camera_name in camera_names:
            image_name = f"{camera_name}_image"

            @sensor(modality="image")
            def camera_image(
                obs_cache: dict[str, Any],
                camera_name: str = camera_name,
            ) -> np.ndarray:
                """渲染 RGB 图像，并转换为常用的左上角原点约定。"""
                image = self.robosuite_sim.render(
                    camera_name=camera_name,
                    width=self.config.camera.width,
                    height=self.config.camera.height,
                )
                return image[::-1].copy()

            camera_image.__name__ = image_name
            observables[image_name] = Observable(
                name=image_name,
                sensor=camera_image,
                sampling_rate=self.config.timing.policy_hz,
            )
        return observables

    def _pre_action(self, action: np.ndarray, policy_step: bool = False) -> None:
        """把一个绝对 TCP 目标转换为 120 Hz 关节位置命令。

        robosuite 在一个 30 Hz ``step`` 内调用本方法 20 次。每隔 5 个
        物理步重新求解一次 IK，从而保持 600 Hz 物理、120 Hz 控制
        和 30 Hz 策略时序。

        Args:
            action: 世界系绝对 TCP 位姿 ``[x, y, z, qw, qx, qy, qz]``。
            policy_step: 当前调用是否为新策略动作的第一个物理步。
        """
        self.observed_contacts.update(self.collision_pairs)
        if policy_step:
            values = np.asarray(action, dtype=np.float64).reshape(self.action_dim)
            quaternion = values[3:] / np.linalg.norm(values[3:])
            self.target_pose = Pose(values[:3].copy(), quaternion)
            self.policy_physics_step = 0
            self.control_index = 0
            self.observed_contacts.clear()

        if self.policy_physics_step % self.config.timing.physics_steps_per_control == 0:
            current = self.tcp_pose()
            controls_left = self.config.timing.controls_per_policy - self.control_index
            fraction = 1.0 / controls_left
            target = self.target_pose
            assert target is not None
            intermediate = Pose(
                current.position + fraction * (target.position - current.position),
                target.quaternion_wxyz,
            )
            current_joint = self.mj_data.qpos[self.qpos_ids]
            # 上一控制目标比带有执行器滞后的实际状态更适合作为连续 IK 初值，
            # 可避免圆弧经过奇异邻域时在关节解分支之间跳转。
            command, residual = self.solve_ik(intermediate, self.last_joint_command)
            max_delta = self.config.robot.joint_velocity_limit / self.config.timing.control_hz
            command = current_joint + np.clip(command - current_joint, -max_delta, max_delta)
            self.last_joint_command = command.copy()
            self.last_ik_residual = residual
            self.control_index += 1

        self.mj_data.ctrl[self.motor_ids] = self.last_joint_command
        self.policy_physics_step += 1

    def _post_action(self, action: np.ndarray) -> tuple[float, bool, dict[str, Any]]:
        """汇总一个策略周期的接触、成功状态和控制诊断。"""
        self.observed_contacts.update(self.collision_pairs)
        self.last_collision_pairs = tuple(sorted(self.observed_contacts))
        self.update_weld_visuals()
        reward, done, info = super()._post_action(action)
        info.update(
            {
                "success": self._check_success(),
                "collision": bool(self.last_collision_pairs),
                "collision_pairs": self.last_collision_pairs,
                "joint_command": self.last_joint_command.copy(),
                "ik_residual_m": self.last_ik_residual,
            }
        )
        return reward, done, info

    def reset_weld_visuals(self) -> None:
        """将所有候选焊缝恢复为未焊接的黑色。"""
        self.mj_model.geom_rgba[self.weld_visual_geom_ids] = SEAM_PENDING_RGBA

    def update_weld_visuals(self, tcp_position: np.ndarray | None = None) -> None:
        """将 TCP 距离阈值内的焊缝分段永久标记为白色。

        Args:
            tcp_position: 可选 TCP 世界坐标；默认读取当前仿真状态。
        """
        tcp = self.tcp_pose().position if tcp_position is None else np.asarray(tcp_position)
        geom_ids = self.weld_visual_geom_ids
        centers = self.mj_data.geom_xpos[geom_ids]
        rotations = self.mj_data.geom_xmat[geom_ids].reshape(-1, 3, 3)
        half_lengths = self.mj_model.geom_size[geom_ids, 1]
        axes = rotations[:, :, 2]
        starts = centers - half_lengths[:, None] * axes
        vectors = 2 * half_lengths[:, None] * axes
        progress = np.sum((tcp - starts) * vectors, axis=1) / np.sum(vectors**2, axis=1)
        closest = starts + np.clip(progress, 0, 1)[:, None] * vectors
        distances = np.linalg.norm(tcp - closest, axis=1)
        welded = geom_ids[distances <= self.config.task.weld_success_distance_m]
        self.mj_model.geom_rgba[welded] = SEAM_WELDED_RGBA

    def reward(self, action: np.ndarray | None = None) -> float:
        """使用稀疏任务奖励：到达焊缝末端且仍在跟踪带内时为 1。"""
        return float(self._check_success())

    def _check_success(self) -> bool:
        """判断 TCP 是否到达配置定义的自然完成区域。"""
        projection = self.active_seam().project(
            self.tcp_pose().position,
            self.seam_progress_hint,
        )
        if projection.distance_m <= self.config.evaluation.tracking_band_m:
            self.seam_progress_hint = projection.progress
        return (
            projection.raw_progress >= self.config.deployment.completion_progress_min
            and projection.distance_m <= self.config.deployment.completion_distance_m
        )

    def active_seam(self) -> SeamPath:
        """返回当前随机化工件位姿下的有向焊缝。"""
        position = self.mj_data.xpos[self.workpiece_id].copy()
        rotation = self.mj_data.xmat[self.workpiece_id].reshape(3, 3).copy()
        return self.workpiece.seam(position, rotation)

    def base_rotation(self) -> np.ndarray:
        """返回机器人底座相对世界系的旋转矩阵。"""
        return yaw_degrees_to_matrix(self.config.scene.robot_base_yaw_deg)

    def name_id(self, kind: mujoco.mjtObj, name: str) -> int:
        """按名称查询底层 MuJoCo 对象 ID。

        Args:
            kind: MuJoCo 对象类型。
            name: MJCF 中的对象名称。

        Returns:
            对象整数 ID。
        """
        value = mujoco.mj_name2id(self.mj_model, kind, name)
        if value < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return value

    def reset(self, seed: int | None = None) -> OrderedDict[str, np.ndarray]:
        """重置环境，并可同时更新 robosuite 随机数生成器。

        Args:
            seed: 新的随机种子；未传入时延续当前生成器。

        Returns:
            robosuite observation dictionary。
        """
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        return super().reset()

    def observe(self) -> OrderedDict[str, np.ndarray]:
        """立即读取与 ``reset`` / ``step`` 同构的最新观测。"""
        return self._get_observations(force_update=True)

    def visualize(self, vis_settings: dict[str, bool]) -> None:
        """保留 robosuite 可视化接口。

        当前不显示 robosuite 辅助 site，因此重置时无需切换对象标记。

        Args:
            vis_settings: robosuite 请求的可视化开关。
        """

    def images_from_observation(
        self,
        observation: OrderedDict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """从 robosuite 观测中提取数据集使用的相机名称映射。

        Args:
            observation: ``reset``、``step`` 或 ``observe`` 返回的观测。

        Returns:
            ``{"global": RGB, "wrist": RGB}`` 形式的图像字典。
        """
        return {
            name: observation[f"{name}_image"]
            for name in (self.config.camera.global_name, self.config.camera.wrist_name)
        }

    def randomize_workpiece(self, rng: np.random.Generator) -> None:
        """在 YAML 配置范围内随机化工件位置和偏航角。

        Args:
            rng: episode 专用随机数生成器。
        """
        randomization = self.config.randomization
        position = np.asarray(self.config.scene.workpiece_position_m, dtype=np.float64).copy()
        position += rng.uniform(
            [-randomization.xy_m, -randomization.xy_m, -randomization.z_m],
            [randomization.xy_m, randomization.xy_m, randomization.z_m],
        )
        yaw = np.radians(rng.uniform(-randomization.yaw_deg, randomization.yaw_deg))
        self.mj_model.body_pos[self.workpiece_id] = position
        self.mj_model.body_quat[self.workpiece_id] = [
            np.cos(yaw / 2),
            0,
            0,
            np.sin(yaw / 2),
        ]
        self.robosuite_sim.forward()

    def site_position(self, name: str) -> np.ndarray:
        """返回指定 site 的世界坐标位置。"""
        site_id = self.name_id(mujoco.mjtObj.mjOBJ_SITE, name)
        return self.mj_data.site_xpos[site_id].copy()

    def body_rotation(self, name: str) -> np.ndarray:
        """返回指定 body 的世界坐标旋转矩阵。"""
        body_id = self.name_id(mujoco.mjtObj.mjOBJ_BODY, name)
        return self.mj_data.xmat[body_id].reshape(3, 3).copy()

    def tcp_pose(self) -> Pose:
        """返回当前 TCP 世界位姿。"""
        matrix = self.mj_data.site_xmat[self.tcp_id].reshape(3, 3)
        return Pose(
            self.mj_data.site_xpos[self.tcp_id].copy(),
            matrix_to_quaternion(matrix),
        )

    def state(self) -> RobotState:
        """返回当前关节状态和 TCP 位姿。"""
        return RobotState(
            joint_position=self.mj_data.qpos[self.qpos_ids].copy(),
            joint_velocity=self.mj_data.qvel[self.dof_ids].copy(),
            tcp=self.tcp_pose(),
        )

    def set_joint_position(self, joint_position: np.ndarray) -> None:
        """直接设置六轴位置，并同步位置执行器目标。"""
        self.mj_data.qpos[self.qpos_ids] = joint_position
        self.mj_data.qvel[:] = 0
        self.mj_data.ctrl[self.motor_ids] = joint_position
        self.robosuite_sim.forward()

    def pose_for_joint_position(self, joint_position: np.ndarray) -> Pose:
        """使用独立 IK 数据计算关节命令对应的 TCP 位姿。"""
        self.ik_data.qpos[:] = self.mj_data.qpos
        self.ik_data.qpos[self.qpos_ids] = joint_position
        mujoco.mj_forward(self.mj_model, self.ik_data)
        matrix = self.ik_data.site_xmat[self.tcp_id].reshape(3, 3)
        return Pose(
            self.ik_data.site_xpos[self.tcp_id].copy(),
            matrix_to_quaternion(matrix),
        )

    def randomize_joint_position(
        self,
        rng: np.random.Generator,
        max_offset_deg: list[float],
        attempts: int,
    ) -> tuple[np.ndarray, int]:
        """在当前构型附近采样无碰撞初始关节位置。

        Args:
            rng: episode 专用随机数生成器。
            max_offset_deg: 六个关节各自允许的最大偏移角。
            attempts: 最大采样次数。

        Returns:
            实际关节偏移（度）和成功采样次数。
        """
        center = self.mj_data.qpos[self.qpos_ids].copy()
        radius = np.radians(max_offset_deg)
        margin = self.config.safety.joint_position_margin_rad
        lower = np.maximum(center - radius, self.joint_ranges[:, 0] + margin)
        upper = np.minimum(center + radius, self.joint_ranges[:, 1] - margin)
        table_top = self.config.scene.table_center_m[2] + self.config.scene.table_half_size_m[2]
        for attempt in range(1, attempts + 1):
            candidate = rng.uniform(lower, upper)
            self.set_joint_position(candidate)
            if not self.collision and self.tcp_pose().position[2] > table_top:
                return np.degrees(candidate - center), attempt
        self.set_joint_position(center)
        raise RuntimeError(f"cannot sample collision-free initial joints after {attempts} attempts")

    def solve_ik(
        self,
        target: Pose,
        seed_joint_position: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """在独立 ``MjData`` 上使用阻尼最小二乘法求解 IK。

        独立数据避免在 robosuite 的 ``mj_step1`` / ``mj_step2`` 之间改写主
        仿真状态。

        Args:
            target: 世界坐标系下的目标 TCP 位姿。
            seed_joint_position: 可选的六轴初值；连续轨迹规划使用上一帧解，
                单步控制默认从当前实际关节位置开始。

        Returns:
            六关节解和最终六维位姿误差范数。
        """
        robot = self.config.robot
        data = self.ik_data
        data.qpos[:] = self.mj_data.qpos
        if seed_joint_position is not None:
            data.qpos[self.qpos_ids] = seed_joint_position
        data.qvel[:] = self.mj_data.qvel
        mujoco.mj_forward(self.mj_model, data)
        residual = float("inf")
        remaining_iterations = robot.ik_iterations
        while remaining_iterations:
            matrix = data.site_xmat[self.tcp_id].reshape(3, 3)
            error = np.concatenate(
                [
                    target.position - data.site_xpos[self.tcp_id],
                    rotation_error(
                        target.quaternion_wxyz,
                        matrix_to_quaternion(matrix),
                    ),
                ]
            )
            residual = float(np.linalg.norm(error))
            if residual <= robot.ik_tolerance:
                break
            jacobian_position = np.zeros((3, self.mj_model.nv))
            jacobian_rotation = np.zeros((3, self.mj_model.nv))
            mujoco.mj_jacSite(
                self.mj_model,
                data,
                jacobian_position,
                jacobian_rotation,
                self.tcp_id,
            )
            jacobian = np.vstack(
                [
                    jacobian_position[:, self.dof_ids],
                    jacobian_rotation[:, self.dof_ids],
                ]
            )
            regularizer = robot.ik_damping**2 * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + regularizer,
                error,
            )
            norm = float(np.linalg.norm(delta))
            if norm > robot.ik_max_step:
                delta *= robot.ik_max_step / norm
            data.qpos[self.qpos_ids] = np.clip(
                data.qpos[self.qpos_ids] + delta,
                self.joint_ranges[:, 0],
                self.joint_ranges[:, 1],
            )
            mujoco.mj_forward(self.mj_model, data)
            remaining_iterations -= 1
        matrix = data.site_xmat[self.tcp_id].reshape(3, 3)
        final_error = np.concatenate(
            [
                target.position - data.site_xpos[self.tcp_id],
                rotation_error(
                    target.quaternion_wxyz,
                    matrix_to_quaternion(matrix),
                ),
            ]
        )
        residual = float(np.linalg.norm(final_error))
        return data.qpos[self.qpos_ids].copy(), residual

    def perturb_tcp(
        self,
        offset: np.ndarray,
        rotation_vector: np.ndarray | None = None,
    ) -> bool:
        """给 TCP 施加位置和姿态扰动，用于生成恢复样本。

        Args:
            offset: 三维世界系位置偏移，单位为米。
            rotation_vector: 三维旋转向量，单位为弧度。

        Returns:
            扰动是否成功且未产生碰撞。
        """
        current = self.tcp_pose()
        current_joints = self.mj_data.qpos[self.qpos_ids].copy()
        rotation = (
            Rotation.from_rotvec(rotation_vector).as_matrix()
            if rotation_vector is not None and np.linalg.norm(rotation_vector)
            else np.eye(3)
        )
        target = Pose(
            current.position + offset,
            matrix_to_quaternion(rotation @ quaternion_to_matrix(current.quaternion_wxyz)),
        )
        solution, residual = self.solve_ik(target)
        if residual > 0.003:
            return False
        self.set_joint_position(solution)
        if self.collision:
            self.set_joint_position(current_joints)
            return False
        return True

    @property
    def collision(self) -> bool:
        """返回当前时刻是否存在需要中止任务的有效碰撞。"""
        return bool(self.collision_pairs)

    @property
    def collision_pairs(self) -> tuple[tuple[str, str], ...]:
        """返回有效碰撞对，忽略焊丝尖端对工件的浅层正常接触。"""
        return self.collision_pairs_for(self.mj_data)

    def collision_pairs_for(self, data: mujoco.MjData) -> tuple[tuple[str, str], ...]:
        """读取指定仿真状态中的有效碰撞对。

        Args:
            data: 主仿真状态或轨迹预检使用的独立 MuJoCo 状态。

        Returns:
            去除焊丝尖端正常擦碰后的几何体名称对。
        """
        pairs: list[tuple[str, str]] = []
        for index, contact in enumerate(data.contact):
            geom_ids = (contact.geom1, contact.geom2)
            if self.torch_tip_id in geom_ids:
                other_id = contact.geom2 if contact.geom1 == self.torch_tip_id else contact.geom1
                if other_id in self.workpiece_geom_ids:
                    # 刚性位置伺服会把亚毫米穿透放大为数百牛的瞬时接触力，
                    # 因此先用几何深度识别焊丝尖端在焊缝上的正常擦碰。
                    if contact.dist >= -self.config.safety.tip_contact_penetration_limit_m:
                        continue
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.mj_model, data, index, force)
                    if np.linalg.norm(force[:3]) < self.config.safety.tip_contact_force_limit_n:
                        continue
            names = tuple(
                mujoco.mj_id2name(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    geom_id,
                )
                or f"geom_{geom_id}"
                for geom_id in geom_ids
            )
            pairs.append((names[0], names[1]))
        return tuple(pairs)


register_env(WeldingEnv)

__all__ = ["WeldingEnv"]
