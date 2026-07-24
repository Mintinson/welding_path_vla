"""MuJoCo 焊接仿真环境: 封装场景配置、IK 解算、碰撞检测与渲染。"""

from __future__ import annotations

from importlib.resources import files

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from welding_path_vla.config import AppConfig
from welding_path_vla.domain import Pose, RobotState
from welding_path_vla.geometry import (
    look_at_quaternion,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_error,
    yaw_degrees_to_matrix,
)


class WeldingSimulation:
    """华研 Elfin5-Pro 焊接仿真环境的 MuJoCo 封装。

    管理场景配置、工件随机化、正逆运动学、碰撞检测和多视角渲染。
    提供 IK 解算、轨迹跟踪执行和 TCP 扰动等核心接口。
    """

    joint_names = tuple(f"elfin_joint{i}" for i in range(1, 7))
    motor_names = tuple(f"motor{i}" for i in range(1, 7))

    def __init__(self, config: AppConfig) -> None:
        """加载模型, 配置场景, 初始化 MuJoCo 数据与渲染器。

        Args:
            config: 全局应用配置, 含机器人、场景、相机等参数。
        """
        self.config = config
        path = files("welding_path_vla").joinpath("assets", config.robot.model_asset)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.model.opt.timestep = 1.0 / config.timing.physics_hz
        self.configure_scene()
        self.data = mujoco.MjData(self.model)
        self.tcp_id = self.name_id(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.workpiece_id = self.name_id(mujoco.mjtObj.mjOBJ_BODY, "workpiece")
        self.joint_ids = [
            self.name_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names
        ]
        self.qpos_ids = [int(self.model.jnt_qposadr[joint_id]) for joint_id in self.joint_ids]
        self.dof_ids = [int(self.model.jnt_dofadr[joint_id]) for joint_id in self.joint_ids]
        self.joint_ranges = self.model.jnt_range[self.joint_ids].copy()
        self.motor_ids = [
            self.name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.motor_names
        ]
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[3] = 0
        self.scene_option.sitegroup[5] = 0
        self.last_collision_pairs: tuple[tuple[str, str], ...] = ()
        self.renderers: dict[str, mujoco.Renderer] = {}
        self.reset()

    def configure_scene(self) -> None:
        """根据配置设置桌子、机器人底座、工件、相机等场景元素。

        将配置中的位置/姿态参数写入 MuJoCo 模型的对应字段。
        """
        scene = self.config.scene
        table_id = self.name_id(mujoco.mjtObj.mjOBJ_GEOM, "table")
        table_frame_id = self.name_id(mujoco.mjtObj.mjOBJ_BODY, "table_frame")
        mount_id = self.name_id(mujoco.mjtObj.mjOBJ_BODY, "robot_mount")
        workpiece_id = self.name_id(mujoco.mjtObj.mjOBJ_BODY, "workpiece")
        camera_id = self.name_id(mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera.global_name)
        wrist_camera_id = self.name_id(mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera.wrist_name)
        seam_start_id = self.name_id(mujoco.mjtObj.mjOBJ_SITE, "seam_start")
        seam_end_id = self.name_id(mujoco.mjtObj.mjOBJ_SITE, "seam_end")
        base = np.asarray(scene.robot_base_position_m)
        base_quaternion = matrix_to_quaternion(yaw_degrees_to_matrix(scene.robot_base_yaw_deg))
        self.model.body_pos[table_frame_id] = scene.table_center_m
        self.model.geom_pos[table_id] = 0
        self.model.geom_size[table_id] = scene.table_half_size_m
        self.model.body_pos[mount_id] = base
        self.model.body_quat[mount_id] = base_quaternion
        self.model.body_pos[workpiece_id] = scene.workpiece_position_m
        camera_position = np.asarray(scene.global_camera_position_table_m)
        self.model.cam_mode[camera_id] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        self.model.cam_targetbodyid[camera_id] = -1
        self.model.cam_pos[camera_id] = camera_position
        self.model.cam_quat[camera_id] = look_at_quaternion(
            camera_position,
            np.asarray(scene.global_camera_target_table_m),
            np.asarray(scene.global_camera_up_table),
        )
        self.model.cam_fovy[camera_id] = self.config.camera.global_fovy_deg
        wrist_position = np.asarray(self.config.camera.wrist_position_link6_m)
        self.model.cam_pos[wrist_camera_id] = wrist_position
        self.model.cam_quat[wrist_camera_id] = look_at_quaternion(
            wrist_position,
            np.asarray(self.config.camera.wrist_target_link6_m),
            np.asarray(self.config.camera.wrist_up_link6),
        )
        self.model.cam_fovy[wrist_camera_id] = self.config.camera.wrist_fovy_deg
        seam_surface = 0.0025 + self.config.task.tcp_clearance_m
        self.model.site_pos[[seam_start_id, seam_end_id], 0] = seam_surface
        self.model.site_pos[[seam_start_id, seam_end_id], 2] = seam_surface

    def base_rotation(self) -> np.ndarray:
        """返回机器人底座的旋转矩阵 (仅偏航角)。

        Returns:
            3x3 旋转矩阵。
        """
        return yaw_degrees_to_matrix(self.config.scene.robot_base_yaw_deg)

    def name_id(self, kind: mujoco.mjtObj, name: str) -> int:
        """通过 MuJoCo 名称查询对象 ID。

        Args:
            kind: MuJoCo 对象类型 (如 mjOBJ_BODY、mjOBJ_SITE)。
            name: 对象名称。

        Returns:
            对象 ID。

        Raises:
            ValueError: 名称不存在于模型中。
        """
        value = mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return value

    def reset(self, seed: int = 0) -> None:
        """重置仿真到初始关节位置, 清除碰撞记录。

        Args:
            seed: 未使用, 保留以兼容接口。
        """
        mujoco.mj_resetData(self.model, self.data)
        joint_position = np.radians(self.config.robot.initial_joint_deg)
        self.data.qpos[self.qpos_ids] = joint_position
        self.data.ctrl[self.motor_ids] = joint_position
        mujoco.mj_forward(self.model, self.data)
        self.last_collision_pairs = ()

    def randomize_workpiece(self, rng: np.random.Generator) -> None:
        """在配置范围内随机化工件位置和偏航角。

        Args:
            rng: 随机数生成器。
        """
        randomization = self.config.randomization
        base = np.asarray(self.config.scene.workpiece_position_m, dtype=np.float64).copy()
        base += rng.uniform(
            [-randomization.xy_m, -randomization.xy_m, -randomization.z_m],
            [randomization.xy_m, randomization.xy_m, randomization.z_m],
        )
        yaw = np.radians(rng.uniform(-randomization.yaw_deg, randomization.yaw_deg))
        self.model.body_pos[self.workpiece_id] = base
        self.model.body_quat[self.workpiece_id] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)

    def site_position(self, name: str) -> np.ndarray:
        """获取指定 site 的世界坐标位置。

        Args:
            name: site 名称 (如 "seam_start")。

        Returns:
            三维位置向量。
        """
        site_id = self.name_id(mujoco.mjtObj.mjOBJ_SITE, name)
        return self.data.site_xpos[site_id].copy()

    def body_rotation(self, name: str) -> np.ndarray:
        """获取指定 body 的世界坐标系旋转矩阵。

        Args:
            name: body 名称 (如 "workpiece")。

        Returns:
            3x3 旋转矩阵。
        """
        body_id = self.name_id(mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xmat[body_id].reshape(3, 3).copy()

    def tcp_pose(self) -> Pose:
        """获取 TCP 当前位姿。

        Returns:
            包含位置和四元数的位姿对象。
        """
        matrix = self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()
        return Pose(self.data.site_xpos[self.tcp_id].copy(), matrix_to_quaternion(matrix))

    def state(self) -> RobotState:
        """获取当前完整机器人状态。

        Returns:
            包含关节位置、速度和 TCP 位姿的状态对象。
        """
        return RobotState(
            joint_position=self.data.qpos[self.qpos_ids].copy(),
            joint_velocity=self.data.qvel[self.dof_ids].copy(),
            tcp=self.tcp_pose(),
        )

    def set_joint_position(self, joint_position: np.ndarray) -> None:
        """设置六轴位置并同步控制目标。"""
        self.data.qpos[self.qpos_ids] = joint_position
        self.data.qvel[:] = 0
        self.data.ctrl[self.motor_ids] = joint_position
        mujoco.mj_forward(self.model, self.data)

    def pose_for_joint_position(self, joint_position: np.ndarray) -> Pose:
        """计算关节命令对应的 TCP 位姿，不改变当前仿真状态。"""
        original_qpos = self.data.qpos.copy()
        original_qvel = self.data.qvel.copy()
        try:
            self.data.qpos[self.qpos_ids] = joint_position
            mujoco.mj_forward(self.model, self.data)
            return self.tcp_pose()
        finally:
            self.data.qpos[:] = original_qpos
            self.data.qvel[:] = original_qvel
            mujoco.mj_forward(self.model, self.data)

    def randomize_joint_position(
        self, rng: np.random.Generator, max_offset_deg: list[float], attempts: int
    ) -> tuple[np.ndarray, int]:
        """在当前构型附近采样无碰撞初始关节位置。

        Args:
            rng: episode 专用随机数生成器。
            max_offset_deg: 六个关节各自允许的最大角度偏移。
            attempts: 最大采样次数。

        Returns:
            实际关节偏移（度）和成功时的采样次数。
        """
        center = self.data.qpos[self.qpos_ids].copy()
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

    def render(self) -> dict[str, np.ndarray]:
        """渲染全局和腕部相机画面。

        相机通过配置中的名称确定, 使用场景选项隐藏碰撞几何体。

        Returns:
            相机名称到 RGB 图像的映射字典。
        """
        output: dict[str, np.ndarray] = {}
        for name in (self.config.camera.global_name, self.config.camera.wrist_name):
            if name not in self.renderers:
                self.renderers[name] = mujoco.Renderer(
                    self.model,
                    height=self.config.camera.height,
                    width=self.config.camera.width,
                )
            renderer = self.renderers[name]
            renderer.update_scene(self.data, camera=name, scene_option=self.scene_option)
            output[name] = renderer.render().copy()
        return output

    def solve_ik(self, target: Pose) -> tuple[np.ndarray, float]:
        """使用阻尼最小二乘法求解逆运动学。

        在当前位置进行迭代, 雅可比转置法配合阻尼正则化, 避免奇异点附近的
        数值不稳定。

        Args:
            target: 目标 TCP 位姿。

        Returns:
            (solution, residual), 分别为关节位置解和最终位姿误差范数。
        """
        robot = self.config.robot
        original_qpos = self.data.qpos.copy()
        original_qvel = self.data.qvel.copy()
        residual = float("inf")
        try:
            for _ in range(robot.ik_iterations):
                current = self.tcp_pose()
                error = np.concatenate(
                    [
                        target.position - current.position,
                        rotation_error(target.quaternion_wxyz, current.quaternion_wxyz),
                    ]
                )
                residual = float(np.linalg.norm(error))
                if residual <= robot.ik_tolerance:
                    break
                jacobian_position = np.zeros((3, self.model.nv))
                jacobian_rotation = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(
                    self.model,
                    self.data,
                    jacobian_position,
                    jacobian_rotation,
                    self.tcp_id,
                )
                jacobian = np.vstack(
                    [jacobian_position[:, self.dof_ids], jacobian_rotation[:, self.dof_ids]]
                )
                regularizer = robot.ik_damping**2 * np.eye(6)
                delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + regularizer, error)
                norm = float(np.linalg.norm(delta))
                if norm > robot.ik_max_step:
                    delta *= robot.ik_max_step / norm
                self.data.qpos[self.qpos_ids] += delta
                self.data.qpos[self.qpos_ids] = np.clip(
                    self.data.qpos[self.qpos_ids],
                    self.joint_ranges[:, 0],
                    self.joint_ranges[:, 1],
                )
                mujoco.mj_forward(self.model, self.data)
            solution = self.data.qpos[self.qpos_ids].copy()
        finally:
            self.data.qpos[:] = original_qpos
            self.data.qvel[:] = original_qvel
            mujoco.mj_forward(self.model, self.data)
        return solution, residual

    def execute_pose(self, target: Pose) -> tuple[np.ndarray, float]:
        """多步执行目标位姿: 逐步插值 IK 解算 + 物理步进。

        将目标位姿拆分为多个 control 步, 每步 IK 求解后限幅关节速度,
        再执行多个 physics 子步收集接触信息。

        Args:
            target: 目标 TCP 位姿。

        Returns:
            (joint_command, ik_residual), 最终关节命令和 IK 残差。
        """
        controls = self.config.timing.controls_per_policy
        physics_steps = self.config.timing.physics_steps_per_control
        residual = float("inf")
        command = self.state().joint_position
        observed_contacts: set[tuple[str, str]] = set()
        for control_index in range(controls):
            current = self.tcp_pose()
            fraction = 1.0 / (controls - control_index)
            intermediate = Pose(
                current.position + fraction * (target.position - current.position),
                target.quaternion_wxyz,
            )
            command, residual = self.solve_ik(intermediate)
            current_joint = self.data.qpos[self.qpos_ids].copy()
            max_delta = self.config.robot.joint_velocity_limit / self.config.timing.control_hz
            command = current_joint + np.clip(command - current_joint, -max_delta, max_delta)
            self.data.ctrl[self.motor_ids] = command
            for _ in range(physics_steps):
                mujoco.mj_step(self.model, self.data)
                observed_contacts.update(self.collision_pairs)
        self.last_collision_pairs = tuple(sorted(observed_contacts))
        return command, residual

    def perturb_tcp(self, offset: np.ndarray, rotation_vector: np.ndarray | None = None) -> bool:
        """给 TCP 施加偏移和旋转扰动, 模拟恢复场景。

        通过 IK 求解新位姿, 若结果碰撞或无解则恢复原构型。

        Args:
            offset: 三维位置偏移 (米)。
            rotation_vector: 旋转向量 (弧度), 可选。

        Returns:
            扰动是否成功应用。
        """
        current = self.tcp_pose()
        current_joints = self.data.qpos[self.qpos_ids].copy()
        rotation = np.eye(3)
        if rotation_vector is not None:
            angle = float(np.linalg.norm(rotation_vector))
            if angle:
                rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
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
        """是否有任何碰撞接触。"""
        return bool(self.data.ncon)

    @property
    def collision_pairs(self) -> tuple[tuple[str, str], ...]:
        """当前碰撞接触的几何体名称对列表。"""
        pairs: list[tuple[str, str]] = []
        for contact in self.data.contact:
            names = tuple(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                or f"geom_{geom_id}"
                for geom_id in (contact.geom1, contact.geom2)
            )
            pairs.append((names[0], names[1]))
        return tuple(pairs)

    def close(self) -> None:
        """释放所有渲染器资源。"""
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()
