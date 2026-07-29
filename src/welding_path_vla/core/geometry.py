"""几何工具: 四元数/旋转矩阵/齐次变换之间的转换和轨迹相关计算。

本模块提供的几何原语涵盖:
  - 四元数与旋转矩阵的双向转换 (wxyz 格式优先)
  - 位姿增量 (delta pose) 计算与坐标系变换
  - 6D 旋转表示 (rotation_6d) 与 SO(3) 的相互转换
  - 逆运动学中使用的旋转误差 (轴角) 计算
  - 部署时将 9D 相对动作解码为世界坐标目标位姿

四元数约定: 全文使用 (w, x, y, z) 顺序, 与 MuJoCo 和 scipy 的
(x, y, z, w) 不同, 因此在接口边界处需要 roll 转换。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from welding_path_vla.core.domain import Pose

type FloatArray = NDArray[np.float64]


def normalize(vector: np.ndarray) -> np.ndarray:
    """归一化向量, 对零长度向量抛出明确的异常信息。

    主要用于旋转表示的归一化, 零长度向量意味着网络预测退化了,
    检测到这种情况直接报错比静默处理更安全。

    Args:
        vector: 任意维度的向量。

    Returns:
        单位向量。

    Raises:
        ValueError: 向量长度 < 1e-8, 网络预测退化。
    """
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("ACT predicted a degenerate rotation")
    return vector / norm


def normalize_quaternion(quaternion_wxyz: FloatArray) -> FloatArray:
    """归一化四元数并确保 w >= 0, 消除四元数的符号歧义。

    四元数 q 和 -q 代表相同的旋转, 但符号变化会导致插值和误差计算
    不连续。约定 w >= 0 确保表示唯一。

    Args:
        quaternion_wxyz: 输入四元数 (w, x, y, z)。

    Returns:
        归一化后的四元数, w >= 0。
    """
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    # 取 w >= 0 的等价表示, 消除符号歧义
    if quaternion[0] < 0:
        quaternion = -quaternion
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_matrix(quaternion_wxyz: FloatArray) -> FloatArray:
    """将 (w, x, y, z) 四元数转换为 3x3 旋转矩阵。

    pip 安装的 scipy 的 Rotation.from_quat 接受 (x, y, z, w) 格式,
    因此需要在接口处 roll 顺序。

    Args:
        quaternion_wxyz: 四元数 (w, x, y, z)。

    Returns:
        3x3 旋转矩阵 (SO(3))。
    """
    w, x, y, z = normalize_quaternion(quaternion_wxyz)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def matrix_to_quaternion(matrix: FloatArray) -> FloatArray:
    """将 3x3 旋转矩阵转换为 (w, x, y, z) 四元数。

    scipy 的 as_quat() 返回 (x, y, z, w), 需要转回本项目的 (w, x, y, z) 约定。

    Args:
        matrix: 3x3 旋转矩阵。

    Returns:
        四元数 (w, x, y, z), 归一化且 w >= 0。
    """
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return normalize_quaternion(np.array([w, x, y, z]))


def look_at_quaternion(position: FloatArray, target: FloatArray, up: FloatArray) -> FloatArray:
    """计算 MuJoCo 相机的 look-at 四元数, 满足 -Z 轴指向目标。

    MuJoCo 相机约定: 视线沿 -Z 方向。该函数通过标准 look-at 算法
    构建坐标系:
      - forward (Z): 从位置指向目标
      - right (X):    forward × up
      - camera_up (Y): right × forward
    然后取 {-Z} 作为实际视线方向。

    Args:
        position: 相机位置 [x, y, z]。
        target: 目标点位置。
        up: 世界系上方向 (通常为 [0, 0, 1])。

    Returns:
        相机姿态四元数 (w, x, y, z)。
    """
    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    return matrix_to_quaternion(np.column_stack([right, camera_up, -forward]))


def rpy_degrees_to_quaternion(rx: float, ry: float, rz: float) -> FloatArray:
    """将 RPY 欧拉角 (度) 转换为四元数。

    旋转顺序: X → Y → Z (内旋/固定轴)。

    Args:
        rx: 绕 X 轴旋转角度 (度)。
        ry: 绕 Y 轴旋转角度 (度)。
        rz: 绕 Z 轴旋转角度 (度)。

    Returns:
        四元数 (w, x, y, z)。
    """
    x, y, z, w = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_quat()
    return normalize_quaternion(np.array([w, x, y, z]))


def rotation_error(target_wxyz: FloatArray, current_wxyz: FloatArray) -> FloatArray:
    """计算两个姿态之间的旋转误差 (轴角表示)。

    error = target * current^{-1}, 结果是一个旋转向量,
    方向 = 旋转轴, 长度 = 旋转角度 (弧度)。

    这个量在 IK 求解中用作姿态误差项, 与位置误差拼接为 6D 误差向量。

    Args:
        target_wxyz: 目标姿态四元数 (w, x, y, z)。
        current_wxyz: 当前姿态四元数 (w, x, y, z)。

    Returns:
        三维轴角向量, 范数为旋转角度差 (弧度)。
    """
    target = Rotation.from_matrix(quaternion_to_matrix(target_wxyz))
    current = Rotation.from_matrix(quaternion_to_matrix(current_wxyz))
    return (target * current.inv()).as_rotvec()


def pose_delta(
    current_position: FloatArray,
    current_quaternion: FloatArray,
    target_position: FloatArray,
    target_quaternion: FloatArray,
) -> FloatArray:
    """计算 6D delta pose: [位置差, 姿态轴角差]。

    delta_pose = [t_target - t_current, rotation_error(q_target, q_current)]

    这是世界坐标系下的 delta pose, 表示从当前位姿到目标位姿需要施加的
    位置偏移和姿态旋转 (轴角)。配合 frame_delta 可变换到任意坐标系。

    Args:
        current_position: 当前位置 [x, y, z]。
        current_quaternion: 当前姿态四元数 (w, x, y, z)。
        target_position: 目标位置 [x, y, z]。
        target_quaternion: 目标姿态四元数 (w, x, y, z)。

    Returns:
        6D 向量: [dx, dy, dz, rx, ry, rz], 前 3 维为位置差,
        后 3 维为旋转轴角向量。
    """
    return np.concatenate(
        [target_position - current_position, rotation_error(target_quaternion, current_quaternion)]
    )


def frame_delta(delta_world: FloatArray, world_from_frame: FloatArray) -> FloatArray:
    """将世界坐标系下的 SE(3) delta pose 变换到另一个坐标系。

    数学原理:
      给定目标坐标系到世界的旋转矩阵 R (world_from_frame),
      其转置 R^T 把世界系向量旋转到目标坐标系。位置和旋转分量
      独立应用相同的旋转变换:
        delta_{frame} = R^T @ delta_{world}

    Args:
        delta_world: 世界系下的 6D delta pose [dx, dy, dz, rx, ry, rz]。
        world_from_frame: 目标坐标系到世界的 3x3 旋转矩阵。

    Returns:
        目标坐标系下的 6D delta pose。
    """
    rotation = np.asarray(world_from_frame, dtype=np.float64).T
    delta = np.asarray(delta_world, dtype=np.float64)
    return np.concatenate([rotation @ delta[:3], rotation @ delta[3:]])


def yaw_degrees_to_matrix(yaw_deg: float) -> FloatArray:
    """将偏航角 (绕 Z 轴) 转换为 3x3 旋转矩阵。

    常用于机器人底座的朝向表示: 底座在世界系中只有偏航自由度,
    俯仰和翻滚固定。

    Args:
        yaw_deg: 绕 Z 轴旋转角度 (度)。

    Returns:
        3x3 旋转矩阵。
    """
    return Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()


def transform_points(
    position: FloatArray, quaternion: FloatArray, points: FloatArray
) -> FloatArray:
    """用位姿 (位置 + 四元数) 变换一组点 (刚体变换)。

    p' = R @ p + t  的批处理版本, 传入的 points 每行是一个点。

    等效于齐次变换: p' = T @ p_homogeneous 在旋转矩阵转置上的实现。

    Args:
        position: 平移向量 [x, y, z]。
        quaternion: 旋转四元数 (w, x, y, z)。
        points: [N, 3] 的点集。

    Returns:
        [N, 3] 变换后的点集。
    """
    return np.asarray(points) @ quaternion_to_matrix(quaternion).T + np.asarray(position)


def rotation_from_6d_rows(values: np.ndarray) -> np.ndarray:
    """从 6D 旋转表示还原 3x3 SO(3) 旋转矩阵。

    6D 旋转表示 (Zhou et al., CVPR 2019) 使用旋转矩阵的前两行,
    通过 Gram-Schmidt 正交化恢复第三行, 保证输出是有效的 SO(3) 矩阵。

    动机: 直接回归 9 个参数的旋转矩阵或 4 个参数的四元数分别存在
    正交性缺失和符号歧义的问题。6D 表示在神经网络中具有更好的连续性。

    数学步骤 (Gram-Schmidt):
      f1 = normalize(x_row)
      f2 = normalize(y_row - (y_row · f1) * f1)   ← 去除 f1 分量后归一化
      f3 = f1 × f2                                 ← 叉积得到第三行
      R = [f1, f2, f3]^T

    Args:
        values: 6D 向量 [r1x, r1y, r1z, r2x, r2y, r2z], 即旋转矩阵前两行。

    Returns:
        3x3 SO(3) 旋转矩阵。
    """
    first = normalize(np.asarray(values[:3], dtype=np.float64))
    second = np.asarray(values[3:6], dtype=np.float64)
    # 从 second 中移除 first 方向的分量, 保证正交性
    second = normalize(second - np.dot(first, second) * first)
    # 第三行由前两行的叉积得到, 自动满足右手系
    third = normalize(np.cross(first, second))
    return np.stack((first, second, third))


def apply_tcp_action_to_world(current: Pose, action: np.ndarray, max_translation_m: float) -> Pose:
    """将策略预测的 9D 相对动作解码为世界坐标系下的目标位姿。

    这是 build_relative_action_chunk 的逆过程:
      build_relative_action_chunk:  T_current → T_target → T_rel
      apply_tcp_action_to_world:    T_current + action → T_target

    数学原理 (与 build_relative_action_chunk 反向):
      action[:3]   = R_current^T @ (t_target - t_current)
      → t_target   = t_current + R_current @ action[:3]

      action[3:]   = rotation_6d(R_current^T @ R_target)
      → R_target   = R_current @ R_rel
      → q_target   = matrix_to_quaternion(R_current @ R_rel)

    其中 R_rel 由 rotation_from_6d_rows(action[3:]) 恢复得到。

    通常部署时用于:
      1. 策略在 TCP 局部坐标系中预测动作 (好处: 与工件位姿解耦)
      2. 本函数将 9D 动作映射回世界坐标系
      3. 映射结果传给 IK 求解器或位置控制器执行

    Args:
        current: 当前 TCP 位姿。
        action: 9D 动作 [dx, dy, dz, r1x, r1y, r1z, r2x, r2y, r2z]。
        max_translation_m: 平移最大允许值, 超限报错防止危险动作。

    Returns:
        世界坐标系下的目标位姿。

    Raises:
        ValueError: action 不是合法的 9D 有限值向量或超出平移限制。
    """
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (9,) or not np.all(np.isfinite(values)):
        raise ValueError("Input action must be a finite 9D vector")
    translation_norm = float(np.linalg.norm(values[:3]))
    if translation_norm > max_translation_m:
        raise ValueError(
            f"Input action {translation_norm:.6f} exceeds the configured "
            f"TCP increment limit {max_translation_m:.6f}"
        )

    current_rotation = quaternion_to_matrix(current.quaternion_wxyz)

    # 从 6D 旋转表示恢复相对旋转矩阵, 应用到世界系
    relative_rotation = rotation_from_6d_rows(values[3:])

    return Pose(
        current.position + current_rotation @ values[:3],
        matrix_to_quaternion(current_rotation @ relative_rotation),
    )
