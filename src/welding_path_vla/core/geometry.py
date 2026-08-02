"""支持 NumPy 与 PyTorch 的批量几何运算和轨迹工具。

本模块提供的几何原语涵盖:
  - 四元数与旋转矩阵的双向转换 (wxyz 格式优先)
  - 位姿增量 (delta pose) 计算与坐标系变换
  - 6D 旋转表示 (rotation_6d) 与 SO(3) 的相互转换
  - 逆运动学中使用的旋转误差 (轴角) 计算
  - 部署时将 processor 恢复后的 9D 世界系目标转换为位姿

四元数约定: 全文使用 (w, x, y, z) 顺序, 与 MuJoCo 和 scipy 的
(x, y, z, w) 不同, 因此在接口边界处需要 roll 转换。
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from welding_path_vla.core.domain import Pose

type FloatArray = NDArray[np.float64]


def concatenate[ArrayType: (np.ndarray, torch.Tensor)](
    values: tuple[ArrayType, ...], axis: int = -1
) -> ArrayType:
    """使用输入对应的 NumPy 或 PyTorch 后端拼接数组。

    Args:
        values: 后端相同的数组或 Tensor。
        axis: 拼接维度。

    Returns:
        拼接结果，后端与输入一致。
    """
    if isinstance(values[0], torch.Tensor):
        tensors = cast(tuple[torch.Tensor, ...], values)
        return cast(ArrayType, torch.cat(tensors, dim=axis))
    return cast(ArrayType, np.concatenate(values, axis=axis))


def stack[ArrayType: (np.ndarray, torch.Tensor)](
    values: tuple[ArrayType, ...], axis: int = -1
) -> ArrayType:
    """使用输入对应的 NumPy 或 PyTorch 后端堆叠数组。

    Args:
        values: 后端相同的数组或 Tensor。
        axis: 新维度的位置。

    Returns:
        堆叠结果，后端与输入一致。
    """
    if isinstance(values[0], torch.Tensor):
        tensors = cast(tuple[torch.Tensor, ...], values)
        return cast(ArrayType, torch.stack(tensors, dim=axis))
    return cast(ArrayType, np.stack(values, axis=axis))


def normalize[ArrayType: (np.ndarray, torch.Tensor)](
    vector: ArrayType, eps: float = 1e-8
) -> ArrayType:
    """沿最后一维归一化单个或批量向量。

    Args:
        vector: NumPy 数组或 Tensor，最后一维为向量维度。
        eps: 判定退化向量的阈值。

    Returns:
        与输入后端、形状相同的单位向量。

    Raises:
        ValueError: 任意向量长度小于 `eps`。
    """
    if isinstance(vector, torch.Tensor):
        values = vector if vector.is_floating_point() else vector.to(torch.float32)
        norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
        degenerate = bool(torch.any(norm < eps))
    else:
        values = np.asarray(vector)
        norm = np.linalg.norm(values, axis=-1, keepdims=True)
        degenerate = bool(np.any(norm < eps))
    if degenerate:
        raise ValueError("cannot normalize a degenerate vector")
    return cast(ArrayType, values / norm)


def normalize_quaternion[ArrayType: (np.ndarray, torch.Tensor)](
    quaternion_wxyz: ArrayType,
) -> ArrayType:
    """归一化四元数并确保 w >= 0, 消除四元数的符号歧义。

    四元数 q 和 -q 代表相同的旋转, 但符号变化会导致插值和误差计算
    不连续。约定 w >= 0 确保表示唯一。

    Args:
        quaternion_wxyz: `[..., 4]` 四元数，最后一维采用 `(w, x, y, z)`。

    Returns:
        与输入后端、dtype、device 和 batch 前缀一致的四元数，且 `w >= 0`。
    """
    quaternion = normalize(quaternion_wxyz)
    # 取 w >= 0 的等价表示, 消除符号歧义
    if isinstance(quaternion, torch.Tensor):
        sign = torch.where(quaternion[..., :1] < 0, -1, 1)
    else:
        sign = np.where(quaternion[..., :1] < 0, -1, 1)
    return quaternion * sign


def quaternion_to_matrix[ArrayType: (np.ndarray, torch.Tensor)](
    quaternion_wxyz: ArrayType,
) -> ArrayType:
    """将单个或批量 `(w, x, y, z)` 四元数转换为旋转矩阵。

    Args:
        quaternion_wxyz: `[..., 4]` NumPy 数组或 Tensor。

    Returns:
        `[..., 3, 3]` 旋转矩阵，保持输入后端、dtype 和 device。
    """
    quaternion = normalize_quaternion(quaternion_wxyz)
    if isinstance(quaternion, torch.Tensor):
        w, x, y, z = quaternion.unbind(-1)
    else:
        w, x, y, z = np.moveaxis(quaternion, -1, 0)
    matrix = stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    )
    return cast(ArrayType, matrix.reshape(*quaternion.shape[:-1], 3, 3))


def matrix_to_quaternion[ArrayType: (np.ndarray, torch.Tensor)](matrix: ArrayType) -> ArrayType:
    """将单个或批量旋转矩阵转换为 `(w, x, y, z)` 四元数。

    scipy 的 as_quat() 返回 (x, y, z, w), 需要转回本项目的 (w, x, y, z) 约定。

    Args:
        matrix: `[..., 3, 3]` NumPy 数组或 Tensor。

    Returns:
        `[..., 4]` 四元数，保持输入后端、dtype 和 device，且 `w >= 0`。
    """
    if not isinstance(matrix, torch.Tensor):
        quaternion_xyzw = Rotation.from_matrix(matrix).as_quat()
        quaternion = normalize_quaternion(np.roll(quaternion_xyzw, 1, axis=-1))
        if np.issubdtype(matrix.dtype, np.floating):
            quaternion = quaternion.astype(matrix.dtype, copy=False)
        return quaternion
    values = matrix if matrix.is_floating_point() else matrix.to(torch.float32)
    m00, m11, m22 = values[..., 0, 0], values[..., 1, 1], values[..., 2, 2]
    w = 0.5 * torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=0))
    x = 0.5 * torch.copysign(
        torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=0)),
        values[..., 2, 1] - values[..., 1, 2],
    )
    y = 0.5 * torch.copysign(
        torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=0)),
        values[..., 0, 2] - values[..., 2, 0],
    )
    z = 0.5 * torch.copysign(
        torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=0)),
        values[..., 1, 0] - values[..., 0, 1],
    )
    return normalize_quaternion(torch.stack((w, x, y, z), dim=-1))


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


def rotation_from_6d_rows[ArrayType: (np.ndarray, torch.Tensor)](
    values: ArrayType,
) -> ArrayType:
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
        values: `[..., 6]` 向量，最后一维是旋转矩阵前两行。

    Returns:
        `[..., 3, 3]` SO(3) 旋转矩阵，保持输入后端、dtype 和 device。
    """
    first = normalize(values[..., :3])
    second = values[..., 3:6]
    if isinstance(first, torch.Tensor):
        second_tensor = cast(torch.Tensor, second)
        projection = torch.sum(first * second_tensor, dim=-1, keepdim=True)
        second = normalize(second_tensor - projection * first)
        third = normalize(torch.linalg.cross(first, second, dim=-1))
    else:
        second_array = cast(np.ndarray, second)
        projection = np.sum(first * second_array, axis=-1, keepdims=True)
        second = normalize(second_array - projection * first)
        third = normalize(np.cross(first, second))
    return cast(ArrayType, stack((first, second, third), axis=-2))


def rotation_to_6d_rows[ArrayType: (np.ndarray, torch.Tensor)](matrix: ArrayType) -> ArrayType:
    """把 `[..., 3, 3]` 旋转矩阵编码为 `[..., 6]` rotation-6D。"""
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def expand_anchor[ArrayType: (np.ndarray, torch.Tensor)](
    anchor: ArrayType, action: ArrayType, trailing_dimensions: int
) -> ArrayType:
    """为动作比锚点多出的时间维插入广播轴。

    例如动作 `[B, H, 9]` 与锚点 `[B, 3]` 会把锚点变为 `[B, 1, 3]`；
    动作 `[H, 9]` 与单个锚点 `[3]` 会把锚点变为 `[1, 3]`。

    Args:
        anchor: 位置或旋转矩阵锚点。
        action: 单步动作、动作块或批量动作块。
        trailing_dimensions: 锚点自身占用的维数；位置为 1，旋转矩阵为 2。

    Returns:
        可沿动作时间维广播的锚点视图。
    """
    action_prefix = action.ndim - 1
    anchor_prefix = anchor.ndim - trailing_dimensions
    extra_dimensions = action_prefix - anchor_prefix
    shape = (*anchor.shape[:-trailing_dimensions], *((1,) * extra_dimensions))
    return anchor.reshape(*shape, *anchor.shape[-trailing_dimensions:])


def relative_ee_actions_from_absolute[ArrayType: (np.ndarray, torch.Tensor)](
    absolute_actions: ArrayType,
    anchor_positions: ArrayType,
    anchor_quaternions_wxyz: ArrayType,
) -> ArrayType:
    """将 absolute EE actions 转到预测时刻的 TCP 坐标系。

    支持 `[9]`、`[H, 9]`、`[B, H, 9]` 等动作形状。锚点可以是
    `[3]/[4]` 或带有相同 batch 前缀的数组，并自动沿动作时间维广播。

    Args:
        absolute_actions: 世界系 9D 末端目标，最后一维为位置和 rotation-6D。
        anchor_positions: 预测时刻的世界系 TCP 位置。
        anchor_quaternions_wxyz: 预测时刻的世界系 TCP 姿态。

    Returns:
        形状和后端不变、以对应 TCP 为共同锚点的 relative actions。
    """
    anchor_rotations = quaternion_to_matrix(anchor_quaternions_wxyz)
    target_rotations = rotation_from_6d_rows(absolute_actions[..., 3:])
    positions = expand_anchor(anchor_positions, absolute_actions, 1)
    rotations = expand_anchor(anchor_rotations, absolute_actions, 2)
    position_delta = absolute_actions[..., :3] - positions
    relative_positions = (rotations.swapaxes(-1, -2) @ position_delta[..., None])[..., 0]
    relative_rotations = rotations.swapaxes(-1, -2) @ target_rotations
    return concatenate(
        (relative_positions, rotation_to_6d_rows(relative_rotations)),
        axis=-1,
    )


def absolute_ee_actions_from_relative[ArrayType: (np.ndarray, torch.Tensor)](
    relative_actions: ArrayType,
    anchor_positions: ArrayType,
    anchor_quaternions_wxyz: ArrayType,
) -> ArrayType:
    """将共享 TCP 锚点的 relative EE actions 恢复到世界系。

    Args:
        relative_actions: TCP 局部系 9D 动作，最后一维为位置和 rotation-6D。
        anchor_positions: 编码动作时使用的世界系 TCP 位置。
        anchor_quaternions_wxyz: 编码动作时使用的世界系 TCP 姿态。

    Returns:
        形状和后端不变的世界系 absolute EE targets。
    """
    anchor_rotations = quaternion_to_matrix(anchor_quaternions_wxyz)
    relative_rotations = rotation_from_6d_rows(relative_actions[..., 3:])
    positions = expand_anchor(anchor_positions, relative_actions, 1)
    rotations = expand_anchor(anchor_rotations, relative_actions, 2)
    absolute_positions = positions + (rotations @ relative_actions[..., :3, None])[..., 0]
    absolute_rotations = rotations @ relative_rotations
    return concatenate(
        (absolute_positions, rotation_to_6d_rows(absolute_rotations)),
        axis=-1,
    )


def absolute_ee_action_to_pose(action: np.ndarray) -> Pose:
    """将 postprocessor 输出的 9D 世界系绝对 EE 目标转换为位姿。

    relative action 的 SE(3) 解码由 processor 对完整 chunk 一次完成；这里不再读取
    每一步变化的当前 TCP，避免同一预测块被错误地重复换锚点。

    Args:
        action: 9D 世界系目标 `[x, y, z, rotation_6d_rows]`。

    Returns:
        世界坐标系下的目标位姿。

    Raises:
        ValueError: action 不是合法的 9D 有限值向量。
    """
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (9,) or not np.all(np.isfinite(values)):
        raise ValueError("Input action must be a finite 9D vector")
    return Pose(values[:3], matrix_to_quaternion(rotation_from_6d_rows(values[3:])))
