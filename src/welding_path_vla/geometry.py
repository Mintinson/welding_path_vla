from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

type FloatArray = NDArray[np.float64]


def normalize_quaternion(quaternion_wxyz: FloatArray) -> FloatArray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion[0] < 0:
        quaternion = -quaternion
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_matrix(quaternion_wxyz: FloatArray) -> FloatArray:
    w, x, y, z = normalize_quaternion(quaternion_wxyz)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def matrix_to_quaternion(matrix: FloatArray) -> FloatArray:
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return normalize_quaternion(np.array([w, x, y, z]))


def look_at_quaternion(position: FloatArray, target: FloatArray, up: FloatArray) -> FloatArray:
    """Return a MuJoCo camera quaternion whose -Z axis looks at target."""
    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    return matrix_to_quaternion(np.column_stack([right, camera_up, -forward]))


def rpy_degrees_to_quaternion(rx: float, ry: float, rz: float) -> FloatArray:
    x, y, z, w = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_quat()
    return normalize_quaternion(np.array([w, x, y, z]))


def rotation_error(target_wxyz: FloatArray, current_wxyz: FloatArray) -> FloatArray:
    target = Rotation.from_matrix(quaternion_to_matrix(target_wxyz))
    current = Rotation.from_matrix(quaternion_to_matrix(current_wxyz))
    return (target * current.inv()).as_rotvec()


def pose_delta(
    current_position: FloatArray,
    current_quaternion: FloatArray,
    target_position: FloatArray,
    target_quaternion: FloatArray,
) -> FloatArray:
    return np.concatenate(
        [target_position - current_position, rotation_error(target_quaternion, current_quaternion)]
    )


def frame_delta(delta_world: FloatArray, world_from_frame: FloatArray) -> FloatArray:
    """Express a world-frame SE(3) delta in another frame."""
    rotation = np.asarray(world_from_frame, dtype=np.float64).T
    delta = np.asarray(delta_world, dtype=np.float64)
    return np.concatenate([rotation @ delta[:3], rotation @ delta[3:]])


def yaw_degrees_to_matrix(yaw_deg: float) -> FloatArray:
    return Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()


def transform_points(
    position: FloatArray, quaternion: FloatArray, points: FloatArray
) -> FloatArray:
    return np.asarray(points) @ quaternion_to_matrix(quaternion).T + np.asarray(position)
