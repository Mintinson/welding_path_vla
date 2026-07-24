import numpy as np
from scipy.spatial.transform import Rotation

from welding_path_vla.core.geometry import (
    frame_delta,
    pose_delta,
    quaternion_to_matrix,
    rpy_degrees_to_quaternion,
)


def test_huayan_tcp_rpy_bootstrap_conversion() -> None:
    quaternion = rpy_degrees_to_quaternion(172.091, -46.203, 173.622)
    expected = np.array([0.387292070, -0.078064102, -0.914695023, -0.085110886])
    np.testing.assert_allclose(quaternion, expected, atol=1e-8)
    np.testing.assert_allclose(
        quaternion_to_matrix(quaternion).T @ quaternion_to_matrix(quaternion), np.eye(3), atol=1e-12
    )


def test_pose_delta_uses_rotation_vector() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    target = rpy_degrees_to_quaternion(0, 0, 10)
    delta = pose_delta(np.zeros(3), identity, np.array([0.1, 0.2, 0.3]), target)
    np.testing.assert_allclose(delta[:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(delta[3:], [0, 0, np.radians(10)], atol=1e-12)


def test_world_delta_is_expressed_in_rotated_robot_base() -> None:
    world_from_base = Rotation.from_euler("z", -90, degrees=True).as_matrix()
    world_delta = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        frame_delta(world_delta, world_from_base), [0, 1, 0, 0, 0, 1], atol=1e-12
    )
