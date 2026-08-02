import numpy as np
import torch
from scipy.spatial.transform import Rotation

from welding_path_vla.core.geometry import (
    absolute_ee_action_to_pose,
    absolute_ee_actions_from_relative,
    frame_delta,
    matrix_to_quaternion,
    pose_delta,
    quaternion_to_matrix,
    relative_ee_actions_from_absolute,
    rotation_from_6d_rows,
    rotation_to_6d_rows,
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


def test_absolute_ee_action_is_decoded() -> None:
    rotation = rotation_from_6d_rows(np.array([1, 0, 0, 0, 1, 0]))
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-8)
    target = absolute_ee_action_to_pose(np.array([0.101, 0.198, 0.303, 1, 0, 0, 0, 1, 0]))
    np.testing.assert_allclose(target.position, [0.101, 0.198, 0.303], atol=1e-8)
    np.testing.assert_allclose(target.quaternion_wxyz, [1, 0, 0, 0], atol=1e-8)


def test_rotation_conversions_preserve_numpy_and_tensor_batches() -> None:
    """旋转原语应自动识别后端和任意 batch 前缀。"""
    matrices = Rotation.from_euler(
        "xyz", [[0, 0, 0], [180, 0, 0], [10, -20, 30]], degrees=True
    ).as_matrix()
    quaternions_xyzw = Rotation.from_matrix(matrices).as_quat()
    quaternions = np.roll(quaternions_xyzw, 1, axis=-1)

    numpy_matrices = quaternion_to_matrix(quaternions)
    tensor_matrices = quaternion_to_matrix(torch.tensor(quaternions, dtype=torch.float64))
    assert isinstance(numpy_matrices, np.ndarray)
    assert isinstance(tensor_matrices, torch.Tensor)
    np.testing.assert_allclose(numpy_matrices, matrices, atol=1e-8)
    np.testing.assert_allclose(tensor_matrices, matrices, atol=1e-8)
    np.testing.assert_allclose(
        quaternion_to_matrix(matrix_to_quaternion(numpy_matrices)), matrices, atol=1e-8
    )
    np.testing.assert_allclose(
        quaternion_to_matrix(matrix_to_quaternion(tensor_matrices)), matrices, atol=1e-8
    )

    numpy_6d = rotation_to_6d_rows(numpy_matrices)
    tensor_6d = rotation_to_6d_rows(tensor_matrices)
    np.testing.assert_allclose(rotation_from_6d_rows(numpy_6d), matrices, atol=1e-8)
    np.testing.assert_allclose(rotation_from_6d_rows(tensor_6d), matrices, atol=1e-8)


def test_relative_ee_actions_accept_single_chunk_and_batch_shapes() -> None:
    """相对动作接口应自动广播单锚点，也能处理 batch 锚点。"""
    identity_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)
    anchor_position = np.array([0.4, -0.2, 0.3])
    anchor_quaternion = np.array([1, 0, 0, 0], dtype=np.float64)
    offsets = np.array([[0.1, 0, 0], [0.2, 0, 0]])
    target_positions = anchor_position + offsets
    chunk = np.column_stack(
        (target_positions, np.broadcast_to(identity_6d, (len(target_positions), 6)))
    )

    relative_chunk = relative_ee_actions_from_absolute(chunk, anchor_position, anchor_quaternion)
    relative_single = relative_ee_actions_from_absolute(
        chunk[0], anchor_position, anchor_quaternion
    )
    relative_batch = relative_ee_actions_from_absolute(
        np.stack((chunk, chunk)),
        np.stack((anchor_position, anchor_position)),
        np.stack((anchor_quaternion, anchor_quaternion)),
    )

    assert relative_single.shape == (9,)
    assert relative_chunk.shape == (2, 9)
    assert relative_batch.shape == (2, 2, 9)
    np.testing.assert_allclose(relative_chunk[:, :3], [[0.1, 0, 0], [0.2, 0, 0]])
    np.testing.assert_allclose(
        absolute_ee_actions_from_relative(relative_chunk, anchor_position, anchor_quaternion),
        chunk,
        atol=1e-8,
    )
