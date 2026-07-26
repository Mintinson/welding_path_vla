import numpy as np
import pytest

from welding_path_vla.evaluation.seam_geometry import (
    interpolate_quaternions,
    interpolate_speed,
    project_to_seam,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture
def simple_line():
    """沿 X 轴的简单直线焊缝"""
    return np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)


@pytest.fixture
def quats_id():
    """单位四元数序列 (w, x, y, z)"""
    return np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=float,
    )


# ------------------------------------------------------------------
# 1. Quaternion Interpolation (7 tests)
# ------------------------------------------------------------------
def test_nlerp_unit_norm(quats_id):
    """插值结果必须保持单位长度"""
    indices = np.array([0])
    fractions = np.array([0.3])
    result = interpolate_quaternions(quats_id, indices, fractions)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_nlerp_endpoints(quats_id):
    """t=0 和 t=1 时必须精确等于端点"""
    indices = np.array([0, 1])
    fractions = np.array([0.0, 1.0])
    result = interpolate_quaternions(quats_id, indices, fractions)
    assert np.allclose(result[0], quats_id[0])
    assert np.allclose(result[1], quats_id[1])


def test_nlerp_shortest_path():
    """必须走最短路径（自动取反）"""
    q0 = np.array([[1, 0, 0, 0]])
    q1 = np.array([[-1, 0, 0, 0]])  # 与 q0 代表同一旋转
    quats = np.vstack([q0, q1])
    result = interpolate_quaternions(quats, np.array([0]), np.array([0.5]))
    # 插值结果应该是 [1,0,0,0]，而不是 [0,0,0,0]
    assert np.allclose(result[0], [1, 0, 0, 0])


def test_nlerp_double_cover_invariance():
    """q 和 -q 插值结果应一致（模符号）"""
    q_pos = np.array([[1, 0, 0, 0]])
    q_neg = np.array([[-1, 0, 0, 0]])
    res1 = interpolate_quaternions(np.vstack([q_pos, q_pos]), np.array([0]), np.array([0.4]))
    res2 = interpolate_quaternions(np.vstack([q_pos, q_neg]), np.array([0]), np.array([0.4]))
    assert np.allclose(res1, res2) or np.allclose(res1, -res2)


def test_nlerp_batch_shape():
    """批量插值形状正确"""
    quats = np.random.randn(10, 4)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    indices = np.array([0, 1, 2, 3])
    fractions = np.array([0.1, 0.5, 0.9, 0.2])
    result = interpolate_quaternions(quats, indices, fractions)
    assert result.shape == (4, 4)


def test_nlerp_non_normalized_input():
    """输入非归一化时应自动归一化"""
    quats = np.array([[2, 0, 0, 0], [0, 2, 0, 0]], dtype=float)
    result = interpolate_quaternions(quats, np.array([0]), np.array([0.5]))
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_nlerp_identity_case():
    """相同四元数插值应保持不变"""
    q = np.array([[0.707, 0.707, 0, 0]])
    q /= np.linalg.norm(q)  # 确保单位长度
    quats = np.vstack([q, q])
    result = interpolate_quaternions(quats, np.array([0]), np.array([0.5]))
    assert np.allclose(result, q)


# ------------------------------------------------------------------
# 2. Speed Interpolation (4 tests)
# ------------------------------------------------------------------
def test_speed_scalar():
    """标量速度应广播到所有采样点"""
    arc = np.array([0.1, 0.5, 0.9])
    seam = np.array([[0, 0, 0], [1, 0, 0]])
    result = interpolate_speed(0.5, arc, seam)
    assert np.all(result == 0.5)


def test_speed_array_linear():
    """数组速度应在端点间线性插值"""
    seam = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    speeds = np.array([0.0, 1.0, 0.0])
    arc = np.array([0.5])  # 位于第一段中点
    result = interpolate_speed(speeds, arc, seam)
    assert np.isclose(result[0], 0.5)


def test_speed_clipping():
    """超出弧长范围应截断到端点值"""
    seam = np.array([[0, 0, 0], [1, 0, 0]])
    speeds = np.array([1.0, 3.0])
    arc_low = np.array([-1.0])
    arc_high = np.array([5.0])
    assert interpolate_speed(speeds, arc_low, seam)[0] == 1.0
    assert interpolate_speed(speeds, arc_high, seam)[0] == 3.0


def test_speed_shape():
    """输出形状应与 arc_lengths 一致"""
    arc = np.linspace(0, 1, 10)
    seam = np.array([[0, 0, 0], [1, 0, 0]])
    result = interpolate_speed(2.0, arc, seam)
    assert result.shape == (10,)


# ------------------------------------------------------------------
# 3. Seam Projection (7 tests)
# ------------------------------------------------------------------
def test_projection_on_vertices(simple_line):
    """投影到顶点应精确匹配"""
    pts = np.array([[0, 0, 0], [2, 0, 0]])
    proj = project_to_seam(pts, simple_line)
    assert np.allclose(proj.points, pts)


def test_projection_midpoint(simple_line):
    """线段中点投影应精确"""
    pts = np.array([[0.5, 0, 0]])
    proj = project_to_seam(pts, simple_line)
    assert np.allclose(proj.points, pts)
    assert np.isclose(proj.segment_fractions[0], 0.5)


def test_arc_length_accumulation(simple_line):
    """弧长应从 0 单调递增到总长度"""
    pts = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]])
    proj = project_to_seam(pts, simple_line)
    assert np.allclose(proj.arc_lengths, [0.0, 1.0, 3.0])


def test_tangent_direction(simple_line):
    """切向量应指向线段方向"""
    pts = np.array([[0.5, 0, 0]])
    proj = project_to_seam(pts, simple_line)
    assert np.allclose(proj.tangents[0], [1, 0, 0])


def test_zero_length_segment():
    """零长度线段不应导致 NaN"""
    seam = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    pts = np.array([[0, 0, 0]])
    proj = project_to_seam(pts, seam)
    assert not np.any(np.isnan(proj.tangents))
    assert not np.any(np.isnan(proj.points))


def test_projection_shape(simple_line):
    """输出数组形状一致性"""
    pts = np.random.randn(5, 3)
    proj = project_to_seam(pts, simple_line)
    assert proj.points.shape == (5, 3)
    assert proj.tangents.shape == (5, 3)
    assert proj.arc_lengths.shape == (5,)
    assert proj.segment_indices.shape == (5,)


def test_total_length(simple_line):
    """总长度应等于折线总长"""
    proj = project_to_seam(np.array([[0, 0, 0]]), simple_line)
    assert proj.total_length == 3.0


# ------------------------------------------------------------------
# 4. Integration (4 tests)
# ------------------------------------------------------------------
def test_pipeline_consistency(simple_line, quats_id):
    """投影 + 插值 pipeline 不应崩溃"""
    tcp = np.array([[0.5, 0, 0], [1.5, 0, 0]])
    proj = project_to_seam(tcp, simple_line)
    speeds = interpolate_speed(0.5, proj.arc_lengths, simple_line)
    quats = interpolate_quaternions(quats_id, proj.segment_indices, proj.segment_fractions)
    assert speeds.shape == (2,)
    assert quats.shape == (2, 4)


def test_arc_length_monotonicity(simple_line):
    """沿路径移动时弧长应单调不减"""
    tcp = np.array([[0, 0, 0], [0.5, 0, 0], [2.5, 0, 0]])
    proj = project_to_seam(tcp, simple_line)
    assert np.all(np.diff(proj.arc_lengths) >= -1e-9)


def test_deterministic_projection(simple_line):
    """相同输入应产生完全相同的结果"""
    tcp = np.array([[0.3, 0, 0]])
    p1 = project_to_seam(tcp, simple_line)
    p2 = project_to_seam(tcp, simple_line)
    assert np.allclose(p1.points, p2.points)
    assert np.allclose(p1.arc_lengths, p2.arc_lengths)


def test_empty_positions(simple_line):
    """空输入应返回空数组而非崩溃"""
    proj = project_to_seam(np.empty((0, 3)), simple_line)
    assert proj.points.shape == (0, 3)
    assert proj.arc_lengths.shape == (0,)
