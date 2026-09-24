"""phase-balanced sampler 的比例、确定性和恢复状态测试。"""

from pathlib import Path

import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.phase_sampling import PhaseBalancedSampler


def make_sampler(tmp_path: Path) -> tuple[PhaseBalancedSampler, np.ndarray]:
    """创建一个包含三个阶段的最小 sampler。"""
    phase = np.array([0] * 6 + [1] * 10 + [2] * 4, dtype=np.uint8)
    index = tmp_path / "phase.npz"
    np.savez_compressed(index, phase=phase)
    sampler = PhaseBalancedSampler(
        [0],
        [len(phase)],
        index,
        target_ratios=(0.3, 0.5, 0.2),
        action_horizon=1,
        seed=7,
    )
    return sampler, phase


def test_phase_sampler_hits_target_ratio_and_is_deterministic(tmp_path: Path) -> None:
    """一个 epoch 的 phase 数量应按配置分配，且相同 seed 可复现。"""
    sampler, phase = make_sampler(tmp_path)
    first = np.asarray(list(sampler))
    second_sampler, _ = make_sampler(tmp_path)
    second = np.asarray(list(second_sampler))

    assert len(first) == 20
    assert np.array_equal(first, second)
    assert np.bincount(phase[first], minlength=3).tolist() == [6, 10, 4]


def test_phase_sampler_resume_reproduces_remaining_indices(tmp_path: Path) -> None:
    """恢复到 epoch 内偏移后，剩余索引必须与未中断顺序一致。"""
    sampler, _ = make_sampler(tmp_path)
    full = list(sampler)
    resumed, _ = make_sampler(tmp_path)
    resumed.load_state_dict({"epoch": 0, "start_index": 5})

    assert list(resumed) == full[5:]


def test_phase_sampling_is_disabled_by_default() -> None:
    """所有模型继承的默认配置不能改变原始采样行为。"""
    assert not AppConfig.load("configs/xvla_a100.yaml").training.phase_sampling.enabled
