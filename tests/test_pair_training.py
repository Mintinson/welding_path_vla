"""PAIR 动作数据与评估的数值类型回归测试。"""

from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from welding_path_vla.policies.traj_vla_qwen.pair import ActionAEConfig, ActionAutoencoder
from welding_path_vla.policies.traj_vla_qwen.pair_training import (
    ActionChunkDataset,
    evaluate_autoencoder,
)


def test_relative_action_chunks_support_fp32_evaluation():
    """经过真实 NumPy 相对位姿转换后，旋转误差评估不得混用 double/float。"""
    torch.manual_seed(7)
    actions = np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0], (3, 1)).astype(np.float32)
    actions[:, 0] = [0.001, 0.002, 0.003]
    states = np.zeros((3, 13), dtype=np.float32)
    states[:, 9] = 1  # TCP 四元数使用 wxyz，单位姿态的 w 为 1。
    dataset = ActionChunkDataset(
        SimpleNamespace(
            hf_dataset={
                "action": actions,
                "observation.state": states,
                "episode_index": np.zeros(3, dtype=np.int64),
            }
        ),
        horizon=3,
        stride=1,
    )
    model = ActionAutoencoder(
        ActionAEConfig(
            horizon=3,
            hidden_dim=16,
            num_heads=2,
            encoder_layers=1,
            decoder_layers=1,
        )
    )
    metrics = evaluate_autoencoder(
        SimpleNamespace(gather_for_metrics=lambda values: values),
        model,
        DataLoader(dataset, batch_size=3),
        torch.from_numpy(actions.mean(0)),
        torch.ones(9),
        limit=3,
    )
    assert dataset[0]["action"].dtype == torch.float32
    assert all(np.isfinite(value) for value in metrics.values())
    assert model.training
