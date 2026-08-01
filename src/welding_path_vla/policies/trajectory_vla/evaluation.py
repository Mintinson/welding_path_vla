"""Trajectory-VLA 在留出数据上的 flow loss 与动作误差评估。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.data import (
    balanced_frame_indices,
    held_out_episode_indices,
    make_dataset,
    scale_uint8_images,
    validate_dataset,
)
from welding_path_vla.policies.trajectory_vla.runtime import TrajectoryVLARuntime


@dataclass(frozen=True, slots=True)
class TrajectoryVLAEvaluationReport:
    """可与其他策略并列保存的离线指标。"""

    checkpoint: str
    episodes: int
    episode_indices: list[int]
    batches: int
    samples: int
    loss: float
    normalized_action_mae: float
    dataset: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        """返回可直接写入 JSON 的字典。"""
        return asdict(self)


def evaluate_checkpoint(
    config: AppConfig,
    checkpoint: str,
) -> TrajectoryVLAEvaluationReport:
    """在确定性留出 episode 上评估 checkpoint。"""
    import torch
    from lerobot.utils.collate import lerobot_collate_fn
    from lerobot.utils.random_utils import set_seed
    from torch.utils.data import DataLoader

    runtime = TrajectoryVLARuntime.from_pretrained(checkpoint, config.policy.device)
    policy_config = replace(
        config.policy,
        action_horizon=runtime.policy.config.chunk_size,
        action_steps=runtime.policy.config.n_action_steps,
    )
    episodes = held_out_episode_indices(
        config.training,
        config.policy_evaluation.held_out_episodes,
    )
    dataset = make_dataset(policy_config, config.training, episodes)
    frame_indices = balanced_frame_indices(
        dataset,
        config.policy_evaluation.max_batches * config.policy_evaluation.batch_size,
    )
    loader = DataLoader(
        torch.utils.data.Subset(dataset, frame_indices),
        batch_size=config.policy_evaluation.batch_size,
        num_workers=config.policy_evaluation.num_workers,
        shuffle=False,
        pin_memory=config.policy.device.startswith("cuda"),
        persistent_workers=config.policy_evaluation.num_workers > 0,
        collate_fn=lerobot_collate_fn,
    )
    losses: list[float] = []
    action_errors: list[float] = []
    samples = 0
    set_seed(config.training.seed)
    runtime.policy.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= config.policy_evaluation.max_batches:
                break
            processed = runtime.preprocessor(scale_uint8_images(batch))
            loss, _ = runtime.policy.forward(processed)
            predicted = runtime.policy.predict_action_chunk(processed)
            valid = ~processed["action_is_pad"].unsqueeze(-1)
            absolute_error = (predicted - processed["action"]).abs() * valid
            losses.append(float(loss))
            denominator = (valid.sum() * predicted.shape[-1]).clamp_min(1)
            action_errors.append(float(absolute_error.sum() / denominator))
            samples += int(processed["action"].shape[0])
    return TrajectoryVLAEvaluationReport(
        checkpoint=str(checkpoint),
        episodes=len(episodes),
        episode_indices=episodes,
        batches=len(losses),
        samples=samples,
        loss=float(np.mean(losses)),
        normalized_action_mae=float(np.mean(action_errors)),
        dataset=asdict(validate_dataset(config.training)),
    )


__all__ = ["TrajectoryVLAEvaluationReport", "evaluate_checkpoint"]
