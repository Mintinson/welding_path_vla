"""ACT 在留出 LeRobot episode 上的离线 loss 与动作误差评估。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.act.runtime import ACTRuntime
from welding_path_vla.policies.data import (
    balanced_frame_indices,
    held_out_episode_indices,
    make_dataset,
    scale_uint8_images,
    validate_dataset,
)


@dataclass(frozen=True, slots=True)
class ACTEvaluationReport:
    """可用于不同 checkpoint 对比的离线指标。"""

    checkpoint: str
    episodes: int
    batches: int
    samples: int
    loss: float
    l1_loss: float
    kld_loss: float | None
    normalized_action_mae: float
    dataset: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_checkpoint(config: AppConfig, checkpoint: str) -> ACTEvaluationReport:
    """用训练时保存的 LeRobot processor 评估留出 episode。"""
    import torch
    from torch.utils.data import DataLoader

    runtime = ACTRuntime.from_pretrained(checkpoint, config.policy.device)
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
    sampled_dataset = torch.utils.data.Subset(dataset, frame_indices)
    loader = DataLoader(
        sampled_dataset,
        batch_size=config.policy_evaluation.batch_size,
        num_workers=config.policy_evaluation.num_workers,
        shuffle=False,
        pin_memory=config.policy.device.startswith("cuda"),
        persistent_workers=config.policy_evaluation.num_workers > 0,
    )
    totals: dict[str, list[float]] = {
        "loss": [],
        "l1_loss": [],
        "kld_loss": [],
        "action_mae": [],
    }
    samples = 0
    runtime.policy.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= config.policy_evaluation.max_batches:
                break
            processed = runtime.preprocessor(scale_uint8_images(batch))
            loss, values = runtime.policy.forward(processed)
            predicted = runtime.policy.predict_action_chunk(processed)
            valid = ~processed["action_is_pad"].unsqueeze(-1)
            absolute_error = (predicted - processed["action"]).abs() * valid
            totals["loss"].append(float(loss.detach()))
            totals["l1_loss"].append(float(values["l1_loss"]))
            if "kld_loss" in values:
                totals["kld_loss"].append(float(values["kld_loss"]))
            totals["action_mae"].append(
                float(absolute_error.sum() / (valid.sum() * predicted.shape[-1]).clamp_min(1))
            )
            samples += int(processed["action"].shape[0])
    report = validate_dataset(config.training)
    return ACTEvaluationReport(
        checkpoint=str(checkpoint),
        episodes=len(episodes),
        batches=len(totals["loss"]),
        samples=samples,
        loss=float(np.mean(totals["loss"])),
        l1_loss=float(np.mean(totals["l1_loss"])),
        kld_loss=float(np.mean(totals["kld_loss"])) if totals["kld_loss"] else None,
        normalized_action_mae=float(np.mean(totals["action_mae"])),
        dataset=asdict(report),
    )


__all__ = ["ACTEvaluationReport", "evaluate_checkpoint"]
