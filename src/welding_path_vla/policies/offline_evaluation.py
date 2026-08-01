"""所有动作块策略共享的确定性离线评估。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, cast

import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.data import (
    balanced_frame_indices,
    held_out_episode_indices,
    make_dataset,
    metadata,
    scale_uint8_images,
    validate_dataset,
)
from welding_path_vla.policies.runtime import LeRobotRuntime
from welding_path_vla.policies.spec import LeRobotPolicySpec


@dataclass(frozen=True, slots=True)
class PolicyEvaluationReport:
    """跨策略保持同一 schema 的离线指标。"""

    policy: str
    checkpoint: str
    episodes: int
    episode_indices: list[int]
    task_counts: dict[str, int]
    sample_task_counts: dict[str, int]
    batches: int
    samples: int
    loss: float
    normalized_action_mae: float
    dataset: dict[str, object]
    l1_loss: float | None = None
    kld_loss: float | None = None

    def as_dict(self) -> dict[str, object]:
        """返回可直接写入 JSON 的指标。"""
        return asdict(self)


def task_distribution(
    config: AppConfig,
    episodes: list[int],
    dataset: Any,
    frame_indices: list[int],
) -> tuple[dict[str, int], dict[str, int]]:
    """统计 episode 和实际采样帧在不同语言任务间的分布。"""
    meta = metadata(config.training)
    selected = set(episodes)
    episode_counts: dict[str, int] = {}
    for raw_episode in meta.episodes:
        episode = cast(dict[str, Any], raw_episode)
        if int(episode["episode_index"]) in selected:
            task = episode["tasks"][0]
            episode_counts[task] = episode_counts.get(task, 0) + 1
    frame_tasks = np.asarray(dataset.hf_dataset["task_index"])[frame_indices]
    names = {int(row.task_index): str(task) for task, row in meta.tasks.iterrows()}
    frame_counts = {
        names[int(task)]: int(np.sum(frame_tasks == task)) for task in np.unique(frame_tasks)
    }
    return episode_counts, frame_counts


def evaluate_checkpoint(
    config: AppConfig,
    checkpoint: str,
    spec: LeRobotPolicySpec,
) -> PolicyEvaluationReport:
    """使用 checkpoint processor 在任务均衡的留出帧上评估。"""
    import torch
    from lerobot.utils.collate import lerobot_collate_fn
    from lerobot.utils.random_utils import set_seed
    from torch.utils.data import DataLoader

    runtime = LeRobotRuntime.from_pretrained(checkpoint, config.policy.device, spec)
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
    sample_limit = config.policy_evaluation.max_batches * config.policy_evaluation.batch_size
    frame_indices = balanced_frame_indices(dataset, sample_limit)
    episode_counts, frame_counts = task_distribution(config, episodes, dataset, frame_indices)
    loader = DataLoader(
        torch.utils.data.Subset(dataset, frame_indices),
        batch_size=config.policy_evaluation.batch_size,
        num_workers=config.policy_evaluation.num_workers,
        shuffle=False,
        pin_memory=config.policy.device.startswith("cuda"),
        persistent_workers=config.policy_evaluation.num_workers > 0,
        collate_fn=lerobot_collate_fn,
    )
    totals = {name: [] for name in ("loss", "action_mae", *spec.evaluation_values)}
    samples = 0
    set_seed(config.training.seed)
    runtime.policy.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= config.policy_evaluation.max_batches:
                break
            processed = runtime.preprocessor(scale_uint8_images(batch))
            loss, values = runtime.policy.forward(processed)
            predicted = runtime.policy.predict_action_chunk(processed)
            valid = ~processed["action_is_pad"].unsqueeze(-1)
            error = (predicted - processed["action"]).abs() * valid
            totals["loss"].append(float(loss.detach()))
            totals["action_mae"].append(
                float(error.sum() / (valid.sum() * predicted.shape[-1]).clamp_min(1))
            )
            for name in spec.evaluation_values:
                if name in values:
                    totals[name].append(float(values[name]))
            samples += int(processed["action"].shape[0])

    def mean(name: str) -> float | None:
        """计算可选指标均值。"""
        return float(np.mean(totals[name])) if totals[name] else None

    return PolicyEvaluationReport(
        policy=spec.family,
        checkpoint=str(checkpoint),
        episodes=len(episodes),
        episode_indices=episodes,
        task_counts=episode_counts,
        sample_task_counts=frame_counts,
        batches=len(totals["loss"]),
        samples=samples,
        loss=cast(float, mean("loss")),
        normalized_action_mae=cast(float, mean("action_mae")),
        dataset=asdict(validate_dataset(config.training)),
        l1_loss=mean("l1_loss") if "l1_loss" in totals else None,
        kld_loss=mean("kld_loss") if "kld_loss" in totals else None,
    )


__all__ = ["PolicyEvaluationReport", "evaluate_checkpoint", "task_distribution"]
