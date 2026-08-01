"""π0 系列在留出 episode 上的离线评估。"""

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
from welding_path_vla.policies.pi_family.runtime import PIRuntime
from welding_path_vla.policies.pi_family.spec import PIFamilySpec


@dataclass(frozen=True, slots=True)
class PIEvaluationReport:
    """可与 ACT、SmolVLA 并列保存的 π0 系列离线指标。

    Attributes:
        policy: 项目策略名称。
        checkpoint: 模型 checkpoint 路径。
        episodes: 留出 episode 数。
        episode_indices: 留出 episode 索引。
        task_counts: 各语言任务在留出集中的 episode 数。
        sample_task_counts: 各语言任务实际参与计算的帧数。
        batches: 实际计算的 batch 数。
        samples: 实际计算的样本数。
        loss: flow-matching 平均损失。
        normalized_action_mae: 归一化动作块的平均绝对误差。
        dataset: 完整数据集的 schema 摘要。
    """

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

    def as_dict(self) -> dict[str, object]:
        """返回可直接序列化为 JSON 的报告。"""
        return asdict(self)


def evaluate_checkpoint(
    config: AppConfig,
    checkpoint: str,
    family: PIFamilySpec,
) -> PIEvaluationReport:
    """使用 checkpoint processor 在确定性、任务均衡的留出集上评估。"""
    import torch
    from lerobot.utils.collate import lerobot_collate_fn
    from lerobot.utils.random_utils import set_seed
    from torch.utils.data import DataLoader

    runtime = PIRuntime.from_pretrained(checkpoint, config.policy.device, family)
    policy_config = replace(
        config.policy,
        action_horizon=runtime.policy.config.chunk_size,
        action_steps=runtime.policy.config.n_action_steps,
    )
    episodes = held_out_episode_indices(
        config.training,
        config.policy_evaluation.held_out_episodes,
    )
    meta = metadata(config.training)
    selected = set(episodes)
    task_counts: dict[str, int] = {}
    for raw_episode in meta.episodes:
        episode = cast(dict[str, Any], raw_episode)
        if int(episode["episode_index"]) in selected:
            task = episode["tasks"][0]
            task_counts[task] = task_counts.get(task, 0) + 1

    dataset = make_dataset(policy_config, config.training, episodes)
    sample_limit = config.policy_evaluation.max_batches * config.policy_evaluation.batch_size
    frame_indices = balanced_frame_indices(dataset, sample_limit)
    frame_tasks = np.asarray(dataset.hf_dataset["task_index"])[frame_indices]
    task_names = {int(row.task_index): str(task) for task, row in meta.tasks.iterrows()}
    sample_task_counts = {
        task_names[int(task)]: int(np.sum(frame_tasks == task)) for task in np.unique(frame_tasks)
    }
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
            action_errors.append(
                float(absolute_error.sum() / (valid.sum() * predicted.shape[-1]).clamp_min(1))
            )
            samples += int(processed["action"].shape[0])

    report = validate_dataset(config.training)
    return PIEvaluationReport(
        policy=family.family,
        checkpoint=str(checkpoint),
        episodes=len(episodes),
        episode_indices=episodes,
        task_counts=task_counts,
        sample_task_counts=sample_task_counts,
        batches=len(losses),
        samples=samples,
        loss=float(np.mean(losses)),
        normalized_action_mae=float(np.mean(action_errors)),
        dataset=asdict(report),
    )


__all__ = ["PIEvaluationReport", "evaluate_checkpoint"]
