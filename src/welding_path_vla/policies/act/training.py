"""把项目配置映射到 LeRobot 官方 ACT 训练流水线。"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from welding_path_vla.core.config import PolicyConfig, TrainingConfig
from welding_path_vla.policies.data import validate_dataset
from welding_path_vla.policies.process import lerobot_training_log


def split_episodes(total: int, fraction: float) -> tuple[list[int], list[int]]:
    """按 episode 尾部确定性划分训练集和测试集。"""
    eval_count = math.ceil(total * fraction) if fraction else 0
    boundary = total - eval_count
    return list(range(boundary)), list(range(boundary, total))


def act_config(policy: PolicyConfig) -> Any:
    """构造 LeRobot ACTConfig，并保留数据驱动的 feature 推断。"""
    from lerobot.policies.act.configuration_act import ACTConfig

    return ACTConfig(
        device=policy.device,
        chunk_size=policy.action_horizon,
        n_action_steps=policy.action_steps,
        push_to_hub=False,
        **policy.parameters,
    )


def lerobot_train_config(policy: PolicyConfig, training: TrainingConfig) -> Any:
    """构造 LeRobot 官方训练器使用的完整配置。"""
    from lerobot.configs.default import DatasetConfig, PeftConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig

    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=training.dataset_repo_id or "",
            root=training.dataset_root,
            video_backend=training.video_backend,
            eval_split=training.eval_split,
        ),
        policy=act_config(policy),
        output_dir=Path(training.output_dir),
        job_name="act_weldpath",
        seed=training.seed,
        num_workers=training.num_workers,
        persistent_workers=training.num_workers > 0,
        batch_size=training.batch_size,
        steps=training.steps,
        env_eval_freq=0,
        log_freq=training.log_freq,
        eval_steps=training.eval_steps,
        max_eval_samples=training.max_eval_samples,
        save_freq=training.save_freq,
        wandb=WandBConfig(enable=training.wandb),
        peft=PeftConfig(**training.peft) if training.peft else None,
    )


def training_plan(policy: PolicyConfig, training: TrainingConfig) -> dict[str, Any]:
    """返回可记录、可复查的 ACT 训练计划。"""
    report = validate_dataset(training)
    train_episodes, eval_episodes = split_episodes(report.episodes, training.eval_split)
    return {
        "backend": "lerobot-train",
        "policy": "act",
        "dataset": asdict(report),
        "train_episodes": len(train_episodes),
        "eval_episodes": len(eval_episodes),
        "video_backend": training.video_backend,
        "batch_size": training.batch_size,
        "steps": training.steps,
        "output_dir": training.output_dir,
        "action_horizon": policy.action_horizon,
        "action_steps": policy.action_steps,
        "device": policy.device,
        "mixed_precision": training.amp_dtype if policy.parameters.get("use_amp") else None,
        "parameters": policy.parameters,
    }


def train(policy: PolicyConfig, training: TrainingConfig) -> Path:
    """使用 LeRobot 官方训练器执行 ACT 训练。"""
    from accelerate import Accelerator
    from lerobot.scripts.lerobot_train import train as lerobot_train

    config = lerobot_train_config(policy, training)
    precision = training.amp_dtype if policy.parameters.get("use_amp") else "no"
    accelerator = Accelerator(
        mixed_precision={"bfloat16": "bf16", "float16": "fp16"}.get(precision, "no")
    )
    with lerobot_training_log(Path(training.output_dir) / "train.log"):
        lerobot_train(config, accelerator=accelerator)
    return Path(training.output_dir)


__all__ = [
    "act_config",
    "lerobot_train_config",
    "split_episodes",
    "train",
    "training_plan",
]
