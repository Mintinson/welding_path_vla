"""Trajectory-VLA 的 LeRobot 训练配置与执行入口。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from welding_path_vla.core.config import PolicyConfig, TrainingConfig
from welding_path_vla.policies.act.training import split_episodes
from welding_path_vla.policies.checkpoint import find_resume_checkpoint
from welding_path_vla.policies.data import validate_dataset
from welding_path_vla.policies.process import lerobot_config_argument, lerobot_training_log
from welding_path_vla.policies.trajectory_vla.configuration_trajectory_vla import (
    TrajectoryVLAConfig,
)

DEFAULT_PRETRAINED_MODEL = "lerobot/smolvla_base"


def trajectory_vla_config(policy: PolicyConfig) -> TrajectoryVLAConfig:
    """构造本地模型配置，并把官方 SmolVLA 权重作为初始化来源。

    Args:
        policy: 项目统一策略配置。

    Returns:
        等待 LeRobot 从数据集推断 feature 的 Trajectory-VLA 配置。
    """
    parameters = dict(policy.parameters)
    source = policy.checkpoint or parameters.pop(
        "pretrained_model",
        DEFAULT_PRETRAINED_MODEL,
    )
    config = TrajectoryVLAConfig(
        pretrained_path=Path(source),
        input_features={},
        output_features={},
        device=policy.device,
        push_to_hub=False,
        chunk_size=policy.action_horizon,
        n_action_steps=policy.action_steps,
        **parameters,
    )
    return config


def resumed_train_config(policy: PolicyConfig, training: TrainingConfig) -> Any:
    """恢复模型、优化器、调度器和已完成步数。"""
    from lerobot.configs.train import TrainPipelineConfig

    checkpoint = find_resume_checkpoint(training.output_dir)
    if training.steps <= checkpoint.step:
        raise ValueError(
            f"training.steps={training.steps} must exceed resumed step {checkpoint.step}"
        )
    config = TrainPipelineConfig.from_pretrained(checkpoint.config)
    if not isinstance(config.policy, TrajectoryVLAConfig):
        raise ValueError(f"checkpoint policy is not Trajectory-VLA: {checkpoint.config}")
    config.resume = True
    config.checkpoint_path = checkpoint.root
    config.output_dir = Path(training.output_dir)
    config.steps = training.steps
    config.seed = training.seed
    config.num_workers = training.num_workers
    config.persistent_workers = training.num_workers > 0
    config.batch_size = training.batch_size
    config.log_freq = training.log_freq
    config.eval_steps = training.eval_steps
    config.max_eval_samples = training.max_eval_samples
    config.save_freq = training.save_freq
    config.wandb.enable = training.wandb
    config.dataset.repo_id = training.dataset_repo_id or config.dataset.repo_id
    config.dataset.root = Path(training.dataset_root) if training.dataset_root else None
    config.dataset.video_backend = training.video_backend
    config.dataset.eval_split = training.eval_split
    config.dataset.return_uint8 = True
    config.policy.device = policy.device
    config.policy.pretrained_path = checkpoint.model
    return config


def lerobot_train_config(policy: PolicyConfig, training: TrainingConfig) -> Any:
    """构造 LeRobot 训练器需要的完整配置。"""
    from lerobot.configs.default import DatasetConfig, PeftConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig

    if training.resume:
        return resumed_train_config(policy, training)
    policy_cfg = trajectory_vla_config(policy)
    if training.lr is not None:
        policy_cfg.optimizer_lr = training.lr
    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=training.dataset_repo_id or "",
            root=training.dataset_root,
            video_backend=training.video_backend,
            eval_split=training.eval_split,
            return_uint8=True,
        ),
        policy=policy_cfg,
        output_dir=Path(training.output_dir),
        job_name="trajectory_vla_weldpath",
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
        resume=False,
    )


def training_plan(policy: PolicyConfig, training: TrainingConfig) -> dict[str, Any]:
    """返回便于复查的数据划分、模型来源与训练规模。"""
    report = validate_dataset(training)
    train_episodes, eval_episodes = split_episodes(report.episodes, training.eval_split)
    config = lerobot_train_config(policy, training)
    checkpoint = find_resume_checkpoint(training.output_dir) if training.resume else None
    return {
        "backend": "lerobot-train",
        "policy": "trajectory_vla",
        "implementation": "local",
        "pretrained_model": str(config.policy.pretrained_path),
        "resume": training.resume,
        "resume_step": checkpoint.step if checkpoint else 0,
        "remaining_steps": training.steps - checkpoint.step if checkpoint else training.steps,
        "dataset": asdict(report),
        "train_episodes": len(train_episodes),
        "eval_episodes": len(eval_episodes),
        "batch_size": training.batch_size,
        "steps": training.steps,
        "output_dir": training.output_dir,
        "log_file": str(Path(training.output_dir) / "train.log"),
        "action_horizon": policy.action_horizon,
        "action_steps": policy.action_steps,
        "image_size": config.policy.resize_imgs_with_padding,
        "flow_steps": config.policy.num_steps,
        "device": policy.device,
        "mixed_precision": training.amp_dtype,
        "parameters": policy.parameters,
    }


def train(policy: PolicyConfig, training: TrainingConfig) -> Path:
    """运行 LeRobot 训练循环，同时持久化终端日志。"""
    from accelerate import Accelerator
    from lerobot.scripts.lerobot_train import train as lerobot_train

    config = lerobot_train_config(policy, training)
    accelerator = Accelerator(
        mixed_precision={"bfloat16": "bf16", "float16": "fp16"}[training.amp_dtype]
    )
    resume_config = (
        config.checkpoint_path / "pretrained_model" / "train_config.json"
        if config.resume and config.checkpoint_path
        else None
    )
    with (
        lerobot_config_argument(resume_config),
        lerobot_training_log(Path(training.output_dir) / "train.log"),
    ):
        lerobot_train(config, accelerator=accelerator)
    return Path(training.output_dir)


__all__ = [
    "DEFAULT_PRETRAINED_MODEL",
    "lerobot_train_config",
    "resumed_train_config",
    "train",
    "training_plan",
    "trajectory_vla_config",
]
