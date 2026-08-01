"""把项目配置映射到 LeRobot 官方 π0 系列训练器。"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from welding_path_vla.core.config import PolicyConfig, TrainingConfig
from welding_path_vla.policies.checkpoint import find_resume_checkpoint
from welding_path_vla.policies.data import validate_dataset
from welding_path_vla.policies.pi_family.spec import PIFamilySpec
from welding_path_vla.policies.process import lerobot_config_argument, lerobot_training_log


def pi_config(policy: PolicyConfig, family: PIFamilySpec) -> Any:
    """加载官方基础模型配置，并应用焊接任务参数。

    Args:
        policy: 项目统一策略配置。
        family: π0 或 π0.5 模型规格。

    Returns:
        清空原机器人 feature、可由焊接数据集重新推断 feature 的官方配置。
    """
    from lerobot.configs import PreTrainedConfig

    config_class = family.config_class()
    parameters = dict(policy.parameters)
    configured_source = parameters.pop("pretrained_model", family.pretrained_model)
    source = str(policy.checkpoint or configured_source)
    config = PreTrainedConfig.from_pretrained(source)
    if not isinstance(config, config_class):
        raise ValueError(f"pretrained model is not {family.display_name}: {source}")

    config.pretrained_path = Path(source)
    config.input_features = {}
    config.output_features = {}
    config.device = policy.device
    config.push_to_hub = False
    config.chunk_size = policy.action_horizon
    config.n_action_steps = policy.action_steps
    for name, value in parameters.items():
        if not hasattr(config, name):
            raise ValueError(f"unknown {family.display_name} parameter: {name}")
        setattr(config, name, value)
    config.__post_init__()
    return config


def resumed_train_config(
    policy: PolicyConfig,
    training: TrainingConfig,
    family: PIFamilySpec,
) -> Any:
    """从 LeRobot checkpoint 恢复模型、优化器、调度器和全局 step。"""
    from lerobot.configs.train import TrainPipelineConfig

    config_class = family.config_class()
    checkpoint = find_resume_checkpoint(training.output_dir)
    if training.steps <= checkpoint.step:
        raise ValueError(
            f"training.steps={training.steps} must exceed resumed step {checkpoint.step}; "
            "LeRobot steps denotes the target total step count"
        )

    config = TrainPipelineConfig.from_pretrained(checkpoint.config)
    resumed_policy = config.policy
    if resumed_policy is None or not isinstance(resumed_policy, config_class):
        raise ValueError(f"checkpoint policy is not {family.display_name}: {checkpoint.config}")
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
    resumed_policy.device = policy.device
    resumed_policy.pretrained_path = checkpoint.model
    return config


def lerobot_train_config(
    policy: PolicyConfig,
    training: TrainingConfig,
    family: PIFamilySpec,
) -> Any:
    """构造 LeRobot 官方训练器使用的完整配置。"""
    from lerobot.configs.default import DatasetConfig, PeftConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig

    if training.resume:
        return resumed_train_config(policy, training, family)
    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=training.dataset_repo_id or "",
            root=training.dataset_root,
            video_backend=training.video_backend,
            eval_split=training.eval_split,
            return_uint8=True,
        ),
        policy=pi_config(policy, family),
        output_dir=Path(training.output_dir),
        job_name=f"{family.family}_weldpath",
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


def training_plan(
    policy: PolicyConfig,
    training: TrainingConfig,
    family: PIFamilySpec,
) -> dict[str, Any]:
    """返回数据规模、预训练来源和显存相关设置。"""
    report = validate_dataset(training)
    eval_episodes = math.ceil(report.episodes * training.eval_split)
    config = lerobot_train_config(policy, training, family)
    checkpoint = find_resume_checkpoint(training.output_dir) if training.resume else None
    return {
        "backend": "lerobot-train",
        "policy": family.family,
        "lerobot_policy_type": family.lerobot_type,
        "pretrained_model": str(config.policy.pretrained_path),
        "resume": training.resume,
        "resume_step": checkpoint.step if checkpoint else 0,
        "remaining_steps": training.steps - checkpoint.step if checkpoint else training.steps,
        "dataset": asdict(report),
        "train_episodes": report.episodes - eval_episodes,
        "eval_episodes": eval_episodes,
        "video_backend": training.video_backend,
        "batch_size_per_process": training.batch_size,
        "steps": training.steps,
        "output_dir": training.output_dir,
        "log_file": str(Path(training.output_dir) / "train.log"),
        "action_horizon": policy.action_horizon,
        "action_steps": policy.action_steps,
        "image_resolution": list(config.policy.image_resolution),
        "inference_steps": config.policy.num_inference_steps,
        "device": policy.device,
        "mixed_precision": config.policy.dtype,
        "peft": training.peft,
        "parameters": policy.parameters,
    }


def train(policy: PolicyConfig, training: TrainingConfig, family: PIFamilySpec) -> Path:
    """运行官方训练器；在 ``accelerate launch`` 下自动使用多张 GPU。"""
    from lerobot.scripts.lerobot_train import train as lerobot_train

    config = lerobot_train_config(policy, training, family)
    resume_config = (
        config.checkpoint_path / "pretrained_model" / "train_config.json"
        if config.resume and config.checkpoint_path
        else None
    )
    with (
        lerobot_config_argument(resume_config),
        lerobot_training_log(Path(training.output_dir) / "train.log"),
    ):
        lerobot_train(config)
    return Path(training.output_dir)


__all__ = [
    "lerobot_train_config",
    "pi_config",
    "resumed_train_config",
    "train",
    "training_plan",
]
