"""所有 LeRobot 策略共享的训练配置、恢复和执行流程。"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from welding_path_vla.core.config import PolicyConfig, TrainingConfig
from welding_path_vla.policies.action_processors import (
    relative_processor_factory,
    require_relative_checkpoint,
)
from welding_path_vla.policies.checkpoint import find_resume_checkpoint
from welding_path_vla.policies.data import validate_dataset
from welding_path_vla.policies.process import lerobot_config_argument, lerobot_training_log
from welding_path_vla.policies.spec import LeRobotPolicySpec


def pretrained_source(
    policy: PolicyConfig,
    spec: LeRobotPolicySpec,
) -> tuple[str | None, dict[str, Any]]:
    """分离公共的预训练来源和模型参数。"""
    parameters = dict(policy.parameters)
    configured = parameters.pop("pretrained_model", spec.pretrained_model)
    return policy.checkpoint or configured, parameters


def make_policy_config(policy: PolicyConfig, spec: LeRobotPolicySpec) -> Any:
    """按策略规格构造配置，同时保留数据驱动的 feature 推断。"""
    from lerobot.configs import PreTrainedConfig

    config_class = spec.config_class()
    source, parameters = pretrained_source(policy, spec)
    common = {
        "input_features": {},
        "output_features": {},
        "device": policy.device,
        "push_to_hub": False,
        "chunk_size": policy.action_horizon,
        "n_action_steps": policy.action_steps,
    }
    if spec.config_mode == "scratch" and source is None:
        return config_class(**common, **parameters)
    if spec.config_mode == "local_pretrained":
        return config_class(pretrained_path=Path(str(source)), **common, **parameters)
    if source is None:
        raise ValueError(f"{spec.display_name} requires a pretrained model")

    config = PreTrainedConfig.from_pretrained(source)
    if not isinstance(config, config_class):
        raise ValueError(f"pretrained model is not {spec.display_name}: {source}")
    config.pretrained_path = Path(source)
    for name, value in {**common, **parameters}.items():
        if not hasattr(config, name):
            raise ValueError(f"unknown {spec.display_name} parameter: {name}")
        setattr(config, name, value)
    config.__post_init__()
    return config


def apply_training_overrides(config: Any, policy: PolicyConfig, training: TrainingConfig) -> Any:
    """把项目 YAML 中允许变化的训练字段覆盖到 LeRobot 配置。"""
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
    config.policy.device = policy.device
    if training.lr is not None and hasattr(config.policy, "optimizer_lr"):
        config.policy.optimizer_lr = training.lr
    return config


def resumed_train_config(
    policy: PolicyConfig,
    training: TrainingConfig,
    spec: LeRobotPolicySpec,
) -> Any:
    """从 checkpoint 恢复模型、optimizer、scheduler 和全局 step。"""
    from lerobot.configs.train import TrainPipelineConfig

    config_class = spec.config_class()
    checkpoint = find_resume_checkpoint(training.output_dir)
    require_relative_checkpoint(checkpoint.model)
    if training.steps <= checkpoint.step:
        raise ValueError(
            f"training.steps={training.steps} must exceed resumed step {checkpoint.step}"
        )
    config = TrainPipelineConfig.from_pretrained(checkpoint.config)
    if config.policy is None or not isinstance(config.policy, config_class):
        raise ValueError(f"checkpoint policy is not {spec.display_name}: {checkpoint.config}")
    config.resume = True
    config.checkpoint_path = checkpoint.root
    config.policy.pretrained_path = checkpoint.model
    config.dataset.return_uint8 = spec.return_uint8
    return apply_training_overrides(config, policy, training)


def make_train_config(
    policy: PolicyConfig,
    training: TrainingConfig,
    spec: LeRobotPolicySpec,
) -> Any:
    """构造 LeRobot 训练器使用的完整配置。"""
    from lerobot.configs.default import DatasetConfig, PeftConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig

    if training.resume:
        return resumed_train_config(policy, training, spec)
    if policy.checkpoint and Path(policy.checkpoint).exists():
        require_relative_checkpoint(policy.checkpoint)
    policy_config = make_policy_config(policy, spec)
    if training.lr is not None and hasattr(policy_config, "optimizer_lr"):
        policy_config.optimizer_lr = training.lr
    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=training.dataset_repo_id or "",
            root=training.dataset_root,
            video_backend=training.video_backend,
            eval_split=training.eval_split,
            return_uint8=spec.return_uint8,
        ),
        policy=policy_config,
        output_dir=Path(training.output_dir),
        job_name=f"{spec.family}_weldpath",
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
    spec: LeRobotPolicySpec,
) -> dict[str, Any]:
    """返回所有策略统一的可复查训练计划。"""
    report = validate_dataset(training, policy)
    eval_episodes = math.ceil(report.episodes * training.eval_split) if training.eval_split else 0
    config = make_train_config(policy, training, spec)
    checkpoint = find_resume_checkpoint(training.output_dir) if training.resume else None
    plan = {
        "backend": "lerobot-train",
        "policy": spec.family,
        "lerobot_policy_type": spec.policy_type,
        "implementation": spec.implementation,
        "pretrained_model": (
            str(config.policy.pretrained_path) if config.policy.pretrained_path else None
        ),
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
        "action_representation": policy.action_representation,
        "action_horizon": policy.action_horizon,
        "action_steps": policy.action_steps,
        "device": policy.device,
        "mixed_precision": training.amp_dtype,
        "peft": training.peft,
        "parameters": policy.parameters,
    }
    for label, attribute in spec.plan_fields:
        plan[label] = getattr(config.policy, attribute)
    return plan


def train(policy: PolicyConfig, training: TrainingConfig, spec: LeRobotPolicySpec) -> Path:
    """运行官方训练循环，并固定保存本地日志。"""
    from lerobot.scripts.lerobot_train import train as lerobot_train

    config = make_train_config(policy, training, spec)
    accelerator = None
    if spec.explicit_mixed_precision:
        from accelerate import Accelerator

        enabled = policy.parameters.get("use_amp", True)
        precision = training.amp_dtype if enabled else "float32"
        accelerator = Accelerator(
            mixed_precision={"bfloat16": "bf16", "float16": "fp16"}.get(precision, "no")
        )
    resume_config = (
        config.checkpoint_path / "pretrained_model" / "train_config.json"
        if config.resume and config.checkpoint_path
        else None
    )
    with (
        lerobot_config_argument(resume_config),
        lerobot_training_log(Path(training.output_dir) / "train.log"),
        relative_processor_factory(),
    ):
        lerobot_train(config, accelerator=accelerator)
    return Path(training.output_dir)


__all__ = [
    "apply_training_overrides",
    "make_policy_config",
    "make_train_config",
    "pretrained_source",
    "resumed_train_config",
    "train",
    "training_plan",
]
