"""把项目配置映射到 LeRobot 官方 SmolVLA 训练流水线。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from welding_path_vla.core.config import PolicyConfig, TrainingConfig
from welding_path_vla.policies.act.training import split_episodes
from welding_path_vla.policies.checkpoint import resolve_checkpoint
from welding_path_vla.policies.data import validate_dataset
from welding_path_vla.policies.process import lerobot_config_argument, lerobot_training_log

DEFAULT_PRETRAINED_MODEL = "lerobot/smolvla_base"


@dataclass(frozen=True, slots=True)
class ResumeCheckpoint:
    """一次可恢复的 LeRobot checkpoint。

    Attributes:
        root: 同时包含模型和训练状态的 step checkpoint 目录。
        model: 包含权重、processor 和训练配置的 ``pretrained_model`` 目录。
        config: LeRobot 保存的 ``train_config.json``。
        step: 已完成的 optimizer update 数量。
    """

    root: Path
    model: Path
    config: Path
    step: int


def find_resume_checkpoint(output_dir: str | Path) -> ResumeCheckpoint:
    """从训练输出目录定位 LeRobot 的 ``checkpoints/last``。

    Args:
        output_dir: 原训练使用的输出目录。

    Returns:
        带模型配置和训练 step 的恢复点。
    """
    from lerobot.common.train_utils import load_training_step

    model = resolve_checkpoint(output_dir)
    root = model.parent
    config = model / "train_config.json"
    if not config.is_file():
        raise FileNotFoundError(f"missing LeRobot train config: {config}")
    step = load_training_step(root / "training_state")
    return ResumeCheckpoint(root, model, config, step)


def smolvla_config(policy: PolicyConfig) -> Any:
    """加载官方 SmolVLA 基线，并覆盖焊接任务需要的运行参数。

    Args:
        policy: 项目统一策略配置。``parameters.pretrained_model`` 可替换基线。

    Returns:
        已清空旧机器人 feature、可由当前数据集重新推断 feature 的 SmolVLAConfig。
    """
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    parameters = dict(policy.parameters)
    configured_source = parameters.pop("pretrained_model", DEFAULT_PRETRAINED_MODEL)
    source = str(policy.checkpoint or configured_source)
    config = PreTrainedConfig.from_pretrained(source)
    if not isinstance(config, SmolVLAConfig):
        raise ValueError(f"pretrained model is not SmolVLA: {source}")
    config.pretrained_path = Path(source)
    config.input_features = {}
    config.output_features = {}
    config.device = policy.device
    config.push_to_hub = False
    config.chunk_size = policy.action_horizon
    config.n_action_steps = policy.action_steps
    for name, value in parameters.items():
        if not hasattr(config, name):
            raise ValueError(f"unknown SmolVLA parameter: {name}")
        setattr(config, name, value)
    config.__post_init__()
    return config


def resumed_train_config(policy: PolicyConfig, training: TrainingConfig) -> Any:
    """从 LeRobot checkpoint 配置恢复训练器、optimizer 和 scheduler 定义。

    LeRobot 的 ``steps`` 是目标总步数。比如 checkpoint 已完成 3,500 步且配置目标为
    5,000 步，本次会继续执行 1,500 步。
    """
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    checkpoint = find_resume_checkpoint(training.output_dir)
    if training.steps <= checkpoint.step:
        raise ValueError(
            f"training.steps={training.steps} must exceed resumed step {checkpoint.step}; "
            "LeRobot steps denotes the target total step count"
        )

    config = TrainPipelineConfig.from_pretrained(checkpoint.config)
    if not isinstance(config.policy, SmolVLAConfig):
        raise ValueError(f"checkpoint policy is not SmolVLA: {checkpoint.config}")
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
    """构造 LeRobot 官方训练器使用的完整配置。"""
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig

    if training.resume:
        return resumed_train_config(policy, training)

    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=training.dataset_repo_id or "",
            root=training.dataset_root,
            video_backend=training.video_backend,
            eval_split=training.eval_split,
            return_uint8=True,
        ),
        policy=smolvla_config(policy),
        output_dir=Path(training.output_dir),
        job_name="smolvla_weldpath",
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
        resume=False,
    )


def training_plan(policy: PolicyConfig, training: TrainingConfig) -> dict[str, Any]:
    """返回包含数据规模、预训练来源和显存相关参数的训练计划。"""
    report = validate_dataset(training)
    train_episodes, eval_episodes = split_episodes(report.episodes, training.eval_split)
    config = lerobot_train_config(policy, training)
    checkpoint = find_resume_checkpoint(training.output_dir) if training.resume else None
    return {
        "backend": "lerobot-train",
        "policy": "smolvla",
        "pretrained_model": str(config.policy.pretrained_path),
        "resume": training.resume,
        "resume_step": checkpoint.step if checkpoint else 0,
        "remaining_steps": training.steps - checkpoint.step if checkpoint else training.steps,
        "dataset": asdict(report),
        "train_episodes": len(train_episodes),
        "eval_episodes": len(eval_episodes),
        "video_backend": training.video_backend,
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
    """用 LeRobot 官方训练器在单卡 BF16 下微调 SmolVLA。"""
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
    log_path = Path(training.output_dir) / "train.log"
    with lerobot_config_argument(resume_config), lerobot_training_log(log_path):
        lerobot_train(config, accelerator=accelerator)
    return Path(training.output_dir)


__all__ = [
    "DEFAULT_PRETRAINED_MODEL",
    "ResumeCheckpoint",
    "find_resume_checkpoint",
    "lerobot_train_config",
    "resumed_train_config",
    "smolvla_config",
    "train",
    "training_plan",
]
