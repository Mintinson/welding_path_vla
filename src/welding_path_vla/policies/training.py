from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from welding_path_vla.core.config import PolicyConfig, TrainingConfig
from welding_path_vla.policies.factory import get_policy_pipeline


def cli_value(value: object) -> str:
    """把 Python 配置值转换为 draccus 命令行表示。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    policy: PolicyConfig
    training: TrainingConfig

    def validate(self) -> None:
        if not self.training.dataset_repo_id:
            raise ValueError("training.dataset_repo_id must identify an exported dataset")

    @property
    def output_dir(self) -> Path:
        return Path(self.training.output_dir)

    def command(self) -> list[str]:
        """生成与统一 YAML 对应的 LeRobot 训练命令。"""
        self.validate()
        options = {
            "dataset.repo_id": self.training.dataset_repo_id,
            "dataset.root": self.training.dataset_root,
            "dataset.video_backend": self.training.video_backend,
            "dataset.eval_split": self.training.eval_split,
            "output_dir": self.training.output_dir,
            "job_name": f"{self.policy.family}_weldpath",
            "batch_size": self.training.batch_size,
            "steps": self.training.steps,
            "num_workers": self.training.num_workers,
            "eval_steps": self.training.eval_steps,
            "max_eval_samples": self.training.max_eval_samples,
            "log_freq": self.training.log_freq,
            "save_freq": self.training.save_freq,
            "seed": self.training.seed,
            "env_eval_freq": 0,
            "save_checkpoint": True,
            "wandb.enable": self.training.wandb,
            **get_policy_pipeline(self.policy.family).training_overrides(self.policy),
        }
        return ["lerobot-train", *(f"--{key}={cli_value(value)}" for key, value in options.items())]

    def run(self) -> Path:
        """通过策略注册表执行对应训练实现。"""
        return get_policy_pipeline(self.policy.family).train(
            self.policy,
            self.training,
        )

    def plan(self) -> dict[str, object]:
        """返回策略专属的训练计划。"""
        return get_policy_pipeline(self.policy.family).training_plan(
            self.policy,
            self.training,
        )


__all__ = ["TrainingRequest", "cli_value"]
