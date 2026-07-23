from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from welding_path_vla.config import PolicyConfig, TrainingConfig


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
        command = [
            "lerobot-train",
            f"--dataset.repo_id={self.training.dataset_repo_id}",
            f"--policy.type={self.policy.family}",
            f"--policy.device={self.policy.device}",
            f"--output_dir={self.training.output_dir}",
            f"--batch_size={self.training.batch_size}",
            f"--steps={self.training.steps}",
        ]
        if self.training.dataset_root:
            command.append(f"--dataset.root={self.training.dataset_root}")
        if self.policy.family == "smolvla":
            command.extend(
                [
                    f"--policy.chunk_size={self.policy.action_horizon}",
                    f"--policy.n_action_steps={self.policy.action_horizon}",
                ]
            )
        return command


__all__ = ["TrainingRequest"]
