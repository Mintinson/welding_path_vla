"""VLA-Adapter 的 LeRobot processor。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from lerobot.processor import ProcessorStepRegistry

from welding_path_vla.policies.traj_vla_qwen.processor_traj_vla_qwen import (
    QwenPromptProcessorStep,
)
from welding_path_vla.policies.trajectory_vla.processor_trajectory_vla import (
    make_trajectory_vla_pre_post_processors,
)


@dataclass
@ProcessorStepRegistry.register(name="vla_adapter_action_prompt_processor")
class VLAAdapterActionPromptProcessorStep(QwenPromptProcessorStep):
    """沿用 Qwen chat 格式，并补齐上游 VLA-Adapter 的动作提问。"""

    def wrap(self, task: str) -> str:
        """将任务指令放入上游使用的 ``What action ...`` 模板。"""
        instruction, *details = task.replace("<image>", "").strip().lower().splitlines()
        question = f"What action should the robot take to {instruction.rstrip('.?')}?"
        return super().wrap("\n".join((question, *details)))


def make_vla_adapter_pre_post_processors(
    config: Any,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """复用统一语言、归一化和动作处理链，仅替换 Qwen prompt。"""
    return make_trajectory_vla_pre_post_processors(
        config,
        dataset_stats,
        task_processor=VLAAdapterActionPromptProcessorStep(),
    )


__all__ = ["VLAAdapterActionPromptProcessorStep", "make_vla_adapter_pre_post_processors"]
