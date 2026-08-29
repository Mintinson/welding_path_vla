"""Prismatic-Qwen Policy 的 LeRobot processor。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    ProcessorStepRegistry,
)

from welding_path_vla.policies.traj_vla_qwen.geometry_grounding import (
    GeometryGroundingTargetProcessorStep,
)
from welding_path_vla.policies.trajectory_vla.processor_trajectory_vla import (
    make_trajectory_vla_pre_post_processors,
)


@dataclass
@ProcessorStepRegistry.register(name="prismatic_qwen_prompt_processor")
class QwenPromptProcessorStep(ComplementaryDataProcessorStep):
    """把任务包装成 MiniVLA 预训练使用的 Qwen chat prompt。"""

    system_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    def wrap(self, task: str) -> str:
        """生成单轮 user prompt，并保留 assistant 起始标记。"""
        task = task.replace("<image>", "").strip()
        return (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{task}<|im_end|>\n<|im_start|>assistant\n"
        )

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        """包装单条或 batch 形式的任务文本。"""
        task = complementary_data.get("task")
        result = dict(complementary_data)
        if isinstance(task, str):
            result["task"] = self.wrap(task)
        elif isinstance(task, list):
            result["task"] = [self.wrap(text) for text in task]
        return result

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """文本包装不改变任何 feature 形状。"""
        return features


def make_traj_vla_qwen_pre_post_processors(
    config: Any,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """复用统一的语言、归一化与 relative-action processor 链。"""
    preprocessor, postprocessor = make_trajectory_vla_pre_post_processors(
        config,
        dataset_stats,
        task_processor=QwenPromptProcessorStep(),
    )
    if config.use_geometry_grounding:
        target_step = GeometryGroundingTargetProcessorStep(
            camera_keys=tuple(config.geometry_camera_keys),
            camera_fovy_deg=tuple(config.geometry_camera_fovy_deg),
            global_camera_pose_world=tuple(config.geometry_global_camera_pose_world),
            wrist_camera_pose_tcp=tuple(config.geometry_wrist_camera_pose_tcp),
            target_size=tuple(config.resize_imgs_with_padding),
            patch_grid=config.vision_patch_grid,
            corridor_radius_px=config.geometry_corridor_radius_px,
        )
        device_index = next(
            index
            for index, step in enumerate(preprocessor.steps)
            if isinstance(step, DeviceProcessorStep)
        )
        # 将 GeometryGroundingTargetProcessorStep 插入到设备转移之前
        preprocessor.steps.insert(device_index, target_step)
    return preprocessor, postprocessor


__all__ = ["QwenPromptProcessorStep", "make_traj_vla_qwen_pre_post_processors"]
