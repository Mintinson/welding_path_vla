# Copyright 2025 Hugging Face Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trajectory-VLA 的输入、输出 processor。

LeRobot 用 ProcessorStep 链把"原始 transition 数据"逐步加工成模型输入
（预处理），再把模型输出加工回物理动作（后处理）。本文件把该链按研究
需要拆成可单独替换的步骤：例如只替换 Normalizer 步骤即可切换归一化
策略，只替换 Tokenizer 步骤即可更换语言编码方式，其余代码不用改动。
"""

from __future__ import annotations

from typing import Any

import torch
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def make_trajectory_vla_pre_post_processors(
    config: Any,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    task_processor: Any | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """构造可单独替换步骤的标准 LeRobot processor。

    Args:
        config: Trajectory-VLA 模型配置。
        dataset_stats: 当前训练数据集的归一化统计。

    Returns:
        输入 processor 和动作反归一化 processor。

    输入链（按序执行）各步骤职责：
        - RenameObservations：观测键名映射（本项目为空映射，键名不变）；
        - AddBatchDimension：给无 batch 维的样本补上 batch 维（[1, ...]）；
        - TaskProcessor：把任务描述转换为当前语言主干的预训练 prompt 格式；
        - Tokenizer：用 VLM 的 tokenizer 把语言指令编码为 token id，
          右侧补齐到 pad_language_to、截断到 tokenizer_max_length；
        - Device：把张量搬到模型所在设备；
        - Normalizer：对输入与输出特征按 dataset_stats 统一归一化。

    输出链（后处理）各步骤职责：
        - Unnormalizer：只反归一化输出特征（动作），把模型预测从归一化
          空间还原回真实物理量纲；
        - Device(cpu)：把动作移回 CPU，供环境步进使用。
    """
    tokenizer_name = (
        config.language_model_name
        if hasattr(config, "language_model_name")
        else config.vlm_model_name
    )
    # ---- 输入预处理链：原始 transition → 模型输入 ----
    input_steps = [
        # 观测键名映射（数据键与模型要求的键不一致时在此改名）
        RenameObservationsProcessorStep(rename_map={}),
        # 单条样本补成 batch=1，便于与批量训练共用同一模型调用
        AddBatchDimensionProcessorStep(),
        # 不同语言主干在这里注入自己的预训练 prompt 格式。
        task_processor or NewLineTaskProcessorStep(),
        # 语言指令 token 化：右侧补齐、超长截断
        TokenizerProcessorStep(
            tokenizer_name=tokenizer_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        # 输入搬到模型所在设备（CPU 训练时为空操作）
        DeviceProcessorStep(device=config.device),
        # 输入与输出特征统一按数据集统计归一化（输出特征也要在此归一化，
        # 模型才是在归一化空间内预测动作）
        NormalizerProcessorStep(
            features={**(config.input_features or {}), **(config.output_features or {})},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    # ---- 输出后处理链：模型输出 → 物理动作 ----
    output_steps = [
        # 只反归一化输出特征（动作）：从归一化空间还原到真实量纲
        UnnormalizerProcessorStep(
            features=config.output_features or {},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        # 动作移回 CPU，方便环境步进且不占用 GPU 显存
        DeviceProcessorStep(device="cpu"),
    ]
    # 后处理链额外注册"动作 <-> transition"的双向转换器：
    # 环境返回的 transition 动作在送入后处理前先转成 PolicyAction 格式，
    # 后处理完再转回 transition 格式交给环境
    return (
        PolicyProcessorPipeline(
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline(
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


__all__ = ["make_trajectory_vla_pre_post_processors"]
