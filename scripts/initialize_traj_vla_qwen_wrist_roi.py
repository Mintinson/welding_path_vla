#!/usr/bin/env python3
"""把已有 direct-geometry PEFT 权重扩展为零门控腕部局部分支初始化 checkpoint。"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from lerobot.configs import PreTrainedConfig
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

from welding_path_vla.policies.traj_vla_qwen.configuration_traj_vla_qwen import TrajVLAQwenConfig
from welding_path_vla.policies.traj_vla_qwen.modeling_traj_vla_qwen import TrajVLAQwenPolicy


def initialize(source: Path, output: Path) -> None:
    """保留旧模型全部可学习权重和归一化器，只新增局部模块，不复制优化器状态。

    Args:
        source: 已训练的 direct-geometry PEFT pretrained_model 目录。
        output: 不存在的新目录；禁止覆盖任何已有训练或初始化结果。
    """
    if output.exists():
        raise FileExistsError(output)
    config = PreTrainedConfig.from_pretrained(source)
    if not isinstance(config, TrajVLAQwenConfig) or config.geometry_attention_mode != "direct":
        raise ValueError("source must be a TrajVLA-Qwen direct geometry checkpoint")
    if config.use_wrist_roi:
        raise ValueError("source already contains a wrist ROI branch")
    config.use_wrist_roi = True
    config.pretrained_path = None
    config.device = "cpu"
    config.compile_model = False
    config.__post_init__()
    torch.manual_seed(42)
    base = TrajVLAQwenPolicy.make_peft_base(config)
    adapter = LoraConfig.from_pretrained(source)
    adapter.modules_to_save = base._get_default_peft_targets()["modules_to_save"]
    policy = base.wrap_with_peft(adapter)
    old = load_file(source / "adapter_model.safetensors")
    current = get_peft_model_state_dict(policy)
    if old.keys() - current.keys():
        raise ValueError(f"source weights cannot be mapped: {old.keys() - current.keys()}")
    added = current.keys() - old.keys()
    if not added or any(".wrist_roi_" not in key for key in added):
        raise ValueError(f"unexpected newly initialized weights: {added}")
    # PEFT 的 modules_to_save 恢复要求所有新增模块键都存在，保留其初始化后再覆盖旧权重。
    current.update(old)
    set_peft_model_state_dict(policy, current)
    camera_index = list(config.image_features).index(config.wrist_roi_camera_key)
    base.model.vlm_with_expert.decoder.initialize_wrist_roi_from_global(camera_index)
    restored = get_peft_model_state_dict(policy)
    for key, value in old.items():
        if not torch.equal(restored[key], value):
            raise ValueError(f"source weight changed during initialization: {key}")
    output.mkdir(parents=True)
    policy.save_pretrained(output)
    policy.config.save_pretrained(output)
    # 保留 relative-action processor 及其训练统计，避免热启动改变动作语义。
    for path in source.glob("policy_*processor*"):
        shutil.copy2(path, output / path.name)
    report = {
        "source": str(source.resolve()),
        "new_tensors": sorted(added),
        "preserved_tensors": len(old),
        "initial_gate": 0.0,
        "local_projection_initialization": "trained_full_view_wrist",
    }
    (output / "initialization.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(output), "preserved_tensors": len(old), "new_tensors": len(added)},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initialize(args.source, args.output)
