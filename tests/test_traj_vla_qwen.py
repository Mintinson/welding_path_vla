from copy import deepcopy

import pytest
import torch

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.factory import get_policy_pipeline
from welding_path_vla.policies.spec import TRAJ_VLA_QWEN
from welding_path_vla.policies.traj_vla_qwen.configuration_traj_vla_qwen import (
    TrajVLAQwenConfig,
)
from welding_path_vla.policies.traj_vla_qwen.modeling_traj_vla_qwen import (
    TrajVLAQwenPolicy,
)
from welding_path_vla.policies.traj_vla_qwen.prismatic import SpatialTokenMerger
from welding_path_vla.policies.traj_vla_qwen.processor_traj_vla_qwen import (
    QwenPromptProcessorStep,
)
from welding_path_vla.policies.traj_vla_qwen.qwen_with_expert import (
    PairedLayerDecoder,
    Qwen25DecoderAdapter,
)
from welding_path_vla.policies.trajectory_vla.flow_matching import make_attention_masks


def make_tiny_paired_decoder(
    checkpoint_qwen: bool = False,
    checkpoint_expert: bool = False,
) -> PairedLayerDecoder:
    """构造无需下载权重的两层 Qwen2.5 双流网络。"""
    transformers = pytest.importorskip("transformers")
    Qwen2Config = transformers.Qwen2Config
    Qwen2Model = transformers.Qwen2Model
    language_config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    expert_config = deepcopy(language_config)
    expert_config.hidden_size = 24
    expert_config.intermediate_size = 48
    language = Qwen2Model(language_config)
    expert = Qwen2Model(expert_config)
    expert.embed_tokens = None
    return PairedLayerDecoder(
        language,
        expert,
        Qwen25DecoderAdapter(),
        checkpoint_qwen=checkpoint_qwen,
        checkpoint_expert=checkpoint_expert,
    )


def test_qwen_config_and_policy_are_registered() -> None:
    """模块化 YAML 应能选择本地 Qwen Policy。"""
    config = AppConfig.load("configs/traj_vla_qwen.yaml")
    assert config.policy.family == "traj_vla_qwen"
    assert config.policy.parameters["num_vlm_layers"] == 16
    assert config.policy.parameters["num_expert_layers"] == 16
    assert config.policy.parameters["frozen_vision_dtype"] == "float32"
    assert not config.policy.parameters["gradient_checkpointing_qwen"]
    assert not config.policy.parameters["gradient_checkpointing_expert"]
    assert get_policy_pipeline("traj_vla_qwen").spec is TRAJ_VLA_QWEN


def test_qwen_config_rejects_unpaired_depth() -> None:
    """第一版必须维持一一配对的 Qwen 与专家深度。"""
    try:
        TrajVLAQwenConfig(num_vlm_layers=16, num_expert_layers=8)
    except ValueError as error:
        assert "equal" in str(error)
    else:
        raise AssertionError("unpaired decoder depth was accepted")


def test_trainable_vision_rejects_frozen_bfloat16_setting() -> None:
    """冻结权重的 BF16 开关不能悄悄改变可训练视觉主干精度。"""
    with pytest.raises(ValueError, match="frozen_vision_dtype"):
        TrajVLAQwenConfig(
            train_vision_encoder=True,
            frozen_vision_dtype="bfloat16",
        )


def test_token_merger_starts_as_local_average() -> None:
    """Token Merger 初值应保留局部平均，随后仍可学习。"""
    merger = SpatialTokenMerger(hidden_size=2, grid_size=4, factor=2)
    tokens = torch.arange(32, dtype=torch.float32).view(1, 16, 2)
    output = merger(tokens)
    grid = tokens.view(1, 4, 4, 2)
    expected = torch.stack(
        [
            grid[:, :2, :2].mean(dim=(1, 2)),
            grid[:, :2, 2:].mean(dim=(1, 2)),
            grid[:, 2:, :2].mean(dim=(1, 2)),
            grid[:, 2:, 2:].mean(dim=(1, 2)),
        ],
        dim=1,
    )
    assert torch.allclose(output, expected)
    assert merger.projection.weight.requires_grad


def test_qwen_prompt_matches_prismatic_pretraining_format() -> None:
    """任务文本必须使用 MiniVLA 的 Qwen 单轮 chat 包装。"""
    prompt = QwenPromptProcessorStep().complementary_data({"task": "Weld the seam."})["task"]
    assert prompt.startswith("<|im_start|>system\nYou are Qwen")
    assert "<|im_start|>user\nWeld the seam.<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_paired_attention_is_asymmetric_and_cache_equivalent() -> None:
    """上下文不能读取动作，KV cache 与完整双流前向应保持一致。"""
    torch.manual_seed(7)
    decoder = make_tiny_paired_decoder().eval()
    context = torch.randn(1, 3, 32)
    action = torch.randn(1, 2, 24)
    padding = torch.ones(1, 5, dtype=torch.bool)
    blocks = torch.tensor([[False, False, False, True, True]])
    mask = make_attention_masks(padding, blocks)
    positions = torch.arange(5).unsqueeze(0)

    full, _ = decoder(mask, positions, None, [context, action], False, False)
    changed, _ = decoder(mask, positions, None, [context, action + 10], False, False)
    assert torch.allclose(full[0], changed[0], atol=1e-5)

    prefix_mask = torch.ones(1, 3, 3, dtype=torch.bool)
    _, cache = decoder(
        prefix_mask,
        positions[:, :3],
        None,
        [context, None],
        True,
        True,
    )
    assert cache is not None
    suffix_mask = mask[:, 3:, :]
    cached, _ = decoder(
        suffix_mask,
        positions[:, 3:],
        cache,
        [None, action],
        True,
        False,
    )
    assert torch.allclose(full[1], cached[1], atol=1e-5)


def test_paired_layer_checkpointing_preserves_gradients() -> None:
    """双流 checkpoint 重算应保持输出和输入梯度不变。"""
    torch.manual_seed(11)
    direct = make_tiny_paired_decoder().train()
    checkpointed = deepcopy(direct)
    checkpointed.checkpoint_qwen = True
    checkpointed.checkpoint_expert = True
    padding = torch.ones(1, 5, dtype=torch.bool)
    blocks = torch.tensor([[False, False, False, True, True]])
    mask = make_attention_masks(padding, blocks)
    positions = torch.arange(5).unsqueeze(0)
    context = torch.randn(1, 3, 32, requires_grad=True)
    action = torch.randn(1, 2, 24, requires_grad=True)
    checkpoint_context = context.detach().clone().requires_grad_(True)
    checkpoint_action = action.detach().clone().requires_grad_(True)

    output, _ = direct(mask, positions, None, [context, action], False, False)
    checked, _ = checkpointed(
        mask,
        positions,
        None,
        [checkpoint_context, checkpoint_action],
        False,
        False,
    )
    assert torch.allclose(output[0], checked[0], atol=1e-5)
    assert torch.allclose(output[1], checked[1], atol=1e-5)
    sum(tensor.sum() for tensor in output if tensor is not None).backward()
    sum(tensor.sum() for tensor in checked if tensor is not None).backward()
    assert context.grad is not None and checkpoint_context.grad is not None
    assert action.grad is not None and checkpoint_action.grad is not None
    assert torch.allclose(context.grad, checkpoint_context.grad, atol=1e-5)
    assert torch.allclose(action.grad, checkpoint_action.grad, atol=1e-5)


def test_lora_targets_can_select_qwen_or_expert() -> None:
    """LoRA 目标范围应由策略配置选择，并保留新建动作头。"""
    policy = TrajVLAQwenPolicy.__new__(TrajVLAQwenPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = TrajVLAQwenConfig(lora_target="qwen")
    targets = policy._get_default_peft_targets()
    assert "qwen" in targets["target_modules"]
    assert "expert" in targets["modules_to_save"][-1]
    assert "model.action_out_proj" in targets["modules_to_save"]

    policy.config.lora_target = "all"
    targets = policy._get_default_peft_targets()
    assert "qwen" in targets["target_modules"]
    assert "expert" in targets["target_modules"]
    assert "model.vlm_with_expert.expert" not in targets["modules_to_save"]
