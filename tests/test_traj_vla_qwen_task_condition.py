"""任务条件直达动作 Expert 的梯度、缓存、混合精度与保存兼容性测试。"""

import json
from copy import deepcopy

import pytest
import torch
from lerobot.configs import PreTrainedConfig
from test_traj_vla_qwen import make_geometry_context, make_tiny_paired_decoder
from test_traj_vla_qwen_direct_geometry import decoder_inputs

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.traj_vla_qwen.configuration_traj_vla_qwen import TrajVLAQwenConfig
from welding_path_vla.policies.traj_vla_qwen.modeling_traj_vla_qwen import TrajVLAQwenPolicy


def condition_decoder(enabled=True, checkpointed=False):
    """构造含两个几何 CA 层的小模型，不需要下载预训练权重。"""
    return make_tiny_paired_decoder(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        num_layers=4,
        geometry_attention_mode="direct",
        use_task_condition_attention=enabled,
        checkpoint_expert=checkpointed,
        checkpoint_qwen=checkpointed,
    )


def test_condition_config_and_prismatic_profile(tmp_path):
    """新实验从 Prismatic 单阶段训练；旧配置缺少字段时不创建新参数。"""
    config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        geometry_attention_mode="direct",
        use_task_condition_attention=True,
    )
    config.save_pretrained(tmp_path)
    assert PreTrainedConfig.from_pretrained(tmp_path).use_task_condition_attention
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    saved.pop("use_task_condition_attention")
    path.write_text(json.dumps(saved))
    assert not PreTrainedConfig.from_pretrained(tmp_path).use_task_condition_attention
    with pytest.raises(ValueError, match="requires direct"):
        TrajVLAQwenConfig(use_task_condition_attention=True)
    profile = AppConfig.load("configs/traj_vla_qwen_task_condition_a100.yaml")
    assert profile.policy.checkpoint is None
    assert not profile.training.resume
    assert profile.policy.parameters["load_prismatic_weights"]
    assert profile.policy.parameters["load_base_weights"]
    assert profile.policy.parameters["use_task_condition_attention"]
    assert not profile.policy.parameters["use_wrist_roi"]
    assert profile.policy.parameters["vision_lora_last_n_blocks"] == 4
    assert profile.training.steps == 200000
    assert profile.training.batch_size == 16
    assert profile.training.include_tasks == ["curve_plate"]


def test_disabled_path_has_unchanged_weights_and_outputs():
    """关闭开关时不消耗额外随机初始化，也不改变原 direct 权重键或数值。"""
    torch.manual_seed(81)
    old = make_tiny_paired_decoder(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        num_layers=4,
        geometry_attention_mode="direct",
    )
    torch.manual_seed(81)
    new = condition_decoder(enabled=False)
    assert not new.task_condition_kv
    assert new.state_dict().keys() == old.state_dict().keys()
    for name, value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
    new.load_state_dict(old.state_dict(), strict=True)
    args, geometry = decoder_inputs(), make_geometry_context()
    torch.testing.assert_close(
        new(*args, expert_context=geometry)[0][1],
        old(*args, expert_context=geometry)[0][1],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("checkpointed", [False, True])
def test_condition_has_immediate_gradients(checkpointed):
    """无需门控预热，首步损失就更新所有条件模块，并向文本/状态输入传播。"""
    model = condition_decoder(checkpointed=checkpointed)
    model.language_model.requires_grad_(False)
    args, geometry = decoder_inputs(batch=2), make_geometry_context(batch_size=2)
    args[3][0].requires_grad_()
    result = model(*args, expert_context=geometry)[0][1]
    (result * torch.randn_like(result)).sum().backward()
    assert args[3][0].grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if "task_condition_kv" in name:
                assert parameter.grad.abs().sum() > 0, name


def test_condition_cache_excludes_images_and_reuses_kv():
    """条件缓存只包含文本/状态；无动作泄漏，且多次去噪不重算条件投影。"""
    model = condition_decoder().eval()
    args, geometry = decoder_inputs(), make_geometry_context()
    geometry.condition_start = 1
    geometry.language_mask[:, 0] = False
    calls = []
    hooks = [
        module.register_forward_hook(
            lambda module, inputs, output: calls.append(inputs[0].shape[1])
        )
        for module in model.task_condition_kv.values()
    ]
    with torch.no_grad():
        full = model(*args, expert_context=geometry)[0]
        changed = list(args)
        changed[3] = [args[3][0], args[3][1] + 7]
        torch.testing.assert_close(full[0], model(*changed, expert_context=geometry)[0][0])
        calls.clear()
        cache = model(
            args[0][:, :3, :3],
            args[1][:, :3],
            None,
            [args[3][0], None],
            True,
            True,
            expert_context=geometry,
        )[1]
        for index in (1, 3):
            assert cache[index]["condition_key"].shape == (1, 2, 2, 8)
            assert cache[index]["condition_mask"].all()
        for _ in range(2):
            cached = model(
                args[0][:, 3:],
                args[1][:, 3:],
                cache,
                [None, args[3][1]],
                True,
                False,
            )[0][1]
            torch.testing.assert_close(cached, full[1], atol=1e-5, rtol=1e-5)
    assert calls == [2, 2]
    for hook in hooks:
        hook.remove()

    # 固定其他路径，仅改变条件缓存，确认新通路真实影响动作；遮挡位置不能泄漏。
    altered = deepcopy(cache)
    for layer in (1, 3):
        cache[layer]["condition_mask"][:, 0] = False
        altered[layer]["condition_mask"][:, 0] = False
        altered[layer]["condition_value"][:, :, 0] += 1000
    suffix = (args[0][:, 3:], args[1][:, 3:])
    expected = model(*suffix, cache, [None, args[3][1]], True, False)[0][1]
    actual = model(*suffix, altered, [None, args[3][1]], True, False)[0][1]
    torch.testing.assert_close(actual, expected)
    altered[1]["condition_value"][:, :, 1] += 1
    assert not torch.allclose(
        model(*suffix, altered, [None, args[3][1]], True, False)[0][1],
        expected,
    )


def test_condition_mixed_precision_inference():
    """PEFT 条件模块为 fp32、Expert 为 bf16 时，部署不依赖 autocast。"""
    model = condition_decoder().to(torch.bfloat16).eval()
    model.task_condition_kv.float()
    args = list(decoder_inputs())
    args[3] = [value.to(torch.bfloat16) for value in args[3]]
    output = model(*args, expert_context=make_geometry_context())[0][1]
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


def test_condition_compile_and_roi_composition():
    """条件路径可完整编译，且不会与独立腕部 ROI 模块混用参数或缓存。"""
    model = condition_decoder()
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    args, geometry = decoder_inputs(), make_geometry_context()
    torch.testing.assert_close(
        compiled(*args, expert_context=geometry)[0][1],
        model(*args, expert_context=geometry)[0][1],
    )
    combined = make_tiny_paired_decoder(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        num_layers=4,
        geometry_attention_mode="direct",
        use_task_condition_attention=True,
        use_wrist_roi=True,
    )
    missing, unexpected = combined.load_state_dict(model.state_dict(), strict=False)
    assert not unexpected and all("wrist_roi_" in name for name in missing)
    geometry.wrist_roi_tokens = torch.randn(1, 4, 12)
    geometry.wrist_roi_mask = torch.ones(1, 4, dtype=torch.bool)
    torch.testing.assert_close(
        combined(*args, expert_context=geometry)[0][1],
        model(*args, expert_context=geometry)[0][1],
        rtol=0,
        atol=0,
    )


def test_condition_peft_roundtrip(tmp_path):
    """逐层条件参数随 adapter 训练和保存，重载输出不依赖随机初始化。"""
    from peft import LoraConfig, PeftModel, get_peft_model

    class TinyPolicy(torch.nn.Module):
        """保留真实 decoder、Qwen 和 Expert 的别名路径，用于 PEFT 回归。"""

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            backbone = self.model.vlm_with_expert = torch.nn.Module()
            backbone.decoder = condition_decoder()
            backbone.qwen = torch.nn.Module()
            backbone.qwen.model = backbone.decoder.language_model
            backbone.expert = backbone.decoder.expert_model

        def forward(self, args, geometry):
            """复用真实 decoder 训练前向。"""
            return self.model.vlm_with_expert.decoder(*args, expert_context=geometry)[0][1]

    base = TinyPolicy()
    initial = deepcopy(base.state_dict())
    policy = TrajVLAQwenPolicy.__new__(TrajVLAQwenPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        geometry_attention_mode="direct",
        use_task_condition_attention=True,
        lora_target="qwen",
        num_vlm_layers=4,
        num_expert_layers=4,
    )
    targets = policy._get_default_peft_targets()
    assert "model.vlm_with_expert.decoder.task_condition_kv.1" in targets["modules_to_save"]
    # Qwen 共享别名在此 tiny 容器中先注册在 decoder，指向实际的可适配层。
    targets["target_modules"] = (
        r"model.vlm_with_expert.decoder.language_model.layers.\d+.self_attn.q_proj"
    )
    wrapped = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, **targets))
    args, geometry = decoder_inputs(), make_geometry_context()
    optimizer = torch.optim.AdamW((p for p in wrapped.parameters() if p.requires_grad), lr=0.01)
    wrapped(args, geometry).square().mean().backward()
    for name, parameter in wrapped.named_parameters():
        if "task_condition_kv" in name and parameter.requires_grad:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    optimizer.step()
    expected = wrapped(args, geometry).detach()
    wrapped.save_pretrained(tmp_path)
    restored_base = TinyPolicy()
    restored_base.load_state_dict(initial)
    restored = PeftModel.from_pretrained(restored_base, tmp_path)
    torch.testing.assert_close(restored(args, geometry), expected)
