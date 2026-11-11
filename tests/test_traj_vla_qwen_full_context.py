"""完整层后 Context 的动作可达性、梯度、缓存及 checkpoint 兼容性测试。"""

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


def context_decoder(enabled=True, checkpointed=False):
    """构造以几何 CA 结束的真实四层 decoder，重现末端 Qwen 输出断开的问题。"""
    return make_tiny_paired_decoder(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        num_layers=4,
        geometry_attention_mode="direct",
        use_full_context_attention=enabled,
        checkpoint_qwen=checkpointed,
        checkpoint_expert=checkpointed,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_terminal_qwen_output_changes_actions_only_when_connected(enabled):
    """修改末端 Qwen MLP 输出；开启分支后必须影响动作，关闭时保持原行为。"""
    torch.manual_seed(42)
    model = context_decoder(enabled).eval()
    args, geometry = decoder_inputs(), make_geometry_context()
    expected = model(*args, expert_context=geometry)[0][1]
    hook = model.language_model.layers[-1].mlp.register_forward_hook(
        lambda module, inputs, output: torch.zeros_like(output)
    )
    changed = model(*args, expert_context=geometry)[0][1]
    hook.remove()
    if enabled:
        assert (changed - expected).abs().max() > 1e-6
    else:
        torch.testing.assert_close(changed, expected, rtol=0, atol=0)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_full_context_immediate_gradient_through_frozen_terminal_qwen(checkpointed):
    """冻结 Qwen 权重时，首步动作梯度仍经过最后一层回到输入及新投影/门控。"""
    torch.manual_seed(43)
    model = context_decoder(checkpointed=checkpointed)
    model.language_model.requires_grad_(False)
    args, geometry = decoder_inputs(batch=2), make_geometry_context(batch_size=2)
    args[3][0].requires_grad_()
    reached = []

    def capture_gradient(module, inputs, output):
        """直接记录末端 MLP 输出收到的反向梯度，不依赖模块是否可训练。"""
        output.register_hook(lambda grad: reached.append(grad.detach()))

    hook = model.language_model.layers[-1].mlp.register_forward_hook(capture_gradient)
    output = model(*args, expert_context=geometry)[0][1]
    (output * torch.randn_like(output)).sum().backward()
    hook.remove()
    assert reached and all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in reached)
    assert args[3][0].grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if "full_context_kv" in name:
                assert parameter.grad.abs().sum() > 0, name


def test_full_context_cache_padding_and_action_isolation():
    """缓存读取全部有效 prefix、复用 K/V，并隔离 padding 和带噪动作。"""
    model = context_decoder().eval()
    args, geometry = list(decoder_inputs()), make_geometry_context()
    # 三个 prefix token 分别代表视觉、padding 文本和状态。
    args[0][:, 1, :] = False
    args[0][:, :, 1] = False
    calls = []
    hooks = [
        m.register_forward_hook(lambda m, i, o: calls.append(i[0].shape[1]))
        for m in model.full_context_kv.values()
    ]
    with torch.no_grad():
        expected = model(*args, expert_context=geometry)[0]
        changed = list(args)
        changed[3] = [args[3][0], args[3][1] + 7]
        # padding 查询行没有语义；只要求有效 Context 不读取带噪动作。
        torch.testing.assert_close(
            model(*changed, expert_context=geometry)[0][0][:, [0, 2]], expected[0][:, [0, 2]]
        )
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
        assert calls == [3, 3]
        for index in (1, 3):
            assert cache[index]["full_context_key"].shape == (1, 2, 3, 8)
            assert cache[index]["full_context_mask"].tolist() == [[True, False, True]]
        suffix = (args[0][:, 3:], args[1][:, 3:])
        for _ in range(2):
            actual = model(*suffix, cache, [None, args[3][1]], True, False)[0][1]
            torch.testing.assert_close(actual, expected[1], atol=1e-5, rtol=1e-5)
        assert calls == [3, 3]
        altered = deepcopy(cache)
        altered[3]["full_context_value"][:, :, 1] += 1000
        actual = model(*suffix, altered, [None, args[3][1]], True, False)[0][1]
        torch.testing.assert_close(actual, expected[1], atol=1e-5, rtol=1e-5)
        # 末端 CA 的视觉位置可直接改变动作，区别于仅文本/状态条件。
        altered[3]["full_context_value"][:, :, 0] += 1
        actual = model(*suffix, altered, [None, args[3][1]], True, False)[0][1]
        assert not torch.allclose(actual, expected[1])
    for hook in hooks:
        hook.remove()


def test_full_context_config_profile_and_disabled_checkpoint(tmp_path):
    """新开关可持久化；旧配置默认为关闭，原 direct 权重严格加载且数值一致。"""
    config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        geometry_attention_mode="direct",
        use_full_context_attention=True,
    )
    config.save_pretrained(tmp_path)
    assert PreTrainedConfig.from_pretrained(tmp_path).use_full_context_attention
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    saved.pop("use_full_context_attention")
    path.write_text(json.dumps(saved))
    assert not PreTrainedConfig.from_pretrained(tmp_path).use_full_context_attention
    with pytest.raises(ValueError, match="requires direct"):
        TrajVLAQwenConfig(use_full_context_attention=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrajVLAQwenConfig(
            attention_mode="cross_attn",
            use_geometry_branch=True,
            geometry_attention_mode="direct",
            use_full_context_attention=True,
            use_task_condition_attention=True,
        )
    profile = AppConfig.load("configs/traj_vla_qwen_full_context_a100.yaml")
    assert profile.policy.parameters["num_vlm_layers"] == 24
    assert profile.policy.parameters["num_expert_layers"] == 24
    assert profile.policy.parameters["use_full_context_attention"]
    assert not profile.policy.parameters["use_state_history"]
    assert not profile.policy.parameters["use_task_condition_attention"]
    assert profile.policy.checkpoint is None and not profile.training.resume
    assert profile.training.steps == 200000
    torch.manual_seed(81)
    old = make_tiny_paired_decoder(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        num_layers=4,
        geometry_attention_mode="direct",
    )
    torch.manual_seed(81)
    new = context_decoder(enabled=False)
    assert not new.full_context_kv
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


def test_full_context_bfloat16_inference_and_peft_roundtrip(tmp_path):
    """新投影/门控在 PEFT 中训练、保存和恢复，混合精度部署不依赖 autocast。"""
    from peft import LoraConfig, PeftModel, get_peft_model

    class TinyPolicy(torch.nn.Module):
        """保留生产模型的共享别名和实际 decoder 路径。"""

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            backbone = self.model.vlm_with_expert = torch.nn.Module()
            backbone.decoder = context_decoder()
            backbone.qwen = torch.nn.Module()
            backbone.qwen.model = backbone.decoder.language_model
            backbone.expert = backbone.decoder.expert_model

        def forward(self, args, geometry):
            """运行真实 decoder。"""
            return self.model.vlm_with_expert.decoder(*args, expert_context=geometry)[0][1]

    base = TinyPolicy()
    initial = deepcopy(base.state_dict())
    policy = TrajVLAQwenPolicy.__new__(TrajVLAQwenPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        geometry_attention_mode="direct",
        use_full_context_attention=True,
        lora_target="qwen",
        num_vlm_layers=4,
        num_expert_layers=4,
    )
    targets = policy._get_default_peft_targets()
    assert "model.vlm_with_expert.decoder.full_context_kv.3" in targets["modules_to_save"]
    targets["target_modules"] = (
        r"model.vlm_with_expert.decoder.language_model.layers.\d+.self_attn.q_proj"
    )
    wrapped = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, **targets))
    args, geometry = decoder_inputs(), make_geometry_context()
    optimizer = torch.optim.AdamW((p for p in wrapped.parameters() if p.requires_grad), lr=0.01)
    wrapped(args, geometry).square().mean().backward()
    for name, parameter in wrapped.named_parameters():
        if "full_context_kv" in name and parameter.requires_grad:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    optimizer.step()
    expected = wrapped(args, geometry).detach()
    wrapped.save_pretrained(tmp_path)
    restored_base = TinyPolicy()
    restored_base.load_state_dict(initial)
    restored = PeftModel.from_pretrained(restored_base, tmp_path)
    torch.testing.assert_close(restored(args, geometry), expected)

    model = context_decoder().to(torch.bfloat16).eval()
    model.full_context_kv.float()
    mixed = list(args)
    mixed[3] = [v.to(torch.bfloat16) for v in args[3]]
    output = model(*mixed, expert_context=geometry)[0][1]
    assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()
