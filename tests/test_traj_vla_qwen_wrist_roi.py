"""腕部原图裁剪、零门控兼容、局部梯度和去噪缓存回归。"""

from types import SimpleNamespace

import pytest
import torch
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from test_traj_vla_qwen import make_geometry_context, make_tiny_paired_decoder
from test_traj_vla_qwen_direct_geometry import decoder_inputs

from welding_path_vla.policies.traj_vla_qwen.configuration_traj_vla_qwen import TrajVLAQwenConfig
from welding_path_vla.policies.traj_vla_qwen.modeling_traj_vla_qwen import TrajVLAQwenPolicy
from welding_path_vla.policies.trajectory_vla.flow_matching import make_attention_masks


def roi_decoder(checkpointed=False):
    """四层 tiny decoder，覆盖两个独立局部 CA。"""
    return make_tiny_paired_decoder(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        num_layers=4,
        geometry_attention_mode="direct",
        use_wrist_roi=True,
        checkpoint_expert=checkpointed,
    )


def local_context():
    """两路全图与一路局部图，均使用 2×2 patch 网格。"""
    context = make_geometry_context()
    context.wrist_roi_tokens = torch.randn(1, 4, 12, requires_grad=True)
    context.wrist_roi_mask = torch.ones(1, 4, dtype=torch.bool)
    return context


def test_roi_config_and_original_pixel_crop(tmp_path):
    """裁剪原始像素而非缩小图；局部 mask 继承腕部 mask，旧配置默认关闭。"""
    assert not TrajVLAQwenConfig().use_wrist_roi
    with pytest.raises(ValueError, match="requires direct"):
        TrajVLAQwenConfig(use_wrist_roi=True)
    config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        geometry_attention_mode="direct",
        use_wrist_roi=True,
        wrist_roi_box=(0.25, 0.25, 0.75, 0.75),
        input_features={
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 448, 448))
        },
    )
    config.save_pretrained(tmp_path)
    assert PreTrainedConfig.from_pretrained(tmp_path).use_wrist_roi
    policy = TrajVLAQwenPolicy.__new__(TrajVLAQwenPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    image = torch.rand(1, 3, 448, 448)
    images, masks = policy.prepare_images(
        {
            "observation.images.wrist": image,
            "observation.images.wrist_padding_mask": torch.tensor([False]),
        }
    )
    assert len(images) == 2
    torch.testing.assert_close(images[-1], image[..., 112:336, 112:336], rtol=0, atol=0)
    assert not masks[-1].any()
    policy.config.use_wrist_roi = False
    old_images, _ = policy.prepare_images({"observation.images.wrist": image})
    torch.testing.assert_close(images[0], old_images[0], rtol=0, atol=0)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_roi_zero_gate_compatibility_and_gradients(checkpointed):
    """零门控严格保留旧动作；门控学习后动作损失到达局部特征与独立 K/V。"""
    model = roi_decoder(checkpointed)
    old = make_tiny_paired_decoder(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        num_layers=4,
        geometry_attention_mode="direct",
    )
    old.load_state_dict({k: v for k, v in model.state_dict().items() if "wrist_roi_" not in k})
    args, context = decoder_inputs(), local_context()
    torch.testing.assert_close(
        model(*args, expert_context=context)[0][1],
        old(*args, expert_context=context)[0][1],
        rtol=0,
        atol=0,
    )
    output = model(*args, expert_context=context)[0][1]
    (output * torch.randn_like(output)).sum().backward()
    assert all(m.gate.grad.abs() > 0 for m in model.wrist_roi_attention.values())
    model.zero_grad(set_to_none=True)
    for module in model.wrist_roi_attention.values():
        module.gate.data.fill_(0.2)
    output = model(*args, expert_context=context)[0][1]
    (output * torch.randn_like(output)).sum().backward()
    assert context.wrist_roi_tokens.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        if "wrist_roi_" in name:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_roi_masks_and_cached_inference():
    """逐层缓存局部 K/V；去噪复用时无需再次提供图像或执行局部投影。"""
    model, context, args = roi_decoder().eval(), local_context(), decoder_inputs()
    for module in model.wrist_roi_attention.values():
        module.gate.data.fill_(0.3)
    expected = model(*args, expert_context=context)[0][1]
    prefix = make_attention_masks(
        torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.bool)
    )
    cache = model(
        prefix, torch.arange(3)[None], None, [args[3][0], None], True, True, expert_context=context
    )[1]
    for index in (1, 3):
        assert cache[index]["wrist_roi_key"].shape == (1, 2, 4, 8)
    handles = [
        m.register_forward_hook(lambda *args: pytest.fail("recomputed local K/V"))
        for m in model.wrist_roi_kv.values()
    ]
    actual = model(args[0][:, -2:], args[1][:, -2:], cache, [None, args[3][1]], True, False)[0][1]
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    context.wrist_roi_mask.zero_()
    expected = model(*args, expert_context=context)[0][1]
    context.wrist_roi_tokens = context.wrist_roi_tokens + 1000
    torch.testing.assert_close(model(*args, expert_context=context)[0][1], expected)


def test_roi_peft_targets_cover_every_new_module():
    """新模块都纳入 PEFT 全量保存，避免优化器漏训或部署恢复丢失。"""
    policy = TrajVLAQwenPolicy.__new__(TrajVLAQwenPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        geometry_attention_mode="direct",
        use_wrist_roi=True,
        lora_target="qwen",
    )
    policy.model = SimpleNamespace(vlm_with_expert=SimpleNamespace(decoder=roi_decoder()))
    saved = policy._get_default_peft_targets()["modules_to_save"]
    assert "model.vlm_with_expert.decoder.wrist_roi_encoder" in saved
    assert "model.vlm_with_expert.decoder.wrist_roi_attention.1" in saved
    assert "model.vlm_with_expert.decoder.wrist_roi_kv.3" in saved


def test_roi_warm_start_copies_projections_without_sharing_weights():
    """局部投影从已学全图初始化，但不共享参数，也不改变全图或门控。"""
    model = roi_decoder()
    model.initialize_wrist_roi_from_global(1)
    local, full = model.wrist_roi_encoder, model.direct_geometry_encoder
    torch.testing.assert_close(local.patch_projection.weight, full.patch_projection.weight)
    assert local.patch_projection.weight is not full.patch_projection.weight
    torch.testing.assert_close(local.camera_embedding.weight[0], full.camera_embedding.weight[1])
    for index, projection in model.wrist_roi_kv.items():
        layer = model.expert_model.layers[int(index)].self_attn
        torch.testing.assert_close(
            projection.weight, torch.cat([layer.k_proj.weight, layer.v_proj.weight])
        )
        assert model.wrist_roi_attention[index].gate.item() == 0


def test_roi_mixed_precision_without_autocast():
    """部署的 bf16 Expert 与 fp32 新模块无需依赖训练 autocast 才能协同。"""
    model = roi_decoder().to(torch.bfloat16).eval()
    model.wrist_roi_encoder.float()
    model.wrist_roi_kv.float()
    args = list(decoder_inputs())
    args[3] = [value.to(torch.bfloat16) for value in args[3]]
    output = model(*args, expert_context=local_context())[0][1]
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


def test_roi_does_not_add_semantic_tokens(monkeypatch):
    """全图仍各执行融合编码，局部只执行一次 DINO，且不进入 Qwen prefix。"""
    from welding_path_vla.policies.traj_vla_qwen import modeling_traj_vla_qwen

    class TinyBackbone(torch.nn.Module):
        """用查表记录融合路径与 DINO-only 路径的调用。"""

        language_hidden_size = 8
        expert_hidden_size = 8

        def __init__(self, config):
            super().__init__()
            self.tokens = torch.nn.Embedding(16, 8)
            self.vision_encoder = SimpleNamespace(forward_dino=self.local_features)
            self.fused_calls = 0
            self.local_calls = 0

        def embed_language_tokens(self, tokens):
            return self.tokens(tokens)

        def embed_image_with_geometry(self, image):
            self.fused_calls += 1
            return self.tokens(image), self.tokens(image)

        def local_features(self, image):
            self.local_calls += 1
            return self.tokens(image)

    monkeypatch.setattr(modeling_traj_vla_qwen, "PrismaticQwenWithExpert", TinyBackbone)
    config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        geometry_attention_mode="direct",
        use_wrist_roi=True,
    )
    model = modeling_traj_vla_qwen.QwenTrajectoryFlowModel(config)
    context = model.encode_context(
        [torch.tensor([[0, 1]]), torch.tensor([[2, 3]]), torch.tensor([[4, 5]])],
        [torch.tensor([True]), torch.tensor([False]), torch.tensor([False])],
        torch.tensor([[6, 7]]),
        torch.tensor([[True, True]]),
        torch.randn(1, 32),
    )
    assert context.embeddings.shape == (1, 7, 8)
    assert context.expert_context.patch_tokens.shape == (1, 2, 2, 8)
    assert context.expert_context.wrist_roi_tokens.shape == (1, 2, 8)
    assert not context.expert_context.wrist_roi_mask.any()
    assert model.vlm_with_expert.fused_calls == 2
    assert model.vlm_with_expert.local_calls == 1
