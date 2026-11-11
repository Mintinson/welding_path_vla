"""完整 patch 几何主通路的梯度、空间信息、缓存和配置回归测试。"""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from lerobot.configs import PreTrainedConfig
from test_traj_vla_qwen import make_geometry_context, make_tiny_paired_decoder

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.traj_vla_qwen.configuration_traj_vla_qwen import TrajVLAQwenConfig
from welding_path_vla.policies.traj_vla_qwen.modeling_traj_vla_qwen import TrajVLAQwenPolicy
from welding_path_vla.policies.traj_vla_qwen.prismatic import (
    DenseGeometryContext,
    DirectGeometryEncoder,
    SpatialTokenMerger,
)
from welding_path_vla.policies.traj_vla_qwen.qwen_with_expert import attend_dense_geometry
from welding_path_vla.policies.trajectory_vla.flow_matching import make_attention_masks


def make_direct_decoder(checkpointed: bool = False):
    """构造四层网络，覆盖两个几何 CA 层以及 SA/CA 的交替。"""
    return make_tiny_paired_decoder(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        num_layers=4,
        geometry_attention_mode="direct",
        checkpoint_qwen=checkpointed,
        checkpoint_expert=checkpointed,
    )


def decoder_inputs(batch: int = 1):
    """三枚 Context token 和两枚动作 token；相机 patch 独立输入。"""
    mask = make_attention_masks(
        torch.ones(batch, 5, dtype=torch.bool),
        torch.tensor([[False, False, False, True, True]]).expand(batch, -1),
    )
    return (
        mask,
        torch.arange(5)[None].expand(batch, -1),
        None,
        [
            torch.randn(batch, 3, 32),
            torch.randn(batch, 2, 24),
        ],
        False,
        False,
    )


def test_direct_config_profile_and_legacy_default(tmp_path: Path) -> None:
    """新模式可持久化；缺少字段的 checkpoint 使用原 latent 路径。"""
    config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        geometry_attention_mode="direct",
    )
    config.save_pretrained(tmp_path)
    assert PreTrainedConfig.from_pretrained(tmp_path).geometry_attention_mode == "direct"
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    saved.pop("geometry_attention_mode")
    saved.pop("train_geometry_encoder")
    path.write_text(json.dumps(saved))
    assert PreTrainedConfig.from_pretrained(tmp_path).geometry_attention_mode == "latent"
    for option in (
        "use_motion_latent",
        "use_geometry_grounding",
        "geometry_readout_from_latents",
        "use_dense_geometry_attention",
    ):
        with pytest.raises(ValueError, match="direct geometry"):
            TrajVLAQwenConfig(
                attention_mode="cross_attn",
                use_geometry_branch=True,
                geometry_attention_mode="direct",
                **{option: True},
            )
    with pytest.raises(ValueError, match="requires use_geometry_branch"):
        TrajVLAQwenConfig(geometry_attention_mode="direct")
    with pytest.raises(ValueError, match="geometry_attention_mode"):
        TrajVLAQwenConfig(geometry_attention_mode="unknown")
    profile = AppConfig.load("configs/traj_vla_qwen_direct_geometry_a100.yaml")
    assert profile.policy.parameters["geometry_attention_mode"] == "direct"
    assert profile.policy.parameters["token_merge_factor"] == 1
    assert not profile.policy.parameters["train_token_merger"]
    assert profile.policy.checkpoint is None
    assert not profile.training.resume
    assert profile.training.image_aug
    assert profile.training.include_tasks == ["curve_plate"]
    assert profile.training.steps == 100000
    assert profile.deployment.action_steps == 8


def test_no_merge_preserves_every_fused_patch() -> None:
    """factor=1 的冻结单位投影保留 256 个 token 及输入梯度，而非局部平均。"""
    merger = SpatialTokenMerger(14, grid_size=16, factor=1).requires_grad_(False)
    patches = torch.randn(2, 256, 14, requires_grad=True)
    actual = merger(patches)
    torch.testing.assert_close(actual, patches, rtol=0, atol=0)
    actual.sum().backward()
    assert torch.equal(patches.grad, torch.ones_like(patches))
    assert merger.projection.weight.grad is None


def test_geometry_encoder_has_spatial_and_camera_identity() -> None:
    """相同内容的 patch 仍可区分二维网格位置及相机，不依赖焊缝坐标标签。"""
    encoder = DirectGeometryEncoder(12, 24, 2, 2)
    geometry = make_geometry_context()
    geometry.patch_tokens.zero_()
    memory = encoder(geometry)
    assert memory.tokens.shape == (1, 8, 24)
    assert torch.equal(memory.mask, geometry.patch_mask.flatten(1, 2))
    assert not torch.allclose(memory.tokens[:, 0], memory.tokens[:, 1])
    assert not torch.allclose(memory.tokens[:, 0], memory.tokens[:, 2])
    assert not torch.allclose(memory.tokens[:, 0], memory.tokens[:, 4])
    torch.testing.assert_close(
        encoder.patch_coordinates,
        torch.tensor([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]),
    )
    with pytest.raises(ValueError, match="patch count"):
        DirectGeometryEncoder(12, 24, 2, 3)(geometry)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_direct_path_has_immediate_gradients_and_no_latent(checkpointed: bool) -> None:
    """无门控预热，第一步动作损失就更新全部几何编码参数和每个 CA 的专用 K/V。"""
    torch.manual_seed(37)
    model = make_direct_decoder(checkpointed)
    # 实际配置冻结 Qwen；此处检查所有可训练 Expert 参数都参与反向，避免 DDP 未使用参数。
    model.language_model.requires_grad_(False)
    geometry = make_geometry_context()
    geometry.patch_tokens.requires_grad_()
    output = model(*decoder_inputs(), expert_context=geometry)[0][1]
    (output * torch.randn_like(output)).sum().backward()
    assert model.geometry_resampler is None
    assert not model.dense_geometry_attention
    assert not any("latent_queries" in key or key.endswith(".gate") for key in model.state_dict())
    assert geometry.patch_tokens.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    for parameter in model.direct_geometry_encoder.parameters():
        assert parameter.grad.abs().sum() > 0
    for index in (1, 3):
        layer = model.expert_model.layers[index].self_attn
        assert layer.k_proj.in_features == 24
        assert layer.k_proj.weight.grad.abs().sum() > 0
        assert layer.v_proj.weight.grad.abs().sum() > 0


def test_direct_cache_matches_full_and_encodes_geometry_once() -> None:
    """只在 prefill 编码几何并缓存完整 mask/K/V；多次动作去噪不重新读取图像。"""
    model = make_direct_decoder().eval()
    args, geometry = decoder_inputs(), make_geometry_context()
    geometry.patch_mask[:, 1] = False
    full = model(*args, expert_context=geometry)[0]
    changed = list(args)
    changed[3] = [args[3][0], args[3][1] + 3]
    torch.testing.assert_close(full[0], model(*changed, expert_context=geometry)[0][0])
    calls = []
    model.direct_geometry_encoder.register_forward_hook(lambda *unused: calls.append(1))
    _, cache, _ = model(
        args[0][:, :3, :3],
        args[1][:, :3],
        None,
        [args[3][0], None],
        True,
        True,
        expert_context=geometry,
    )
    for index in (1, 3):
        assert cache[index]["key_states"].shape[2] == 8
        assert torch.equal(cache[index]["geometry_patch_mask"], geometry.patch_mask.flatten(1, 2))
    for _ in range(2):
        cached = model(args[0][:, 3:], args[1][:, 3:], cache, [None, args[3][1]], True, False)[0][1]
        torch.testing.assert_close(cached, full[1], atol=1e-5, rtol=1e-5)
    assert len(calls) == 1


def test_direct_attention_masks_missing_cameras() -> None:
    """遮挡相机不能影响结果；全遮挡的几何输出严格为零且无 NaN。"""
    query, key, value = torch.randn(2, 4, 2, 8), torch.randn(2, 2, 8, 8), torch.randn(2, 2, 8, 8)
    mask = torch.tensor([[True] * 4 + [False] * 4, [False] * 8])
    output = attend_dense_geometry(query, key, value, mask)
    assert torch.count_nonzero(output[1]) == 0
    key[:, :, 4:] += 100
    value[:, :, 4:] -= 100
    torch.testing.assert_close(attend_dense_geometry(query, key, value, mask), output)
    model, args, geometry = make_direct_decoder().eval(), decoder_inputs(), make_geometry_context()
    geometry.patch_mask[:, 1] = False
    expected = model(*args, expert_context=geometry)[0][1]
    geometry.patch_tokens[:, 1] += 1000
    torch.testing.assert_close(model(*args, expert_context=geometry)[0][1], expected)


def test_direct_compile_and_weights_roundtrip() -> None:
    """direct 路径可整图编译、严格加载权重，且反向与 eager 一致。"""
    eager = make_direct_decoder()
    restored = make_direct_decoder()
    restored.load_state_dict(eager.state_dict(), strict=True)
    compiled = torch.compile(restored, backend="aot_eager", fullgraph=True)
    args, geometry = decoder_inputs(), make_geometry_context()
    expected = eager(*args, expert_context=geometry)[0][1]
    actual = compiled(*args, expert_context=geometry)[0][1]
    torch.testing.assert_close(actual, expected)
    target = torch.randn_like(expected)
    (expected * target).sum().backward()
    (actual * target).sum().backward()
    for name, parameter in eager.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, dict(restored.named_parameters())[name].grad)


def test_direct_peft_targets_save_encoder_not_resampler() -> None:
    """后续开启 Qwen LoRA 时，应保存新的几何编码器，不引用不存在的 Resampler。"""
    policy = TrajVLAQwenPolicy.__new__(TrajVLAQwenPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        geometry_attention_mode="direct",
        lora_target="qwen",
        train_token_merger=False,
    )
    targets = policy._get_default_peft_targets()["modules_to_save"]
    assert "model.vlm_with_expert.decoder.direct_geometry_encoder" in targets
    assert not any("geometry_resampler" in target for target in targets)
    assert not any("token_merger" in target for target in targets)
    assert "model.vlm_with_expert.decoder.expert_model" in targets


def test_vision_lora_profile_and_legacy_config(tmp_path: Path) -> None:
    """新实验只切换视觉适配；旧配置缺少新字段时保留四层和 expert 默认值。"""
    from lerobot.configs.default import PeftConfig

    profile = AppConfig.load("configs/traj_vla_qwen_direct_geometry_vision_lora_a100.yaml")
    assert PeftConfig(**profile.training.peft).r == 32
    assert profile.policy.parameters["lora_target"] == "vision"
    assert profile.policy.parameters["vision_lora_last_n_blocks"] == 4
    assert profile.policy.parameters["geometry_attention_mode"] == "direct"
    assert profile.policy.parameters["token_merge_factor"] == 1
    assert not profile.policy.parameters["train_token_merger"]
    assert not profile.policy.parameters["train_language_model"]
    assert profile.training.peft["r"] == 32
    assert profile.training.include_tasks == ["curve_plate"]
    assert not TrajVLAQwenConfig().vision_lora_include_mlp
    expanded = AppConfig.load("configs/traj_vla_qwen_direct_geometry_vision_lora_plus_a100.yaml")
    assert expanded.policy.parameters["vision_lora_last_n_blocks"] == 8
    assert expanded.policy.parameters["vision_lora_include_mlp"]
    assert expanded.policy.parameters["lora_target"] == "vision"
    assert not expanded.policy.parameters["train_language_model"]
    assert expanded.training.peft == profile.training.peft
    assert expanded.training.output_dir != profile.training.output_dir
    config = TrajVLAQwenConfig(
        lora_target="vision", vision_lora_last_n_blocks=2, vision_lora_include_mlp=True
    )
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.lora_target == "vision"
    assert restored.vision_lora_last_n_blocks == 2
    assert restored.vision_lora_include_mlp
    saved = json.loads((tmp_path / "config.json").read_text())
    saved.pop("vision_lora_last_n_blocks")
    saved.pop("lora_target")
    saved.pop("vision_lora_include_mlp")
    (tmp_path / "config.json").write_text(json.dumps(saved))
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.lora_target == "expert"
    assert restored.vision_lora_last_n_blocks == 4
    assert not restored.vision_lora_include_mlp
    with pytest.raises(ValueError, match="vision_lora_last_n_blocks"):
        TrajVLAQwenConfig(vision_lora_last_n_blocks=0)


@pytest.mark.parametrize("include_mlp", [False, True])
def test_vision_lora_gradients_and_checkpoint(tmp_path: Path, include_mlp: bool) -> None:
    """验证视觉 LoRA、直接几何和共享 Expert 都更新，而基座冻结且可完整恢复。"""
    from peft import LoraConfig, PeftModel, get_peft_model

    class TinyPolicy(torch.nn.Module):
        """复现真实视觉命名、直接几何和 decoder 的 Expert 别名调用链。"""

        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            backbone = torch.nn.Module()
            self.model.vlm_with_expert = backbone
            vision = torch.nn.Module()
            backbone.vision_encoder = vision
            for name in ("dino", "siglip"):
                encoder = torch.nn.Module()
                encoder.blocks = torch.nn.ModuleList()
                for _ in range(4):
                    block = torch.nn.Module()
                    block.attn = torch.nn.Module()
                    block.attn.qkv = torch.nn.Linear(4, 4)
                    block.attn.proj = torch.nn.Linear(4, 4)
                    block.mlp = torch.nn.Module()
                    block.mlp.fc1 = torch.nn.Linear(4, 8)
                    block.mlp.fc2 = torch.nn.Linear(8, 4)
                    encoder.blocks.append(block)
                setattr(vision, f"{name}_featurizer", encoder)
                setattr(vision, f"{name}_layer", 2)
            backbone.qwen = torch.nn.Linear(4, 4)
            backbone.expert = torch.nn.Module()
            backbone.expert.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
            backbone.decoder = torch.nn.Module()
            backbone.decoder.expert_model = backbone.expert
            backbone.decoder.direct_geometry_encoder = DirectGeometryEncoder(4, 4, 2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """两塔特征经几何 memory 进入 decoder，最后一层不参与视觉输出。"""
            backbone = self.model.vlm_with_expert
            features = []
            for name in ("dino", "siglip"):
                hidden = inputs
                for block in getattr(backbone.vision_encoder, f"{name}_featurizer").blocks[:3]:
                    hidden = hidden + block.attn.proj(block.attn.qkv(hidden).tanh())
                    hidden = hidden + block.mlp.fc2(block.mlp.fc1(hidden).tanh())
                features.append(hidden)
            patches = torch.stack(features).mean(0)
            mask = torch.ones(patches.shape[:3], dtype=torch.bool)
            geometry = DenseGeometryContext(patches, mask, mask[:, 0], mask[:, 0])
            memory = backbone.decoder.direct_geometry_encoder(geometry)
            return backbone.decoder.expert_model.layers[0](memory.tokens).mean(1) + backbone.qwen(
                inputs.mean((1, 2))
            )

    torch.manual_seed(7)
    base = TinyPolicy()
    initial = deepcopy(base.state_dict())
    policy = TrajVLAQwenPolicy.__new__(TrajVLAQwenPolicy)
    torch.nn.Module.__init__(policy)
    policy.model = base.model
    policy.config = TrajVLAQwenConfig(
        lora_target="vision",
        vision_lora_last_n_blocks=2,
        vision_lora_include_mlp=include_mlp,
        attention_mode="cross_attn",
        use_geometry_branch=True,
        geometry_attention_mode="direct",
        train_token_merger=False,
    )
    targets = policy._get_default_peft_targets()
    wrapped = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, **targets))
    trainable = {n: p for n, p in wrapped.named_parameters() if p.requires_grad}
    assert not any("qwen" in n for n in trainable)
    assert not any("blocks.0." in n or "blocks.3." in n for n in trainable)
    assert any(".mlp." in n for n in trainable) == include_mlp
    assert sum("lora_B" in n for n in trainable) == (16 if include_mlp else 8)
    assert any("direct_geometry_encoder.modules_to_save.default" in n for n in trainable)
    assert any("expert_model.modules_to_save.default" in n for n in trainable)
    before = {n: p.detach().clone() for n, p in trainable.items()}
    inputs = torch.randn(2, 2, 4, 4)
    optimizer = torch.optim.AdamW(trainable.values(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        wrapped(inputs).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
        optimizer.step()
    assert all(not torch.equal(p, before[n]) for n, p in trainable.items() if "lora_B" in n)
    expected = wrapped(inputs).detach()
    wrapped.save_pretrained(tmp_path)
    restored_base = TinyPolicy()
    restored_base.load_state_dict(initial)
    restored = PeftModel.from_pretrained(restored_base, tmp_path)
    torch.testing.assert_close(restored(inputs), expected)
