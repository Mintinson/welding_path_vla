from copy import deepcopy
from pathlib import Path

import pytest
import torch
from lerobot.configs import PreTrainedConfig

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.factory import get_policy_pipeline
from welding_path_vla.policies.spec import TRAJ_VLA_QWEN
from welding_path_vla.policies.traj_vla_qwen.configuration_traj_vla_qwen import (
    TrajVLAQwenConfig,
)
from welding_path_vla.policies.traj_vla_qwen.modeling_traj_vla_qwen import (
    TrajVLAQwenPolicy,
)
from welding_path_vla.policies.traj_vla_qwen.prismatic import (
    DenseGeometryContext,
    PrismaticVisionEncoder,
    SeamQueryResampler,
    SpatialTokenMerger,
)
from welding_path_vla.policies.traj_vla_qwen.processor_traj_vla_qwen import (
    QwenPromptProcessorStep,
)
from welding_path_vla.policies.traj_vla_qwen.qwen_with_expert import (
    InterleavedSACADecoder,
    PairedLayerDecoder,
    Qwen25DecoderAdapter,
)
from welding_path_vla.policies.trajectory_vla.flow_matching import make_attention_masks


def make_tiny_paired_decoder(
    checkpoint_qwen: bool = False,
    checkpoint_expert: bool = False,
    attention_mode: str = "self_attn",
    use_geometry_branch: bool = False,
    num_layers: int = 2,
) -> PairedLayerDecoder:
    """构造无需下载权重的两层 Qwen2.5 双流网络。"""
    transformers = pytest.importorskip("transformers")
    Qwen2Config = transformers.Qwen2Config
    Qwen2Model = transformers.Qwen2Model
    language_config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
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
    common = {
        "language_model": language,
        "expert_model": expert,
        "adapter": Qwen25DecoderAdapter(),
        "checkpoint_qwen": checkpoint_qwen,
        "checkpoint_expert": checkpoint_expert,
    }
    if attention_mode == "cross_attn":
        return InterleavedSACADecoder(
            **common,
            self_attn_every_n_layers=2,
            geometry_input_size=12 if use_geometry_branch else None,
            geometry_num_cameras=2,
            geometry_num_queries=4,
            geometry_num_heads=4,
        )
    return PairedLayerDecoder(**common)


def make_geometry_context(batch_size: int = 1) -> DenseGeometryContext:
    """构造两相机、每相机四个 patch 的 tiny 几何上下文。"""
    return DenseGeometryContext(
        patch_tokens=torch.randn(batch_size, 2, 4, 12),
        patch_mask=torch.tensor([[True, True]] * batch_size)[:, :, None].expand(-1, -1, 4).clone(),
        language_mask=torch.tensor([[True, True, False]] * batch_size),
        state_mask=torch.tensor([[False, False, True]] * batch_size),
    )


def test_qwen_config_and_policy_are_registered() -> None:
    """模块化 YAML 应能选择本地 Qwen Policy。"""
    config = AppConfig.load("configs/traj_vla_qwen.yaml")
    assert config.policy.family == "traj_vla_qwen"
    assert config.policy.parameters["num_vlm_layers"] == 16
    assert config.policy.parameters["num_expert_layers"] == 16
    assert config.policy.parameters["attention_mode"] == "cross_attn"
    assert config.policy.parameters["self_attn_every_n_layers"] == 2
    assert config.policy.parameters["frozen_vision_dtype"] == "float32"
    assert not config.policy.parameters["gradient_checkpointing_qwen"]
    assert not config.policy.parameters["gradient_checkpointing_expert"]
    assert config.policy.parameters["use_geometry_branch"] is True
    assert config.policy.parameters["geometry_num_queries"] == 16
    assert config.policy.parameters["geometry_num_heads"] == 8
    assert get_policy_pipeline("traj_vla_qwen").spec is TRAJ_VLA_QWEN


def test_qwen_config_rejects_unpaired_depth() -> None:
    """第一版必须维持一一配对的 Qwen 与专家深度。"""
    try:
        TrajVLAQwenConfig(num_vlm_layers=16, num_expert_layers=8)
    except ValueError as error:
        assert "equal" in str(error)
    else:
        raise AssertionError("unpaired decoder depth was accepted")


def test_qwen_attention_config_preserves_old_checkpoint_default() -> None:
    """缺少新字段的第一版 checkpoint 应继续使用逐层联合注意力。"""
    assert TrajVLAQwenConfig().attention_mode == "self_attn"
    with pytest.raises(ValueError, match="attention_mode"):
        TrajVLAQwenConfig(attention_mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        TrajVLAQwenConfig(self_attn_every_n_layers=0)
    with pytest.raises(ValueError, match="cross_attn"):
        TrajVLAQwenConfig(use_geometry_branch=True)
    with pytest.raises(ValueError, match="cross-attention"):
        TrajVLAQwenConfig(
            use_geometry_branch=True,
            attention_mode="cross_attn",
            self_attn_every_n_layers=1,
        )


def test_geometry_config_round_trip(tmp_path: Path) -> None:
    """几何结构字段应随 LeRobot checkpoint 配置稳定保存和恢复。"""
    config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        geometry_num_queries=12,
        geometry_num_heads=4,
        train_geometry_resampler=False,
        device="cpu",
    )
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(restored, TrajVLAQwenConfig)
    assert restored.use_geometry_branch
    assert restored.geometry_num_queries == 12
    assert restored.geometry_num_heads == 4
    assert not restored.train_geometry_resampler


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


def test_seam_query_resampler_conditions_and_masks_dense_patches() -> None:
    """Resampler 应输出固定 query，并完全屏蔽缺失相机。"""
    torch.manual_seed(3)
    resampler = SeamQueryResampler(12, 32, 24, 2, num_heads=4)
    context = torch.randn(2, 3, 32)
    geometry = make_geometry_context(batch_size=2)
    geometry.patch_mask[1, 1] = False
    output = resampler(context, geometry)

    assert output.patch_tokens.shape == (2, 2, 4, 24)
    assert output.latent_tokens.shape == (2, 16, 24)
    assert output.readout_tokens.shape == (2, 3, 24)
    assert output.attention_weights.shape == (2, 19, 2, 4)
    assert torch.allclose(output.attention_weights[1, :, 1], torch.zeros(19, 4))
    assert torch.allclose(output.attention_weights.sum(dim=(2, 3)), torch.ones(2, 19))

    changed = make_geometry_context(batch_size=2)
    changed.patch_tokens = geometry.patch_tokens
    changed.patch_mask = geometry.patch_mask.clone()
    changed.language_mask = torch.tensor([[True, False, False], [False, True, False]])
    conditioned = resampler(context, changed)
    assert not torch.allclose(output.latent_tokens, conditioned.latent_tokens)

    changed.state_mask = torch.tensor([[False, True, False], [True, False, False]])
    state_conditioned = resampler(context, changed)
    assert not torch.allclose(conditioned.latent_tokens, state_conditioned.latent_tokens)


class FakePatchFeaturizer(torch.nn.Module):
    """记录调用次数并返回固定 patch 的最小 ViT 替身。"""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.embed_dim = width
        self.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = 0

    def get_intermediate_layers(self, image: torch.Tensor, n: set[int]) -> tuple[torch.Tensor]:
        """返回与 224 像素 ViT/14 一致的 256 个 patch。"""
        self.calls += 1
        tokens = torch.ones(image.shape[0], 256, self.embed_dim, device=image.device)
        return (tokens * self.weight,)


def test_prismatic_encoder_splits_dense_dino_without_second_forward() -> None:
    """语义融合和稠密支路必须共享同一次 DINO 前向。"""
    encoder = PrismaticVisionEncoder.__new__(PrismaticVisionEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.dino_featurizer = FakePatchFeaturizer(8)
    encoder.siglip_featurizer = FakePatchFeaturizer(6)
    encoder.dino_layer = 0
    encoder.siglip_layer = 0
    encoder.register_buffer("dino_mean", torch.zeros(1, 3, 1, 1), persistent=False)
    encoder.register_buffer("dino_std", torch.ones(1, 3, 1, 1), persistent=False)

    dino, fused = encoder.forward_features(torch.zeros(2, 3, 224, 224))
    merged = SpatialTokenMerger(fused.shape[-1], grid_size=16, factor=2)(fused)
    assert dino.shape == (2, 256, 8)
    assert fused.shape == (2, 256, 14)
    assert merged.shape == (2, 64, 14)
    assert encoder.dino_featurizer.calls == 1
    assert encoder.siglip_featurizer.calls == 1


def test_qwen_prompt_matches_prismatic_pretraining_format() -> None:
    """任务文本必须使用 MiniVLA 的 Qwen 单轮 chat 包装。"""
    prompt = QwenPromptProcessorStep().complementary_data({"task": "Weld the seam."})["task"]
    assert prompt.startswith("<|im_start|>system\nYou are Qwen")
    assert "<|im_start|>user\nWeld the seam.<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize("attention_mode", ["self_attn", "cross_attn"])
def test_paired_attention_is_asymmetric_and_cache_equivalent(attention_mode: str) -> None:
    """两种 Expert 结构都应保持非对称可见性和 KV cache 等价。"""
    torch.manual_seed(7)
    decoder = make_tiny_paired_decoder(attention_mode=attention_mode).eval()
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


def test_interleaved_decoder_alternates_sa_and_ca_projections() -> None:
    """Interleaved 模式应保留 SA 层，并仅改造 CA 层的 Expert K/V。"""
    decoder = make_tiny_paired_decoder(attention_mode="cross_attn")
    assert isinstance(decoder, InterleavedSACADecoder)
    assert decoder.uses_self_attention(0)
    assert not decoder.uses_self_attention(1)
    assert decoder.expert_model.layers[0].self_attn.k_proj.in_features == 24
    assert decoder.expert_model.layers[1].self_attn.k_proj.in_features == 16


def test_geometry_decoder_is_asymmetric_and_cache_equivalent() -> None:
    """几何 CA 只更新 Expert，且缓存前向应与完整前向等价。"""
    torch.manual_seed(17)
    decoder = make_tiny_paired_decoder(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        num_layers=4,
    ).eval()
    assert isinstance(decoder, InterleavedSACADecoder)
    assert decoder.expert_model.layers[1].self_attn.k_proj.in_features == 24
    context = torch.randn(1, 3, 32)
    action = torch.randn(1, 2, 24)
    geometry = make_geometry_context()
    padding = torch.ones(1, 5, dtype=torch.bool)
    blocks = torch.tensor([[False, False, False, True, True]])
    mask = make_attention_masks(padding, blocks)
    positions = torch.arange(5).unsqueeze(0)

    full, _, auxiliary = decoder(
        mask,
        positions,
        None,
        [context, action],
        False,
        False,
        expert_context=geometry,
    )
    changed_geometry = make_geometry_context()
    changed, _, _ = decoder(
        mask,
        positions,
        None,
        [context, action],
        False,
        False,
        expert_context=changed_geometry,
    )
    assert torch.allclose(full[0], changed[0], atol=1e-5)
    assert not torch.allclose(full[1], changed[1])
    assert auxiliary["geometry.latent_tokens"].shape == (1, 4, 24)
    assert auxiliary["geometry.readout_tokens"].shape == (1, 3, 24)

    changed_action, _, _ = decoder(
        mask,
        positions,
        None,
        [context, action + 10],
        False,
        False,
        expert_context=geometry,
    )
    assert torch.allclose(full[0], changed_action[0], atol=1e-5)

    resampler_calls: list[None] = []
    geometry_projection_calls: list[None] = []
    decoder.geometry_resampler.register_forward_hook(
        lambda module, inputs, output: resampler_calls.append(None)
    )
    decoder.expert_model.layers[1].self_attn.k_proj.register_forward_hook(
        lambda module, inputs, output: geometry_projection_calls.append(None)
    )
    prefix_mask = torch.ones(1, 3, 3, dtype=torch.bool)
    _, cache, _ = decoder(
        prefix_mask,
        positions[:, :3],
        None,
        [context, None],
        True,
        True,
        expert_context=geometry,
    )
    assert cache is not None
    calls_after_cache = (len(resampler_calls), len(geometry_projection_calls))
    cached, _, cached_auxiliary = decoder(
        mask[:, 3:, :],
        positions[:, 3:],
        cache,
        [None, action],
        True,
        False,
    )
    assert torch.allclose(full[1], cached[1], atol=1e-5)
    assert cached_auxiliary == {}
    assert (len(resampler_calls), len(geometry_projection_calls)) == calls_after_cache

    restored = make_tiny_paired_decoder(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        num_layers=4,
    ).eval()
    restored.load_state_dict(decoder.state_dict())
    round_trip, _, _ = restored(
        mask,
        positions,
        None,
        [context, action],
        False,
        False,
        expert_context=geometry,
    )
    assert torch.allclose(full[1], round_trip[1], atol=1e-5)


@pytest.mark.parametrize("attention_mode", ["self_attn", "cross_attn"])
def test_paired_layer_checkpointing_preserves_gradients(attention_mode: str) -> None:
    """两种 Expert 结构的 checkpoint 重算都应保持输出和梯度。"""
    torch.manual_seed(11)
    direct = make_tiny_paired_decoder(attention_mode=attention_mode).train()
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

    policy.config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
    )
    targets = policy._get_default_peft_targets()
    assert "model.vlm_with_expert.decoder.geometry_resampler" in targets["modules_to_save"]
