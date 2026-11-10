import json
import re
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from lerobot.configs import PreTrainedConfig

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.factory import get_policy_pipeline
from welding_path_vla.policies.lerobot_training import require_grounding_checkpoint
from welding_path_vla.policies.spec import TRAJ_VLA_QWEN
from welding_path_vla.policies.traj_vla_qwen.configuration_traj_vla_qwen import (
    TrajVLAQwenConfig,
)
from welding_path_vla.policies.traj_vla_qwen.geometry_grounding import (
    ORIENTATION_TARGET,
    ORIENTATION_VALID,
    SEAM_LABELS,
    SEAM_VALID,
    TANGENT_TARGET,
    TANGENT_VALID,
    GeometryGroundingHeads,
    GeometryGroundingTargetProcessorStep,
    geometry_grounding_losses,
)
from welding_path_vla.policies.traj_vla_qwen.modeling_traj_vla_qwen import (
    QwenTrajectoryFlowModel,
    TrajVLAQwenPolicy,
)
from welding_path_vla.policies.traj_vla_qwen.motion_latent import (
    ConditionalMotionLatent,
    diagonal_gaussian_kl,
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
    GatedDenseGeometryAttention,
    InterleavedSACADecoder,
    PairedLayerDecoder,
    Qwen25DecoderAdapter,
    make_expert_decoder,
)
from welding_path_vla.policies.trajectory_vla.flow_matching import (
    TrajectoryFlowModel,
    make_attention_masks,
)


def make_tiny_paired_decoder(
    checkpoint_qwen: bool = False,
    checkpoint_expert: bool = False,
    attention_mode: str = "self_attn",
    use_geometry_branch: bool = False,
    use_motion_latent: bool = False,
    num_layers: int = 2,
    geometry_readout_from_latents: bool = False,
    dense_geometry_attention_layers: tuple[int, ...] = (),
    geometry_attention_mode: str = "latent",
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
            geometry_readout_from_latents=geometry_readout_from_latents,
            dense_geometry_attention_layers=dense_geometry_attention_layers,
            geometry_attention_mode=geometry_attention_mode,
            geometry_patch_grid=2,
            motion_action_size=32 if use_motion_latent else None,
            motion_latent_size=8,
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


def test_context_scaling_config_and_profile(tmp_path: Path) -> None:
    """缩放开关随 checkpoint 恢复；缺少字段时保留旧行为，短实验保留原 LR 周期。"""
    TrajVLAQwenConfig(scale_context_embeddings=False).save_pretrained(tmp_path)
    assert not PreTrainedConfig.from_pretrained(tmp_path).scale_context_embeddings
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    saved.pop("scale_context_embeddings")
    path.write_text(json.dumps(saved))
    assert PreTrainedConfig.from_pretrained(tmp_path).scale_context_embeddings

    profile = AppConfig.load("configs/traj_vla_qwen_unscaled_a100.yaml")
    parameters = profile.policy.parameters
    assert not parameters["scale_context_embeddings"]
    assert parameters["geometry_num_queries"] == 16
    assert not parameters["use_dense_geometry_attention"]
    assert not parameters["use_geometry_grounding"]
    assert not parameters["use_motion_latent"]
    assert parameters["scheduler_warmup_steps"] == 4000
    assert parameters["scheduler_decay_steps"] == 200000
    assert profile.policy.checkpoint is None
    assert profile.training.image_aug
    assert profile.training.steps == 40000
    assert profile.training.save_freq == 20000
    assert profile.training.include_tasks == ["curve_plate"]
    assert profile.training.peft is None
    assert not profile.training.resume
    assert profile.deployment.mode == "scene"
    assert profile.deployment.action_steps == 8


@pytest.mark.parametrize("geometry", [False, True])
def test_context_scaling_changes_only_semantic_embeddings(monkeypatch, geometry: bool) -> None:
    """真实 prefix 组装仅改变语义 token 及其梯度，状态、动作、几何和 mask 保持不变。"""
    import math

    from welding_path_vla.policies.traj_vla_qwen import modeling_traj_vla_qwen

    class TinyBackbone(torch.nn.Module):
        """以可训练查表替代大模型加载，保留语义与几何两种图像接口。"""

        language_hidden_size = 8
        expert_hidden_size = 8

        def __init__(self, config) -> None:
            super().__init__()
            self.tokens = torch.nn.Embedding(8, self.language_hidden_size)

        def embed_language_tokens(self, tokens):
            """返回原始文本 embedding。"""
            return self.tokens(tokens)

        def embed_image(self, image):
            """把输入当成 patch 索引，便于精确检查幅值和梯度。"""
            return self.tokens(image)

        def embed_image_with_geometry(self, image):
            """几何 patch 不应跟随语义 token 一起缩放。"""
            tokens = self.embed_image(image)
            return tokens, tokens

    monkeypatch.setattr(modeling_traj_vla_qwen, "PrismaticQwenWithExpert", TinyBackbone)
    scaled = QwenTrajectoryFlowModel(
        TrajVLAQwenConfig(
            attention_mode="cross_attn",
            use_geometry_branch=geometry,
            prefix_length=10,
        )
    )
    unscaled = deepcopy(scaled)
    unscaled.config.scale_context_embeddings = False
    # 开关没有新参数，两个方向都能严格加载同一份 state dict。
    unscaled.load_state_dict(scaled.state_dict(), strict=True)
    scaled.load_state_dict(unscaled.state_dict(), strict=True)
    args = (
        [torch.tensor([[0, 1]]), torch.tensor([[2, 3]])],
        [torch.tensor([True]), torch.tensor([False])],
        torch.tensor([[4, 5]]),
        torch.tensor([[True, False]]),
        torch.randn(1, 32),
    )
    original, changed = scaled.encode_context(*args), unscaled.encode_context(*args)
    factor = math.sqrt(TinyBackbone.language_hidden_size)
    torch.testing.assert_close(original.embeddings[:, :6], changed.embeddings[:, :6] * factor)
    torch.testing.assert_close(
        original.embeddings[:, 6:], changed.embeddings[:, 6:], rtol=0, atol=0
    )
    assert torch.equal(original.padding_mask, changed.padding_mask)
    assert torch.equal(original.attention_ar, changed.attention_ar)
    if geometry:
        for name in ("patch_tokens", "patch_mask", "language_mask", "state_mask"):
            assert torch.equal(
                getattr(original.expert_context, name), getattr(changed.expert_context, name)
            )
    # 开关开启时必须逐位等于公共 Smol 路径；该路径不读取 Qwen 配置项。
    monkeypatch.setattr(
        unscaled,
        "scale_context_embedding",
        lambda value: TrajectoryFlowModel.scale_context_embedding(unscaled, value),
    )
    torch.testing.assert_close(
        unscaled.encode_context(*args).embeddings, original.embeddings, rtol=0, atol=0
    )
    original.embeddings.sum().backward()
    changed.embeddings.sum().backward()
    torch.testing.assert_close(
        scaled.vlm_with_expert.tokens.weight.grad,
        unscaled.vlm_with_expert.tokens.weight.grad * factor,
    )
    torch.testing.assert_close(
        scaled.state_proj.weight.grad, unscaled.state_proj.weight.grad, rtol=0, atol=0
    )
    action_args = (torch.randn(1, 2, 32), torch.tensor([0.5]))
    torch.testing.assert_close(
        scaled.encode_action_tokens(*action_args).embeddings,
        unscaled.encode_action_tokens(*action_args).embeddings,
        rtol=0,
        atol=0,
    )


def test_dense_geometry_config_and_profile(tmp_path: Path) -> None:
    """稠密连接开关应保存恢复，拒绝无效层，且缺省不影响已有配置。"""
    config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        use_dense_geometry_attention=True,
        dense_geometry_attention_layers=(7, 15),
    )
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.use_dense_geometry_attention
    assert tuple(restored.dense_geometry_attention_layers) == (7, 15)
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    for key in ("use_dense_geometry_attention", "dense_geometry_attention_layers"):
        saved.pop(key)
    path.write_text(json.dumps(saved))
    assert not PreTrainedConfig.from_pretrained(tmp_path).use_dense_geometry_attention
    for layers in ((), (0,), (16,), (-1,), (1, 1)):
        with pytest.raises(ValueError, match="dense geometry"):
            TrajVLAQwenConfig(
                attention_mode="cross_attn",
                use_geometry_branch=True,
                use_dense_geometry_attention=True,
                dense_geometry_attention_layers=layers,
            )
    with pytest.raises(ValueError, match="dense geometry"):
        TrajVLAQwenConfig(use_dense_geometry_attention=True)
    profile = AppConfig.load("configs/traj_vla_qwen_dense_a100.yaml")
    assert profile.policy.parameters["use_dense_geometry_attention"]
    assert profile.policy.parameters["geometry_num_queries"] == 16
    assert not profile.policy.parameters["use_geometry_grounding"]
    assert not profile.policy.parameters["use_motion_latent"]


@pytest.mark.parametrize("checkpointed", [False, True])
def test_dense_geometry_is_gated_and_bypasses_latents(checkpointed: bool) -> None:
    """零门控保留原结果；打开门控后，动作梯度可绕过 latent 直接抵达 patch。"""
    from dataclasses import replace

    torch.manual_seed(71)
    legacy = make_tiny_paired_decoder(attention_mode="cross_attn", use_geometry_branch=True)
    dense = make_tiny_paired_decoder(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        dense_geometry_attention_layers=(1,),
        checkpoint_qwen=checkpointed,
        checkpoint_expert=checkpointed,
    )
    missing, unexpected = dense.load_state_dict(legacy.state_dict(), strict=False)
    assert missing == ["dense_geometry_attention.1.gate"]
    assert unexpected == []
    off = make_tiny_paired_decoder(attention_mode="cross_attn", use_geometry_branch=True)
    off.load_state_dict(legacy.state_dict(), strict=True)
    context, action = torch.randn(1, 3, 32), torch.randn(1, 2, 24)
    geometry = make_geometry_context()
    mask = make_attention_masks(
        torch.ones(1, 5, dtype=torch.bool), torch.tensor([[False, False, False, True, True]])
    )
    args = (mask, torch.arange(5)[None], None, [context, action], False, False)
    expected = legacy(*args, expert_context=geometry)[0][1]
    actual = dense(*args, expert_context=geometry)[0][1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    target = torch.randn_like(actual)
    (actual * target).sum().backward()
    gate = dense.dense_geometry_attention["1"].gate
    assert gate.grad.abs() > 0
    dense.zero_grad()
    with torch.no_grad():
        gate.fill_(0.5)
    dense.geometry_resampler.register_forward_hook(
        lambda module, inputs, output: replace(output, latent_tokens=output.latent_tokens.detach())
    )
    changed = dense(*args, expert_context=geometry)[0][1]
    assert not torch.allclose(changed, expected)
    (changed * target).sum().backward()
    assert dense.geometry_resampler.patch_projection.weight.grad.abs().sum() > 0
    assert dense.geometry_resampler.latent_queries.grad is None


def test_dense_geometry_attention_masks_and_peft(tmp_path: Path) -> None:
    """无效相机不得影响稠密读取，门控须在 PEFT 下可训练且能保存恢复。"""
    from peft import LoraConfig, PeftModel, get_peft_model

    gate = GatedDenseGeometryAttention()
    with torch.no_grad():
        gate.gate.fill_(0.5)
    query, key, value = torch.randn(2, 4, 3, 8), torch.randn(2, 2, 8, 8), torch.randn(2, 2, 8, 8)
    mask = torch.tensor([[True] * 4 + [False] * 4, [False] * 8])
    expected = gate(query, key, value, mask)
    key[:, :, 4:] += 100
    value[:, :, 4:] += 100
    torch.testing.assert_close(gate(query, key, value, mask), expected)
    assert torch.count_nonzero(expected[1]) == 0

    class TinyGateModel(torch.nn.Module):
        """仅保留真实稠密模块和一个 LoRA 投影的 PEFT 测试边界。"""

        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)
            self.dense = GatedDenseGeometryAttention()

        def forward(self, query: torch.Tensor) -> torch.Tensor:
            """以相同 patch 构造键值，使门控参与动作输出。"""
            return self.dense(self.proj(query), key, value, mask)

    base = TinyGateModel()
    original = deepcopy(base.state_dict())
    wrapped = get_peft_model(
        base, LoraConfig(r=2, target_modules=["proj"], modules_to_save=["dense"])
    )
    output = wrapped(query)
    (output * torch.randn_like(output)).sum().backward()
    trained_gate = wrapped.base_model.model.dense.modules_to_save.default.gate
    assert trained_gate.requires_grad and trained_gate.grad.abs() > 0
    with torch.no_grad():
        trained_gate.fill_(0.3)
    expected = wrapped(query)
    wrapped.save_pretrained(tmp_path)
    restored_base = TinyGateModel()
    restored_base.load_state_dict(original)
    restored = PeftModel.from_pretrained(restored_base, tmp_path)
    torch.testing.assert_close(restored(query), expected)


def test_dense_geometry_compilation_matches_eager() -> None:
    """实际双流 decoder 的新路径应支持整图编译，且输出与梯度保持一致。"""
    torch.manual_seed(83)
    eager = make_tiny_paired_decoder(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        dense_geometry_attention_layers=(1,),
    )
    with torch.no_grad():
        eager.dense_geometry_attention["1"].gate.fill_(0.3)
    compiled_base = deepcopy(eager)
    compiled = torch.compile(compiled_base, backend="aot_eager", fullgraph=True)
    mask = make_attention_masks(
        torch.ones(1, 5, dtype=torch.bool),
        torch.tensor([[False, False, False, True, True]]),
    )
    args = (
        mask,
        torch.arange(5)[None],
        None,
        [torch.randn(1, 3, 32), torch.randn(1, 2, 24)],
        False,
        False,
    )
    geometry = make_geometry_context()
    expected = eager(*args, expert_context=geometry)[0][1]
    actual = compiled(*args, expert_context=geometry)[0][1]
    torch.testing.assert_close(actual, expected)
    target = torch.randn_like(expected)
    (expected * target).sum().backward()
    (actual * target).sum().backward()
    for name, parameter in eager.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad, dict(compiled_base.named_parameters())[name].grad
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
    assert config.policy.parameters["scale_context_embeddings"]
    assert config.policy.parameters["geometry_num_heads"] == 8
    assert not config.policy.parameters["use_dense_geometry_attention"]
    assert not config.policy.parameters["use_geometry_grounding"]
    assert not config.policy.parameters["geometry_readout_from_latents"]
    assert not config.policy.parameters["use_motion_latent"]
    assert config.policy.parameters["training_stage"] == "standard"
    assert config.training.video_backend == "torchcodec"
    assert get_policy_pipeline("traj_vla_qwen").spec is TRAJ_VLA_QWEN


def test_two_stage_a100_profiles_enable_features_in_order() -> None:
    """Stage 1 只做接地预热，Stage 2 再加载其权重并启用 motion latent。"""
    stage1 = AppConfig.load("configs/traj_vla_qwen_grounding_stage1_a100.yaml")
    stage2 = AppConfig.load("configs/traj_vla_qwen_grounding_stage2_a100.yaml")
    assert stage1.policy.parameters["training_stage"] == "grounding_warmup"
    assert stage1.policy.parameters["geometry_aux_loss_weights"] == [1.0, 0.5, 0.5]
    assert stage1.training.steps == 10_000
    assert stage2.policy.parameters["training_stage"] == "policy_joint"
    assert stage2.policy.parameters["use_motion_latent"]
    assert stage2.policy.parameters["geometry_aux_loss_weights"] == [0.05, 0.02, 0.02]
    assert stage2.policy.checkpoint and "grounding_stage1" in stage2.policy.checkpoint


def test_policy_joint_requires_grounding_weights(tmp_path: Path) -> None:
    """Stage 2 可随机初始化 motion，但不能静默丢失 Stage 1 grounding 权重。"""
    from safetensors.torch import save_file

    path = tmp_path / "model.safetensors"
    save_file({"model.action_out_proj.weight": torch.ones(1)}, path)
    with pytest.raises(ValueError, match="Stage 1 weights"):
        require_grounding_checkpoint(str(tmp_path))
    save_file(
        {
            "model.vlm_with_expert.decoder.geometry_resampler.latent_queries": torch.ones(1),
            "model.geometry_grounding_heads.tangent.weight": torch.ones(1),
        },
        path,
    )
    require_grounding_checkpoint(str(tmp_path))


def test_inference_preparation_discards_training_only_modules() -> None:
    """部署裁剪应删除 grounding heads/posterior，同时保留 motion prior。"""

    class FakeDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.motion_latent = ConditionalMotionLatent(32, 32, 24, 8)

    class FakeBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = FakeDecoder()

    class FakeFlow(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.geometry_grounding_heads = GeometryGroundingHeads(24)
            self.vlm_with_expert = FakeBackbone()

    policy = TrajVLAQwenPolicy.__new__(TrajVLAQwenPolicy)
    torch.nn.Module.__init__(policy)
    policy.model = FakeFlow()
    policy.prepare_for_inference()
    assert policy.model.geometry_grounding_heads is None
    assert policy.model.vlm_with_expert.decoder.motion_latent.posterior is None
    assert policy.model.vlm_with_expert.decoder.motion_latent.prior is not None


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


@pytest.mark.parametrize("readout_from_latents", [False, True])
def test_geometry_config_round_trip(tmp_path: Path, readout_from_latents: bool) -> None:
    """几何结构字段应随 LeRobot checkpoint 配置稳定保存和恢复。"""
    config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        geometry_num_queries=12,
        geometry_num_heads=4,
        train_geometry_resampler=False,
        geometry_readout_from_latents=readout_from_latents,
        device="cpu",
    )
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(restored, TrajVLAQwenConfig)
    assert restored.use_geometry_branch
    assert restored.geometry_num_queries == 12
    assert restored.geometry_num_heads == 4
    assert not restored.train_geometry_resampler
    assert restored.geometry_readout_from_latents is readout_from_latents


def test_geometry_readout_config_defaults_to_legacy_checkpoint(tmp_path: Path) -> None:
    """缺少新字段的 checkpoint 应保持关闭，不改变既有动作几何结构。"""
    config = TrajVLAQwenConfig(attention_mode="cross_attn", use_geometry_branch=True)
    config.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    saved.pop("geometry_readout_from_latents")
    path.write_text(json.dumps(saved))
    assert not PreTrainedConfig.from_pretrained(tmp_path).geometry_readout_from_latents
    with pytest.raises(ValueError, match="geometry_readout_from_latents"):
        TrajVLAQwenConfig(geometry_readout_from_latents=True)


@pytest.mark.parametrize("readout_from_latents", [False, True])
@pytest.mark.parametrize("dense_enabled", [False, True])
def test_geometry_readout_switch_reaches_decoder_factory(
    readout_from_latents: bool,
    dense_enabled: bool,
) -> None:
    """模型工厂必须把保存的开关传入 Resampler，而非只支持手动构造。"""
    tiny = make_tiny_paired_decoder()
    config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        geometry_readout_from_latents=readout_from_latents,
        use_dense_geometry_attention=dense_enabled,
        dense_geometry_attention_layers=(1,),
    )
    decoder = make_expert_decoder(
        config,
        tiny.language_model,
        tiny.expert_model,
        Qwen25DecoderAdapter(),
        geometry_input_size=12,
    )
    assert decoder.geometry_resampler.readout_from_latents is readout_from_latents
    assert list(decoder.dense_geometry_attention) == (["1"] if dense_enabled else [])


def test_grounding_stage_configs_enable_latent_readout() -> None:
    """两阶段配置显式开启新路径，普通 A100 配置保持关闭。"""
    for stage in (1, 2):
        config = AppConfig.load(f"configs/traj_vla_qwen_grounding_stage{stage}_a100.yaml")
        assert config.policy.parameters["use_geometry_grounding"]
        assert config.policy.parameters["geometry_readout_from_latents"]
    config = AppConfig.load("configs/traj_vla_qwen_a100.yaml")
    assert not config.policy.parameters["geometry_readout_from_latents"]


def test_xsvla_features_default_off_and_validate_dependencies(tmp_path: Path) -> None:
    """旧 checkpoint 默认不创建 grounding、motion 或两阶段训练结构。"""
    config = TrajVLAQwenConfig(device="cpu")
    assert not config.use_geometry_grounding
    assert not config.use_motion_latent
    assert config.training_stage == "standard"
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(restored, TrajVLAQwenConfig)
    assert not restored.use_geometry_grounding
    assert not restored.use_motion_latent

    enabled_path = tmp_path / "enabled"
    enabled = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        use_geometry_grounding=True,
        use_motion_latent=True,
        training_stage="policy_joint",
        motion_latent_dim=12,
        device="cpu",
    )
    enabled.save_pretrained(enabled_path)
    enabled_restored = PreTrainedConfig.from_pretrained(enabled_path)
    assert isinstance(enabled_restored, TrajVLAQwenConfig)
    assert enabled_restored.use_geometry_grounding
    assert enabled_restored.use_motion_latent
    assert enabled_restored.motion_latent_dim == 12
    assert enabled_restored.training_stage == "policy_joint"

    with pytest.raises(ValueError, match="geometry grounding"):
        TrajVLAQwenConfig(use_geometry_grounding=True)
    with pytest.raises(ValueError, match="motion latent"):
        TrajVLAQwenConfig(use_motion_latent=True)
    with pytest.raises(ValueError, match="grounding_warmup"):
        TrajVLAQwenConfig(training_stage="grounding_warmup")


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


@pytest.mark.parametrize("readout_from_latents", [False, True])
def test_seam_query_resampler_conditions_and_masks_dense_patches(
    readout_from_latents: bool,
) -> None:
    """Resampler 应输出固定 query，并完全屏蔽缺失相机。"""
    torch.manual_seed(3)
    resampler = SeamQueryResampler(
        12, 32, 24, 2, num_heads=4, readout_from_latents=readout_from_latents
    )
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


@pytest.mark.parametrize("readout_from_latents", [False, True])
@pytest.mark.parametrize("loss_name", ["seam", "tangent", "orientation"])
def test_grounding_loss_reaches_action_latents(loss_name: str, readout_from_latents: bool) -> None:
    """每项真实辅助损失都应在开启后直接训练动作 latent，而非只更新共享参数。"""
    torch.manual_seed(37)
    resampler = SeamQueryResampler(
        12,
        32,
        24,
        2,
        num_queries=4,
        num_heads=4,
        readout_from_latents=readout_from_latents,
    )
    output = resampler(torch.randn(2, 3, 32), make_geometry_context(batch_size=2))
    output.latent_tokens.retain_grad()
    auxiliary = output.auxiliary_outputs()
    auxiliary.update(GeometryGroundingHeads(24)(auxiliary))
    batch = {
        SEAM_LABELS: torch.zeros(2, 2, 4),
        SEAM_VALID: torch.ones(2, 2, dtype=torch.bool),
        TANGENT_TARGET: torch.tensor([[1.0, 0, 0]]).expand(2, -1),
        TANGENT_VALID: torch.ones(2, dtype=torch.bool),
        ORIENTATION_TARGET: torch.tensor([[1.0, 0, 0, 0, 1, 0]]).expand(2, -1),
        ORIENTATION_VALID: torch.ones(2, dtype=torch.bool),
    }
    geometry_grounding_losses(auxiliary, batch)[loss_name].mean().backward()
    gradient = resampler.latent_queries.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    if readout_from_latents:
        assert (gradient.abs().sum(dim=-1) > 0).all()
        assert output.latent_tokens.grad is not None
        assert output.latent_tokens.grad.abs().sum() > 0
    else:
        assert torch.count_nonzero(gradient) == 0
        assert output.latent_tokens.grad is None


def test_geometry_readout_switch_preserves_weights_and_action_latents() -> None:
    """切换 readout 路径不新增参数、不改变动作 latent，关闭后可严格恢复原权重。"""
    torch.manual_seed(41)
    legacy = SeamQueryResampler(12, 32, 24, 2, num_queries=4, num_heads=4).eval()
    routed = SeamQueryResampler(
        12, 32, 24, 2, num_queries=4, num_heads=4, readout_from_latents=True
    ).eval()
    routed.load_state_dict(legacy.state_dict(), strict=True)
    context, geometry = torch.randn(2, 3, 32), make_geometry_context(batch_size=2)
    expected, actual = legacy(context, geometry), routed(context, geometry)
    torch.testing.assert_close(actual.latent_tokens, expected.latent_tokens)
    assert not torch.allclose(actual.readout_tokens, expected.readout_tokens)
    legacy.load_state_dict(routed.state_dict(), strict=True)
    routed.readout_from_latents = False
    restored = routed(context, geometry)
    for name, value in expected.auxiliary_outputs().items():
        torch.testing.assert_close(restored.auxiliary_outputs()[name], value, rtol=0, atol=0)


def test_grounding_targets_align_two_cameras_and_ignore_padding() -> None:
    """在线标签应投影到双相机 patch，并从有效 future targets 生成局部姿态。"""
    processor = GeometryGroundingTargetProcessorStep(
        camera_keys=("observation.images.global", "observation.images.wrist"),
        camera_fovy_deg=(90.0, 90.0),
        global_camera_pose_world=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        wrist_camera_pose_tcp=(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        corridor_radius_px=2,
    )
    state = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0]],
        dtype=torch.float32,
    )
    identity = [1, 0, 0, 0, 1, 0]
    actions = torch.tensor(
        [[[0.1, 0, -1, *identity], [0.2, 0, -1, *identity], [9, 9, -1, *identity]]],
        dtype=torch.float32,
    )
    targets = processor.targets(
        state,
        actions,
        torch.tensor([[False, False, True]]),
        [(480, 640), (480, 640)],
    )
    assert targets[SEAM_LABELS].shape == (1, 2, 256)
    assert torch.all(targets[SEAM_VALID])
    torch.testing.assert_close(targets[TANGENT_TARGET], torch.tensor([[1.0, 0, 0]]))
    torch.testing.assert_close(
        targets[ORIENTATION_TARGET],
        torch.tensor([identity], dtype=torch.float32),
    )
    assert targets[TANGENT_VALID].item()
    assert targets[ORIENTATION_VALID].item()
    assert targets[SEAM_LABELS].sum() < 40  # padding 的远端目标不能拉长 corridor。


def test_grounding_heads_and_losses_are_masked_and_differentiable() -> None:
    """三个 readout head 应输出稳定形状，并把监督梯度传回 query/patch。"""
    torch.manual_seed(5)
    heads = GeometryGroundingHeads(8)
    patches = torch.randn(2, 2, 4, 8, requires_grad=True)
    readout = torch.randn(2, 3, 8, requires_grad=True)
    auxiliary = {
        "geometry.patch_tokens": patches,
        "geometry.patch_mask": torch.ones(2, 2, 4, dtype=torch.bool),
        "geometry.readout_tokens": readout,
    }
    auxiliary.update(heads(auxiliary))
    batch = {
        SEAM_LABELS: torch.zeros(2, 2, 4),
        SEAM_VALID: torch.tensor([[True, True], [False, False]]),
        TANGENT_TARGET: auxiliary["geometry.tangent_prediction"].detach(),
        TANGENT_VALID: torch.tensor([True, False]),
        ORIENTATION_TARGET: auxiliary["geometry.orientation_6d"].detach(),
        ORIENTATION_VALID: torch.tensor([True, False]),
    }
    losses = geometry_grounding_losses(auxiliary, batch)
    assert auxiliary["geometry.seam_logits"].shape == (2, 2, 4)
    assert losses["seam"].shape == (2,)
    assert losses["tangent"][0] < 1e-5
    assert losses["tangent"][1] == 0
    assert losses["orientation"][1] == 0
    sum(loss.mean() for loss in losses.values()).backward()
    assert patches.grad is not None
    assert readout.grad is not None


def test_grounding_losses_prefer_matching_geometry() -> None:
    """正确 corridor、方向和姿态的三项损失都应低于相反预测。"""
    identity = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float32)
    matched_auxiliary = {
        "geometry.seam_logits": torch.tensor([[[6.0, -6.0]]]),
        "geometry.patch_mask": torch.ones(1, 1, 2, dtype=torch.bool),
        "geometry.tangent_prediction": torch.tensor([[1.0, 0, 0]]),
        "geometry.orientation_6d": identity,
    }
    matched_batch = {
        SEAM_LABELS: torch.tensor([[[1.0, 0]]]),
        SEAM_VALID: torch.ones(1, 1, dtype=torch.bool),
        TANGENT_TARGET: torch.tensor([[1.0, 0, 0]]),
        TANGENT_VALID: torch.ones(1, dtype=torch.bool),
        ORIENTATION_TARGET: identity,
        ORIENTATION_VALID: torch.ones(1, dtype=torch.bool),
    }
    wrong_auxiliary = {
        **matched_auxiliary,
        "geometry.seam_logits": -matched_auxiliary["geometry.seam_logits"],
        "geometry.tangent_prediction": -matched_auxiliary["geometry.tangent_prediction"],
        "geometry.orientation_6d": torch.tensor([[-1, 0, 0, 0, -1, 0.0]]),
    }
    matched = geometry_grounding_losses(matched_auxiliary, matched_batch)
    wrong = geometry_grounding_losses(wrong_auxiliary, matched_batch)
    assert all(matched[name].item() < wrong[name].item() for name in matched)


def test_conditional_motion_latent_uses_posterior_only_in_training() -> None:
    """训练 posterior 应随 action 改变，推理则确定性使用 conditional prior mean。"""
    torch.manual_seed(13)
    module = ConditionalMotionLatent(32, 32, 24, 8)
    context = torch.randn(2, 3, 32)
    geometry_context = make_geometry_context(batch_size=2)
    geometry_context.clean_actions = torch.randn(2, 5, 32)
    geometry_context.action_padding_mask = torch.tensor(
        [[False, False, False, True, True], [False, False, False, False, True]]
    )
    geometry = SeamQueryResampler(12, 32, 24, 2, num_queries=4, num_heads=4)(
        context,
        geometry_context,
    )
    module.train()
    first = module(context, geometry_context, geometry)
    original_actions = geometry_context.clean_actions.clone()
    geometry_context.clean_actions[geometry_context.action_padding_mask] += 20
    padding_changed = module(context, geometry_context, geometry)
    torch.testing.assert_close(
        first.auxiliary_outputs["motion.posterior_mean"],
        padding_changed.auxiliary_outputs["motion.posterior_mean"],
    )
    geometry_context.clean_actions = original_actions + 2
    second = module(context, geometry_context, geometry)
    assert not torch.allclose(
        first.auxiliary_outputs["motion.posterior_mean"],
        second.auxiliary_outputs["motion.posterior_mean"],
    )
    assert first.auxiliary_outputs["motion.kl"].shape == (2,)
    zeros = torch.zeros(2, 8)
    torch.testing.assert_close(diagonal_gaussian_kl(zeros, zeros, zeros, zeros), torch.zeros(2))

    module.eval()
    inferred = module(context, geometry_context, geometry)
    repeated = module(context, geometry_context, geometry)
    torch.testing.assert_close(inferred.token, repeated.token)
    torch.testing.assert_close(
        inferred.auxiliary_outputs["motion.latent"],
        inferred.auxiliary_outputs["motion.prior_mean"],
    )
    task_changed = context.clone()
    task_changed[:, 0] += 2
    changed_prior = module(task_changed, geometry_context, geometry)
    assert not torch.allclose(
        inferred.auxiliary_outputs["motion.prior_mean"],
        changed_prior.auxiliary_outputs["motion.prior_mean"],
    )


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


@pytest.mark.parametrize("readout_from_latents", [False, True])
@pytest.mark.parametrize("dense_layers", [(), (1, 3)])
def test_geometry_decoder_is_asymmetric_and_cache_equivalent(
    readout_from_latents: bool,
    dense_layers: tuple[int, ...],
) -> None:
    """几何 CA 只更新 Expert，且缓存前向应与完整前向等价。"""
    torch.manual_seed(17)
    decoder = make_tiny_paired_decoder(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        num_layers=4,
        geometry_readout_from_latents=readout_from_latents,
        dense_geometry_attention_layers=dense_layers,
    ).eval()
    for module in decoder.dense_geometry_attention.values():
        with torch.no_grad():
            module.gate.fill_(0.5)
    assert isinstance(decoder, InterleavedSACADecoder)
    assert decoder.geometry_resampler.readout_from_latents is readout_from_latents
    assert decoder.expert_model.layers[1].self_attn.k_proj.in_features == 24
    context = torch.randn(1, 3, 32)
    action = torch.randn(1, 2, 24)
    geometry = make_geometry_context()
    padding = torch.ones(1, 5, dtype=torch.bool)
    blocks = torch.tensor([[False, False, False, True, True]])
    mask = make_attention_masks(padding, blocks)
    positions = torch.arange(5).unsqueeze(0)

    expert_calls: list[None] = []
    decoder.expert_model.layers[0].self_attn.q_proj.register_forward_hook(
        lambda module, inputs, output: expert_calls.append(None)
    )
    geometry_only = decoder.forward_geometry(
        torch.ones(1, 3, 3, dtype=torch.bool),
        positions[:, :3],
        context,
        geometry,
    )
    assert geometry_only["geometry.readout_tokens"].shape == (1, 3, 24)
    assert expert_calls == []
    if readout_from_latents:
        # warm-up 不运行动作专家，但 readout 的监督仍应抵达专家使用的 latent query。
        readout = geometry_only["geometry.readout_tokens"]
        (readout * torch.randn_like(readout)).sum().backward()
        assert decoder.geometry_resampler.latent_queries.grad.abs().sum() > 0
        decoder.zero_grad()

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
    for index in dense_layers:
        assert cache[index]["dense_key_states"].shape[2] == 8
        assert cache[index]["dense_patch_mask"].shape == (1, 8)
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
        dense_geometry_attention_layers=dense_layers,
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


@pytest.mark.parametrize("readout_from_latents", [False, True])
def test_motion_decoder_is_asymmetric_and_cache_equivalent(readout_from_latents: bool) -> None:
    """Motion prior 只影响 Expert，并在 Context cache 后不重复计算。"""
    torch.manual_seed(19)
    decoder = make_tiny_paired_decoder(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        use_motion_latent=True,
        num_layers=4,
        geometry_readout_from_latents=readout_from_latents,
    ).eval()
    assert isinstance(decoder, InterleavedSACADecoder)
    context = torch.randn(1, 3, 32)
    action = torch.randn(1, 3, 24)  # motion slot + 两个动作 token
    geometry = make_geometry_context()
    padding = torch.ones(1, 6, dtype=torch.bool)
    blocks = torch.tensor([[False, False, False, True, True, True]])
    mask = make_attention_masks(padding, blocks)
    positions = torch.arange(6).unsqueeze(0)

    full, _, auxiliary = decoder(
        mask,
        positions,
        None,
        [context, action],
        False,
        False,
        expert_context=geometry,
    )
    assert "motion.prior_mean" in auxiliary
    assert decoder.motion_latent is not None
    prior_output = decoder.motion_latent.prior[-1]
    assert isinstance(prior_output, torch.nn.Linear)
    original_bias = prior_output.bias.detach().clone()
    with torch.no_grad():
        prior_output.bias.add_(2)
    changed_motion, _, _ = decoder(
        mask,
        positions,
        None,
        [context, action],
        False,
        False,
        expert_context=geometry,
    )
    torch.testing.assert_close(full[0], changed_motion[0], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(full[1], changed_motion[1])
    with torch.no_grad():
        prior_output.bias.copy_(original_bias)

    prior_calls: list[None] = []
    decoder.motion_latent.prior.register_forward_hook(
        lambda module, inputs, output: prior_calls.append(None)
    )
    _, cache, _ = decoder(
        torch.ones(1, 3, 3, dtype=torch.bool),
        positions[:, :3],
        None,
        [context, None],
        True,
        True,
        expert_context=geometry,
    )
    assert cache is not None and -1 in cache
    calls_after_cache = len(prior_calls)
    cached, _, cached_auxiliary = decoder(
        mask[:, 3:],
        positions[:, 3:],
        cache,
        [None, action],
        True,
        False,
    )
    torch.testing.assert_close(full[1], cached[1], atol=1e-5, rtol=1e-5)
    assert cached_auxiliary == {}
    assert len(prior_calls) == calls_after_cache


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
    policy.model = torch.nn.Module()
    policy.model.vlm_with_expert = torch.nn.Module()
    vision = torch.nn.Module()
    vision.dino_layer = 22
    vision.siglip_layer = 25
    policy.model.vlm_with_expert.vision_encoder = vision
    policy.config = TrajVLAQwenConfig(lora_target="qwen")
    targets = policy._get_default_peft_targets()
    assert "qwen" in targets["target_modules"]
    assert targets["modules_to_save"][-1] == "model.vlm_with_expert.decoder.expert_model"
    assert "model.action_out_proj" in targets["modules_to_save"]
    assert "vision_encoder" not in targets["target_modules"]

    policy.config.lora_target = "all"
    targets = policy._get_default_peft_targets()
    assert "qwen" in targets["target_modules"]
    assert "expert\\.layers" not in targets["target_modules"]
    assert "model.vlm_with_expert.decoder.expert_model" in targets["modules_to_save"]
    assert "model.vlm_with_expert.expert" not in targets["modules_to_save"]
    assert re.fullmatch(
        targets["target_modules"],
        "model.vlm_with_expert.vision_encoder.dino_featurizer.blocks.22.attn.qkv",
    )
    assert re.fullmatch(
        targets["target_modules"],
        "model.vlm_with_expert.vision_encoder.siglip_featurizer.blocks.25.attn.proj",
    )
    assert not re.fullmatch(
        targets["target_modules"],
        "model.vlm_with_expert.vision_encoder.siglip_featurizer.blocks.26.attn.qkv",
    )

    policy.config.lora_target = "vision"
    policy.config.vision_lora_last_n_blocks = 2
    targets = policy._get_default_peft_targets()
    assert "qwen" not in targets["target_modules"]
    assert "model.vlm_with_expert.decoder.expert_model" in targets["modules_to_save"]
    for encoder, last in (("dino", 22), ("siglip", 25)):
        for index in range(last + 2):
            name = (
                f"model.vlm_with_expert.vision_encoder.{encoder}_featurizer.blocks.{index}.attn.qkv"
            )
            assert bool(re.fullmatch(targets["target_modules"], name)) == (
                last - 1 <= index <= last
            )

    # 八层 attention + MLP 只覆盖实际参与特征输出的 block。
    policy.config.vision_lora_last_n_blocks = 8
    policy.config.vision_lora_include_mlp = True
    targets = policy._get_default_peft_targets()
    for encoder, last in (("dino", 22), ("siglip", 25)):
        for index in range(last + 2):
            for projection in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
                name = (
                    f"model.vlm_with_expert.vision_encoder.{encoder}_featurizer"
                    f".blocks.{index}.{projection}"
                )
                assert bool(re.fullmatch(targets["target_modules"], name)) == (
                    last - 7 <= index <= last
                )
    assert not re.fullmatch(
        targets["target_modules"], "model.vlm_with_expert.qwen.model.layers.15.mlp.up_proj"
    )

    policy.config.train_expert = False
    assert (
        "model.vlm_with_expert.decoder.expert_model"
        not in policy._get_default_peft_targets()["modules_to_save"]
    )

    policy.config.lora_target = "expert"
    targets = policy._get_default_peft_targets()
    assert "expert\\.layers" in targets["target_modules"]
    assert "model.vlm_with_expert.decoder.expert_model" not in targets["modules_to_save"]

    policy.config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        use_geometry_grounding=True,
        use_motion_latent=True,
        use_dense_geometry_attention=True,
    )
    targets = policy._get_default_peft_targets()
    assert "model.vlm_with_expert.decoder.geometry_resampler" in targets["modules_to_save"]
    assert "model.geometry_grounding_heads" in targets["modules_to_save"]
    assert "model.vlm_with_expert.decoder.motion_latent" in targets["modules_to_save"]
    assert "model.vlm_with_expert.decoder.dense_geometry_attention.7" in targets["modules_to_save"]


def test_peft_trains_and_restores_decoder_expert(tmp_path: Path) -> None:
    """共享 Expert 应在 decoder 前向中全量训练，并能随 adapter 保存恢复。"""
    from peft import LoraConfig, PeftModel, get_peft_model

    class TinyModel(torch.nn.Module):
        """复现主干与 decoder 共享 Expert 的模块路径。"""

        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            backbone = torch.nn.Module()
            self.model.vlm_with_expert = backbone
            backbone.expert = torch.nn.Module()
            backbone.expert.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
            backbone.expert.norm = torch.nn.LayerNorm(4)
            backbone.decoder = torch.nn.Module()
            backbone.decoder.expert_model = backbone.expert
            backbone.qwen = torch.nn.Module()
            backbone.qwen.model = torch.nn.Module()
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.self_attn.q_proj = torch.nn.Linear(4, 4)
            layer.self_attn.v_proj = torch.nn.Linear(4, 4)
            backbone.qwen.model.layers = torch.nn.ModuleList([layer])

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """由 decoder 的 Expert 路径计算动作，并使 Qwen LoRA 参与前向。"""
            backbone = self.model.vlm_with_expert
            expert = backbone.decoder.expert_model
            output = expert.norm(expert.layers[0](inputs))
            return output + backbone.qwen.model.layers[0].self_attn.q_proj(inputs)

    policy = TrajVLAQwenPolicy.__new__(TrajVLAQwenPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = TrajVLAQwenConfig(lora_target="qwen", train_expert=True)
    targets = policy._get_default_peft_targets()
    torch.manual_seed(7)
    base = TinyModel()
    base_state = deepcopy(base.state_dict())
    adapter = LoraConfig(r=2, lora_alpha=4, **targets)
    wrapped = get_peft_model(base, adapter)
    expert_parameters = [
        parameter
        for name, parameter in wrapped.named_parameters()
        if "decoder.expert_model.modules_to_save.default" in name
    ]
    assert sum(parameter.numel() for parameter in expert_parameters) == 28
    assert all(parameter.requires_grad for parameter in expert_parameters)
    inputs = torch.randn(2, 4)
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.1)
    wrapped(inputs).square().sum().backward()
    assert all(parameter.grad is not None for parameter in expert_parameters)
    optimizer.step()
    expected = wrapped(inputs).detach()
    wrapped.save_pretrained(tmp_path)

    restored_base = TinyModel()
    restored_base.load_state_dict(base_state)
    restored = PeftModel.from_pretrained(restored_base, tmp_path, is_trainable=True)
    torch.testing.assert_close(restored(inputs), expected)


def test_peft_base_can_be_rebuilt_without_policy_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧 LoRA checkpoint 没有基座路径时应从 Prismatic 配置重建基座。"""
    config = TrajVLAQwenConfig(load_prismatic_weights=False, load_base_weights=False)

    def fake_init(self: TrajVLAQwenPolicy, received: TrajVLAQwenConfig) -> None:
        torch.nn.Module.__init__(self)
        self.config = received

    monkeypatch.setattr(TrajVLAQwenPolicy, "__init__", fake_init)
    policy = TrajVLAQwenPolicy.make_peft_base(config)

    assert policy.config is config
    assert config.load_prismatic_weights is True
    assert config.load_base_weights is True


def test_peft_checkpoint_records_and_recognizes_prismatic_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新 LoRA checkpoint 应记录并识别其内部加载的 Prismatic 基座。"""
    from welding_path_vla.policies.trajectory_vla.modeling_trajectory_vla import (
        TrajectoryVLAPolicy,
    )

    config = TrajVLAQwenConfig(load_prismatic_weights=True, load_base_weights=True)
    policy = TrajVLAQwenPolicy.__new__(TrajVLAQwenPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    monkeypatch.setattr(
        TrajectoryVLAPolicy,
        "wrap_with_peft",
        lambda self, peft_config=None, peft_cli_overrides=None: self,
    )

    wrapped = policy.wrap_with_peft()

    assert wrapped is policy
    assert policy.name_or_path == config.prismatic_repo_id

    sentinel = object()
    monkeypatch.setattr(
        TrajVLAQwenPolicy,
        "make_peft_base",
        classmethod(lambda cls, received: sentinel),
        raising=False,
    )

    def reject_policy_checkpoint(*args: object, **kwargs: object) -> None:
        raise AssertionError("Prismatic 基座不应按完整 policy checkpoint 加载")

    monkeypatch.setattr(
        TrajectoryVLAPolicy,
        "from_pretrained",
        classmethod(reject_policy_checkpoint),
    )
    restored = TrajVLAQwenPolicy.from_pretrained(config.prismatic_repo_id, config=config)

    assert restored is sentinel
