import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.processor import PolicyProcessorPipeline
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from safetensors.torch import load_file

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.checkpoint import ensure_peft_base_reference
from welding_path_vla.policies.factory import get_policy_pipeline
from welding_path_vla.policies.lerobot_training import apply_training_overrides, make_train_config
from welding_path_vla.policies.spec import VLA_ADAPTER
from welding_path_vla.policies.vla_adapter.bridge import ProBridgeBlock
from welding_path_vla.policies.vla_adapter.configuration_vla_adapter import (
    VLAAdapterConfig,
)
from welding_path_vla.policies.vla_adapter.modeling_vla_adapter import (
    BridgeContext,
    VLAAdapterModel,
)
from welding_path_vla.policies.vla_adapter.processor_vla_adapter import (
    make_vla_adapter_pre_post_processors,
)


def tiny_features() -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    """构造双相机、13D 状态和 9D 动作的最小 feature schema。"""
    inputs = {
        "observation.images.global": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
        "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
        "observation.state": PolicyFeature(FeatureType.STATE, (13,)),
    }
    outputs = {"action": PolicyFeature(FeatureType.ACTION, (9,))}
    return inputs, outputs


class TinyBackbone(torch.nn.Module):
    """避免下载预训练权重的逐层条件替身。"""

    def __init__(self, config: VLAAdapterConfig) -> None:
        super().__init__()
        self.hidden_size = 32
        self.config = config
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.projector = torch.nn.Linear(32, 32)
        self.vision_encoder = torch.nn.Sequential(torch.nn.Linear(32, 32))
        self.qwen = torch.nn.Module()
        self.qwen.model = torch.nn.Module()
        self.qwen.model.layers = torch.nn.ModuleList()
        for _ in range(config.num_bridge_layers):
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.self_attn.q_proj = torch.nn.Linear(32, 32)
            layer.self_attn.k_proj = torch.nn.Linear(32, 32)
            layer.self_attn.v_proj = torch.nn.Linear(32, 32)
            layer.self_attn.o_proj = torch.nn.Linear(32, 32)
            self.qwen.model.layers.append(layer)

    def forward(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        action_queries: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor]:
        batch = language_tokens.shape[0]
        image_features = images[0].flatten(1).mean(1, keepdim=True).expand(-1, self.hidden_size)
        visual = self.projector(self.vision_encoder(image_features))
        raw = visual[:, None].expand(-1, 8, -1) + self.anchor
        query = action_queries[None].expand(batch, -1, -1)
        mask = torch.ones(batch, 8, dtype=torch.bool, device=language_tokens.device)
        return (
            tuple(raw for _ in range(self.config.num_bridge_layers)),
            tuple(query for _ in range(self.config.num_bridge_layers)),
            mask,
        )


def make_tiny_config(action_head: str = "l1") -> VLAAdapterConfig:
    """返回可直接实例化 tiny 动作模型的配置。"""
    inputs, outputs = tiny_features()
    return VLAAdapterConfig(
        input_features=inputs,
        output_features=outputs,
        action_head=action_head,
        chunk_size=5,
        n_action_steps=2,
        num_action_queries=4,
        num_bridge_layers=2,
        bridge_num_heads=4,
        num_steps=2,
        device="cpu",
        load_prismatic_weights=False,
        load_base_weights=False,
    )


def test_vla_adapter_configs_register_both_heads() -> None:
    """默认入口使用 L1，Flow 示例只覆盖动作头。"""
    l1 = AppConfig.load("configs/vla_adapter.yaml")
    a100 = AppConfig.load("configs/vla_adapter_a100.yaml")
    flow = AppConfig.load("configs/vla_adapter_flow.yaml")
    assert l1.policy.family == "vla_adapter"
    assert l1.policy.parameters["action_head"] == "l1"
    assert flow.policy.parameters["action_head"] == "flow_matching"
    assert l1.policy.parameters["num_action_queries"] == 64
    assert l1.policy.parameters["num_bridge_layers"] == 24
    assert l1.policy.parameters["use_vision_lora"] is True
    assert l1.policy.parameters["compile_model"] is False
    assert l1.policy.parameters["compile_mode"] == "reduce-overhead"
    assert a100.policy.parameters["compile_model"] is False
    assert l1.training.image_aug and a100.training.image_aug and flow.training.image_aug
    assert get_policy_pipeline("vla_adapter").spec is VLA_ADAPTER


@pytest.mark.parametrize(
    ("action_head", "expected_methods"),
    [("l1", ["forward"]), ("flow_matching", ["forward", "sample_flow"])],
)
def test_compile_model_wraps_expected_methods(
    action_head: str,
    expected_methods: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """编译开关只包装训练前向，并在 Flow 头时编译采样路径。"""
    from welding_path_vla.policies.vla_adapter import modeling_vla_adapter
    from welding_path_vla.policies.vla_adapter.modeling_vla_adapter import VLAAdapterPolicy

    monkeypatch.setattr(modeling_vla_adapter, "VLAAdapterBackbone", TinyBackbone)
    calls: list[tuple[str, str]] = []

    def compile_method(method: Any, *, mode: str) -> Any:
        calls.append((method.__name__, mode))
        return method

    monkeypatch.setattr(modeling_vla_adapter.torch, "compile", compile_method)
    monkeypatch.setattr(modeling_vla_adapter.torch, "set_float32_matmul_precision", lambda _: None)

    config = make_tiny_config(action_head)
    config.compile_model = True
    config.compile_mode = "reduce-overhead"
    VLAAdapterPolicy(config)

    assert [name for name, _ in calls] == expected_methods
    assert all(mode == "reduce-overhead" for _, mode in calls)


def test_compile_settings_override_resumed_config() -> None:
    """恢复训练时应使用当前 YAML 指定的编译开关与模式。"""
    app = AppConfig.load("configs/vla_adapter.yaml")
    app.policy.parameters["compile_model"] = True
    app.policy.parameters["compile_mode"] = "reduce-overhead"
    config = SimpleNamespace(
        output_dir=Path("outputs/train/test"),
        steps=1,
        seed=0,
        num_workers=0,
        persistent_workers=False,
        batch_size=1,
        log_freq=1,
        eval_steps=1,
        max_eval_samples=1,
        save_freq=1,
        resume=True,
        wandb=SimpleNamespace(
            enable=False,
            mode="offline",
            project="test",
            disable_artifact=True,
            run_id="existing-run",
        ),
        dataset=SimpleNamespace(
            repo_id=None,
            root=None,
            video_backend=None,
            episodes=None,
            eval_split=0.0,
            image_transforms=None,
        ),
        policy=SimpleNamespace(
            device="cpu",
            scheduler_warmup_steps=0,
            scheduler_decay_steps=0,
            compile_model=False,
            compile_mode="default",
            optimizer_lr=0.0,
        ),
    )

    apply_training_overrides(config, app.policy, app.training)

    assert config.policy.compile_model
    assert config.policy.compile_mode == "reduce-overhead"


def test_image_augmentation_uses_photometric_transforms_only() -> None:
    """训练开启颜色增强，不对焊接位置和姿态引入随机几何变换。"""
    app = AppConfig.load("configs/vla_adapter.yaml")
    train_config = make_train_config(app.policy, app.training, VLA_ADAPTER)
    transform_config = train_config.dataset.image_transforms
    assert transform_config.enable
    assert tuple(transform_config.tfs) == ("brightness", "contrast", "saturation", "hue")
    image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    augmented = ImageTransforms(transform_config)(image)
    assert augmented.shape == image.shape
    assert augmented.dtype == image.dtype
    assert not torch.equal(augmented, image)


def test_vla_adapter_config_round_trip(tmp_path: Path) -> None:
    """动作头与 Bridge 结构字段应随 checkpoint 稳定保存。"""
    config = make_tiny_config("flow_matching")
    config.compile_model = True
    config.compile_mode = "reduce-overhead"
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(restored, VLAAdapterConfig)
    assert restored.action_head == "flow_matching"
    assert restored.num_action_queries == 4
    assert restored.num_bridge_layers == 2
    assert restored.compile_model
    assert restored.compile_mode == "reduce-overhead"


def test_pro_bridge_uses_three_sources_and_backpropagates_gate() -> None:
    """Pro Bridge 应保持动作长度并让 raw gate 获得梯度。"""
    block = ProBridgeBlock(hidden_size=32, num_heads=4)
    action = torch.randn(2, 5, 32, requires_grad=True)
    query = torch.randn(2, 4, 32)
    raw = torch.randn(2, 8, 32)
    state = torch.randn(2, 1, 32)
    output = block(action, query, raw, state, torch.ones(2, 8, dtype=torch.bool))
    output.square().mean().backward()
    assert output.shape == action.shape
    assert block.raw_gate.item() == pytest.approx(0)
    assert block.raw_gate.grad is not None


@pytest.mark.parametrize("action_head", ["l1", "flow_matching"])
def test_shared_bridge_heads_predict_9d_chunks(
    action_head: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两个动作头应共享逐层条件并输出当前数据的 5×9 动作块。"""
    from welding_path_vla.policies.vla_adapter import modeling_vla_adapter

    monkeypatch.setattr(modeling_vla_adapter, "VLAAdapterBackbone", TinyBackbone)
    config = make_tiny_config(action_head)
    model = VLAAdapterModel(config)
    images = [torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)]
    masks = [torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)]
    language = torch.ones(2, 6, dtype=torch.long)
    context = model.encode_context(images, masks, language, language.bool(), torch.randn(2, 13))
    if action_head == "l1":
        prediction = model.predict_l1(context)
    else:
        prediction = model.sample_flow(context, noise=torch.zeros(2, 5, 9))
    assert prediction.shape == (2, 5, 9)


def test_flow_target_and_euler_steps_are_deterministic() -> None:
    """固定 noise/time 时应复现直线 Flow 目标与反向 Euler 更新。"""

    class TinyFlow(VLAAdapterModel):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.config = SimpleNamespace(
                flow_beta_alpha=1.5,
                flow_beta_beta=1.0,
                flow_time_scale=0.999,
                flow_time_offset=0.001,
                chunk_size=2,
                num_steps=2,
            )
            self.action_dim = 3

        def predict_velocity(
            self,
            context: BridgeContext,
            actions: torch.Tensor,
            time: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ones_like(actions)

    model = TinyFlow()
    context = BridgeContext((), (), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, 4))
    actions = torch.zeros(1, 2, 3)
    noise = torch.full_like(actions, 2)
    output = model.flow_matching_output(context, actions, noise, torch.tensor([0.25]))
    assert torch.allclose(output.noisy_actions, torch.full_like(actions, 0.5))
    assert torch.allclose(output.target_velocity, torch.full_like(actions, 2))
    assert torch.allclose(output.losses, torch.ones_like(actions))
    sampled = model.sample_flow(context, noise=torch.zeros_like(actions))
    assert torch.allclose(sampled, -torch.ones_like(actions))


def test_processor_reuses_qwen_prompt_and_standard_normalization(tmp_path: Path) -> None:
    """VLA-Adapter 应补齐上游动作提问，且不改变公共 processor 链。"""
    config = make_tiny_config()
    preprocessor, postprocessor = make_vla_adapter_pre_post_processors(config)
    names = [type(step).__name__ for step in preprocessor.steps]
    assert "VLAAdapterActionPromptProcessorStep" in names
    assert "NormalizerProcessorStep" in names
    assert type(postprocessor.steps[0]).__name__ == "UnnormalizerProcessorStep"
    prompt_step = next(step for step in preprocessor.steps if hasattr(step, "wrap"))
    prompt = prompt_step.wrap("Weld the seam?")
    assert "<|im_start|>user\nWhat action should the robot take to weld the seam?" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    detailed_prompt = prompt_step.wrap("Weld the seam.\nMove in the forward direction.")
    assert "weld the seam?\nmove in the forward direction." in detailed_prompt
    preprocessor.save_pretrained(tmp_path, config_filename="vla_processor.json")
    restored = PolicyProcessorPipeline.from_pretrained(
        tmp_path, config_filename="vla_processor.json"
    )
    assert type(restored.steps[2]).__name__ == "VLAAdapterActionPromptProcessorStep"


def test_peft_keeps_bridge_queries_and_selected_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """LoRA 应覆盖 Qwen 与视觉线性层，并完整保存新增动作模块。"""
    from welding_path_vla.policies.vla_adapter import modeling_vla_adapter
    from welding_path_vla.policies.vla_adapter.modeling_vla_adapter import VLAAdapterPolicy

    monkeypatch.setattr(modeling_vla_adapter, "VLAAdapterBackbone", TinyBackbone)
    policy = VLAAdapterPolicy(make_tiny_config())
    targets = policy._get_default_peft_targets()
    assert "model.action_queries" in targets["modules_to_save"]
    assert "model.bridge" in targets["modules_to_save"]
    assert "model.l1_output" in targets["modules_to_save"]
    assert "model.backbone.qwen.model.layers.0.self_attn.q_proj" in targets["target_modules"]
    assert "model.backbone.vision_encoder.0" in targets["target_modules"]
    policy.config.use_vision_lora = False
    frozen_targets = policy._get_default_peft_targets()["target_modules"]
    assert "model.backbone.vision_encoder.0" not in frozen_targets


@pytest.mark.parametrize("action_head", ["l1", "flow_matching"])
def test_tiny_policy_can_run_forward_after_lora_wrapping(
    action_head: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """PEFT 包装后两个动作头都应能完成前向、反向与保存。"""
    from peft import PeftModel

    from welding_path_vla.policies.vla_adapter import modeling_vla_adapter
    from welding_path_vla.policies.vla_adapter.modeling_vla_adapter import VLAAdapterPolicy

    monkeypatch.setattr(modeling_vla_adapter, "VLAAdapterBackbone", TinyBackbone)
    config = make_tiny_config(action_head)
    config.load_prismatic_weights = True
    policy = VLAAdapterPolicy(config)
    wrapped = policy.wrap_with_peft(
        peft_cli_overrides={"method_type": "LORA", "r": 2, "lora_alpha": 4}
    )
    assert isinstance(wrapped, PeftModel)
    assert any(
        "lora_" in name for name, parameter in wrapped.named_parameters() if parameter.requires_grad
    )
    assert any(
        "vision_encoder.0.lora_" in name
        for name, parameter in wrapped.named_parameters()
        if parameter.requires_grad
    )
    assert not any(
        "vision_encoder.0.base_layer" in name and parameter.requires_grad
        for name, parameter in wrapped.named_parameters()
    )
    batch = {
        "observation.images.global": torch.rand(2, 3, 32, 32),
        "observation.images.wrist": torch.rand(2, 3, 32, 32),
        OBS_STATE: torch.randn(2, 13),
        OBS_LANGUAGE_TOKENS: torch.ones(2, 6, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 6, dtype=torch.bool),
        ACTION: torch.randn(2, config.chunk_size, 9),
    }
    loss, _ = wrapped(batch)
    loss.backward()
    assert any(
        "vision_encoder.0.lora_" in name and parameter.grad is not None
        for name, parameter in wrapped.named_parameters()
    )
    wrapped.save_pretrained(tmp_path)
    adapter_weights = load_file(tmp_path / "adapter_model.safetensors")
    assert any("vision_encoder.0.lora_" in name for name in adapter_weights)
    adapter_config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert adapter_config["base_model_name_or_path"] == config.prismatic_repo_id


def test_peft_resume_rebuilds_prismatic_base_and_loads_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """PEFT 恢复应通过 Prismatic 来源重建基座并应用 adapter。"""
    from peft import PeftModel

    from welding_path_vla.policies.vla_adapter import modeling_vla_adapter
    from welding_path_vla.policies.vla_adapter.modeling_vla_adapter import VLAAdapterPolicy

    monkeypatch.setattr(modeling_vla_adapter, "VLAAdapterBackbone", TinyBackbone)
    config = make_tiny_config()
    config.load_prismatic_weights = True
    policy = VLAAdapterPolicy(config)
    wrapped = policy.wrap_with_peft(
        peft_cli_overrides={"method_type": "LORA", "r": 2, "lora_alpha": 4}
    )
    wrapped.save_pretrained(tmp_path)

    base = VLAAdapterPolicy.from_pretrained(config.prismatic_repo_id, config=config)
    resumed = PeftModel.from_pretrained(base, tmp_path, is_trainable=True)
    assert any("lora_" in name for name, _ in resumed.named_parameters())


def test_old_peft_checkpoint_gets_only_missing_base_reference(tmp_path: Path) -> None:
    """旧 checkpoint 只补基座标识，保留其余 PEFT 元数据。"""
    adapter = tmp_path / "adapter_config.json"
    adapter.write_text(json.dumps({"base_model_name_or_path": None, "r": 64}))

    ensure_peft_base_reference(tmp_path, "prismatic/base")

    restored = json.loads(adapter.read_text())
    assert restored == {"base_model_name_or_path": "prismatic/base", "r": 64}


def test_tiny_policy_checkpoint_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整 LeRobot checkpoint 应恢复配置与新增动作模块权重。"""
    from welding_path_vla.policies.vla_adapter import modeling_vla_adapter
    from welding_path_vla.policies.vla_adapter.modeling_vla_adapter import VLAAdapterPolicy

    monkeypatch.setattr(modeling_vla_adapter, "VLAAdapterBackbone", TinyBackbone)
    policy = VLAAdapterPolicy(make_tiny_config())
    with torch.no_grad():
        policy.model.action_queries.weight.fill_(0.25)
    policy.save_pretrained(tmp_path)
    restored = VLAAdapterPolicy.from_pretrained(tmp_path)
    assert restored.config.action_head == "l1"
    assert torch.allclose(
        restored.model.action_queries.weight,
        torch.full_like(restored.model.action_queries.weight, 0.25),
    )
