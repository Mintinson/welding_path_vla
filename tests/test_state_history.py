"""历史运动的因果采样、动作锚点、在线缓存和配置兼容性回归。"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.processor import (
    PolicyProcessorPipeline,
    batch_to_transition,
    transition_to_batch,
)
from lerobot.processor.converters import TransitionKey
from lerobot.utils.constants import OBS_STATE

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.action_processors import RelativeEEActionsProcessorStep
from welding_path_vla.policies.base import Observation
from welding_path_vla.policies.data import lerobot_task_filter
from welding_path_vla.policies.runtime import LeRobotRuntime
from welding_path_vla.policies.spec import TRAJ_VLA_QWEN
from welding_path_vla.policies.state_history import (
    DEFAULT_HISTORY_OFFSETS,
    HISTORY_TIMES,
    MOTION_FEATURES,
    RobotStateHistory,
    StateHistoryProcessorStep,
)
from welding_path_vla.policies.traj_vla_qwen.configuration_traj_vla_qwen import TrajVLAQwenConfig
from welding_path_vla.policies.traj_vla_qwen.modeling_traj_vla_qwen import (
    QwenTrajectoryFlowModel,
    TrajVLAQwenPolicy,
)


def state_sequence(times):
    """生成已知速度：关节 0.1 rad/s、TCP x 0.01 m/s、姿态恒定。"""
    times = torch.as_tensor(times, dtype=torch.float32)
    states = torch.zeros(len(times), 13)
    states[:, :6] = times[:, None] * 0.1
    states[:, 6] = 0.4 + times * 0.01
    states[:, 9] = 1
    return states


def history_transition(states, padding=None, times=None):
    """构造原始 transition，默认历史均有效。"""
    obs = {OBS_STATE: states[None]}
    if times is not None:
        obs[HISTORY_TIMES] = torch.as_tensor(times)[None]
    return {
        TransitionKey.OBSERVATION: obs,
        TransitionKey.COMPLEMENTARY_DATA: {
            f"{OBS_STATE}_is_pad": torch.tensor([padding or [False] * len(states)])
        },
    }


def test_history_velocities_sign_invariance_padding_and_current_anchor():
    """旋转符号翻转不产生伪运动；无效历史归零，相对动作仍锚定当前 TCP。"""
    states = state_sequence(DEFAULT_HISTORY_OFFSETS)
    states[1, 9] = -1
    transition = history_transition(states)
    transition[TransitionKey.ACTION] = torch.tensor([[[0.41, 0, 0, 1, 0, 0, 0, 1, 0]]])
    processed = StateHistoryProcessorStep()(transition)
    features = processed[TransitionKey.OBSERVATION][MOTION_FEATURES].reshape(1, 2, 17)
    torch.testing.assert_close(features[0, :, :6], torch.ones(2, 6))
    torch.testing.assert_close(features[0, :, 6], torch.ones(2), atol=2e-5, rtol=0)
    assert not features[..., 7:15].any()
    relative = RelativeEEActionsProcessorStep()(processed)
    torch.testing.assert_close(relative[TransitionKey.ACTION][..., 0], torch.tensor([[0.01]]))
    torch.testing.assert_close(processed[TransitionKey.OBSERVATION][OBS_STATE], states[-1:])
    padded = StateHistoryProcessorStep()(history_transition(states, [True, False, False]))
    assert not padded[TransitionKey.OBSERVATION][MOTION_FEATURES][:, :17].any()


def test_live_dt_and_episode_start_are_not_fake_motion():
    """在线采用实际 dt；不足历史和时间断层不会被解释成巨大速度。"""
    times = (-0.27, -0.14, 0)
    step = StateHistoryProcessorStep()
    result = step(history_transition(state_sequence(times), times=times))
    motion = result[TransitionKey.OBSERVATION][MOTION_FEATURES].reshape(2, 17)
    torch.testing.assert_close(motion[:, :6], torch.ones(2, 6))
    torch.testing.assert_close(motion[:, 15], torch.tensor([0.13, 0.14]))
    result = step(history_transition(state_sequence(times), times=(-5, -0.14, 0)))
    assert not result[TransitionKey.OBSERVATION][MOTION_FEATURES][:, :17].any()
    history = RobotStateHistory(DEFAULT_HISTORY_OFFSETS)
    history.record(1_800_000_000.0, state_sequence([0])[0].numpy())
    sample = history.observation()
    assert sample[f"{OBS_STATE}_is_pad"].tolist() == [[True, True, False]]
    pipeline = PolicyProcessorPipeline(
        [step], to_transition=batch_to_transition, to_output=transition_to_batch
    )
    assert not pipeline(sample)[MOTION_FEATURES].any()


def test_history_processor_roundtrip_keeps_nondefault_settings(tmp_path):
    """保存后的 processor 必须恢复自定义采样点和速度尺度。"""
    step = StateHistoryProcessorStep(offsets_s=(-0.2, 0), velocity_scales=(0.2, 0.03, 0.4))
    pipeline = PolicyProcessorPipeline(
        [step], name="history", to_transition=batch_to_transition, to_output=transition_to_batch
    )
    pipeline.save_pretrained(tmp_path)
    restored = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="history.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    batch = {
        OBS_STATE: state_sequence([-0.2, 0])[None],
        f"{OBS_STATE}_is_pad": torch.zeros(1, 2, dtype=torch.bool),
    }
    torch.testing.assert_close(pipeline(batch)[MOTION_FEATURES], restored(batch)[MOTION_FEATURES])


def test_runtime_records_every_tick_with_eight_step_action_queue():
    """第二次预测须读取第 0/4/8 帧；不能只保留两次调用模型时的状态。"""
    batches = []
    policy = SimpleNamespace(
        config=SimpleNamespace(
            use_state_history=True,
            state_history_offsets_s=DEFAULT_HISTORY_OFFSETS,
            n_action_steps=8,
        ),
        predict_action_chunk=lambda batch: torch.zeros(1, 8, 9),
        reset=lambda: None,
    )

    def capture(batch):
        batches.append(batch)
        return batch

    runtime = LeRobotRuntime(policy, capture, lambda actions: actions, "cpu", TRAJ_VLA_QWEN)
    base = Observation(0.0, {}, state_sequence([0])[0].numpy(), "weld")
    for frame in range(9):
        time = frame / 30
        runtime.select_action(
            replace(base, timestamp=time, state=state_sequence([time])[0].numpy())
        )
    assert len(batches) == 2
    torch.testing.assert_close(batches[-1][OBS_STATE][0], state_sequence([0, 4 / 30, 8 / 30]))
    assert not batches[-1][f"{OBS_STATE}_is_pad"].any()
    runtime.discard_pending_actions()
    assert runtime.state_history.samples
    # 直接调用 observation_sample 也会记录；重复时间不会增加缓冲长度。
    latest = replace(base, timestamp=8 / 30)
    runtime.observation_sample(latest)
    assert len(runtime.state_history.samples) == 9
    runtime.preprocessor = SimpleNamespace(reset=lambda: None)
    runtime.postprocessor = SimpleNamespace(reset=lambda: None)
    runtime.reset()
    assert not runtime.state_history.samples and not runtime.action_queue


def test_dataset_factory_changes_only_state_and_restores_itself():
    """训练工厂保留每路单帧图像、动作窗口和标量 timestamp，退出作用域还原。"""
    from lerobot.datasets import factory

    profile = AppConfig.load("configs/traj_vla_qwen_state_history_a100.yaml")
    cfg = TrajVLAQwenConfig(use_state_history=True)
    meta = SimpleNamespace(
        fps=30,
        features={
            OBS_STATE: {},
            "observation.images.wrist": {},
            "action": {},
            "timestamp": {},
        },
    )
    original = factory.resolve_delta_timestamps
    with lerobot_task_filter(profile.policy):
        deltas = factory.resolve_delta_timestamps(cfg, meta)
        assert deltas[OBS_STATE] == list(DEFAULT_HISTORY_OFFSETS)
        assert deltas["observation.images.wrist"] == [0.0]
        assert "timestamp" not in deltas
        assert deltas["action"] == [index / 30 for index in cfg.action_delta_indices]
    assert factory.resolve_delta_timestamps is original
    assert profile.policy.checkpoint is None and not profile.training.resume
    assert profile.policy.parameters["vision_lora_last_n_blocks"] == 4
    assert profile.policy.parameters["use_task_condition_attention"]


def tiny_history_model(monkeypatch, enabled):
    """用微型视觉查表替换大骨干，保持真实 Flow 上下文拼接与条件 mask。"""
    from welding_path_vla.policies.traj_vla_qwen import modeling_traj_vla_qwen

    class TinyBackbone(torch.nn.Module):
        language_hidden_size = 8
        expert_hidden_size = 8

        def __init__(self, config):
            super().__init__()
            self.tokens = torch.nn.Embedding(16, 8)
            self.image_calls = 0

        def embed_language_tokens(self, tokens):
            return self.tokens(tokens)

        def embed_image_with_geometry(self, image):
            self.image_calls += 1
            return self.tokens(image), self.tokens(image)

    monkeypatch.setattr(modeling_traj_vla_qwen, "PrismaticQwenWithExpert", TinyBackbone)
    config = TrajVLAQwenConfig(
        attention_mode="cross_attn",
        use_geometry_branch=True,
        geometry_attention_mode="direct",
        use_task_condition_attention=True,
        use_state_history=enabled,
    )
    return QwenTrajectoryFlowModel(config)


def test_history_token_gradient_and_old_path_compatibility(monkeypatch):
    """新增一个可训练条件 token，旧状态投影数值不变，视觉仍只前向两次。"""
    model = tiny_history_model(monkeypatch, True)
    state = torch.randn(1, 66)
    tokens = model.embed_context_state(state)
    assert tokens.shape == (1, 2, 8)
    torch.testing.assert_close(tokens[:, 0], model.state_proj(state[:, :32]), rtol=0, atol=0)
    (tokens * torch.randn_like(tokens)).sum().backward()
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.state_history_encoder.parameters()
    )
    context = model.encode_context(
        [torch.tensor([[0, 1]]), torch.tensor([[2, 3]])],
        [torch.tensor([True]), torch.tensor([True])],
        torch.tensor([[4, 5]]),
        torch.ones(1, 2, dtype=torch.bool),
        state,
    )
    assert context.embeddings.shape == (1, 8, 8)
    assert context.expert_context.state_mask.sum() == 2
    assert model.vlm_with_expert.image_calls == 2
    old = tiny_history_model(monkeypatch, False)
    old.load_state_dict(
        {k: v for k, v in model.state_dict().items() if not k.startswith("state_history_encoder.")},
        strict=True,
    )
    assert not hasattr(old, "state_history_encoder")
    torch.testing.assert_close(old.embed_context_state(state[:, :32]), tokens[:, 0], rtol=0, atol=0)
    policy = TrajVLAQwenPolicy.__new__(TrajVLAQwenPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = model.config
    batch = {OBS_STATE: torch.zeros(1, 13), MOTION_FEATURES: torch.randn(1, 34)}
    assert policy.prepare_state(batch).shape == (1, 66)
    assert "model.state_history_encoder" in policy._get_default_peft_targets()["modules_to_save"]


@pytest.mark.parametrize("offsets", [(0,), (-0.1, 0.1), (-0.1, -0.2, 0), (float("nan"), 0)])
def test_invalid_or_future_history_rejected(offsets):
    """配置拒绝未来帧和时间倒序，避免训练信息泄漏。"""
    with pytest.raises(ValueError, match="offsets"):
        TrajVLAQwenConfig(use_state_history=True, state_history_offsets_s=offsets)


def test_history_config_roundtrip_and_legacy_default(tmp_path):
    """新配置可完整恢复；没有历史字段的旧模型仍默认关闭。"""
    import json

    config = TrajVLAQwenConfig(use_state_history=True, state_history_offsets_s=(-0.2, 0.0))
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.use_state_history and tuple(restored.state_history_offsets_s) == (-0.2, 0.0)
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    path.write_text(
        json.dumps(
            {
                key: value
                for key, value in saved.items()
                if key != "use_state_history" and not key.startswith("state_history_")
            }
        )
    )
    assert not PreTrainedConfig.from_pretrained(tmp_path).use_state_history


def test_peft_restored_history_respects_freezing(monkeypatch):
    """PEFT 将保存模块解冻后，优化器构建仍遵守 train_state_proj=false。"""
    from peft import LoraConfig, get_peft_model

    model = tiny_history_model(monkeypatch, True)
    model.config.train_state_proj = False
    model.state_history_encoder.requires_grad_(False)
    wrapped = get_peft_model(
        model,
        LoraConfig(
            r=2,
            target_modules=["action_in_proj"],
            modules_to_save=["state_history_encoder"],
        ),
    )
    assert any(p.requires_grad for p in wrapped.state_history_encoder.parameters())
    policy = TrajVLAQwenPolicy.__new__(TrajVLAQwenPolicy)
    torch.nn.Module.__init__(policy)
    policy.config, policy.model = model.config, wrapped
    parameters = list(policy.get_optim_params())
    assert parameters
    assert not any(p.requires_grad for p in wrapped.state_history_encoder.parameters())
