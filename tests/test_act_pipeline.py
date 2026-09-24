import json
from pathlib import Path

import av
import numpy as np
import pytest
import torch

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.deployment_rules import (
    COLLISION_FRAME_HOLD_S,
    COMPLETION_DISTANCE_M,
    COMPLETION_EXIT_DISTANCE_M,
    COMPLETION_PROGRESS_MIN,
    deployment_rules_metadata,
)
from welding_path_vla.core.geometry import quaternion_to_matrix, rotation_to_6d_rows
from welding_path_vla.dataset.video import VideoRecorder, video_codec_settings
from welding_path_vla.policies import simulation_rollout
from welding_path_vla.policies.checkpoint import resolve_checkpoint
from welding_path_vla.policies.data import scale_uint8_images
from welding_path_vla.policies.lerobot_training import make_train_config
from welding_path_vla.policies.rollout_diagnostics import (
    RolloutCompletion,
    seam_endpoint_reached,
)
from welding_path_vla.policies.simulation_rollout import (
    rollout_episode,
    terminal_video_frame_count,
)
from welding_path_vla.policies.spec import ACT
from welding_path_vla.policies.training import TrainingRequest
from welding_path_vla.simulation.evaluation_scenes import config_for_scene, load_scene_bank


def test_act_training_command_uses_lerobot_dataset_tools() -> None:
    config = AppConfig.load("configs/act.yaml")
    command = TrainingRequest(config.policy, config.training).command()
    assert command[0] == "lerobot-train"
    assert "--dataset.video_backend=torchcodec" in command
    assert "--dataset.eval_split=0.1" in command
    assert "--policy.type=act" in command
    assert "--policy.chunk_size=30" in command
    assert "--policy.n_action_steps=8" in command


def test_act_uses_official_lerobot_training_config() -> None:
    config = AppConfig.load("configs/act.yaml")
    training = make_train_config(config.policy, config.training, ACT)
    assert training.policy.type == "act"
    assert training.dataset.video_backend == "torchcodec"
    assert training.dataset.eval_split == 0.1
    assert training.use_policy_training_preset


def test_rollout_video_is_browser_compatible_and_pyav_decodable(tmp_path: Path) -> None:
    config = AppConfig.load("configs/act.yaml")
    names = (config.camera.global_name, config.camera.wrist_name)
    recorder = VideoRecorder.start(tmp_path, names, config.timing.policy_hz)
    assert recorder.encoder._encoder_threads == 2
    assert len(recorder.encoder._threads) == len(names)
    assert all(thread.is_alive() for thread in recorder.encoder._threads.values())
    images = {
        name: np.zeros((32, 32, 3), dtype=np.uint8)
        for name in (config.camera.global_name, config.camera.wrist_name)
    }
    for _ in range(3):
        recorder.append(images)
    videos = recorder.finish()
    recorder.close()

    with av.open(videos[0]) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.pix_fmt == "yuv420p"
        assert not stream.codec_context.has_b_frames
    with av.open(videos[0]) as container:
        frame = next(container.decode(video=0))
        assert frame.to_ndarray(format="rgb24").shape == (32, 32, 3)


def test_rollout_video_uses_explicit_intra_frame_safe_h264_settings() -> None:
    """rollout 视频必须固定软件编码器和短 GOP，避免异步/长参考链污染画面。"""
    assert video_codec_settings() == {
        "vcodec": "h264",
        "pix_fmt": "yuv420p",
        "g": 2,
        "bf": 0,
        "crf": 23,
        "preset": "veryfast",
        "threads": 2,
    }


class StaticRuntime:
    """输出恒等位姿增量的最小策略运行时。"""

    def reset(self) -> None:
        pass

    def select_action(self, observation: object) -> np.ndarray:
        return np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)


class ClippedRuntime:
    """始终输出超过单步平移上限的绝对目标。"""

    def __init__(self) -> None:
        self.discarded = 0

    def reset(self) -> None:
        pass

    def select_action(self, observation: object) -> np.ndarray:
        position = observation.state[6:9] + np.array([0.04, 0.0, 0.0])
        return np.concatenate((position, [1, 0, 0, 0, 1, 0]))

    def discard_pending_actions(self) -> None:
        self.discarded += 1


class HoldPoseRuntime:
    """保持当前 TCP 位姿，用于验证固定场景的完成状态机调用。"""

    def reset(self) -> None:
        pass

    def select_action(self, observation: object) -> np.ndarray:
        state = np.asarray(observation.state)
        rotation = quaternion_to_matrix(state[9:13])
        return np.concatenate((state[6:9], rotation_to_6d_rows(rotation)))


def test_clipped_action_discards_stale_chunk(tmp_path: Path) -> None:
    config = AppConfig.load("configs/act.yaml")
    config.camera.width = 32
    config.camera.height = 32
    config.deployment.max_steps = 1
    config.deployment.record_video = False
    runtime = ClippedRuntime()
    report = rollout_episode(config, runtime, 0, tmp_path / "clipped_episode")
    with np.load(report.trace_path) as trace:
        assert bool(trace["action_increment"][0])
    assert runtime.discarded == 1


def test_act_rollout_writes_diagnostics_and_terminal_video_frame(tmp_path: Path) -> None:
    config = AppConfig.load("configs/act.yaml")
    config.camera.width = 32
    config.camera.height = 32
    config.deployment.max_steps = 1
    config.deployment.record_video = True
    config.evaluation.report_detailed_metrics = True

    report = rollout_episode(config, StaticRuntime(), 0, tmp_path / "episode_000000")
    trajectory = np.load(report.trace_path)
    required = {
        "collision_pairs",
        "command_tcp_position",
        "ik_residual_m",
        "joint_command",
        "seam_distance_m",
        "seam_progress",
    }
    derived = {
        "action_translation_norm_m",
        "completion_mask",
        "joint_margin_min_rad",
        "seam_progress_raw",
        "tcp_displacement_m",
        "tcp_speed_m_s",
    }
    assert required <= set(trajectory.files)
    assert derived.isdisjoint(trajectory.files)
    assert all(len(trajectory[name]) == report.steps for name in trajectory.files)
    assert report.diagnostics["video"]["includes_terminal_state"]
    assert report.diagnostics["tracking"]["frames"] == 1
    assert report.diagnostics["deployment_rules"] == deployment_rules_metadata()
    assert (
        json.loads((tmp_path / "episode_000000/config.json").read_text())["policy"]["family"]
        == "act"
    )
    saved_summary = json.loads(
        (tmp_path / "episode_000000/summary.json").read_text(encoding="utf-8")
    )
    assert saved_summary["diagnostics"]["deployment_rules"] == deployment_rules_metadata()

    with av.open(report.videos[0]) as container:
        assert sum(1 for _ in container.decode(video=0)) == report.steps + 1


def test_rollout_video_only_holds_collision_frame() -> None:
    """只有碰撞视频会按固定时长停留，其他终止只保留一个终态帧。"""
    assert pytest.approx(1.0) == COLLISION_FRAME_HOLD_S
    assert terminal_video_frame_count("completed", 30) == 1
    assert terminal_video_frame_count("timeout", 30) == 1
    assert terminal_video_frame_count("safety_violation: joint limit", 30) == 1
    assert terminal_video_frame_count("collision", 30) == 30


def test_rollout_completion_rules_are_fixed_across_policy_configs() -> None:
    """模型配置不能覆盖部署完成规则，论文 evaluation 阈值仍独立。"""
    config = AppConfig.load("configs/act.yaml")
    qwen_config = AppConfig.load("configs/traj_vla_qwen.yaml")
    fields = (
        "completion_progress_min",
        "completion_distance_m",
        "completion_exit_distance_m",
        "collision_frame_hold_s",
    )
    assert all(not hasattr(config.deployment, name) for name in fields)
    assert all(not hasattr(qwen_config.deployment, name) for name in fields)
    assert pytest.approx(0.95) == COMPLETION_PROGRESS_MIN
    assert pytest.approx(0.01) == COMPLETION_DISTANCE_M
    assert pytest.approx(0.01) == COMPLETION_EXIT_DISTANCE_M
    assert config.evaluation.pcr_min == pytest.approx(0.95)
    assert seam_endpoint_reached(0.95, 0.01)
    assert not seam_endpoint_reached(0.949, 0.01)
    assert not seam_endpoint_reached(1.0, 0.0101)


def test_rollout_does_not_finish_at_the_seam_endpoint() -> None:
    """到达焊缝末端后仍应给策略留下执行退出动作的时间。"""
    completion = RolloutCompletion()

    assert not completion.update(0.95, 0.005)
    assert completion.endpoint_reached
    assert not completion.update(1.0, 0.014)
    assert completion.update(1.0, 0.016)


def test_rollout_completion_accepts_clear_exit_without_reaching_expert_post() -> None:
    """离开焊缝即可完成，不应要求策略复现专家退出方向。"""
    completion = RolloutCompletion()

    assert not completion.update(0.98, 0.002)
    assert completion.update(1.0, 0.013)


def test_scene_rollout_uses_exit_completion_state_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """固定场景也必须经过“到达末端→离开焊缝”两阶段完成判定。"""
    scene = next(
        item
        for item in load_scene_bank("configs/evaluation_scenes_v2.json")["scenes"]
        if item["task"]["task_id"] == "l_joint"
    )
    config = config_for_scene(AppConfig.load("configs/act.yaml"), scene)
    config.deployment.max_steps = 1
    config.deployment.record_video = False
    updates: list[tuple[float, float]] = []

    class CapturingCompletion:
        """记录 rollout 是否调用两阶段完成判定器。"""

        def __init__(self) -> None:
            pass

        def update(self, progress: float, seam_distance_m: float) -> bool:
            updates.append((progress, seam_distance_m))
            return False

    monkeypatch.setattr(simulation_rollout, "RolloutCompletion", CapturingCompletion)
    report = simulation_rollout.rollout_episode(
        config,
        HoldPoseRuntime(),
        int(scene["episode"]),
        tmp_path / "scene_episode",
        scene,
    )

    assert report.termination_reason == "timeout"
    assert len(updates) == 1


def test_act_rgb_preprocessing_scales_uint8() -> None:
    batch = {
        "observation.images.global": torch.full((1, 3, 2, 2), 255, dtype=torch.uint8),
        "observation.images.wrist": torch.zeros((1, 3, 2, 2), dtype=torch.uint8),
    }
    converted = scale_uint8_images(batch)
    assert converted["observation.images.global"].dtype == torch.float32
    assert converted["observation.images.global"].max() == 1
    assert converted["observation.images.wrist"].min() == 0


def test_checkpoint_resolver_accepts_training_run(tmp_path: Path) -> None:
    model = tmp_path / "checkpoints" / "last" / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_checkpoint(tmp_path) == model.resolve()
