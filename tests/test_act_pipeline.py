from pathlib import Path

import numpy as np
import pytest
import torch

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.domain import Pose
from welding_path_vla.policies.act.data_adapter import scale_uint8_images
from welding_path_vla.policies.act.rollout import RolloutVideoRecorder
from welding_path_vla.policies.act.runtime import resolve_checkpoint
from welding_path_vla.policies.act.training import lerobot_train_config
from welding_path_vla.policies.training import TrainingRequest


def test_act_training_command_uses_lerobot_dataset_tools() -> None:
    config = AppConfig.load("configs/act.yaml")
    command = TrainingRequest(config.policy, config.training).command()
    assert command[0] == "lerobot-train"
    assert "--dataset.video_backend=torchcodec" in command
    assert "--dataset.eval_split=0.1" in command
    assert "--policy.type=act" in command
    assert "--policy.chunk_size=20" in command
    assert "--policy.n_action_steps=5" in command


def test_act_uses_official_lerobot_training_config() -> None:
    config = AppConfig.load("configs/act.yaml")
    training = lerobot_train_config(config.policy, config.training)
    assert training.policy.type == "act"
    assert training.dataset.video_backend == "torchcodec"
    assert training.dataset.eval_split == 0.1
    assert training.use_policy_training_preset


def test_rollout_video_is_browser_compatible_and_torchcodec_decodable(tmp_path: Path) -> None:
    import av
    from torchcodec.decoders import VideoDecoder

    config = AppConfig.load("configs/act.yaml")
    recorder = RolloutVideoRecorder.start(tmp_path, config)
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
    assert tuple(VideoDecoder(videos[0])[0].shape) == (3, 32, 32)



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
