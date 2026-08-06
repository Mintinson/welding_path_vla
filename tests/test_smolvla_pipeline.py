from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.base import Observation
from welding_path_vla.policies.checkpoint import find_resume_checkpoint, resolve_checkpoint
from welding_path_vla.policies.data import balanced_frame_indices
from welding_path_vla.policies.lerobot_training import make_policy_config, make_train_config
from welding_path_vla.policies.process import lerobot_config_argument, lerobot_training_log
from welding_path_vla.policies.runtime import LeRobotRuntime
from welding_path_vla.policies.spec import SMOLVLA
from welding_path_vla.policies.training import TrainingRequest


def test_smolvla_config_uses_official_pretrained_model() -> None:
    """SmolVLA 应复用官方权重，并让当前数据集重新推断 feature。"""
    config = AppConfig.load("configs/smolvla.yaml")
    policy = make_policy_config(config.policy, SMOLVLA)
    assert str(policy.pretrained_path) == "lerobot/smolvla_base"
    assert policy.input_features == {}
    assert policy.output_features == {}
    expected_size = config.policy.parameters.get("resize_imgs_with_padding", (512, 512))
    assert tuple(policy.resize_imgs_with_padding) == tuple(expected_size)
    assert policy.chunk_size == 30
    assert policy.n_action_steps == 8


def test_smolvla_uses_lerobot_training_config() -> None:
    """训练计划应直接使用 LeRobot 的 SmolVLA 和视频数据集配置。"""
    config = AppConfig.load("configs/smolvla.yaml")
    training = make_train_config(
        config.policy,
        replace(config.training, resume=False),
        SMOLVLA,
    )
    assert training.policy.type == "smolvla"
    assert str(training.policy.pretrained_path) == "lerobot/smolvla_base"
    assert training.dataset.return_uint8
    assert training.dataset.eval_split == 0.1
    assert training.batch_size == 16


def test_smolvla_command_uses_pretrained_path() -> None:
    """命令行预览应与程序化训练使用相同官方 checkpoint。"""
    config = AppConfig.load("configs/smolvla.yaml")
    command = TrainingRequest(config.policy, replace(config.training, resume=False)).command()
    assert "--policy.path=lerobot/smolvla_base" in command
    assert "--policy.input_features=null" in command
    assert "--policy.chunk_size=30" in command


def test_smolvla_training_can_continue_from_checkpoint() -> None:
    """统一 checkpoint 字段应覆盖官方基线，便于追加微调。"""
    config = AppConfig.load("configs/smolvla.yaml")
    policy = replace(config.policy, checkpoint="outputs/train/previous/pretrained_model")
    command = TrainingRequest(policy, replace(config.training, resume=False)).command()
    assert "--policy.path=outputs/train/previous/pretrained_model" in command


def test_policy_evaluation_balances_frames_across_tasks() -> None:
    """有限测试预算不应被数据集中排在最前面的任务独占。"""

    class Dataset:
        def __init__(self) -> None:
            self.hf_dataset = {"task_index": [0] * 10 + [1] * 20 + [2] * 30}

        def __len__(self) -> int:
            return 60

    dataset = cast(Any, Dataset())
    indices = balanced_frame_indices(dataset, 9)
    tasks = np.asarray(dataset.hf_dataset["task_index"])[indices]
    assert [(tasks == task).sum() for task in range(3)] == [3, 3, 3]


class CaptureProcessor:
    """记录 runtime 送入 LeRobot processor 的单样本 batch。"""

    def __init__(self) -> None:
        self.batch: dict[str, object] = {}

    def __call__(self, batch: dict[str, object]) -> dict[str, object]:
        self.batch = batch
        return batch


class StaticSmolPolicy:
    """返回固定 9D 动作的最小 SmolVLA 替身。"""

    config = type("PolicyConfig", (), {"n_action_steps": 1})()

    def reset(self) -> None:
        pass

    def predict_action_chunk(self, batch: dict[str, object]) -> torch.Tensor:
        return torch.zeros((1, 1, 9), dtype=torch.float32)


def test_smolvla_runtime_keeps_language_instruction() -> None:
    """运行时必须把焊接指令和双相机图像交给官方 processor。"""
    processor = CaptureProcessor()
    runtime = LeRobotRuntime(
        cast(Any, StaticSmolPolicy()),
        processor,
        lambda action: action,
        "cpu",
        SMOLVLA,
    )
    observation = Observation(
        0.0,
        {
            "global": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        },
        np.zeros(13, dtype=np.float32),
        "Weld around the top rim of the pipe.",
    )
    action = runtime.select_action(observation)
    assert action.shape == (9,)
    assert processor.batch["task"] == [observation.instruction]
    assert processor.batch["observation.state"].shape == (1, 13)
    assert processor.batch["observation.images.global"].shape == (1, 3, 8, 8)


def test_smolvla_checkpoint_resolver_accepts_run_directory(tmp_path: Path) -> None:
    """SmolVLA runtime 与 ACT 使用相同的 LeRobot checkpoint 目录约定。"""
    model = tmp_path / "checkpoints" / "last" / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")

    assert resolve_checkpoint(tmp_path) == model.resolve()


def test_smolvla_resume_finds_model_and_training_step(tmp_path: Path) -> None:
    """恢复训练必须同时定位模型权重和 optimizer 对应的全局 step。"""
    checkpoint = tmp_path / "checkpoints" / "003500"
    model = checkpoint / "pretrained_model"
    state = checkpoint / "training_state"
    model.mkdir(parents=True)
    state.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "train_config.json").write_text("{}", encoding="utf-8")
    (state / "training_step.json").write_text('{"step": 3500}', encoding="utf-8")
    (tmp_path / "checkpoints" / "last").symlink_to("003500")

    resume = find_resume_checkpoint(tmp_path)

    assert resume.root == checkpoint.resolve()
    assert resume.model == model.resolve()
    assert resume.step == 3500


def test_lerobot_resume_receives_checkpoint_config(tmp_path: Path) -> None:
    """项目 YAML 不应遮蔽 LeRobot 恢复所需的 checkpoint config_path。"""
    import sys

    config = tmp_path / "train_config.json"
    arguments = sys.argv
    with lerobot_config_argument(config):
        assert sys.argv == [arguments[0], f"--config_path={config}"]
    assert sys.argv is arguments


def test_lerobot_local_log_uses_official_file_handler(monkeypatch: Any, tmp_path: Path) -> None:
    """关闭 WandB 时仍应把日志路径交给 LeRobot 自带 logging。"""
    from importlib import import_module

    module = import_module("lerobot.scripts.lerobot_train")
    received: dict[str, object] = {}

    def capture_init_logging(*args: object, **kwargs: object) -> None:
        received.update(kwargs)

    monkeypatch.setattr(module, "init_logging", capture_init_logging)
    log_path = tmp_path / "train.log"
    with lerobot_training_log(log_path):
        cast(Any, module).init_logging()

    assert received["log_file"] == log_path
