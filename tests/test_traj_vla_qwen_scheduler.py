"""验证 Qwen 短实验使用的实际学习率曲线，而不只核对 YAML 字段。"""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import draccus
import pytest
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig, LRSchedulerConfig

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.lerobot_training import make_policy_config
from welding_path_vla.policies.spec import TRAJ_VLA_QWEN
from welding_path_vla.policies.traj_vla_qwen.configuration_traj_vla_qwen import TrajVLAQwenConfig


def build_scheduler(config: TrajVLAQwenConfig, steps: int):
    """通过训练器实际使用的 factory 构建单参数 optimizer 和 scheduler。"""
    parameter = torch.nn.Parameter(torch.ones(1))
    pipeline = SimpleNamespace(
        use_policy_training_preset=True,
        optimizer=config.get_optimizer_preset(),
        scheduler=config.get_scheduler_preset(),
        steps=steps,
    )
    policy = SimpleNamespace(get_optim_params=lambda: [parameter])
    return make_optimizer_and_scheduler(pipeline, policy)


def test_short_run_preserves_explicit_schedule() -> None:
    """关闭自动压缩时，40k 短实验必须与 200k 训练的前 40k 学习率逐点相同。"""
    config = TrajVLAQwenConfig(
        optimizer_lr=5e-5,
        scheduler_warmup_steps=4000,
        scheduler_decay_steps=200000,
        scheduler_auto_scale=False,
    )
    _, short = build_scheduler(config, 40000)
    _, long = build_scheduler(config, 200000)
    for step in (0, 1, 800, 3999, 4000, 20000, 40000):
        assert short.lr_lambdas[0](step) == pytest.approx(long.lr_lambdas[0](step)), step
    assert short.lr_lambdas[0](40000) * config.optimizer_lr == pytest.approx(4.5464153616405e-5)


def test_legacy_auto_scaling_is_unchanged(tmp_path: Path) -> None:
    """无新字段的 checkpoint 保持自动压缩；不默默改变已有实验的训练过程。"""
    config = TrajVLAQwenConfig(
        optimizer_lr=5e-5,
        scheduler_warmup_steps=4000,
        scheduler_decay_steps=200000,
    )
    config.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    saved.pop("scheduler_auto_scale")
    path.write_text(json.dumps(saved))
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.scheduler_auto_scale
    assert type(restored.get_scheduler_preset()) is CosineDecayWithWarmupSchedulerConfig
    _, scheduler = build_scheduler(restored, 40000)
    assert scheduler.lr_lambdas[0](0) == pytest.approx(1 / 801)
    assert scheduler.lr_lambdas[0](40000) * restored.optimizer_lr == pytest.approx(2.5e-6)


def test_fixed_scheduler_serialization_and_resume(tmp_path: Path) -> None:
    """保存恢复 scheduler 类型和状态后，延长训练仍继续原曲线，不重新 warmup。"""
    config = TrajVLAQwenConfig(
        optimizer_lr=5e-5,
        scheduler_warmup_steps=4000,
        scheduler_decay_steps=200000,
        scheduler_auto_scale=False,
    )
    config.save_pretrained(tmp_path)
    assert not PreTrainedConfig.from_pretrained(tmp_path).scheduler_auto_scale
    encoded = draccus.encode(config.get_scheduler_preset())
    assert encoded["type"] == "fixed_cosine_decay_with_warmup"
    restored_config = draccus.decode(LRSchedulerConfig, encoded)
    optimizer, scheduler = build_scheduler(config, 40000)
    # 用训练器的真实 step 路径走到 40k，避免只检查调度配置的假阳性。
    for _ in range(40000):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(4.5464153616405e-5)
    resumed_optimizer, _ = build_scheduler(config, 100000)
    resumed = restored_config.build(resumed_optimizer, 100000)
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed.load_state_dict(scheduler.state_dict())
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        resumed_optimizer.step()
        resumed.step()
        assert resumed.get_last_lr() == pytest.approx(scheduler.get_last_lr())


def test_fixedlr_profile_is_an_isolated_rerun() -> None:
    """新实验只更改调度约定和输出目录，不加载短周期动作权重。"""
    old = AppConfig.load("configs/traj_vla_qwen_unscaled_a100.yaml")
    new = AppConfig.load("configs/traj_vla_qwen_unscaled_fixedlr_a100.yaml")
    parameters = dict(old.policy.parameters, scheduler_auto_scale=False)
    assert new.policy.parameters == parameters
    assert new.policy.checkpoint is None
    assert new.training.output_dir != old.training.output_dir
    assert new.training.steps == 40000
    assert new.training.image_aug
    assert not new.training.resume
    assert new.training.include_tasks == ["curve_plate"]


def test_training_config_restores_fixed_scheduler(tmp_path: Path) -> None:
    """训练配置完整序列化后仍恢复自定义调度器，覆盖 checkpoint 恢复入口。"""
    config = TrajVLAQwenConfig(scheduler_auto_scale=False, push_to_hub=False)
    pipeline = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/local"),
        policy=config,
        output_dir=tmp_path / "run",
        steps=40000,
    )
    pipeline.validate()
    pipeline.save_pretrained(tmp_path)
    restored = TrainPipelineConfig.from_pretrained(tmp_path / "train_config.json")
    assert restored.scheduler.type == "fixed_cosine_decay_with_warmup"
    assert not restored.policy.scheduler_auto_scale


def test_scaled_shortlr_matches_unscaled_actual_schedule() -> None:
    """补齐单变量对照：训练设置不变，全部 40k 步的实际 LR 等同已完成的短周期。"""
    old = AppConfig.load("configs/traj_vla_qwen_unscaled_a100.yaml")
    new = AppConfig.load("configs/traj_vla_qwen_scaled_shortlr_a100.yaml")
    assert new.policy.checkpoint is None
    assert not new.training.resume
    assert new.policy.parameters == dict(
        old.policy.parameters,
        scale_context_embeddings=True,
        scheduler_auto_scale=False,
        scheduler_warmup_steps=800,
        scheduler_decay_steps=40000,
    )
    assert new.training.output_dir != old.training.output_dir
    assert asdict(new.training) == dict(asdict(old.training), output_dir=new.training.output_dir)
    assert new.deployment == old.deployment
    schedulers = []
    for profile in (old, new):
        config = make_policy_config(profile.policy, TRAJ_VLA_QWEN)
        config.optimizer_lr = profile.training.lr
        _, scheduler = build_scheduler(config, profile.training.steps)
        schedulers.append(scheduler)
    # 检查实际 factory 生成的函数，避免只核对配置数值而漏掉运行时变换。
    for step in range(40001):
        assert schedulers[0].lr_lambdas[0](step) == schedulers[1].lr_lambdas[0](step)
    assert schedulers[1].lr_lambdas[0](40000) * new.training.lr == pytest.approx(2.5e-6)
