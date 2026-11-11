"""PAIR 阶段一：只读取动作/状态列的轻量自编码器训练，不解码视频。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.geometry import relative_ee_actions_from_absolute, rotation_from_6d_rows
from welding_path_vla.policies.data import (
    dataset_feature_array,
    episode_row_ranges,
    lerobot_task_filter,
    metadata,
    validate_dataset,
)
from welding_path_vla.policies.lerobot_training import make_train_config
from welding_path_vla.policies.spec import TRAJ_VLA_QWEN
from welding_path_vla.policies.traj_vla_qwen.pair import (
    ActionAEConfig,
    ActionAutoencoder,
    masked_time_mean,
)


class ActionChunkDataset(Dataset):
    """从 Arrow 数值列构造共享 TCP 锚点的相对动作，不调用视频 __getitem__。"""

    def __init__(self, dataset, horizon: int, stride: int) -> None:
        self.actions = dataset_feature_array(dataset, "action", np.float32)
        self.states = dataset_feature_array(dataset, "observation.state", np.float32)
        episodes = dataset_feature_array(dataset, "episode_index", np.int64)
        self.ends = np.empty(len(episodes), dtype=np.int64)
        for _, start, end in episode_row_ranges(episodes):
            self.ends[start:end] = end
        self.offsets = np.arange(horizon) * stride

    def __len__(self) -> int:
        """每帧对应一个未来动作块，包含带有效步 mask 的 episode 末尾。"""
        return len(self.actions)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """复用项目 SE(3) 转换，与阶段二 processor 的共同锚点语义一致。"""
        indices = index + self.offsets
        padding = indices >= self.ends[index]
        actions = self.actions[np.minimum(indices, self.ends[index] - 1)]
        state = self.states[index]
        relative = relative_ee_actions_from_absolute(actions, state[6:9], state[9:13])
        # NumPy 位姿运算可能提升到 float64；与 AE 和归一化统计统一为 FP32。
        return {
            "action": torch.from_numpy(relative.astype(np.float32, copy=False)),
            "padding": torch.from_numpy(padding),
        }


@torch.no_grad()
def evaluate_autoencoder(
    accelerator: Accelerator, model, loader: DataLoader, mean: Tensor, std: Tensor, limit: int
) -> dict[str, float]:
    """记录干净/受扰动重建及物理误差，不把 AE loss 当作 VLA 部署指标。

    Args:
        accelerator: 负责分布式样本聚合。
        model: 动作自编码器，可为 DDP 包装。
        loader: 按 episode 留出的动作数据。
        mean: 训练集动作均值。
        std: 训练集动作标准差。
        limit: 最多评估的全局样本数，必须为正。

    Returns:
        样本平均的 clean/corrupt L1、位置毫米误差和旋转角误差。
    """
    model.eval()
    records = []
    count = 0
    for batch in loader:
        actions, padding = batch["action"], batch["padding"]
        normalized = (actions - mean) / (std + 1e-8)
        clean = model(normalized, padding, corrupt=False)
        corrupt = model(normalized, padding, corrupt=True)
        predicted = clean.float() * (std + 1e-8) + mean
        position = (predicted[..., :3] - actions[..., :3]).norm(dim=-1) * 1000
        target_rotation = rotation_from_6d_rows(actions[..., 3:])
        predicted_rotation = rotation_from_6d_rows(predicted[..., 3:])
        relative = predicted_rotation.transpose(-1, -2) @ target_rotation
        cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        angle = torch.rad2deg(cosine.clamp(-1, 1).acos())
        values = torch.stack(
            [
                masked_time_mean((clean.float() - normalized).abs().mean(-1), padding),
                masked_time_mean((corrupt.float() - normalized).abs().mean(-1), padding),
                masked_time_mean(position, padding),
                masked_time_mean(angle, padding),
            ],
            dim=-1,
        )
        values = accelerator.gather_for_metrics(values)
        records.append(values[: max(0, limit - count)].cpu())
        count += len(values)
        if count >= limit:
            break
    result = torch.cat(records).mean(0).tolist()
    model.train()
    return dict(
        zip(("clean_l1", "corrupt_l1", "position_mm", "orientation_deg"), result, strict=True)
    )


def train_action_autoencoder(app: AppConfig, ae_config: ActionAEConfig) -> None:
    """复用任务过滤/训练划分/统计，用独立小模型训练并保存教师及恢复状态。"""
    from lerobot.scripts import lerobot_train

    if app.policy.family != "traj_vla_qwen" or app.policy.checkpoint:
        raise ValueError("Stage 1 requires traj_vla_qwen configuration without policy.checkpoint")
    if app.policy.parameters.get("use_pair_bridge", False):
        raise ValueError("Stage 1 must not enable the policy Bridge")
    if app.training.phase_sampling.enabled:
        raise ValueError("Action AE uses all phases; disable phase_sampling for Stage 1")
    training = app.training
    if min(training.steps, training.batch_size, training.log_freq, training.save_freq) < 1:
        raise ValueError("Stage 1 steps, batch_size, log_freq and save_freq must be positive")
    if training.eval_steps and (training.eval_split <= 0 or training.max_eval_samples < 1):
        raise ValueError("Stage 1 evaluation requires held-out episodes and max_eval_samples > 0")
    accelerator = Accelerator(
        mixed_precision="no",
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True, data_seed=training.seed
        ),
    )
    set_seed(training.seed)
    output = Path(training.output_dir)
    if not training.resume and output.exists():
        raise FileExistsError(f"refusing to overwrite Stage 1 output: {output}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(output / "train.log"),
            ],
            force=True,
        )
    accelerator.wait_for_everyone()
    report = validate_dataset(training, app.policy)
    if (ae_config.horizon, ae_config.action_dim) != (
        app.policy.action_horizon,
        report.action_dimension,
    ):
        raise ValueError("Action AE horizon/action_dim must match the policy and dataset")
    # 独立 AE 不使用 VLA 的恢复逻辑；仅把相同 YAML 构造成数据工厂所需配置。
    config = make_train_config(app.policy, replace(training, resume=False), TRAJ_VLA_QWEN)
    if (
        list(config.policy.action_delta_indices)
        != (np.arange(ae_config.horizon) * app.policy.action_stride).tolist()
    ):
        raise ValueError("Action AE and policy action stride must match")
    if config.dataset.episodes is None:
        config.dataset.episodes = list(range(metadata(training).total_episodes))
    with lerobot_task_filter(app.policy):
        train_data, eval_data = lerobot_train.make_train_eval_datasets(config)
    stats = train_data.meta.stats["action"]
    mean = torch.as_tensor(stats["mean"], device=accelerator.device, dtype=torch.float32)
    std = torch.as_tensor(stats["std"], device=accelerator.device, dtype=torch.float32)
    teacher_metadata = {
        "dataset_repo_id": training.dataset_repo_id,
        "dataset_root": str(Path(training.dataset_root).resolve()),
        "train_episodes": sorted(train_data.episodes),
        "eval_episodes": sorted(eval_data.episodes) if eval_data is not None else [],
        "action_delta_indices": list(config.policy.action_delta_indices),
        "action_representation": app.policy.action_representation,
        "action_stats": {name: np.asarray(stats[name]).tolist() for name in ("mean", "std")},
    }
    loader_options = dict(
        batch_size=training.batch_size,
        num_workers=training.num_workers,
        pin_memory=accelerator.device.type == "cuda",
        persistent_workers=training.num_workers > 0,
    )
    train_loader = DataLoader(
        ActionChunkDataset(train_data, ae_config.horizon, app.policy.action_stride),
        shuffle=True,
        **loader_options,
    )
    eval_loader = (
        None
        if eval_data is None
        else DataLoader(
            ActionChunkDataset(eval_data, ae_config.horizon, app.policy.action_stride),
            shuffle=False,
            **loader_options,
        )
    )
    del train_data, eval_data
    model = ActionAutoencoder(ae_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.lr or 3e-4, weight_decay=1e-4)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if eval_loader is not None:
        eval_loader = accelerator.prepare(eval_loader)
    step, epoch, offset = 0, 0, 0
    if training.resume:
        last = (output / "checkpoints" / "last").resolve(strict=True)
        saved = json.loads((last / "config.json").read_text())
        if saved["model"] != asdict(ae_config) or any(
            saved[key] != value for key, value in teacher_metadata.items()
        ):
            raise ValueError("Stage 1 resume requires identical model/data/normalization")
        progress = json.loads((last / "progress.json").read_text())
        if (
            progress["world_size"] != accelerator.num_processes
            or progress["batch_size"] != training.batch_size
        ):
            raise ValueError("Stage 1 resume requires unchanged world size and batch size")
        accelerator.load_state(last / "training_state")
        step, epoch, offset = progress["step"], progress["epoch"], progress["offset"]
    if step >= training.steps:
        raise ValueError("training.steps must exceed the resumed step")
    if accelerator.is_main_process:
        logging.info(
            "Action AE: %s; train from step %d to %d; global batch %d",
            asdict(ae_config),
            step,
            training.steps,
            training.batch_size * accelerator.num_processes,
        )
    last_metrics = {}
    while step < training.steps:
        train_loader.set_epoch(epoch)
        batches = accelerator.skip_first_batches(train_loader, offset) if offset else train_loader
        model.train()
        for batch_index, batch in enumerate(batches, start=offset):
            optimizer.zero_grad(set_to_none=True)
            normalized = (batch["action"] - mean) / (std + 1e-8)
            reconstruction = model(normalized, batch["padding"])
            loss = masked_time_mean(
                F.l1_loss(reconstruction, normalized, reduction="none").mean(-1),
                batch["padding"],
            ).mean()
            accelerator.backward(loss)
            norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite Action AE gradient; update skipped")
            optimizer.step()
            step += 1
            if step % training.log_freq == 0:
                logged = accelerator.reduce(loss.detach(), reduction="mean")
                if accelerator.is_main_process:
                    logging.info(
                        "step:%d reconstruction_l1:%.6f grad_norm:%.4f", step, logged, norm
                    )
            if (
                eval_loader is not None
                and training.eval_steps
                and (step % training.eval_steps == 0 or step == training.steps)
            ):
                # 验证的随机 corruption 不改变后续训练随机数状态。
                # 单卡设备可能是无 index 的 cuda；传设备对象让 PyTorch 解析当前设备。
                devices = [accelerator.device] if accelerator.device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(training.seed + 1)
                    last_metrics = evaluate_autoencoder(
                        accelerator, model, eval_loader, mean, std, training.max_eval_samples
                    )
                if accelerator.is_main_process:
                    logging.info("step:%d validation:%s", step, last_metrics)
            if step % training.save_freq == 0 or step == training.steps:
                checkpoint = output / "checkpoints" / f"{step:06d}"
                accelerator.save_state(checkpoint / "training_state")
                if accelerator.is_main_process:
                    accelerator.unwrap_model(model).save(checkpoint, teacher_metadata)
                    (checkpoint / "progress.json").write_text(
                        json.dumps(
                            {
                                "step": step,
                                "epoch": epoch,
                                "offset": batch_index + 1,
                                "world_size": accelerator.num_processes,
                                "batch_size": training.batch_size,
                                "validation": last_metrics,
                            },
                            indent=2,
                        )
                    )
                    last = output / "checkpoints" / "last"
                    temporary = last.with_name("last.tmp")
                    temporary.unlink(missing_ok=True)
                    temporary.symlink_to(checkpoint.name, target_is_directory=True)
                    temporary.replace(last)
                accelerator.wait_for_everyone()
            if step >= training.steps:
                break
        epoch += 1
        offset = 0
    accelerator.end_training()
