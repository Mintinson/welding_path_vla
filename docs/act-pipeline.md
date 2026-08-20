# ACT 训练、评估与仿真部署

ACT 是当前代码和数据链路的轻量对比基线。它直接使用 LeRobot 的 `ACTPolicy`、训练器、
processor 和 checkpoint，不读取自然语言任务。

## 数据契约

- `observation.images.global`、`observation.images.wrist`：双路 RGB；
- `observation.state`：6 个关节角 + TCP 位置 + `wxyz` 四元数，共 13 维；
- policy action：当前 TCP 局部系的 3D 平移 + rotation-6D，共 9 维；
- 频率：30 Hz；默认 future horizon 为 30 帧，执行 8 帧后重新规划。

图像解码和 `[0, 255] → [0, 1]` 由 LeRobot 完成。ACT 忽略 `task`，因此只能作为固定任务的
行为克隆基线，不能用它论证语言理解能力。LeRobot Parquet 中的 action 仍是世界系 absolute
target，公共 preprocessor 会在送入 ACT 前完成 relative 变换。

```bash
pixi run -e train policy-data-check --config_path=configs/act.yaml
```

## 训练

先预览配置：

```bash
pixi run -e train train-policy \
  --config_path=configs/act.yaml \
  --dry_run=true
```

建议先运行四步 smoke test：

```bash
pixi run -e train train-policy \
  --config_path=configs/act.yaml \
  --training.steps=4 \
  --training.batch_size=1 \
  --training.num_workers=0 \
  --training.log_freq=1 \
  --training.eval_steps=2 \
  --training.max_eval_samples=2 \
  --training.save_freq=4 \
  --training.output_dir=outputs/train/act_smoke
```

正式训练：

```bash
pixi run -e train train-policy --config_path=configs/act.yaml
pixi run -e train train-policy-2gpu --config_path=configs/act_a100.yaml
```

输出目录包含 `train.log`、offline W&B 日志以及：

```text
checkpoints/<step>/
├── pretrained_model/   # 模型、配置和前后处理器
├── training_state/     # optimizer、scheduler、随机状态和 step
└── train_config.json
```

恢复训练使用相同 `training.output_dir` 和 `--training.resume=true`。`training.steps` 是最终总
step，不是本次追加量。

## 离线评估

```bash
pixi run -e train policy-evaluate \
  --config_path=configs/act.yaml \
  --policy.checkpoint=outputs/train/RUN/checkpoints/last/pretrained_model
```

报告包含总 loss、ACT L1/KL loss 和归一化动作 MAE。它只能说明留出数据拟合质量。

## 仿真部署

```bash
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/act.yaml \
  --policy.checkpoint=outputs/train/RUN/checkpoints/last/pretrained_model \
  --deployment.episodes=5
```

rollout 依次完成双相机观测、processor、ACT chunk、relative-action 解码、安全门、IK 和仿真
执行，并保存：

- `config.json`：完整有效配置；
- `rollout.npz`：复现状态转移和安全判定所需的原始量；
- `summary.json`：终止、完成、跟踪、控制、安全和论文指标；
- `global.mp4`、`wrist.mp4`：30 FPS H.264 视频。

ACT 配置使用较宽松的自然退出门槛，便于基线正常结束；该门槛不改变论文统一的 ESR 和 CTE
判定。应分别解读 `summary.json` 的 rollout `completed` 与 `evaluation.success`。

## 公共策略边界

ACT 与 SmolVLA、π0、π0.5 和 Trajectory-VLA 共用 `LeRobotPipeline`。模型差异集中在
`policies/spec.py`；新增策略不要复制训练、评估、runtime 或 rollout。详见
[策略扩展方式](policy-extension.md)。
