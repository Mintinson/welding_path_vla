# ACT 训练、测试与仿真部署

## 1. 数据契约

ACT 直接读取 `datasets/weldpath_lerobot_v2`，不复制数据。项目适配器会检查：

- `observation.images.global` 与 `observation.images.wrist`：双路 RGB；
- `observation.state`：6 轴关节角 + 3D TCP 位置 + 4D `wxyz` 四元数，共 13 维；
- `action`：TCP 局部坐标系中的 3D 平移 + 6D 旋转，共 9 维；
- 采样率：30 Hz。

LeRobot 根据 `policy.action_horizon` 自动查询 future action，并生成 `action_is_pad`。自然语言
`task` 当前不进入 ACT。Pixi 将 CPU 环境的 PyTorch 2.9 固定搭配 TorchCodec 0.9，将训练环境
的 PyTorch 2.11 固定搭配 TorchCodec 0.11；训练默认使用 LeRobot 的 `torchcodec` 后端。图像
的 `uint8 [0,255]` 到 `float32 [0,1]` 转换由 LeRobot 官方训练循环完成。

检查数据：

```bash
pixi run -e train policy-data-check --config_path=configs/act.yaml
```

## 2. 训练

先检查最终的数据划分、模型和训练计划：

```bash
pixi run -e train train-policy --config_path=configs/act.yaml --dry_run=true
```

执行一个四步 smoke test：

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
  --training.output_dir=outputs/train/act_smoke_01
```

smoke test 使用 batch size 1、单进程 DataLoader，在第 2 步计算少量留出集 loss，并在第 4
步保存 checkpoint。正式训练：

```bash
pixi run -e train train-policy --config_path=configs/act.yaml
```

RTX 4060 Laptop 8GB 的起点配置是 batch size 2 和 BF16 autocast。显存不足时先把 batch
size 降为 1，不要改变 action 定义。训练直接调用 LeRobot 官方训练器，复用其 Accelerate、
episode-aware sampler、optimizer preset、日志、评估和 checkpoint 恢复。输出目录包含：

- `checkpoints/*/train_config.json`：完整训练配置；
- `checkpoints/*/pretrained_model/`：权重和前后处理器；
- `checkpoints/*/training_state/`：optimizer、scheduler 和随机状态；
- 终端日志：loss、梯度范数、显存、数据读取与更新耗时；启用 `training.wandb` 后同步记录。

LeRobot checkpoint 位于：

```text
outputs/train/act_weldpath_v2/checkpoints/last/pretrained_model/
```

模型默认预测 30 步（1 秒）动作块，每执行 8 步（约 0.27 秒）后重新观察并规划。视觉骨干使用
ImageNet ResNet-18 权重；首次运行可能下载该权重。

本机以官方训练器完成的四步 smoke test（RTX 4060 Laptop、BF16、batch size 1）结果为：

- 51,566,473 个可训练参数，峰值显存约 1.08 GiB；
- 训练 loss 从 70.994 降至 54.193；
- 留出集 loss 从 0.8448 降至 0.8328；
- TorchCodec 解码、反向更新、离线评估和 checkpoint 恢复全部成功。

该结果只证明数据流和训练链路有效，不代表模型已经收敛。

## 3. 离线测试

离线测试使用数据集末尾 9 个 episode，不参与训练时的前 90% episode：

```bash
pixi run -e train policy-evaluate \
  --config_path=configs/act.yaml \
  --policy.checkpoint=outputs/train/act_weldpath_v2
```

报告包含总 loss、ACT L1/KL loss、归一化 action MAE、样本数和数据 schema。快速检查可加
`--policy_evaluation.max_batches=2`。这是行为克隆拟合质量，不等价于焊缝跟踪成功率。

## 4. robosuite 闭环部署与轨迹评价

```bash
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/act.yaml \
  --policy.checkpoint=outputs/train/act_weldpath_v2 \
  --deployment.episodes=5 \
  --deployment.log_dir=outputs/deploy/act_weldpath_v2_eval
```

每个 rollout 会：

1. 随机化工件、初始关节和 TCP；
2. 从 robosuite observation dictionary 读取与训练一致的双相机和机器人状态；
3. 用 checkpoint 中保存的 LeRobot processor 完成归一化和反归一化；
4. 将 9D 局部 action 恢复为世界坐标 TCP 目标；
5. 经过 TCP 增量、IK、关节范围与碰撞安全门后执行；
6. 保存完整有效配置、逐步诊断轨迹、结构化摘要和双相机视频；
7. 复用项目评价模块计算 PCR、CTE、姿态误差、SpeedMAPE、安全条件和 ESR。

每个 episode 的排查产物如下：

- `config.json`：包含命令行覆盖在内的完整有效配置，可直接复现实验；
- `rollout.npz`：同时保存 observation、action、目标 TCP、IK 命令与残差、执行状态、焊缝
  进度/距离、安全信号、碰撞对及逐步错误；
- `summary.json`：`diagnostics` 分为 termination、completion rule、reference、tracking、
  control、safety 和 video，未进入焊缝跟踪区时也能给出原因；
- `global.mp4`、`wrist.mp4`：第 0 帧是初态，第 n 帧是第 n 个 action 执行后的状态，因此
  包含碰撞或安全终止后的最终画面。

`rollout.npz` 只持久化复现状态转移和安全判定所需的数据。动作范数、TCP 速度、完成区掩码等
派生量不重复保存，由 `summary.json` 生成时从原始轨迹和有效配置计算。

视频由 LeRobot 的流式 PyAV 编码器生成，固定为 H.264 与 `yuv420p`，可在 VS Code 和主流
浏览器中直接播放。

ACT 在本项目中主要承担 pipeline 验证和对比基线，因此 `configs/act.yaml` 使用独立的部署完成
条件：焊缝进度达到 85%，且 TCP 距焊缝不超过 15 mm 时自然退出。该条件只控制 rollout 生命周期；
论文评价仍使用统一的 PCR 95%、CTE、姿态和安全阈值，`summary.json` 中的 `completed` 与
`evaluation.success` 应分别解读。

smoke checkpoint 尚未学会任务时通常会被 action 安全门提前终止，这是正确行为。使用
`--deployment.record_video=false --deployment.max_steps=2 --deployment.episodes=1`
可以只验证部署数据流。

由于当前 ACT 忽略语言且场景只有一条目标焊缝，仿真 ICR 仅表示固定任务和方向正确，不能作为
语言理解能力结论。

## 5. 策略切换边界

`policies/factory.py` 是公共注册入口。每种策略实现同一组能力：

- `training_overrides`：生成 LeRobot 训练参数；
- `train`：执行对应策略的训练入口；
- `load`：加载策略和前后处理器；
- `evaluate`：离线 checkpoint 评价；
- `deploy_simulation`：robosuite 闭环 rollout。

训练、评估和部署脚本只依赖该接口。后续加入 Diffusion、SmolVLA 或 Trajectory VLA 时，不需要
修改数据采集协议或 ACT 内部代码。
