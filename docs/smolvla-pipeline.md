# SmolVLA 多任务基线

项目直接使用 LeRobot 0.6 的 `SmolVLAPolicy`、`lerobot/smolvla_base`、processor、训练器和
checkpoint 格式。项目层只负责统一焊接 observation、relative action、配置和 robosuite rollout。

## 数据契约

- 双路 RGB：`observation.images.global`、`observation.images.wrist`；
- 13D 状态：6 个关节角、TCP 位置和 `wxyz` 四元数；
- 9D 动作：TCP 局部系平移与 rotation-6D future target；
- 英文任务：LeRobot `task`；
- 频率：30 Hz，默认预测 30 帧并执行 8 帧后重新规划。

LeRobot Parquet 中保留世界系 absolute target；共享 processor 在归一化前转换为预测时刻 TCP
坐标系下的 relative chunk。训练、评估和部署必须使用 checkpoint 内保存的同一组 processor。

检查数据：

```bash
pixi run -e train policy-data-check --config_path=configs/smolvla.yaml
```

## 训练与恢复

本机配置位于 `configs/policies/smolvla.yaml`，双 A100 覆盖位于
`configs/policies/smolvla_a100.yaml`。常用组合入口：

```bash
# 预览最终 LeRobot 配置
pixi run -e train train-policy \
  --config_path=configs/smolvla.yaml \
  --dry_run=true

# 新建单进程训练（当前本机入口保留 resume=true，故显式关闭）
pixi run -e train train-policy \
  --config_path=configs/smolvla.yaml \
  --training.resume=false

# 双 A100 DDP
pixi run -e train train-policy-2gpu --config_path=configs/smolvla_a100.yaml
```

默认冻结视觉编码器和 VLM 主干，训练动作专家与状态投影。实际可训练参数、分辨率、batch、
scheduler 和 step 应以 `--dry_run=true` 输出为准，不在文档中复制一份容易过期的配置。

当前 `configs/policies/smolvla.yaml` 为继续本地已有实验保留了 `training.resume: true`；创建新输出
目录时应显式覆盖为 `false`。`Ctrl+C` 中断后，保持原 `training.output_dir` 并恢复：

```bash
pixi run -e train train-policy \
  --config_path=configs/smolvla.yaml \
  --training.resume=true
```

恢复会读取 `checkpoints/last` 中的模型、processor、optimizer、scheduler、随机状态和全局 step。
`training.steps` 是最终总 step；如果 checkpoint 已达到该值，需要覆盖为更大的总目标。

只想从已有权重开始新实验，而不恢复 optimizer 和 step 时使用：

```bash
pixi run -e train train-policy \
  --config_path=configs/smolvla.yaml \
  --policy.checkpoint=CHECKPOINT/pretrained_model \
  --training.output_dir=outputs/train/smolvla_new_experiment
```

日志固定写入 `<training.output_dir>/train.log`，W&B 默认 offline 写入同目录 `wandb/`。

## 离线评估

```bash
pixi run -e train policy-evaluate \
  --config_path=configs/smolvla.yaml \
  --policy.checkpoint=CHECKPOINT/pretrained_model
```

报告在任务间均衡抽样，并包含 flow-matching loss 与归一化动作 MAE。它衡量行为克隆拟合，
不能替代闭环 ESR、CTE、碰撞和完成度。

## 仿真部署

部署入口把基础场景、公共 checkpoint 和具体任务组合在一起。当前提供五类任务：

```bash
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla_l_joint.yaml
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla_pipe_bottom.yaml
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla_pipe_top.yaml
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla_curve_plate.yaml
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla_trihedral_horizontal.yaml
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla_trihedral_vertical.yaml
```

一次运行全部任务并自动写入 `outputs/deploy/smolvla_{task_id}`：

```bash
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla.yaml \
  --deployment.run_all_tasks=true
```

公共 checkpoint 位于 `configs/deploy/smolvla.yaml`。也可在命令行临时覆盖：

```bash
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla_pipe_top.yaml \
  --policy.checkpoint=outputs/train/RUN/checkpoints/last/pretrained_model \
  --deployment.episodes=5
```

服务器无法访问 Hugging Face、且模型已经完整缓存时，可在命令前加
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，避免 Transformers 探测可选远端文件。

每个 rollout 保存 `config.json`、`rollout.npz`、`summary.json` 和双相机 H.264 视频。
策略安全门可能裁剪过大的平移增量并继续闭环；部署自然退出条件与论文统一成功条件应分别解释，
完整过程与排错方式见[仿真策略部署](simulation-deployment.md)，指标定义见[评估规范](evaluation.md)。
