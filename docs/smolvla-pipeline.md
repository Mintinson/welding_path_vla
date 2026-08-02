# SmolVLA 三任务基线

本项目直接复用 LeRobot 0.6 的 `SmolVLAPolicy`、`lerobot/smolvla_base`、
processor、数据集和训练器。项目代码只负责把统一 YAML、焊接 observation/action
语义和 robosuite rollout 接到官方接口上。

## 1. 采集三个任务

每条命令中的 `collection.episodes=100` 表示本次必须新增 100 条通过质量门槛的
episode；规划、碰撞或跟踪失败的尝试会保留用于诊断，但不会计入 100 条。

```bash
pixi run -e sim sim-collect \
  --config_path=configs/default.yaml \
  --collection.episodes=100

pixi run -e sim sim-collect \
  --config_path=configs/pipe_bottom.yaml \
  --collection.episodes=100

pixi run -e sim sim-collect \
  --config_path=configs/pipe_top.yaml \
  --collection.episodes=100
```

## 2. 合并为一个 LeRobot 数据集

第一次创建数据集，后两次通过增量清单把新任务追加到同一目标。导出器只选择
`valid_success` 和 `valid_recovery`，并把 episode 的中文任务指令写入 LeRobot
`task`。

```bash
pixi run -e data export-lerobot \
  --config_path=configs/smolvla.yaml \
  --dataset=datasets/weldpath_raw_v2 \
  --output=datasets/weldpath_lerobot_relative_v1 \
  --repo_id=huayan/weldpath_relative_v1 \
  --lerobot_export.incremental=true

pixi run -e data export-lerobot \
  --config_path=configs/smolvla.yaml \
  --dataset=datasets/weldpath_pipe_bottom_raw_v2 \
  --output=datasets/weldpath_lerobot_relative_v1 \
  --repo_id=huayan/weldpath_relative_v1 \
  --lerobot_export.incremental=true

pixi run -e data export-lerobot \
  --config_path=configs/smolvla.yaml \
  --dataset=datasets/weldpath_pipe_top_raw_v2 \
  --output=datasets/weldpath_lerobot_relative_v1 \
  --repo_id=huayan/weldpath_relative_v1 \
  --lerobot_export.incremental=true

pixi run -e train policy-data-check --config_path=configs/smolvla.yaml
```

当前已生成的数据为：

| 任务 | 有效成功 episode | 原始目录 |
|---|---:|---|
| L 形直线角焊缝 | 113 | `datasets/weldpath_raw_v2` |
| 圆管下沿四分之一圆弧 | 100 | `datasets/weldpath_pipe_bottom_raw_v2` |
| 圆管上沿完整圆弧 | 100 | `datasets/weldpath_pipe_top_raw_v2` |

`valid_recovery` 表示成功完成扰动恢复的有效示范，也计入成功数据。统一数据集包含
313 个 episode、436,928 帧和 3 条任务指令；图像只保存为双视角视频，不重复保存逐帧
图片。

## 3. 训练与离线评估

RTX 4060 8GB 的默认配置使用 batch size 16、BF16、256×256 双相机输入，并冻结
视觉编码器及 VLM 主干，只训练动作专家和状态投影。实测峰值约 2.6 GiB、持续吞吐约
80 samples/s；batch size 32 的吞吐反而稍低，因此不采用。训练前可以先检查完整计划：

```bash
pixi run -e train train-policy \
  --config_path=configs/smolvla.yaml \
  --dry_run=true

pixi run -e train train-policy \
  --config_path=configs/smolvla.yaml
```

训练被 `Ctrl+C` 中断后，使用原来的 `training.output_dir` 并启用 LeRobot 原生恢复：

```bash
pixi run -e train train-policy \
  --config_path=configs/smolvla.yaml \
  --training.resume=true
```

程序会解析 `training.output_dir/checkpoints/last`，恢复模型、optimizer、scheduler、
随机数状态、数据采样位置和全局 step。`training.steps` 遵循 LeRobot 语义，表示训练
结束时的总步数，而不是本次追加步数。例如从 step 3,500 恢复且
`training.steps=5000`，本次继续训练 1,500 步；如果 checkpoint 已达到 5,000，则应
把总目标覆盖为更大的值：

```bash
pixi run -e train train-policy \
  --config_path=configs/smolvla.yaml \
  --training.resume=true \
  --training.steps=10000
```

每次训练都会把与终端相同的 LeRobot 指标追加到
`<training.output_dir>/train.log`，此功能不依赖 WandB。

如需加载权重并在新的输出目录开始另一轮微调，而不恢复 optimizer 和全局 step，则
使用 `policy.checkpoint`：

```bash
pixi run -e train train-policy \
  --config_path=configs/smolvla.yaml \
  --policy.checkpoint=outputs/train/smolvla_weldpath_relative_v1/checkpoints/last/pretrained_model \
  --training.output_dir=outputs/train/smolvla_weldpath_relative_v1_finetune \
  --training.save_freq=2500
```

训练完成后，`checkpoints/last/pretrained_model` 包含模型权重、processor 配置和训练
数据归一化统计；语言 tokenizer 按模型配置从本地 Hugging Face 缓存解析。离线测试
固定从每个任务选取 3 个留出 episode，并在三类任务间均衡抽帧：

```bash
pixi run -e train policy-evaluate \
  --config_path=configs/smolvla.yaml \
  --policy.checkpoint=outputs/train/smolvla_weldpath_relative_v1_finetune/checkpoints/best/pretrained_model \
  --output=outputs/evaluation/policies/smolvla_weldpath_relative_v1_finetune.json
```

## 4. 仿真闭环部署

部署入口已经组合好基础场景、SmolVLA checkpoint、工件和任务。切换工件时只需切换
配置文件，策略 runtime 会把该任务的 `task.instruction` 交给官方 tokenizer：

```bash
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla_l_joint.yaml

pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla_pipe_bottom.yaml

pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla_pipe_top.yaml
```

公共 checkpoint 和 episode 数量位于 `configs/deploy/smolvla.yaml`；三个入口只选择任务、
最大步数和输出目录。临时实验仍可使用 Draccus 覆盖，例如
`--deployment.episodes=10`。

rollout 会保存双视角 H.264 视频、逐步轨迹、碰撞对、IK 诊断和论文评价指标。
策略平移增量超过仿真安全上限时会等比例裁剪、记录
`action_increment=true` 并继续闭环；该 episode 不会因此被误报为碰撞，但正式安全
评价仍会把裁剪计为动作越界。

## 5. 本次训练与测试结果

本次先训练 5,000 步，再从其最终 checkpoint 继续训练 5,000 步。最终模型位于：

```text
outputs/train/smolvla_weldpath_relative_v1_finetune/checkpoints/best/pretrained_model
```

离线评估固定使用 9 个留出 episode，每个任务 3 个，并均衡抽取 50 帧。指标使用
LeRobot 归一化动作空间；数值越低越好。

| checkpoint | 均衡 loss | 归一化动作 MAE |
|---|---:|---:|
| 初训 500 步 | 0.4206 | 0.4054 |
| 初训 5,000 步 | 0.2019 | 0.2735 |
| 续训 2,500 步 | 0.1828 | 0.2940 |
| 续训 5,000 步（最终） | **0.1474** | **0.2356** |

最终 checkpoint 的三任务闭环 smoke test 结果如下。SmolVLA 在此项目中是轻量对比
基线，当前训练量用于证明数据流、学习能力和部署接口正确，不将离线拟合结果误报为
闭环成功。

| 任务 | 结果 | 最佳现象 | 安全状态 |
|---|---|---|---|
| L 形直线 | 1,000 步超时 | 最大投影进度 0.447 | 无碰撞 |
| 管件下圆弧 | 第 1,165 步碰撞退出 | 最大投影进度 1.000，最近距离 21.5 mm | 腕部相机外壳碰桌 |
| 管件上圆周 | 3,300 步超时 | 最大进度 0.741，最近距离 12.75 mm，200 帧进入 15 mm 区域 | 无碰撞 |

完整离线报告在 `outputs/evaluation/policies/`，三项 rollout 位于
`outputs/deploy/smolvla_weldpath_relative_v1_final_{straight,bottom,top}/`。每个目录均包含
`summary.json`、`rollout.npz` 和双视角视频。视频已验证为 640×480、30 FPS、
H.264/yuv420p，并可由 TorchCodec 解码。
