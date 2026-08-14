# π0 / π0.5 训练、评估与仿真部署

项目直接使用 LeRobot 0.6 的 `PI0Policy`、`PI05Policy`、官方 PaliGemma tokenizer、
processor、训练器和 checkpoint 格式。两者共用项目的双相机、13D 状态、9D 相对末端
动作和语言任务；项目名 `pi0_5` 映射到 LeRobot 内部名称 `pi05`。

## 配置层次

```text
configs/base.yaml                 公共机器人和环境
configs/tasks/*.yaml              工件、焊缝与自然语言指令
configs/policies/pi0*.yaml        π0 本机 / 双 A100 参数
configs/policies/pi0_5*.yaml      π0.5 本机 / 双 A100 参数
configs/deploy/{model}.yaml       模型 checkpoint
configs/deploy/{model}_{task}.yaml 可直接部署的完整入口
```

本机 RTX 4060 8GB 配置使用 batch 1、BF16、梯度检查点和 rank 4 LoRA。LoRA 只适配
动作专家的注意力与投影层，动作块缩短到 15 帧，flow 推理缩短到 5 步。这仍然加载
完整官方 VLM 权重，首次运行需要从 Hugging Face 下载模型。应在训练前关闭占用 GPU
的桌面推理进程；如果完整模型仍无法装入剩余显存，应先使用 SmolVLA 或 ACT 验证数据，
π0 系列正式训练放到服务器。

双 A100 配置每卡使用 batch 4，全局 batch 为 8；同时启用 `torch.compile`，并
保持完整 30 帧动作块、10 个 flow 步和梯度检查点。它训练视觉编码器以外的 VLM 与动作
专家，完整遍历当前训练集约需 459,253 step。80 GiB A100 经短程显存测试确认有余量后，可以在
对应 A100 YAML 中关闭 `gradient_checkpointing`，进一步减少每一步的重计算。

## 数据检查与训练

```bash
pixi run -e train policy-data-check --config_path=configs/pi0.yaml
pixi run -e train train-policy --config_path=configs/pi0.yaml --dry_run=true
pixi run -e train train-policy --config_path=configs/pi0.yaml

pixi run -e train policy-data-check --config_path=configs/pi0_5.yaml
pixi run -e train train-policy --config_path=configs/pi0_5.yaml
```

π0.5 使用分位数归一化。当前数据集的状态和动作统计已包含
`q01/q10/q50/q90/q99`，不需要重新导出。

服务器只需换入口文件，不需要写一串命令行覆盖：

```bash
pixi run -e train train-policy-2gpu --config_path=configs/pi0_a100.yaml
pixi run -e train train-policy-2gpu --config_path=configs/pi0_5_a100.yaml
```

训练日志固定写入 `training.output_dir/train.log`。恢复训练时把目标总步数调大，并覆盖
一次 `resume`：

```bash
pixi run -e train train-policy \
  --config_path=configs/pi0.yaml \
  --training.resume=true \
  --training.steps=10000
```

## 离线评估与部署

离线评估使用任务均衡的留出 episode，报告 flow-matching loss 和归一化动作 MAE：

```bash
pixi run -e train policy-evaluate \
  --config_path=configs/pi0_5.yaml \
  --policy.checkpoint=outputs/train/pi0_5_weldpath_relative_v1/checkpoints/last/pretrained_model
```

每个部署入口已经组合模型、场景和任务。切换工件只切换文件：

```bash
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/pi0_5_l_joint.yaml
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/pi0_5_pipe_bottom.yaml
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/pi0_5_pipe_top.yaml
```

如果训练输出目录不同，只需在 `configs/deploy/pi0.yaml` 或
`configs/deploy/pi0_5.yaml` 修改一次 checkpoint，三个任务入口会自动继承。
