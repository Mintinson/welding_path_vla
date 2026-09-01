# π0 / π0.5 训练、评估与部署

项目使用 LeRobot 0.6 的 `PI0Policy`、`PI05Policy`、PaliGemma tokenizer、processor 和
checkpoint 格式。项目名 `pi0_5` 对应 LeRobot 内部类型 `pi05`。

## 共同数据接口

π0 和 π0.5 与其他策略读取相同的双相机、13D 状态、英文任务和 9D relative action chunk。
π0.5 使用分位数归一化，因此数据集 `meta/stats.json` 必须包含对应 action/state quantile。

两者也读取 `task.direction` 和 `task.parameters`。公共 `WeldingPromptBuilder` 先生成焊接方向、
速度和工具角文本，随后才进入 π0 / π0.5 自己的 PaliGemma prompt、状态离散化与 tokenizer。
字段由 `policy.welding_prompt_fields` 独立选择，空列表可用于原始任务文本基线；该配置不是
PaliGemma 或 π 模型参数。

训练和部署都通过项目公共 processor 将世界系 absolute target 与 TCP 局部系 relative chunk
互相转换。旧 LeRobot checkpoint 可能没有顶层 policy `type`；运行时会按所选策略的具体配置类
用 Draccus 恢复 `PolicyFeature`，不应手工修改 checkpoint JSON。

## 本机与 A100 配置

本机入口：

- `configs/pi0.yaml`；
- `configs/pi0_5.yaml`。

它们以低显存验证为目标，使用 batch 1、BF16、冻结视觉主干、较少 flow 推理步和 LoRA。
完整模型仍需要加载，8 GiB GPU 不一定有足够余量；数据链路可先用 ACT 或 SmolVLA 验证。

双 A100 入口：

- `configs/pi0_a100.yaml`：每卡 batch 4，当前不使用 PEFT；
- `configs/pi0_5_a100.yaml`：每卡 batch 4，动作专家 LoRA；
- 二者恢复 30 帧动作块和 10 个推理步，并启用 `torch.compile`。

π0.5 A100 当前关闭 gradient checkpointing；π0 A100 的其他训练边界以 YAML 为准。40 GiB 与
80 GiB A100 的余量不同，正式训练前必须用短程 smoke test 读取实际 `gpu_mem_gb`，不能仅凭
“A100”名称判断是否可以继续解冻或增大 batch。

## 训练与恢复

```bash
# 数据和配置检查
pixi run -e train policy-data-check --config_path=configs/pi0_5.yaml
pixi run -e train train-policy \
  --config_path=configs/pi0_5.yaml \
  --dry_run=true

# 本机单进程
pixi run -e train train-policy --config_path=configs/pi0_5.yaml

# 服务器双 A100
pixi run -e train train-policy-2gpu --config_path=configs/pi0_5_a100.yaml
```

恢复训练使用相同输出目录；`training.steps` 是最终总 step：

```bash
pixi run -e train train-policy-2gpu \
  --config_path=configs/pi0_5_a100.yaml \
  --training.resume=true \
  --training.steps=500000
```

恢复 checkpoint 会同时加载 processor、LoRA adapter（若存在）、optimizer、scheduler 和全局
step。仅开始新微调实验时才使用 `policy.checkpoint` 加新输出目录。

## 离线评估

```bash
pixi run -e train policy-evaluate \
  --config_path=configs/pi0_5.yaml \
  --policy.checkpoint=outputs/train/RUN/checkpoints/last/pretrained_model
```

报告包含 flow-matching loss 和归一化动作 MAE。留出集结果不代表闭环焊接成功。

## 仿真部署

每个策略的公共 checkpoint 分别配置在 `configs/deploy/pi0.yaml` 和
`configs/deploy/pi0_5.yaml`。任务入口包括：

```text
l_joint
pipe_bottom
pipe_top
curve_plate
trihedral_horizontal
trihedral_vertical
```

例如：

```bash
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/pi0_5_pipe_top.yaml \
  --policy.checkpoint=outputs/train/pi0_5_weldpath_relative_v1_a100/checkpoints/last/pretrained_model \
  --deployment.episodes=5
```

运行时会把 processor 缓存的 TCP 锚点对齐到 policy action 的 device 和 dtype，因此 CPU
观测与 CUDA / BF16 模型输出可以安全完成 relative-action 解码。服务器模型已完整缓存但无法联网时，
可加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。
