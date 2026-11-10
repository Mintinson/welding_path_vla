# Trajectory-VLA Qwen

该策略把 MiniVLA 的预对齐 Prismatic 视觉—语言主干与 Trajectory-VLA 的
Flow Matching 动作生成结合起来，并从现有冻结 DINOv2-L 分叉独立的稠密几何支路。
当前实验在 Full-Token + Direct Geometry 上增加可选视觉末端 LoRA；默认 latent 模式结构如下：

```text
[双相机 RGB] ─► [DINOv2-L + SigLIP，一次视觉前向]
                    │                         │
             DINOv2 稠密 patch          融合视觉 patch
            [B, camera, 256, 1024]             │
                    │                    [TokenMerger + Projector]
                    │                         │
                    │                  [Qwen Context Stream]
                    │                         │
                    └─► [Seam Query Resampler]│
                         16 latent + 3 readout │
                           │       │           │
             可选 SEAM/TANGENT/ORIENTATION 监督 │
                           │       └─► [Motion Prior]
                           │                 │ z / motion token
                           └────────┬────────┘
                                    ▼
                         [Interleaved Expert Decoder]
                       偶数层：Context/Action Paired SA
                       奇数层：Action → Geometry CA
                                        │
                             [Flow Matching 速度场]
```

上下文始终不能读取带噪动作；动作通过偶数层读取语义 Context，通过奇数层读取
稠密几何。推理时视觉、Resampler 和 Context Stream 只计算一次，后续去噪步骤同时
复用语义与几何的逐层 KV cache。

## Full-Token + Direct Geometry：当前架构实验

`configs/traj_vla_qwen_direct_geometry_a100.yaml` 同时保留全量语义 patch，并在所有
几何 CA 层让动作直接读取稠密 DINO memory；不依赖精确焊缝坐标、相机外参或阶段标签。

```text
双相机 RGB → DINO + SigLIP（共享一次视觉前向）
                ├→ 每相机 256 个融合 token → Prismatic Projector → Qwen Context
                │                                               ↓ 偶数层联合 SA
                └→ 512 个 DINO patch → 几何投影 + 二维位置 + 相机编码
                                             ↓ 奇数层直接 CA
                                  Action Expert → Flow Matching 速度场
```

配置接口：

```yaml
policy:
  parameters:
    token_merge_factor: 1
    train_token_merger: false
    use_geometry_branch: true
    geometry_attention_mode: direct
    train_geometry_encoder: true
    use_dense_geometry_attention: false
    use_geometry_grounding: false
    geometry_readout_from_latents: false
    use_motion_latent: false
    training_stage: standard
```

- `token_merge_factor=1` 的冻结 merger 为单位投影，保留每相机的 256 个 DINO+SigLIP
  融合 token，不是把 DINO 单独塞进原语义 Projector。语义图像分辨率仍为 224×224。
- `geometry_attention_mode=direct` 不创建 Resampler，也没有 16-query 压缩或零门控旁路。
  第 1/3/5/7/9/11/13/15 层直接读完整 512 个 patch；偶数层保持原语义/动作联合注意力。
- `DirectGeometryEncoder` 使用线性投影、相机 embedding、归一化二维网格坐标的可学习
  投影和 LayerNorm。坐标是 patch 中心 `(x,y)`，不是焊缝真值或 TCP 的图像投影。
- 每个几何 CA 层自己的 K/V 仅服务该层几何 memory，与 Qwen 及其他层不共享，不额外
  保留一套不参与前向的 latent K/V。CA 使用 SDPA，有效相机 mask 随 K/V 缓存；全部
  相机无效时几何注意力输出为零。
- 推理 prefill 编码一次几何并准备逐层 K/V；后续 Flow 去噪只计算动作查询与注意力。
  `train_geometry_encoder` 控制几何编码器训练；CA 的 K/V 跟随 `train_expert`。
- 默认 `geometry_attention_mode=latent`，缺少新字段的 checkpoint 行为不变。
  direct 不兼容 grounding/readout/motion 或旧 dense bypass 的组合，配置会明确拒绝。
  后续 Qwen LoRA 的保存列表已包含新几何编码器，而不是不存在的 Resampler。

该独立配置保持 16 层 Qwen/Expert、原输入缩放与原 4k/200k 学习率轨迹，先训练 100k，
每 20k 保存一次；启用图像增强，仅训练 `curve_plate`。没有同时加入 LoRA 或更多层。
这是联合架构升级，不声称单独归因 no-merge 或 direct CA 的收益。

```bash
micromamba run -n welding_vla zsh -c '
accelerate launch --num_processes=2 scripts/train_policy.py \
  --config_path=configs/traj_vla_qwen_direct_geometry_a100.yaml
'
```

训练输出：`outputs/train_v2/traj_vla_qwen_direct_geometry_curve_plate`。
必须从 Prismatic 初始化新动作策略，不要加载或 resume 原 latent/merge checkpoint；
空间 merger 和几何分支的参数结构均已改变。评估 40k/80k/100k，例如：

```bash
micromamba run -n welding_vla zsh -c '
python scripts/deploy_simulation_policy.py \
  --config_path=configs/traj_vla_qwen_direct_geometry_a100.yaml \
  --policy.checkpoint=outputs/train_v2/traj_vla_qwen_direct_geometry_curve_plate/checkpoints/100000/pretrained_model
'
```

配置默认固定 scene、仅 `curve_plate`、每次执行 8 步。结果保留
`outputs/deploy/traj_vla_qwen_direct_geometry_curve_plate/日期到分钟/curve_plate` 结构。

相比每相机 64 个语义 token，新模式 Context 长度约从 289 增至 673（含 160 个文本及
1 个状态 token），几何 CA 的 memory 从 16 增至 512。视觉编码前向次数不变，但 Qwen
激活、注意力和几何缓存占用增加；不要把参数量变化小理解成计算成本不变。
首先比较跟踪覆盖、CTE、停滞场景和安全停止，不只用 complete 或离线 loss 选模型。

## Direct Geometry + 视觉末端 LoRA

`configs/traj_vla_qwen_direct_geometry_vision_lora_a100.yaml` 保持不 merge、全部几何 CA
和 16 层动作 Expert，仅为视觉特征增加任务适配，不需要焊缝坐标或阶段标签。

```yaml
policy:
  parameters:
    lora_target: vision
    vision_lora_last_n_blocks: 4
    train_vision_encoder: false
    train_language_model: false
    train_expert: true
    compile_model: false
training:
  peft:
    method_type: LORA
    r: 32
    lora_alpha: 64
```

- `vision` 默认只对 DINO/SigLIP 实际输出特征之前的末端 attention `qkv/proj` 加 LoRA。
  当前四层范围分别为 DINO 的 19–22 层、SigLIP 的 22–25 层（零基编号），排除未参与
  特征输出的末层。`vision_lora_last_n_blocks` 可以调整范围。
- `vision_lora_include_mlp=true` 额外适配这些 block 的 `mlp.fc1/fc2`；默认关闭，
  已有配置和 adapter 仍使用原来的 attention-only 范围。
- `train_vision_encoder=false` 冻结视觉基座权重，不会冻结 PEFT 注入的 LoRA 参数。
  Qwen 继续冻结；动作 Expert、动作头、状态投影、视觉投影器、直接几何编码器继续训练，
  并随 adapter 保存。两路相机共享同一组视觉 LoRA。
- 原有 `expert/qwen/all` 模式保留；`all` 默认仍适配四层视觉和 Qwen。
  缺少新字段的配置默认 `vision_lora_last_n_blocks=4`，非 PEFT 配置不受影响。
- 当前训练前向不经过 `model.forward` 的编译包装，实验显式关闭编译以避免歧义。
  这不是通过编译提升精度的实验，也没有修改原配置的编译行为。
- 新实验从 Prismatic 基座开始，曲板数据和增强不变，训练 200k，每 10k 保存和验证。
  以场景跟踪覆盖、CTE、停滞和安全停止选 checkpoint，不承诺最后一步效果最好。

已用两张 A100 40 GiB、每卡 batch=16 验证真实数据训练 40 步、两次验证/保存，
再从第 40 步恢复训练到第 45 步；分配显存峰值约 24.9 GiB/卡，保留当前 batch。
保存的全部 16 组视觉 LoRA B 权重均已更新，部署入口也能加载 adapter 和 processor
完成缓存去噪推理。短运行用于验证执行与恢复，不代表已验证长时间稳定性或 rollout 收益。

训练命令（项目根目录）：

```bash
micromamba run -n welding_vla zsh -c '
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 accelerate launch --num_processes=2 --mixed_precision=bf16 \
  scripts/train_policy.py --config_path=configs/traj_vla_qwen_direct_geometry_vision_lora_a100.yaml
'
```

输出目录：`outputs/train_v2/traj_vla_qwen_direct_geometry_vision_lora_curve_plate`。
中断后从最近一次完整 checkpoint 恢复，用同一命令追加 `--training.resume=true`；
不要把 LoRA adapter 当作完整非 PEFT 权重加载。部署沿用统一接口，例如：

```bash
micromamba run -n welding_vla zsh -c '
python scripts/deploy_simulation_policy.py \
  --config_path=configs/traj_vla_qwen_direct_geometry_vision_lora_a100.yaml \
  --policy.checkpoint=outputs/train_v2/traj_vla_qwen_direct_geometry_vision_lora_curve_plate/checkpoints/100000/pretrained_model
'
```

`SIGINT`/`SIGTERM` 不会自动保存当前步；恢复位置是最后一个已写完的 checkpoint。
后台训练应使用独立进程组启动。Accelerate 的 worker 可能拥有各自的进程组，停止前应
核对启动器与 worker 的 PID/PGID；向这些训练进程组发送 SIGTERM，避免留下 GPU 子进程。

### 扩展视觉适配：八层 attention + MLP

`configs/traj_vla_qwen_direct_geometry_vision_lora_plus_a100.yaml` 设置：

```yaml
policy:
  parameters:
    lora_target: vision
    vision_lora_last_n_blocks: 8
    vision_lora_include_mlp: true
```

实际范围为 DINO 15–22、SigLIP 18–25，每个 block 的 `attn.qkv/attn.proj/mlp.fc1/mlp.fc2`
共 64 组 LoRA，rank=32、alpha=64。保持 Qwen 冻结、完整 patch、Direct Geometry、
16 层 Expert、图像增强与单阶段训练；无需焊缝坐标和阶段标签。
从 Prismatic 基座开始训练 200k，不直接恢复四层 attention-only 实验的优化器状态。
该配置继承现有 eval 设置，但不以训练期 eval loss 选择 checkpoint；本实验不修改验证逻辑。
更多视觉 block 需要反向传播，短训练应检查峰值显存和梯度，不能按 LoRA 参数量推算耗时。

2026-10-19 已在双 A100 40 GiB 上验证每卡 batch=16：120 步真实数据训练，
从第 120 步恢复到第 125 步，保存/恢复和部署入口缓存推理均通过。
短测试用 20 步 warmup 提前达到峰值学习率 `5e-5`，正式配置仍为 4000 步 warmup。
可训练参数为 118,294,912；全部 64 组视觉 LoRA B 都已更新，稳定更新约 0.69–0.74 秒/步，
分配显存峰值约 28.6 GiB/卡。短测试验证执行可用性，不代表 rollout 收益或长期稳定性保证。

```bash
micromamba run -n welding_vla zsh -c '
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 accelerate launch \
  --num_processes=2 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no \
  scripts/train_policy.py \
  --config_path=configs/traj_vla_qwen_direct_geometry_vision_lora_plus_a100.yaml
'
```

输出根目录为 `outputs/train_v2/traj_vla_qwen_direct_geometry_vision_lora_plus_curve_plate`；
恢复同一实验时追加 `--training.resume=true`。部署使用同一配置并通过
`--policy.checkpoint=outputs/train_v2/traj_vla_qwen_direct_geometry_vision_lora_plus_curve_plate/checkpoints/200000/pretrained_model`
指定实际已保存的 checkpoint。


## Qwen 输入缩放对照实验

`policy.parameters.scale_context_embeddings` 控制进入 Qwen 前的**语义图像 token 和
文本 embedding** 是否额外乘以 `sqrt(hidden_size)`。当前 Qwen 宽度为 896，缩放约为
29.93 倍；这不是像素归一化，也不是 attention 中的 `1/sqrt(head_dim)`。

- `true`：保留本项目原有缩放行为，也是配置缺省值。
- `false`：直接使用 Projector 和文本查表输出，不再施加该额外缩放。

状态 token、动作 token、原始 DINO 几何 patch 以及归一化统计均不变。开关不增加参数，
不改变 token 数量或注意力层数；训练和部署共用此逻辑，值随 checkpoint 配置保存。
缺少该字段的已有 checkpoint 默认使用 `true`，保持原来的前向结果；公共 SmolVLM
路径也保持原缩放。**不能把旧权重仅在部署时改为 `false` 当成优化结果**，需要重新训练
动作策略以适应不同输入幅值。这里测试的是输入约定差异是否影响学习，并不保证 rollout 改善。

独立配置 `configs/traj_vla_qwen_unscaled_fixedlr_a100.yaml` 对照
`traj_vla_qwen_aug_curve_plate` 的 16-query 实验：保持图像增强、16 层双流、
token merge factor=2、冻结 Qwen、无 LoRA，以及原有数据集和训练划分；关闭 dense bypass、
grounding 和 motion latent，只取消输入缩放。从 Prismatic 预训练主干初始化，
不续训已有动作 checkpoint。每卡 batch=16，两卡共 32，学习率 `5e-5`，种子 `20260806`。

先训练 40k 步，在 20k/40k 保存检查点。该配置显式设置
`scheduler_auto_scale: false`，使用本地注册的 `fixed_cosine_decay_with_warmup`
调度器，复用 LeRobot 余弦公式，但阻止它按训练总步数压缩周期。因此 warmup 为 4k、
decay 为 200k，40k 时实际 LR 约为 `4.5464e-5`。请在项目根目录执行：

```bash
micromamba run -n welding_vla zsh -c '
accelerate launch --num_processes=2 scripts/train_policy.py \
  --config_path=configs/traj_vla_qwen_unscaled_fixedlr_a100.yaml
'
```

训练输出为 `outputs/train_v2/traj_vla_qwen_unscaled_fixedlr_curve_plate`。训练后分别评估
`020000` 和 `040000`，例如：

```bash
micromamba run -n welding_vla zsh -c '
python scripts/deploy_simulation_policy.py \
  --config_path=configs/traj_vla_qwen_unscaled_fixedlr_a100.yaml \
  --policy.checkpoint=outputs/train_v2/traj_vla_qwen_unscaled_fixedlr_curve_plate/checkpoints/040000/pretrained_model
'
```

该配置按固定 scene 清单只评估 `curve_plate`，每次执行 8 步，结果写入
`outputs/deploy/traj_vla_qwen_unscaled_fixedlr_curve_plate/日期到分钟/curve_plate`。
用相同步数的原 16-query checkpoint、相同场景和执行步数进行对照，重点比较跟踪误差、
覆盖率、成功率和安全停止；不要只看离线 loss，也不要只与原模型的 200k 结果比较。
部署的随机动作噪声会带来波动，边际改善需重复 rollout 确认。

`scheduler_auto_scale` 缺省为 `true`，不改变已有 checkpoint 的调度行为。
`traj_vla_qwen_unscaled_a100.yaml` 对应已完成的自动压缩实验：虽然保存参数写着
4k/200k，LeRobot 实际按 800/40k 调度，40k 时 LR 已到 `2.5e-6`。
该实验是“取消输入缩放 + 短周期调度”的联合改动，不能当作纯缩放消融；保留其配置和
结果供比较，不能用它的权重续训来补做固定周期对照。详见
[部署审计与后续计划](../../../../docs/traj-vla-qwen-unscaled-analysis.md)。

### 固定周期结果与当前实验

固定周期 40k 已完成：覆盖率 28.44%、成功 2/25、CTE P95 8.95 mm，相比原 scaled
40k 没有稳定的跟踪精度收益，目前不优先续训至 100k。完整证据和单场景问题见
[固定周期消融审计](../../../../docs/traj-vla-qwen-fixedlr-analysis.md)。

此前准备的 `configs/traj_vla_qwen_scaled_shortlr_a100.yaml` 保留输入缩放，显式使用
800 warmup / 40k decay，与已完成的 unscaled 短周期逐步 LR 完全相同，只对照输入幅值。
原数据、增强、16-query 和 8 步执行频率不变，不引入新模型模块。研究方向现已转向
上文的 Full-Token + Direct Geometry，不再要求执行这组缩放/周期对照。

```bash
micromamba run -n welding_vla zsh -c '
accelerate launch --num_processes=2 scripts/train_policy.py \
  --config_path=configs/traj_vla_qwen_scaled_shortlr_a100.yaml
'
```

输出为 `outputs/train_v2/traj_vla_qwen_scaled_shortlr_curve_plate`，从 Prismatic 初始化，
不要恢复任何已有动作 checkpoint。训练后评估 `020000` 和 `040000`，与
`traj_vla_qwen_unscaled_curve_plate` 的同一步数短周期结果比较。不要在同一 checkpoint
上临时切换缩放或调度，再将其解释为单变量实验；也不要只挑最低 eval loss 的模型。

## 稠密几何与辅助监督接口

`use_geometry_branch: true` 时，每路相机的 256 个 DINO patch 不经过空间压缩，
由第 0 个 paired layer 后的语言和状态 hidden 调制 19 个 learned query。前 16 个
latent token 进入 Action Expert，后三个 readout token 固定对应 `SEAM`、`TANGENT`、
`ORIENTATION`。默认 `use_geometry_grounding: false`，因此旧训练仍不创建预测头或
辅助 loss；显式启用后才把 readout 用作轨迹监督的几何接地。

稠密 patch 设计参考 [Patch Policy](https://arxiv.org/html/2607.18236v1) 及其
[官方 DINOv2 patch 配置](https://github.com/gaoyuezhou/patch_policy/blob/main/configs/encoder/dino_patch.yaml)；
Task-Conditioned Seam Query 是本项目针对双相机焊缝控制增加的计算瓶颈。

训练期 `forward_intermediates()` 通过 `FlowMatchingOutput.auxiliary_outputs` 暴露：

- `geometry.patch_tokens` 与 `geometry.patch_mask`；
- `geometry.latent_tokens`；
- `geometry.readout_tokens`，顺序为 seam、tangent、orientation；
- `geometry.attention_weights`。

Grounding 开启后还会输出 `geometry.seam_logits`、
`geometry.tangent_prediction` 与 `geometry.orientation_6d`。常规配置仍只优化
Flow Matching；辅助监督不会改变 query 数量、动作表示或部署输出形状。

### 辅助监督经过动作 latent

`geometry_readout_from_latents: true` 时，先由动作 latent query 读取 DINO patch，
再由三个辅助 readout query 读取同一组动作 latent。没有 `detach`，因此 seam、tangent、
orientation 三项辅助损失都直接更新 Action Expert 使用的 latent query；Stage 1 即使
不运行 Action Expert，也会训练这些 query。这里的 latent 是几何动作 token，
不是 `use_motion_latent` 控制的 CVAE 运动隐变量，二者可以独立配置。

```text
DINO patch → 动作 latent ┬→ Action Expert → 动作
                        └→ 辅助 readout → grounding heads → 辅助损失
```

两级读取复用 Resampler 的注意力、归一化和前馈参数，不增加 state dict 键，也不改变
动作 latent 的数量。seam head 仍用 patch 特征对 readout 做逐 patch 打分；它的 readout
视觉输入经由 latent，不再与 latent 并行直接读取 patch。诊断输出
`geometry.attention_weights` 仍为 `[B,K+3,C,P]`：最后三行改为
`readout→latent` 与 `latent→patch` 的平均头注意力乘积，仅用于可视化，不是精确归因。

主配置及缺少新字段的已有 checkpoint 默认关闭该选项，保留原来的并行读取方式、
参数布局和前向结果，因此现有 `traj_vla_qwen_a100.yaml` 训练的权重仍可正常加载。
独立 grounding Stage 1 / Stage 2 配置均已开启。若只想在普通动作训练中增加此监督：

```yaml
policy:
  parameters:
    use_geometry_branch: true
    use_geometry_grounding: true
    geometry_readout_from_latents: true
    use_motion_latent: false
    training_stage: standard
```

设置 `geometry_readout_from_latents: false` 即恢复并行 readout；该开关随模型配置保存。
只有启用 `use_geometry_grounding` 并训练时才会产生辅助监督收益，不能仅在旧模型部署时
打开开关就期待性能改善。部署仍删除 grounding heads，动作输出接口与缓存规则不变。

### 轨迹监督标签

`GeometryGroundingTargetProcessorStep` 在 relative-action 转换和归一化之前读取当前
TCP、future absolute `safe_command` 和 `action_is_pad`，不修改 LeRobot 数据 schema：

1. 用固定 global 外参与当前 TCP 推导出的 wrist 外参投影未来轨迹；
2. 按原始相机 FOV 投影后，使用与模型相同的非等比缩放映射到 `224×224`；
3. 生成双相机 `16×16` patch corridor，以及当前 TCP 局部系的有向 tangent；
4. 把第一个有效 future target 的姿态转换为相对 TCP rotation-6D。

相机位姿、FOV 与 corridor 半径全部保存在 policy config 中，可在相机重新标定后覆盖。
视野外轨迹、缺失相机和 episode 尾部 padding 均通过独立 mask 排除。

### Conditional Motion Latent

`use_motion_latent: true` 时，训练期两层 tiny Transformer posterior 从 clean action
chunk 和状态预测 `q(z|A,s)`；prior 从第 0 层 Qwen task/state hidden 与 geometry
latent 预测 `p(z|o,l,s)`。训练使用重参数采样和解析 KL，推理固定取 prior mean，
不使用 `z=0`。

动作块前增加一个 motion slot，第 0 层后注入 `z`，速度输出仍只读取最后 30 个动作
token。Context cache 同时保存 motion token；Euler 去噪期间不会重复运行视觉编码、
Resampler、prior 或几何投影。

## Expert Attention 结构

Expert Decoder 提供两种可互换结构，配置名称及语义与 `trajectory_vla` 一致：

| `attention_mode` | 实现 | 注意力结构 |
| --- | --- | --- |
| `self_attn` | `PairedLayerDecoder` | 每层对 Context/Expert token 执行一次非对称联合 SA |
| `cross_attn` | `InterleavedSACADecoder` | 周期性联合 SA；启用几何支路时其余层改为 Expert query 读取 Geometry K/V |

当前训练配置采用 Interleaved SA/CA，每两层用一次联合 SA：

```yaml
policy:
  parameters:
    attention_mode: cross_attn
    self_attn_every_n_layers: 2
    use_geometry_branch: true
    geometry_num_queries: 16
    geometry_num_heads: 8
```

第 `0, 2, 4, ...` 层为联合 SA，第 `1, 3, 5, ...` 层为 Context SA 加
Action→Geometry CA。关闭几何支路时恢复原 Context→Expert CA。第一版 checkpoint
没有 `use_geometry_branch`，配置默认值为 `false`，因此旧权重仍构造原结构；当前训练
YAML 显式开启新支路，新结构需要重新训练，不能直接加载旧 Action Expert 权重。

## 可选：少量层直接读取稠密 patch

`use_dense_geometry_attention: true` 在指定的几何 CA 层增加动作到稠密 DINO patch
的连接。区别于增加 `geometry_num_queries`，动作查询可以绕过 latent 重采样瓶颈，
直接读取双相机的 512 个 patch；它们不会进入 Qwen 的长序列，也不会新增视觉编码器。

```yaml
policy:
  parameters:
    use_geometry_branch: true
    geometry_num_queries: 16
    use_dense_geometry_attention: true
    dense_geometry_attention_layers: [7, 15]
    use_geometry_grounding: false
    use_motion_latent: false
    training_stage: standard
```

层编号从 0 开始，必须是不重复、未越界的 CA 层；默认周期下即奇数层。
`[7, 15]` 表示仅第 8、16 层启用。设 `use_dense_geometry_attention: false` 即关闭，
不创建新参数或计算新注意力，缺少该字段的已有模型也默认关闭。

选定层保留原 latent 注意力，额外计算：

```text
原 latent 注意力 + tanh(gate) × 稠密 patch 注意力 → 原 Expert 输出投影与 MLP
```

新路径复用 Resampler 的 patch 投影、相机编码和归一化，以及当前 Expert 的 Q/K/V
和输出投影，每个连接仅新增一个可学习标量 `gate`，两层合计两个参数。门控从零开始，
在相同已有权重下初始输出不变，但门控可以收到梯度。稠密注意力使用 SDPA 并遵守相机
有效性 mask；推理预填充时缓存稠密 K/V 和 mask，后续去噪只重新计算动作查询和注意力。
新增计算并不为零，训练吞吐需实测，参数增量小不等于计算开销为零。

门控随 `train_expert` 冻结或训练，启用 PEFT 时也保存并训练这些门控。此分支不依赖
辅助标签、精确焊缝坐标或阶段标注；适用于包含图像和机器人状态的实机示教数据。
它是待 rollout 验证的结构实验，并不保证超过 16-query 或 Adapter。

独立配置保留 16-query，关闭 grounding/motion，仅改变稠密连接，按单阶段训练：

```bash
# 在已激活的 micromamba welding_vla 环境中执行。
accelerate launch --num_processes=2 scripts/train_policy.py \
  --config_path=configs/traj_vla_qwen_dense_a100.yaml
```

该配置默认训练 `curve_plate`，输出到 `outputs/train_v2/traj_vla_qwen_dense_curve_plate`。
建议先评估 40k、100k，再决定是否继续，不只依赖离线 eval loss 选择模型。
原主配置保持关闭；已有 16/32-query 权重仍按其保存配置部署。不能仅给旧权重打开开关
就获得新能力；新增门控需要训练。关闭开关兼容原权重布局，开启后保存的权重则包含门控。

## 默认训练显存估算

以下结果对应当前默认配置：双相机、`batch_size=4`、96 个语言 token、
30 步动作块、16 层 Qwen2.5 Context Stream、16 层 Action Expert、每两层一次
联合 SA、BF16
混合精度和 AdamW。参数量由当前环境中的实际模型配置统计，而非按模型名称估算。

### 1. 参数量与参数显存

当前实现显式把 Qwen 和 Action Expert 转成 BF16；冻结的 DINOv2、SigLIP
以及 Token Merger、Projector、Flow 投影仍为 FP32。AMP 只控制算子自动转换，
不会自动把这些 FP32 参数永久转换成 BF16。

| 模块 | 参数量 | 是否训练 | 常驻精度 | 参数显存 |
| --- | ---: | --- | --- | ---: |
| DINOv2 ViT-L | 303.231 M | 否 | FP32 | 1.130 GiB |
| SigLIP ViT-SO400M | 427.681 M | 否 | FP32 | 1.593 GiB |
| 截断后的 16 层 Qwen2.5 | 374.734 M | 否 | BF16 | 0.698 GiB |
| 16 层 Action Expert | 79.863 M | 是 | BF16 | 0.149 GiB |
| 2×2 Token Merger | 18.942 M | 是 | FP32 | 0.071 GiB |
| Prismatic Projector | 27.552 M | 是 | FP32 | 0.103 GiB |
| Seam Query Resampler | 5.531 M | 是 | FP32 | 0.021 GiB |
| 状态、动作与时间投影 | 1.429 M | 是 | FP32 | 0.005 GiB |
| **合计** | **1,238.964 M** | **133.318 M 可训练** | — | **3.769 GiB** |

上述合计对应三项增强关闭的主配置。按当前 `D_expert=672`、`D_qwen=896` 和
`motion_latent_dim=16` 实际实例化后，可选模块的增量为：

| 可选模块 | 参数量 | FP32 参数显存 | 训练/部署边界 |
| --- | ---: | ---: | --- |
| Geometry Grounding Heads | 0.909 M | 0.0034 GiB | 两阶段训练保留，部署删除 |
| Conditional Motion Latent | 0.720 M | 0.0027 GiB | 部署删除 posterior，保留 prior/slot/projection |
| **两者合计** | **1.629 M** | **0.0061 GiB** | 不改变动作 token 输出形状 |

Stage 1 只有 Resampler 与 heads 共 6.440 M 参数可训练；按 FP32 参数、梯度和
AdamW 两份状态估算，三者合计约 0.096 GiB。Stage 2 同时启用 grounding 与 motion
时，比现有 dense policy 增加约 0.024 GiB 的 FP32 参数/梯度/AdamW 常驻量；实际峰值
还包含 posterior Transformer 和监督 head 激活，应以 A100 训练日志为准。

以参数个数 `N` 和每个元素的字节数 `b` 计算：

```text
parameter_memory = N × b
FP32: b = 4 bytes
BF16: b = 2 bytes
```

例如两个视觉编码器虽然被冻结，仍必须常驻 GPU：

```text
(303,230,976 + 427,680,704) × 4 / 1024³ = 2.723 GiB
```

默认可训练参数由 `53.455 M FP32 + 79.863 M BF16` 组成，因此梯度需要：

```text
53,454,688 × 4 + 79,863,456 × 2 = 0.348 GiB
```

当前 PyTorch 2.11 的标准 AdamW 为每个参数保存一阶、二阶矩；状态精度与参数
精度相同，没有额外 FP32 master weights：

```text
2 × (53,454,688 × 4 + 79,863,456 × 2) = 0.696 GiB
```

由此得到不含激活的稳态下限：

```text
参数 3.769 + 梯度 0.348 + AdamW 状态 0.696 ≈ 4.812 GiB
```

### 2. 激活与实际峰值

每路相机由 256 个 patch token 经 Token Merger 压缩为 64 个 token。联合 SA
层的注意力长度为：

```text
2 × 64 个视觉 token + 96 个语言 token + 1 个状态 token + 30 个动作 token
= 255 tokens
```

注意力分数显式使用 FP32。仅一层联合 SA 的注意力矩阵就需要：

```text
batch × heads × tokens² × 4 bytes
= 4 × 14 × 255² × 4
= 13.9 MiB
```

默认配置中的几何 CA 层分别计算 225-token Context SA，以及 `30 × 16` 的
Expert→Geometry 注意力；因此其注意力激活显著小于联合 SA 层。每路相机仍保留
256 个 DINO patch，但它们只在一次 Resampler Cross-Attention 中出现，不进入
Qwen 或 Action Expert 的长序列。

此外还要保留 Q/K/V、Qwen 与 Expert 的 MLP 中间量、残差和反向传播状态。
冻结 Qwen 只能省去其梯度和优化器状态；为了把梯度传回可训练的 Projector，
Qwen 各层的部分激活仍需保留。视觉编码器的输入和参数都不求导，因此不会
保留完整的视觉反向图。

以下 RTX 4060 Laptop 实测来自第一版逐层联合 SA 结构，可视为当前默认结构的
保守参考；切换结构后应以新训练日志重新测量峰值：

| 项目 | 峰值 |
| --- | ---: |
| 除去两个视觉编码器权重的训练峰值 | 2.792 GiB |
| 两个 FP32 视觉编码器权重 | 2.723 GiB |
| **合并后的 Tensor allocated 峰值** | **约 5.52 GiB** |
| 预计 PyTorch reserved 显存 | **约 5.8–6.2 GiB** |
| 加上 CUDA context、内核 workspace 和输入 batch | **约 6.3–7.0 GiB** |

这里的 `allocated` 是模型真正占用的 Tensor 显存；`nvidia-smi` 通常更接近最后
一行。不同 CUDA、Triton、cuDNN 和 allocator 版本可能带来约 0.5–1 GiB 波动。

### 3. 当前机器是否能训练

本次测量时 RTX 4060 Laptop 总显存为 8188 MiB，空闲约 5520 MiB。桌面和其他
进程已经占用约 2.6 GiB，因此默认配置的理论 Tensor 峰值就已超过可用显存，
实际训练很可能 OOM。独占的 8 GiB GPU 理论上可能勉强运行，但余量很小；实际
建议至少使用 12 GiB，16 GiB 可以为评估、视频解码和 allocator 波动保留更合理
的余量。

仅把 `training.batch_size` 从 4 改为 1 不能解决大部分问题，因为 4.73 GiB
属于与 batch 无关的参数、梯度和优化器状态。可按收益从高到低考虑：

1. 将冻结的 DINOv2 与 SigLIP 以 BF16 常驻，可节省约 1.36 GiB；
2. 冻结 Token Merger 与 Projector，可减少约 0.52 GiB 梯度和 AdamW 状态；
3. 使用 LoRA 训练 Expert 或 Qwen，进一步减少可训练参数和优化器状态；
4. 再把 `training.batch_size` 降为 1，接受较小的有效 batch；
5. 为 Qwen/Expert 增加 gradient checkpointing，以计算时间换激活显存。

第 1、3、5 项会改变训练实现或实验定义，因此接口已经提供，但默认全部关闭。
正式实验应以 `train.log` 中 LeRobot 记录的 `gpu_mem_gb` 和 `nvidia-smi` 峰值
为最终依据。

### 4. 可选显存优化接口

默认值保持上述显存计算不变：视觉主干为 FP32、不开 gradient checkpointing，
并且 `training.peft: null` 不启用 LoRA。

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| `frozen_vision_dtype` | `float32` | 可设为 `bfloat16`，仅改变冻结视觉权重的常驻精度 |
| `gradient_checkpointing_qwen` | `false` | 重算 Qwen 的 QKV 和 MLP 中间量 |
| `gradient_checkpointing_expert` | `false` | 重算 Expert 的 QKV 和 MLP 中间量 |
| `lora_target` | `expert` | PEFT 启用后选择 `expert`、`qwen` 或 `all` |
| `training.peft` | `null` | 非空时由 LeRobot 原生 PEFT 创建 LoRA adapter |

冻结视觉编码器以 BF16 常驻：

```yaml
policy:
  parameters:
    train_vision_encoder: false
    frozen_vision_dtype: bfloat16
```

如果 `train_vision_encoder: true`，必须使用 `frozen_vision_dtype: float32`，避免
这个仅面向冻结权重的开关无意中改变视觉微调精度。

分别或同时启用双流 gradient checkpointing：

```yaml
policy:
  parameters:
    gradient_checkpointing_qwen: true
    gradient_checkpointing_expert: true
```

两条流拥有独立的 QKV 和 MLP checkpoint；联合注意力是共享计算，只要任一开关
启用就会重算。该功能仅作用于需要梯度的训练前向，不改变评估、部署和 KV cache。

使用 LeRobot 原生 PEFT 启用 LoRA：

```yaml
policy:
  parameters:
    lora_target: qwen  # expert、qwen 或 all

training:
  peft:
    method_type: LORA
    r: 16
    lora_alpha: 16
```

新建的状态/动作/时间投影保持完整训练。选择 `qwen` 且 `train_expert: true` 时，
Expert 也保持完整训练；选择 `expert` 或 `all` 时，Expert 基础权重被冻结，只训练
其 LoRA adapter。由于第一轮训练的 Expert 是随机初始化的，`expert` / `all` 更适合
从已经训练过的 TrajVLA-Qwen checkpoint 继续微调，而不适合作为首次训练默认值。
仅设置 `lora_target` 不会启用 PEFT，必须同时提供非空的 `training.peft`。

## 焊接参数 Prompt

TrajVLA-Qwen 不维护专用的焊接 prompt 逻辑。项目公共 `WeldingPromptBuilder` 在 Qwen chat
template 之前读取 `task.direction` 和 `task.parameters`，与 SmolVLA、Trajectory-SmolVLM、
π0 和 π0.5 共用同一实现。字段级消融配置位于模型参数之外：

```yaml
policy:
  welding_prompt_fields:
    - direction
    - welding_speed
    - work_angle
    - travel_angle
    - tool_roll
  parameters:
    tokenizer_max_length: 160
```

将列表设为 `[]` 即只保留原始任务文本。builder 会随 preprocessor 写入 checkpoint，部署时不应
另外手工拼接指令。

## 运行

首次训练会下载约 2.6 GB 的 Prismatic checkpoint：

```bash
pixi run -e train train-policy --config_path=configs/traj_vla_qwen.yaml
```

三项 XS-VLA 增强在主配置中默认关闭。两阶段训练使用独立配置：

```bash
# Stage 1：10k steps，仅训练 Resampler 与 grounding heads
pixi run -e train train-policy-2gpu \
  --config_path=configs/traj_vla_qwen_grounding_stage1_a100.yaml

# Stage 2：加载 Stage 1，联合 FM、低权重 grounding 与 motion KL
pixi run -e train train-policy-2gpu \
  --config_path=configs/traj_vla_qwen_grounding_stage2_a100.yaml
```

Stage 1 使用 `L_seam + 0.5 L_tangent + 0.5 L_orientation`，且不运行 Action Expert；
Stage 2 使用 `L_FM + 0.05 L_seam + 0.02 L_tangent + 0.02 L_orientation + 1e-3 L_KL`。
启动 `policy_joint` 时会检查 checkpoint 确实包含 Stage 1 的 Resampler 与 grounding
heads。部署运行时在加载权重后删除三个辅助 head 和 motion posterior，仅保留 prior。

离线评估与仿真部署沿用项目公共入口：

```bash
pixi run -e train policy-evaluate \
  --config_path=configs/traj_vla_qwen.yaml \
  --policy.checkpoint=outputs/train/traj_vla_qwen_weldpath_relative_v1/checkpoints/last/pretrained_model

pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/traj_vla_qwen.yaml \
  --policy.checkpoint=outputs/train/traj_vla_qwen_weldpath_relative_v1/checkpoints/last/pretrained_model
```

## 腕部局部高分辨率分支

`configs/traj_vla_qwen_wrist_roi_a100.yaml` 保留双相机全图、512 个全图几何 patch
和原 Qwen Context，另从腕部原始 RGB 裁剪尖端附近区域，缩放为 224×224，
仅通过共享 DINO（含已有视觉 LoRA）生成 256 个局部 patch。不重复运行 SigLIP，
不把局部 patch 加入 Qwen，不使用焊缝坐标或阶段标注。

```yaml
policy:
  parameters:
    use_wrist_roi: true
    wrist_roi_camera_key: observation.images.wrist
    # 原图归一化 left/top/right/bottom；640×480 对应 256×256。
    wrist_roi_box: [0.35, 0.25, 0.75, 0.7833333333]
```

局部编码器保留二维位置，每个几何 CA 层使用独立局部 K/V 和零初始化的
`tanh(gate)` 输出残差。零门控初始化保持原模型预测，第一步梯度更新 gate，
随后动作损失更新局部编码器、K/V 和共享 DINO。局部 attention 与全图 attention
分别归一化，推理时按观测缓存局部 K/V，全部 Flow 去噪步复用。
局部编码器服从 `train_geometry_encoder`，局部 K/V 与 gate 服从 `train_expert`。

开关默认关闭，关闭时不创建局部参数、不改变旧模型输入。开启要求
`geometry_attention_mode: direct`。实机必须依据腕部相机和工具安装位置校准裁剪框；
不得用仅在仿真中可得的焊缝位置动态选择裁剪区域。保留全图可避免接近阶段或焊缝
离开局部视野时丢失全局信息，但局部分支不保证能从单帧 RGB 精确恢复深度。

从已有四层视觉 LoRA 模型热启动时，先创建包含新模块的初始化 checkpoint：

```bash
# 在已经激活的 micromamba welding_vla 环境中执行。
python scripts/initialize_traj_vla_qwen_wrist_roi.py \
  --source outputs/train_v2/traj_vla_qwen_direct_geometry_vision_lora_curve_plate/checkpoints/200000/pretrained_model \
  --output outputs/train_v2/traj_vla_qwen_wrist_roi_initial/pretrained_model

accelerate launch --num_processes=2 scripts/train_policy.py \
  --config_path=configs/traj_vla_qwen_wrist_roi_a100.yaml
```

初始化工具拒绝覆盖已有目录，逐张量验证旧 adapter/Expert/投影器权重完全保留，
只允许新增 `wrist_roi_*` 参数，并复制原 relative-action 处理器和归一化统计。
新增局部编码器和各层 K/V 从已学全图投影初始化，但参数独立，避免局部模块从随机
特征空间起步；所有局部门控仍为零，因此初始动作保持不变。
初始化工具仅负责权重扩展；训练学习率等由 YAML 指定，不恢复旧优化器。
新实验为单阶段训练，新增步数从零计数。当前配置从旧 200k 权重再训练 100k，
每卡 batch 16、学习率 2e-5、warmup 1000，每 10k 保存 checkpoint。
新增一次 DINO 前向和局部注意力会增加计算量与显存，应以实际双卡测试为准。

部署仍使用同一个配置，并显式选择新训练的 checkpoint：

```bash
python scripts/deploy_simulation_policy.py \
  --config_path=configs/traj_vla_qwen_wrist_roi_a100.yaml \
  --policy.checkpoint=outputs/train_v2/traj_vla_qwen_wrist_roi_curve_plate/checkpoints/last/pretrained_model \
  --deployment.include_tasks='["curve_plate"]'
```

## 解冻主干

默认只训练 Token Merger、Prismatic Projector、Seam Query Resampler、状态/动作投影
与动作专家。
所有训练边界均为 YAML 参数：

```yaml
policy:
  parameters:
    train_vision_encoder: false
    train_token_merger: true
    train_projector: true
    train_geometry_resampler: true
    train_language_model: false
    train_language_last_n_layers: 0
    train_expert: true
    train_state_proj: true
```

- 微调 Qwen 最后四层：设置 `train_language_last_n_layers: 4`；
- 解冻全部 Qwen：设置 `train_language_model: true`；
- 解冻 DINOv2 与 SigLIP：设置 `train_vision_encoder: true`。
- Token Merger、Projector、Geometry Resampler、动作专家和状态投影也可用各自的
  `train_*` 开关独立冻结；
- 动作输入、时间条件和速度输出投影始终随动作专家训练，不额外制造一组细碎开关。

`DecoderAdapter` 是 Qwen 版本边界。Qwen3/Qwen3.5 应新增 adapter 处理各自的
Q/K Norm、RoPE 或混合层，不修改视觉编码、动作专家输入和 Flow Matching。
