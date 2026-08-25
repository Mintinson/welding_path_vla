# Trajectory-VLA Qwen

该策略把 MiniVLA 的预对齐 Prismatic 视觉—语言主干与 Trajectory-VLA 的
Flow Matching 动作生成结合起来，并从现有冻结 DINOv2-L 分叉独立的稠密几何支路：

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
                                  │           │
                                  └─────┬─────┘
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

## 稠密几何与辅助监督接口

`use_geometry_branch: true` 时，每路相机的 256 个 DINO patch 不经过空间压缩，
由第 0 个 paired layer 后的语言和状态 hidden 调制 19 个 learned query。前 16 个
latent token 进入 Action Expert，后三个 readout token 固定对应 `SEAM`、`TANGENT`、
`POSTURE`，本阶段不接预测头或辅助 loss。

稠密 patch 设计参考 [Patch Policy](https://arxiv.org/html/2607.18236v1) 及其
[官方 DINOv2 patch 配置](https://github.com/gaoyuezhou/patch_policy/blob/main/configs/encoder/dino_patch.yaml)；
Task-Conditioned Seam Query 是本项目针对双相机焊缝控制增加的计算瓶颈。

训练期 `forward_intermediates()` 通过 `FlowMatchingOutput.auxiliary_outputs` 暴露：

- `geometry.patch_tokens` 与 `geometry.patch_mask`；
- `geometry.latent_tokens`；
- `geometry.readout_tokens`，顺序为 seam、tangent、posture；
- `geometry.attention_weights`。

默认 Flow Matching loss 不读取这些字段。后续可以直接在 readout token 或稠密 patch
上增加焊缝区域、切向和焊枪姿态 head，而不改变 query 数量或 checkpoint 结构。

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

## 运行

首次训练会下载约 2.6 GB 的 Prismatic checkpoint：

```bash
pixi run -e train train-policy --config_path=configs/traj_vla_qwen.yaml
```

离线评估与仿真部署沿用项目公共入口：

```bash
pixi run -e train policy-evaluate \
  --config_path=configs/traj_vla_qwen.yaml \
  --policy.checkpoint=outputs/train/traj_vla_qwen_weldpath_relative_v1/checkpoints/last/pretrained_model

pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/traj_vla_qwen.yaml \
  --policy.checkpoint=outputs/train/traj_vla_qwen_weldpath_relative_v1/checkpoints/last/pretrained_model
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
