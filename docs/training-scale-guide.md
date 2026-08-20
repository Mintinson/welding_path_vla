# Batch、Step 与训练时间计算

本文统一解释单卡和 Accelerate DDP 配置中的 batch、optimizer step、数据遍历量和训练时间。
目标是充分利用 GPU，同时明确实验比较保持的是“相同步数”还是“相同数据量”。

## 1. 定义

设：

- $N$：训练划分的帧数；
- $G$：DDP 进程数，通常等于 GPU 数；
- $B_{gpu}$：`training.batch_size`，即每个进程、每张 GPU 的 batch；
- $A$：梯度累积次数；当前项目为 1；
- $S$：optimizer update 数，即 `training.steps`；
- $E$：目标数据遍历轮数。

全局 batch：

$$B_{global}=G\times B_{gpu}\times A.$$

固定 step 时实际处理的训练样本数和等效 epoch：

$$N_{seen}=S\times B_{global},\qquad E_{effective}=\frac{N_{seen}}{N}.$$

固定数据遍历量时所需 step：

$$S=\left\lceil\frac{E\times N}{B_{global}}\right\rceil.$$

`training.steps` 是 optimizer update 数，不是单卡 sample 数，也不是每个进程各自独立累计的
step。DDP 每个 step 同步一次梯度。

## 2. 两种合理的比较目标

### 固定 optimizer step

保持 $S$ 不变并增大全局 batch，会让模型在相同 update 数内看到更多样本。这通常能提高 GPU
利用率并减少达到某个 step 的 wall time，但训练数据量和优化轨迹已经变化，不能称为“相同训练量”。

适合：固定 update budget 的消融、验证更大 batch 是否改善稳定性、希望同样 step 更快完成。

### 固定数据遍历量

增大 $B_{global}$ 后按公式降低 $S$，使 $N_{seen}$ 基本不变。这是在相同数据覆盖下比较硬件吞吐
最直接的方法。

适合：以一轮或若干轮数据为预算、比较不同 GPU 数或每卡 batch 的 wall time。

无论选哪一种，实验记录都应同时保存 $G$、$B_{gpu}$、$A$、$S$ 和 $N_{seen}$，不能只写
`batch_size`。

## 3. 当前配置快照

2026-08-10 的本地数据集训练划分为 $N=3,674,023$ 帧。当前双 A100 入口均使用两个 DDP
进程，配置值如下：

| 策略 | 每卡 batch | 全局 batch | step | 约处理帧数 | 等效 epoch |
|---|---:|---:|---:|---:|---:|
| ACT | 32 | 64 | 57,407 | 3,674,048 | 1.000 |
| SmolVLA | 32 | 64 | 60,000 | 3,840,000 | 1.045 |
| Trajectory-VLA | 32 | 64 | 57,407 | 3,674,048 | 1.000 |
| Traj-VLA-Qwen | 16 | 32 | 114,814 | 3,674,048 | 1.000 |
| π0 | 4 | 8 | 459,253 | 3,674,024 | 1.000 |
| π0.5 | 4 | 8 | 459,253 | 3,674,024 | 1.000 |

SmolVLA 使用整齐的 60,000 step，而不是精确一轮的 57,407；因此它比一轮多约 4.5%。比较
精确一轮 wall time 时应临时覆盖为 57,407，比较现有实验复现时则保留 60,000。

数据增加后先执行：

```bash
pixi run -e train policy-data-check --config_path=configs/POLICY.yaml
```

再使用报告中的训练帧数重算，不能把本节快照永久当作默认事实。

## 4. 如何选择更快的 A100 配置

1. 用 100～200 个预热后 step 测量，忽略 `torch.compile` 首次编译开销。
2. 逐步提高每卡 batch，直到 GPU 利用率趋于稳定、数据解码成为瓶颈或接近显存上限。
3. 比较稳定区间的 `updt_s`、`smp/s` 和 `gpu_mem_gb`。
4. 根据实验目标决定保持 step，还是按全局 batch 重算 step。
5. 调整 scheduler 总长度、warmup、评估和保存频率，使它们与新 step 尺度一致。

训练时间近似为：

$$T\approx S\times t_{update}.$$

固定数据量时应比较 `steps × updt_s`；固定 step 时直接比较稳定后的 `updt_s`。`smp/s` 适合
判断 GPU 与数据流水线利用率，但不能单独说明总实验时间。

`num_workers` 只负责 DataLoader 解码。增加它不会扩大模型 batch；过高会增加 CPU、内存和视频
解码竞争。先增大每卡 batch，再根据 GPU 等待数据的比例调 `num_workers`。

## 5. OOM 与优化变量

发生 OOM 时按以下顺序处理：

1. 降低每卡 batch；
2. 保留或启用模型已有的 gradient checkpointing；
3. 使用 BF16 和冻结主干；
4. 对适合的模块使用 LoRA；
5. 最后才降低图像分辨率、动作 horizon 或模型层数。

前两项主要改变吞吐和计算量；后几项可能改变模型容量或实验定义，必须作为独立实验记录。
π0 / π0.5 是否关闭 gradient checkpointing 应以实际 A100 容量和峰值为准，不能仅凭 GPU 名称推断。

增大 batch 后是否缩放学习率也是独立优化问题。当前配置不自动线性缩放学习率；若改变学习率，
应通过验证集单独确认，而不要与硬件加速结论混在一起。
