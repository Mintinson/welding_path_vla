# 多 GPU Batch、Step 与训练时间计算

本文统一说明训练配置中 `batch_size`、GPU 数量、`steps` 和数据集遍历次数的关系，主要用于
双 A100 配置。当前训练入口使用 Accelerate DDP，模型在每张 GPU 上各保留一份。

## 1. 基本定义

配置中的 `training.batch_size` 是每个 DDP 进程、也就是每张 GPU 的 batch。设：

- $N$：训练划分的总帧数；当前为 3,674,023；
- $G$：GPU 数量；双 A100 时为 2；
- $B_{gpu}$：`training.batch_size`；
- $A$：梯度累积次数；当前训练器为 1；
- $E$：希望遍历训练集的轮数。

一次 optimizer update 实际处理的全局 batch 为：

$$
B_{global}=G\times B_{gpu}\times A.
$$

完整遍历 $E$ 轮训练集所需的 update 数为：

$$
S=\left\lceil\frac{E\times N}{B_{global}}\right\rceil.
$$

必须向上取整，否则最后不足一个 batch 的数据不会被覆盖。实际处理的样本数为
$S\times B_{global}$，因此最多只会比目标多 $B_{global}-1$ 个样本。

## 2. 当前双 A100 配置

以下配置均以 $E=1$ 计算：

| 策略 | 每卡 batch $B_{gpu}$ | GPU 数 $G$ | 全局 batch $B_{global}$ | `training.steps` | 实际覆盖帧数 |
|---|---:|---:|---:|---:|---:|
| ACT | 32 | 2 | 64 | 57,407 | 3,674,048 |
| SmolVLA | 32 | 2 | 64 | 57,407 | 3,674,048 |
| Trajectory VLA | 32 | 2 | 64 | 57,407 | 3,674,048 |
| Traj-VLA-Qwen | 16 | 2 | 32 | 114,814 | 3,674,048 |
| π0 | 4 | 2 | 8 | 459,253 | 3,674,024 |
| π0.5 | 4 | 2 | 8 | 459,253 | 3,674,024 |

例如 Trajectory VLA：

$$
B_{global}=2\times32=64,
\qquad
S=\left\lceil\frac{3,674,023}{64}\right\rceil=57,407.
$$

如果改为 4 张 GPU、每卡 batch 32，则全局 batch 为 128，一轮数据需要：

$$
S=\left\lceil\frac{3,674,023}{128}\right\rceil=28,704.
$$

## 3. 固定 Step 与固定数据遍历量的区别

固定 `steps` 时，增大 batch 会增加训练看到的样本总数：

$$
N_{seen}=S\times B_{global},
\qquad
E_{effective}=\frac{N_{seen}}{N}.
$$

因此，“batch 翻倍但 step 不变”相当于训练轮数翻倍，并不是同等训练工作量下的加速。
这适用于研究固定 update 数的实验，但不能直接与一轮数据的配置比较 wall time。

若目标是更快完成相同的数据遍历量，应在增大 batch 后按上一节公式降低 `steps`。当前 A100
配置采用这种方式。

## 4. 估算和比较训练时间

忽略评估和 checkpoint 时，总训练时间近似为：

$$
T\approx S\times t_{update},
$$

其中 $t_{update}$ 对应 `train.log` 中稳定后的 `updt_s`。`torch.compile` 会让最初几十步包含
编译开销，因此应在预热后比较：

- `updt_s`：完成固定 step 或一轮数据所需时间的直接依据，越低越好；
- `smp/s`：GPU 和输入流水线的样本吞吐，越高越好；
- `mem_gb`：峰值显存，用于判断能否继续提高每卡 batch。

一轮数据的配置优劣应比较 `steps × updt_s`，不能只比较 batch 或 `smp/s`。

## 5. 在新硬件或新数据集上重算

1. 从 LeRobot metadata 和实际 eval split 得到训练帧数 $N$。
2. 用短程训练逐步提高每卡 batch，直到 GPU 利用率稳定或接近显存上限。
3. 计算 $B_{global}=G\times B_{gpu}$。
4. 使用 $S=\lceil E\times N/B_{global}\rceil$ 更新 `training.steps`。
5. 将 scheduler decay 设为新的总 step；warmup 保持约总 step 的 2%。
6. 相应缩短 `eval_steps` 和 `save_freq`，确保一轮训练中仍有足够的评估和 checkpoint。
7. 运行 100～200 个预热后 step，通过 `updt_s`、`smp/s` 和显存决定是否继续调整。

发生 OOM 时，先把每卡 batch 减半，再重新计算全局 batch 和 step。π0 / π0.5 应优先保留
梯度检查点；只有在 80 GiB A100 上确认显存充足后，才考虑关闭它减少重计算。

增大 batch 后是否调整学习率属于单独的优化实验。当前配置为了稳定性保持原学习率，不自动
进行线性缩放；如果修改学习率，应单独记录并通过验证集确认，而不要与硬件加速同时混为一个变量。
