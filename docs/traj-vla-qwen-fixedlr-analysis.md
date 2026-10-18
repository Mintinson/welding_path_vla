# Qwen 固定周期消融结果与下一步

审计日期：2026-10-15。本报告接续 `traj-vla-qwen-unscaled-analysis.md`，以新结果更新
下一步计划；未启动训练或部署，未修改数据、已有 checkpoint 和部署记录。

## 1. 结论

在当前 40k 预算和固定场景评估下，**取消输入缩放没有表现出稳定、值得默认启用的
跟踪精度收益**。有局部改善，但 success 和 CTE 没有改善，不能将其作为已验证优化。
这也不是“证明永远无效”：只有一个训练种子、一次每场景 rollout，且 40k 仍是早期模型。

目前更有希望的是短学习率周期，或短周期与取消缩放之间的组合效果。还缺少
scaled + 短周期这一组，不能直接下结论说上次收益全部来自学习率。
下一步只补齐这一组；不优先续训 fixedlr 到 100k，不叠加 dense、更多 query 或 LoRA。

## 2. 实验可比性

固定周期 20k/40k 结果分别位于：

- `outputs/deploy/traj_vla_qwen_unscaled_fixedlr_curve_plate/2026-10-15-06-31`
- `outputs/deploy/traj_vla_qwen_unscaled_fixedlr_curve_plate/2026-10-15-06-23`

训练 `scheduler_state.json` 中 20k 的实际 LR 为 `4.88375922620099e-5`，40k 为
`4.5464153616405e-5`；调度类型为 `fixed_cosine_decay_with_warmup`，本轮没有再次
发生自动压缩。原 16-query 的 200k 周期前缀与此相同。

原基线与本次 checkpoint 的整个 dataset 配置（含 episode 列表、增强、划分设置）
相同，optimizer 配置相同，seed=20260806、batch=16/卡、workers=8 相同。
输入/输出 processor 的 JSON 完全相同，normalizer 与 unnormalizer 的 safetensors
各字段逐张量完全相同，未发现 prompt 处理或归一化统计变化。
模型字段差异中的 dense/readout=false 等价于原缺省关闭值；scheduler 类型不同但
实际 LR 相同，模型前向的主要实验变量为 `scale_context_embeddings=false`。
历史代码版本、非确定性 GPU 运算与动作采样未被完全控制，因此仍不是多种子因果证明。

25 个 scene 的初始 TCP 和关节位置完全相同；任务、工件、相机、评价阈值等配置一致。
原部署 action_steps=null 读取 checkpoint 的 8，新部署显式为 8，实际执行频率相同。

## 3. 同步数结果

原 40k 指 `traj_vla_qwen_aug_curve_plate/2026-10-09-01-18`；短周期 40k 指
`traj_vla_qwen_unscaled_curve_plate/2026-10-14-12-43`。

| 指标 | 原 scaled 长周期 40k | unscaled 短周期 40k | 本次 unscaled 长周期 40k |
| --- | ---: | ---: | ---: |
| 覆盖率 | 22.44% | 41.76% | 28.44% |
| success | 3/25 | 4/25 | 2/25 |
| complete | 10/25 | 14/25 | 12/25 |
| 到达起点 | 8/25 | 11/25 | 9/25 |
| 平均 episode CTE P95 | 8.91 mm | 8.42 mm | 8.95 mm |
| 有效 CTE episode 数 | 21 | 24 | 21 |
| 方向一致率均值 | 76.31% | 84.43% | 73.46% |
| timeout | 12/25 | 10/25 | 12/25 |
| collision | 2/25 | 0/25 | 0/25 |
| 安全门停止 | 1/25 | 1/25 | 1/25 |

本次 20k 覆盖率为 21.80%、success=0/25、CTE P95=8.97 mm，有 3 次碰撞和 2 次
安全门停止；40k 安全表现更好，但没有看到跟踪尾部精度的明显进步。

本次 40k 相对原 40k，覆盖率 14 个提高、9 个下降、2 个持平，平均差 +6.00 个百分点。
对 25 个配对场景重采样 20000 次（RNG seed=123）的探索性 95% 区间为约
[-9.80, +21.60] 个百分点，覆盖 0；不能声称总体覆盖率已经可靠提升。
此区间不包含训练种子和动作噪声的不确定性，也不代表整个真实任务分布。

与短周期 40k 相比，7 个提高、15 个下降、3 个持平，均值差 -13.32 个百分点；
同样的重采样区间为 [-28.28, +1.64] 个百分点。短周期是更好的实验候选，但目前不能
视为具有统计确定性的赢家。

## 4. 单独看跟踪阶段

原 40k 和本次 40k 都到达起点的 episode 为 0、4、6、22，仅 4 例：

- 覆盖率：61.00% → 53.00%。
- CTE P95：7.31 → 8.71 mm。

该共同到达子集是模型输出筛选的，不是无偏估计，但足以说明“整体不佳只是 approach
阶段拖累，跟踪已经显著变好”没有证据支持。

从每个有效 episode 的 tracking 帧计算距离中位数与 ≤3 mm 帧占比，再等权平均：

| 指标 | 原 40k | 短周期 40k | 本次 40k |
| --- | ---: | ---: | ---: |
| 距离中位数均值 | 5.94 mm | 4.85 mm | 5.17 mm |
| ≤3 mm 帧占比均值 | 20.97% | 29.86% | 25.58% |

典型帧存在小幅贴缝改善，但没有转化为更好的尾部精度和稳定成功。
tracking mask 只保留 ≤10 mm 的帧，远离焊缝的失败帧不计入这些误差指标；而且停滞在
某个近缝位置会贡献大量帧。因此这些数值必须和沿弧长覆盖率一起解读。

具体轨迹与视频抽样：

- **episode 16：确实有成功跟踪。** 覆盖 100%、CTE P95=5.51 mm；17.8 秒时进度
  13%、距离 1.79 mm，22.23 秒进度 58%、距离 1.63 mm。wrist 视频抽帧显示连续推进。
- **episode 4：能够推进，但沿缝精度不稳定。** 本次覆盖 80%、complete 但失败；
  原基线覆盖 98% 且成功，短周期只覆盖 3%。不同设置各有擅长场景，不能只挑改善例。
- **episode 22：靠近起点后仍无法稳定启动和推进。** 短周期覆盖 96% 且成功，
  本次只有 8%；36.67 秒位于约 4% 进度，110 秒仍仅约 17%。视频不同时间抽帧也显示
  大部分路径未被推进覆盖，而不是录制文件冻结。
- **episode 12：停在起点旁偏移位置。** 约 36.67 秒后处于 7%–10% 进度附近，
  距离多为 14–21 mm，最终 timeout；原基线覆盖 95% 且成功。
- **episode 14：无法可靠接近焊缝。** 短周期覆盖 94% 且成功，本次覆盖 0%，
  无有效 tracking 帧。这一例不进入 CTE 均值，不能因为 CTE 未计入它而忽略失败。

本次 12 个 complete 中 10 个不成功。complete 只是终止/退出动作判定，不能作为
跟踪质量替代；本次没有更改安全门、完成规则或评价阈值。

## 5. 离线 loss 与可能解释

| 训练 step | 原 scaled 长周期 | unscaled 短周期 | 本次 unscaled 长周期 |
| --- | ---: | ---: | ---: |
| 10k eval loss | 0.8182 | 0.7630 | 0.7086 |
| 20k eval loss | 0.7356 | 0.5568 | 0.6300 |
| 30k eval loss | 0.7652 | 0.6701 | 0.7455 |
| 40k eval loss | 0.7699 | 0.6519 | 0.6458 |

本次 40k loss 最低但 rollout 不及短周期，不能用这个离线指标选择部署模型。
短周期在 40k 时已经降到 `2.5e-6`，长周期此时仍为 `4.5464e-5`；更早降低 LR
可能有利于当前小模型的闭环稳定性，也可能与输入幅值存在交互。这是待验证解释，
不是已确认原因。不能把一个完整短周期和长周期早期点的不同解释成长期架构上限。

取消缩放不增加视觉 token、动作历史或恢复偏差的数据，因此不会自动解决已有的
停滞和纠偏问题。当前证据只支持它影响了训练行为，不支持其已解决主要模型瓶颈。

## 6. 下一步只做 scaled 短周期对照

补齐最后一格后停止这一轮缩放/调度消融，再根据结果选择后续模型实验基线：

| 实际调度 | 保留缩放 | 取消缩放 |
| --- | --- | --- |
| 4k warmup / 200k decay，在 40k 评估 | 已有：22.44% 覆盖 | 本次：28.44% 覆盖 |
| 800 warmup / 40k decay，在 40k 评估 | **下一步** | 已有：41.76% 覆盖 |

新配置 `configs/traj_vla_qwen_scaled_shortlr_a100.yaml` 保留输入缩放，并显式指定
800/40000 周期、`scheduler_auto_scale=false`；没有依赖自动压缩。测试通过真实
optimizer/scheduler factory 构建配置，逐点验证全部 0–40000 步与上一轮 unscaled
短周期的实际 LR 完全相同。训练数据、增强、seed、batch、层数、query 和部署步数不变。
不增加模型参数，不启用 dense/grounding/motion/LoRA。
新配置已通过实际训练配置构建检查；相关 70 项测试通过，Ruff 与 diff 格式检查通过。

从预训练 Prismatic 初始化新动作策略，**不加载现有动作 checkpoint、不 resume**：

```bash
micromamba run -n welding_vla zsh -c '
accelerate launch --num_processes=2 scripts/train_policy.py \
  --config_path=configs/traj_vla_qwen_scaled_shortlr_a100.yaml
'
```

输出为 `outputs/train_v2/traj_vla_qwen_scaled_shortlr_curve_plate`。评估 20k/40k：

```bash
micromamba run -n welding_vla zsh -c '
python scripts/deploy_simulation_policy.py \
  --config_path=configs/traj_vla_qwen_scaled_shortlr_a100.yaml \
  --policy.checkpoint=outputs/train_v2/traj_vla_qwen_scaled_shortlr_curve_plate/checkpoints/040000/pretrained_model
'
```

将 `040000` 换成 `020000` 可评估另一检查点。主对照是旧 unscaled 短周期 40k，
不是本次长周期 40k。建议两边各做 3 次同场景部署（重复运行对应命令），报告均值和
波动；scene seed 不等于 Flow 动作采样 seed，因此不要把单次 1–2 个成功差别当成定论。
旧短周期的复评命令为：

```bash
micromamba run -n welding_vla zsh -c '
python scripts/deploy_simulation_policy.py \
  --config_path=configs/traj_vla_qwen_unscaled_a100.yaml \
  --policy.checkpoint=outputs/train_v2/traj_vla_qwen_unscaled_curve_plate/checkpoints/040000/pretrained_model
'
```

后续决策：

1. 若 scaled 短周期与 unscaled 短周期相当或更好，保留原缩放，优先采用短周期作为
   当前 40k 实验基线；这支持短调度的贡献，不支持取消缩放是必要条件。
2. 若 unscaled 短周期在重复评估中稳定更好，保留该组合，认为取消缩放的收益可能
   依赖调度，不能推广为普遍有效。
3. 若差异处于重复评估波动范围内，缩放默认保留 true，不再投入更多同类消融。
   选定固定实验基线后，下一轮再单独验证 Qwen LoRA；不同时改层数、token merge
   或 dense 连接。这是后续候选，不是本次已经验证或实现的精度优化。

本次暂不推荐续训 fixedlr 到 100k：上一轮设定的“稳定改善后再续训”条件尚未满足。
也不能直接恢复已完成短周期的 checkpoint 并把 decay 拉长，当成公平的连续长周期。

总结：效果判断依据[同步数结果](#3-同步数结果)，跟踪不足依据
[跟踪阶段](#4-单独看跟踪阶段)，loss 与机制限制见[离线分析](#5-离线-loss-与可能解释)，
现在执行本节的 scaled 短周期实验，不先扩大模型。
