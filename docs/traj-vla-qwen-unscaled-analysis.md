# Qwen unscaled 部署审计与下一步实验

后续 fixedlr 实验已完成，当前结论与执行计划见
[固定周期消融结果](traj-vla-qwen-fixedlr-analysis.md)。本文保留第一次实验的审计记录。

审计日期：2026-10-15。审计已有记录，未启动模型训练或仿真部署。

## 1. 结论与实验混杂

现有 unscaled 40k 的覆盖和贴近焊缝能力优于原 16-query 40k，但不能认定是取消
输入缩放单独造成的收益。原实验计划只改变 `scale_context_embeddings`，实际还改变了
学习率周期：LeRobot 的 `CosineDecayWithWarmupSchedulerConfig.build()` 在
`num_training_steps < num_decay_steps` 时自动等比例压缩 warmup/decay。

`outputs/train_v2/traj_vla_qwen_unscaled_curve_plate/train.log:2199` 明确记录：

```text
Scaling warmup: 4000 → 800, decay: 200000 → 40000 (scale factor: 0.200)
```

保存的模型及训练 JSON 仍写 4000/200000，不能反映这个运行时分支。
40k checkpoint 的 `training_state/scheduler_state.json` 则记录实际 LR 为 `2.5e-6`。
原 200k 周期的 40k LR 应为 `4.5464153616405e-5`，相差约 18.2 倍。
因此，前一次说明“短训练不压缩学习率曲线”错误；配置字段测试不足以覆盖真实行为。

已核对两次训练的数据路径、2000 个选中 episode、seed、每卡 batch=16、图像增强相同。
两卡有效 batch=32，训练/验证分别为 1800/200 个 episode。新增的 dense/readout 默认
关闭字段不改变结构，真正的混杂是实际调度周期。较低的 loss/梯度也不能归因于缩放一项。

训练日志中的 held-out eval loss：

| step | 原 16-query | unscaled 短周期 |
| --- | ---: | ---: |
| 10k | 0.8182 | 0.7630 |
| 20k | 0.7356 | 0.5568 |
| 30k | 0.7652 | 0.6701 |
| 40k | 0.7699 | 0.6519 |

新实验 20k 的 eval loss 比 40k 更低，但 rollout success 从 0/25 变成 4/25、覆盖率
从 22.60% 升到 41.76%；这直接说明本次不能按最低离线 loss 选部署 checkpoint。
新实验末段约 58–60 samples/s，原 40k 附近约 61–62 samples/s；没有证据说明取消
一次乘法显著加速了整体训练，也没有新增参数或注意力计算导致的大幅变慢。

## 2. 相同步数部署对照

数据来源（均在 `outputs/deploy` 下）：

| 标签 | 目录 | checkpoint |
| --- | --- | --- |
| 原 16-query | `traj_vla_qwen_aug_curve_plate/2026-10-09-01-18` | 40k |
| dense | `traj_vla_qwen_dense_curve_plate/2026-10-13-12-21` | 40k |
| unscaled 短周期 | `traj_vla_qwen_unscaled_curve_plate/2026-10-14-12-45` | 20k |
| unscaled 短周期 | `traj_vla_qwen_unscaled_curve_plate/2026-10-14-12-43` | 40k |

每组 25 个相同场景。逐场景核对任务、工件、相机、评价阈值等配置无差异，初始 TCP
位置完全一致；原模型 checkpoint 的 `n_action_steps=8`，新模型部署显式覆盖为 8，
不存在闭环频率变化。部署 JSON 的 training 部分来自启动 YAML，不能代替实际训练
checkpoint 配置，例如旧部署 JSON 中的 image_aug=false 并不代表训练没用增强。

| 指标 | 原 40k | dense 40k | unscaled 20k | unscaled 40k |
| --- | ---: | ---: | ---: | ---: |
| 平均覆盖率 | 22.44% | 27.04% | 22.60% | 41.76% |
| success | 3/25 | 4/25 | 0/25 | 4/25 |
| complete | 10/25 | 12/25 | 7/25 | 14/25 |
| 到达起点 | 8/25 | 10/25 | 10/25 | 11/25 |
| 平均 episode CTE P95 | 8.91 mm | 9.07 mm | 8.99 mm | 8.42 mm |
| 有效 CTE episode 数 | 21 | 23 | 23 | 24 |
| timeout | 12/25 | 11/25 | 16/25 | 10/25 |
| collision | 2/25 | 1/25 | 2/25 | 0/25 |
| 安全门停止 | 1/25 | 1/25 | 0/25 | 1/25 |

“无碰撞”不等于没有安全问题；unscaled 40k 的 episode 18 因关节指令超安全范围停止。

相对原 40k，配对覆盖率 18 个提高、6 个下降、1 个持平，平均提高 19.32 个百分点。
按这 25 个场景配对重采样 20000 次（NumPy RNG seed=123），均值差的探索性 95%
bootstrap 区间约为 +0.80 到 +36.68 个百分点。它只描述本组场景重采样的不确定性，
不包含训练种子、部署噪声或学习率混杂，不能当作缩放因果效应的置信区间。

双方都到达起点的 6 个 episode 为 0、4、6、8、14、22：覆盖率 52.00% → 68.83%，
CTE P95 7.93 → 7.78 mm。收益不完全来自到达率，但尾部跟踪误差的改善很有限。
该子集由模型结果共同筛选且只有 6 例，不能替代全部场景的主要指标。

## 3. 跟踪质量与视频抽样

主评价中的 tracking mask 是 `seam_distance <= 10 mm`。因此 CTE P95 只统计 10 mm
带内帧，远离焊缝的失败帧被排除；不能单看 8.91 → 8.42 mm 就判断完整任务精度。
它也不是每一帧的全局 pooled P95，而是各有效 episode 的 P95 再平均。

补充从 `rollout.npz` 重算的统计：每个有效 episode 在 tracking mask 内分别计算距离
中位数和距离 ≤3 mm 的帧占比，然后对 episode 等权平均。原 40k 与新 40k 分别为：

- 距离中位数均值：5.94 → 4.85 mm。
- ≤3 mm 帧占比均值：20.97% → 29.86%。

这说明典型贴缝程度有所改善，但长时间停在同一点也会贡献帧，不能替代沿弧长覆盖率。

已查看 episode 4、14 的 wrist 视频抽帧，并与逐步进度、距离对应：

- **episode 14：改善实例。** 原覆盖率 60%、timeout；新覆盖率 94%、success。
  新轨迹在 14.07/16.90/19.70 秒分别前进至约 32%/52%/78%，距离约 3.22/1.85/2.93 mm，
  视频抽帧能看到连续沿曲线推进。
- **episode 4：明显回退。** 原覆盖率 98%、success；新覆盖率 3%、timeout。
  约 27.5 秒后长时间停在进度 14% 附近，距离多为 8–11 mm，最终运行满 110 秒。
  视频不同时间抽帧位置变化很小，与轨迹停滞一致；不是仅仅视频尾部冻结。
- **episode 12：启动后停滞。** 原覆盖率 95%、success；新覆盖率 3%。
  27.5–55 秒在进度 7% 附近徘徊，之后逐渐偏离，末尾距离约 24 mm。
- **episode 22：最终成功仍有迟疑。** 原覆盖率 2%，新 96%；18–30 秒仍在起点附近，
  之后才稳定推进。成功率不能说明启动行为已消失。
- **episode 3：跟踪与任务成败要分开。** 覆盖率由 0 升到 91%，CTE P95 由
  9.46 降到 4.47 mm，但 `reached_start=false`，仍被判为失败。鉴于研究重点是跟踪，
  应把这一例计入跟踪改善的证据，而不是据此调松任务成功阈值。

14 个 complete 中有 10 个 success=false。complete 只表达到末端附近后明显离缝，
不保证路径高覆盖或精度合格；本次不修改完成规则或放宽评价阈值。

## 4. 与长期基线的距离

原 16-query 200k（`2026-10-10-00-21`）覆盖率 42.16%、success 6/25、CTE P95
8.44 mm；新 40k 已接近其平均覆盖，但成功数更少。不能据此断言以 1/5 训练成本得到
完全同等模型，也不能预测新模型训练到 200k 必然更强。

Adapter 200k（`vla_adapter_flow_curve_plate/2026-10-09-02-36`）覆盖率 74.76%、
success 17/25、CTE P95 5.20 mm；在与新模型都到达起点的 11 个场景中，Adapter/
unscaled 覆盖率为 90.18%/56.36%，CTE P95 为 4.01/8.19 mm。差距不只是 approach
和 retreat，沿缝跟踪本身还有明显不足。两种模型此处训练步数不同，只是现有可达到
水平的参考，不是公平的效率或架构消融。

## 5. 已修复的代码

在 Qwen 配置中新增 `scheduler_auto_scale`，默认 true，保留旧实验行为。
false 时返回本地注册的 `FixedCosineDecayWithWarmupSchedulerConfig`，复用官方
余弦公式，只阻止 `training.steps` 把调度周期缩短；没有修改 site-packages。

`tests/test_traj_vla_qwen_scheduler.py` 先在修复前通过真实训练 optimizer/scheduler
factory 复现失败：40k 与 200k 构建出的第 0 步 LR 不同。修复后覆盖：

- 短实验与长实验前 40k 的实际 LR 一致。
- 旧配置缺少新字段仍保留 800/40k 自动压缩行为。
- 真实执行 scheduler 至 40k，验证 LR 为 `4.5464153616405e-5`。
- 模型和完整 TrainPipelineConfig 保存恢复保留自定义调度类型。
- scheduler/optimizer 状态恢复并延长至 100k 后继续原曲线，不重启 warmup。

现有 checkpoint、部署记录与主配置的输入缩放默认值均未修改。新配置
`configs/traj_vla_qwen_unscaled_fixedlr_a100.yaml` 使用独立输出目录。

## 6. 明确的下一步

**第一优先：补齐 unscaled + 固定 200k 周期的 40k 对照。** 从 Prismatic 初始化，
不加载现有 unscaled 40k 动作权重。只做这一项，不同时改变 query 数量、dense、层数或 LoRA。

```bash
micromamba run -n welding_vla zsh -c '
accelerate launch --num_processes=2 scripts/train_policy.py \
  --config_path=configs/traj_vla_qwen_unscaled_fixedlr_a100.yaml
'
```

训练后评估 20k、40k；以下命令中的 `040000` 可以替换为 `020000`：

```bash
micromamba run -n welding_vla zsh -c '
python scripts/deploy_simulation_policy.py \
  --config_path=configs/traj_vla_qwen_unscaled_fixedlr_a100.yaml \
  --policy.checkpoint=outputs/train_v2/traj_vla_qwen_unscaled_fixedlr_curve_plate/checkpoints/040000/pretrained_model
'
```

主要对照是原 16-query 40k；已有 unscaled 短周期 40k 用于判断收益是否依赖更早降 LR。
建议新、原 40k 分别重复 3 次固定 25 场景评估。scene 固定不代表 Flow 随机噪声固定，
不要把相同场景 seed 当成完全相同的动作采样；统计每次结果和波动，不只挑最好的一次。

如果固定周期后覆盖/贴缝改善仍稳定存在，且未增加安全停止，再从**新 fixedlr 实验**
恢复至总计 100k，比较 60k/80k/100k，不直接跑满 200k：

```bash
micromamba run -n welding_vla zsh -c '
accelerate launch --num_processes=2 scripts/train_policy.py \
  --config_path=configs/traj_vla_qwen_unscaled_fixedlr_a100.yaml \
  --training.resume=true --training.steps=100000
'
```

如果固定周期无改善而短周期仍好，下一项是 scaled + 同样 800/40k 的短周期对照，
分离调度收益，不宣布取消缩放有效。若固定周期对照稳定改善但 100k 仍明显欠缺跟踪
精度，再考虑单独开放 Qwen LoRA；届时固定已经选好的缩放和调度设置，不能同时加层、
去 token merge、换 bridge 并把联合变化当作单模块收益。

总结：有效证据见[部署对照](#2-相同步数部署对照)，剩余跟踪问题见
[轨迹与视频](#3-跟踪质量与视频抽样)，归因限制见[实验混杂](#1-结论与实验混杂)，
先执行本节的 fixedlr 对照；目前没有足够证据支持直接扩模型或续训现有短周期权重。
