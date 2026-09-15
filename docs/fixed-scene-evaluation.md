# 固定可行场景的闭环评价

正式模型对比使用 `configs/evaluation_scenes_v2.json`。清单由
`scripts/generate_evaluation_scenes.py` 一次生成，每个任务 25 个场景、共 150 个；所有模型
重放同一配置、同一环境随机种子和同一初始状态。每条场景都先经过专家完整轨迹的连续 IK、关节
限位和碰撞预检。清单保存初态指纹，重放不一致时立即报错，不把场景错误计为模型失败。

当前清单已生成。在项目根目录、`zsh` 与 `welding_vla` 环境下运行模型评价：

```bash
/root/.local/bin/micromamba run -n welding_vla python scripts/deploy_simulation_policy.py \
  --config_path=configs/deploy/smolvla.yaml --deployment.run_all_tasks=true
```

如需创建新版本清单，用生成脚本指定新的 `--output` 路径，并在部署时用
`--deployment.scene_bank_path` 指向该文件；生成器不会覆盖已完成的清单。

更换模型时只更换部署 YAML 或 checkpoint。`configs/base.yaml` 默认指向固定场景清单；模型
YAML 中旧的 `seed`、`episodes`、`max_steps`、IK/动作限制、评价阈值不能覆盖清单。所有模型
固定执行 8 步动作块后重规划。模型本身的权重、processor 和推理去噪步数仍随 checkpoint。
若需运行旧式即时随机场景，显式指定 `--deployment.scene_bank_path=null`。

## 场景与训练的区别

评价使用与训练采集不同的种子；空间、焊缝任务参数和外观随机范围扩大至训练 YAML 的
1.2 倍，初始关节偏移扩大至 1.1 倍。
25 条场景中 13 条 forward、12 条 reverse，且任务参数逐条重采样。若扩大范围造成不可达，
生成器换种子重试，只有完整专家轨迹预检合格才写入清单。该设计保证相对当前训练 YAML
的配置级分布区别；发表前仍需核对实际训练数据所用配置与清单版本。

为避免视觉输入欠曝，每个评价场景固定开启两个顶灯，奇偶 episode 分别从 1.00–1.15 和
1.15–1.30 的强度区间采样。灯光色温、位置和方向仍然随机。代表性场景检查中，两路相机的
平均亮度从 72.61 提高到 81.22，亮度低于 35 的像素比例从 39.2% 降到 28.6%。

## 主指标与成功判定

默认 episode `evaluation` 仅含四个字段：

| 字段 | 定义 |
| --- | --- |
| `coverage_ratio` | 将焊缝分为 100 个等弧长区间；整条 rollout 中 TCP 距离中心线不超过 5 mm 的已访问区间比例。相邻轨迹点均贴近焊缝且位移不超过 25 mm 时，计入两点之间经过的区间；大跨度跳跃不填充中间区间。它与是否先到起点、运动方向解耦。 |
| `direction_ratio` | 整条 rollout 中，连续两帧均在 10 mm 跟踪带内时，正向弧长位移与正反向位移总和的比值；没有有效位移时为 0。 |
| `cte_p95_m` | 整条已执行轨迹的 TCP 到焊缝有限折线距离 P95，不先按起点或误差筛掉坏帧。只有没有任何执行点时才为 `null`。 |
| `success` | 曾进入起端 5% 区域，之后到达末端 5% 区域，覆盖率≥90%、方向一致率≥85%，且无碰撞或硬安全终止。起终点条件只影响成功判定，不阻止三项轨迹指标计算。 |

姿态、速度、jerk 等旧指标默认关闭，不参与成功判定；需要排障时显式设置
`evaluation.report_detailed_metrics=true`，它们会写在 `diagnostics.detailed_evaluation` 中。
动作限幅只在原始日志中记录，不自动使主指标失败。碰撞、非法动作和安全门停止仍是硬失败。
5 mm 是这版仿真路径执行协议的固定容差，不代表实际焊缝质量容差。

## v2 闭环控制限制

v2 对所有模型统一使用 25 mm 单步目标位移上限、10 mm IK 残差停止线和 10 mrad
关节软限位余量。绝对目标被位移限幅后，运行时清空动作块中基于旧 TCP 锚点生成的剩余动作，
下一步用真实执行状态重新推理。它避免连续执行旧目标导致后续相对距离逐步增大。物理关节硬限位、
碰撞和非有限动作仍立即终止。v1 文件及既有结果保留用于追溯，不能与 v2 结果合并统计。

视频在 rollout 结束后保留最终仿真状态 1 秒。这不增加策略步数，只使完成位置在 MP4 中可见。

每个模型任务目录有 `summary.json`（逐 episode）与 `metrics_summary.json`（四项主结果汇总）。
跨任务文件位于 `outputs/deploy/evaluation_scenes_v2/`，文件名含模型家族和 checkpoint
路径哈希；平均覆盖、方向、成功率的分母包含全部 25 条场景。所有已开始执行的 episode
都有 CTE；只有初始化失败等没有任何轨迹点的情况才不计入 `cte_observed_episodes`。
