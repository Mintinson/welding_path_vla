# 焊接短时轨迹评估规范

评估命令由 `scripts/evaluate.py` 提供；Pixi 分别暴露
`evaluate-episode` 和 `evaluate-dataset` 两个任务。

## 1. 模块边界

论文级评估位于 `src/welding_path_vla/evaluation/`，分为：

- `schema.py`：统一的焊缝、轨迹、指令、安全和终止数据模型；
- `seam_geometry.py`：把 TCP 投影到有序三维折线焊缝，并插值切向、姿态和速度；
- `motion_metrics.py`：PCR、CTE、姿态、SpeedMAPE 和 jerk；
- `evaluator.py`：组合 ESR 条件并聚合 ESR/ICR；
- `adapters.py`：项目 raw episode 与真机日志适配；
- `real_robot_eval.py`：真机离线评估入口；
- `trajectory_metrics.py`：仿真采集期间的旧版快速质量门，保持兼容但不作为论文主表。

指标核心只依赖 `EvaluationTrace`。以后把折线替换为样条曲线、把人工 ICR 替换为结构化 VLM 输出，或者增加力控安全条件时，不需要修改其他指标。

## 2. 主表指标

单条 episode 报告：

- `success`：ESR 的单条布尔值；
- `instruction`：焊缝选择、方向、顺序三个 ICR 子条件；
- `completion.pcr`：沿正确方向从初始投影位置达到的最远弧长比例；
- `completion.direction_ratio`：正向弧长增量占总弧长变化的比例；
- `tracking.cte_rmse_m / cte_p95_m / cte_max_m`；
- `tracking.orientation_p95_deg / orientation_max_deg`；
- `tracking.speed_mape`：比例值，乘以 100 后以百分数报告；
- `tracking.jerk_rms_m_s3 / expert_jerk_rms_m_s3 / jerk_ratio`；
- `safety.violations / missing_signals`；
- 每个 ESR 子条件的通过结果。

数据集聚合报告：

- `ESR`；
- `ICR`；
- `PCR_mean`；
- CTE RMSE、CTE95、OE95、SpeedMAPE 和 jerk ratio 的 episode 均值；
- 每类条件的失败次数，供错误分析使用。

PCR 使用“相对于 track 起点达到的最远正向弧长”，而不是把每一步路程累加。因此反复前后振荡不会把 PCR 人为累积到 1，反向执行也会得到接近 0 的 PCR。

## 3. ESR 判定

默认要求同时满足：

```text
instruction
and completion
and position
and orientation
and speed
and safety
and termination
```

平滑性默认只报告，不纳入 ESR，避免模型通过降低速度获得虚假的低 jerk。若实验明确需要，把 YAML 中 `require_smoothness_for_success` 设为 `true`。

所有阈值位于统一配置：

```yaml
evaluation:
  pcr_min: 0.95
  direction_ratio_min: 0.90
  cte_rmse_m: 0.0015
  cte_p95_m: 0.002
  cte_max_m: 0.005
  orientation_p95_deg: 2.0
  speed_mape_max: 0.20
  jerk_ratio_max: 2.0
  jerk_min_sample_rate_hz: 80.0
  jerk_reference_floor_m_s3: 0.001
  joint_acceleration_limit_rad_s2: 10.0
  require_smoothness_for_success: false
```

这些是初始工程阈值，不应直接当作论文最终阈值。正式阈值应根据焊接工艺允许偏差、机器人重复定位精度和传感器噪声预注册。

## 4. 仿真 episode 评估

专家数据用于验证指标链路时，可显式声明任务选择正确：

```bash
pixi run -e dev evaluate-episode \
  --episode datasets/weldpath_raw_v1/episodes/episode_000000 \
  --config configs/default.yaml \
  --assume-reference-task
```

评估完整数据集：

```bash
pixi run -e dev evaluate-dataset \
  --dataset datasets/weldpath_raw_v1 \
  --config configs/default.yaml \
  --assume-reference-task \
  --output outputs/evaluation/simulation-summary.json
```

评估真实策略输出时不要使用 `--assume-reference-task`。应在 episode 元数据中写入：

```json
{
  "instruction_assessment": {
    "seam_correct": true,
    "direction_correct": true,
    "sequence_correct": true
  }
}
```

如果该字段缺失，ICR 按失败处理，避免从 TCP 恰好经过目标焊缝错误推断“模型理解了指令”。

## 5. 真机应如何记录

真机评估目录：

```text
real_eval_000001/
├── control.npz
└── task.json
```

`control.npz` 至少包含：

```text
timestamp                 (N,)       单调时间，秒
tcp_position              (N, 3)     项目 world frame，米
tcp_quaternion_wxyz       (N, 4)
track_mask                (N,)       或 phase (N,)

collision                 (N,) bool
joint_limit               (N,) bool
joint_velocity            (N,) bool
joint_acceleration        (N,) bool
action_increment          (N,) bool
```

计算 jerk ratio 时再加入：

```text
expert_timestamp          (Ne,)
expert_position           (Ne, 3)
```

`task.json` 示例：

```json
{
  "seam": {
    "seam_id": "upper_fillet",
    "points_world": [[0.40, -0.10, 0.31], [0.40, 0.10, 0.31]],
    "quaternions_wxyz": [[1, 0, 0, 0], [1, 0, 0, 0]],
    "desired_speed_mps": 0.02
  },
  "instruction_assessment": {
    "seam_correct": true,
    "direction_correct": true,
    "sequence_correct": true
  },
  "required_safety_signals": [
    "collision",
    "joint_limit",
    "joint_velocity",
    "joint_acceleration",
    "action_increment"
  ],
  "termination": {
    "completed": true,
    "timed_out": false,
    "operator_stopped": false
  }
}
```

执行评估：

```bash
pixi run -e dev evaluate-episode \
  --source real \
  --episode outputs/real_eval/real_eval_000001 \
  --config configs/default.yaml \
  --output outputs/real_eval/real_eval_000001/report.json
```

缺少任意 `required_safety_signals` 时安全条件按失败处理，而不是默认无违规。

## 6. 真机测量来源

- TCP 位姿：机器人控制器反馈，统一变换到项目 world frame；评估前完成基座外参与 TCP 标定。
- 焊缝真值：优先使用 CAD + 夹具标定，其次使用独立三维测量；不能直接使用被评估模型自己的焊缝预测作为真值。
- 碰撞：安全 PLC、保护停止、力/力矩阈值或经验证的电流阈值组合。
- 关节限位/速度/加速度：使用控制器状态和安全门事件，不只依赖命令已被下层裁剪这一事实。
- 动作增量：记录安全投影前后的命令，超限或被裁剪都应留下事件。
- 超时/人工终止：由状态机和操作员面板写入，不从轨迹长度猜测。

控制日志建议 100 Hz，图像仍可保持 20 Hz，通过时间戳关联。低于
`jerk_min_sample_rate_hz` 时，jerk 三项输出为 `null`。若专家轨迹近似直线匀速，
其 jerk 小于 `jerk_reference_floor_m_s3`，分母退化，此时只报告双方 jerk，
`jerk_ratio` 输出为 `null`，而不是给出误导性的巨大比值。

100 Hz TCP 数值三阶微分会放大编码器和标定噪声；正式 jerk 对比必须对模型轨迹和
专家轨迹使用同一滤波器、截止频率和边界处理，并同时保存未滤波原始数据。当前首版
指标不内置滤波，避免隐藏实验处理。

## 7. ICR 与辅助任务头

推荐让上层视觉语言模块输出可审计的结构化计划：

```text
seam_id
start_id
end_id
direction
ordered_subtasks
desired_speed
work_angle
travel_angle
obstacle_constraints
```

训练时可加入该结构化计划的辅助损失，再由轨迹头预测短时动作。部署时即使不让该输出直接控制机器人，也建议保留低频日志或评测模式，否则无法区分“模型理解错任务”和“低层轨迹控制失败”。主表只报告 ICR，焊缝选择、方向和顺序准确率放入错误分析。
