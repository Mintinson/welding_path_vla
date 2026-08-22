# 焊接短时轨迹评估规范

本页描述用于论文实验的闭环轨迹指标。它与 `policy-evaluate` 的职责不同：后者在
LeRobot 留出数据上计算模型 loss 和动作 MAE，用于判断模型是否拟合数据；本页的
`evaluate-episode` / `evaluate-dataset` 则根据实际执行后的 TCP 轨迹，评价任务完成度、
焊缝跟踪精度、姿态、速度、平滑性、安全性和最终成功率。

## 1. 评估边界与数据口径

论文级评估位于 `src/welding_path_vla/evaluation/`：

- `schema.py`：统一的焊缝、轨迹、指令、安全和终止数据模型；
- `seam_geometry.py`：把 TCP 投影到有序三维折线焊缝，并插值切向、姿态和速度；
- `motion_metrics.py`：PCR、方向性、CTE、姿态误差、SpeedMAPE 和 jerk；
- `evaluator.py`：组合 episode 成功条件并聚合数据集指标；
- `adapters.py`：把项目 raw episode 或真机日志转换为 `EvaluationTrace`；
- `trajectory_metrics.py`：采集期间使用的旧版快速质量门，不作为论文主表。

所有运动学指标只使用 `track_mask == true` 的焊接阶段样本。靠近焊缝的 approach 和离开
焊缝的 retreat 不参与 PCR、CTE、姿态、速度及 jerk 计算。安全信号则检查完整 episode，
因此 approach/retreat 阶段发生碰撞或越界仍会导致安全条件失败。

报告中的长度、速度、时间、角度和 jerk 单位分别为 m、m/s、s、degree 和 m/s³。项目中
四元数统一采用 `(w, x, y, z)` 顺序。参考焊缝点必须按照任务要求的执行方向排列；该顺序
定义了“正向运动”，不是只用于绘图。

## 2. 焊缝投影与共同符号

离散参考焊缝由按顺序排列的点

$$
\mathcal C=\{\mathbf c_0,\mathbf c_1,\ldots,\mathbf c_{M-1}\}
$$

构成。第 $i$ 条线段的向量、长度和单位切向量为

$$
\mathbf u_i=\mathbf c_{i+1}-\mathbf c_i,\qquad
\ell_i=\lVert\mathbf u_i\rVert_2,\qquad
\mathbf t_i=\frac{\mathbf u_i}{\ell_i}.
$$

对于第 $t$ 个实际 TCP 位置 $\mathbf p_t$，程序先分别投影到每条有限线段。在线段 $i$
上的投影比例为

$$
\alpha_{t,i}=\operatorname{clip}\left(
\frac{(\mathbf p_t-\mathbf c_i)^\top\mathbf u_i}{\ell_i^2+\epsilon},0,1
\right),
$$

候选投影点为

$$
\hat{\mathbf c}_{t,i}=\mathbf c_i+\alpha_{t,i}\mathbf u_i.
$$

选择欧氏距离最近的线段

$$
i_t=\arg\min_i\lVert\mathbf p_t-\hat{\mathbf c}_{t,i}\rVert_2^2,
$$

从而得到最近焊缝点 $\hat{\mathbf c}_t$、局部切向量 $\mathbf t_t$ 和弧长坐标

$$
s_t=\sum_{j<i_t}\ell_j+\alpha_{t,i_t}\ell_{i_t},\qquad
L=\sum_{i=0}^{M-2}\ell_i.
$$

后续完成度、方向、位置、期望姿态和期望速度都使用同一组 $s_t$ 与 $i_t$，避免每个指标
采用不同的最近点定义。当前实现把焊缝视为折线；曲线工件在采集时已离散成足够密集的折线点。

## 3. 单个 episode 指标

### 3.1 路径完成度 PCR

PCR（Path Completion Ratio）回答“从开始焊接的位置出发，沿参考方向最远走到了焊缝的
什么位置”。当前实现为

$$
\boxed{
\mathrm{PCR}=\operatorname{clip}\left(
\frac{\max_t s_t-s_0}{L},0,1
\right)
}.
$$

对应字段为 `completion.pcr`，取值范围为 $[0,1]$，越大越好。默认要求
`PCR >= 0.95`。

这里使用“最远弧长减起始弧长”，而不是累计每一步的路程，因此在同一区域来回振荡不会把
PCR 虚增到 1。它也会惩罚从焊缝中间才开始执行的轨迹：即使从中点走到终点，PCR 也只有
约 0.5。PCR 本身不反映横向精度、姿态或速度，必须与后续指标一起使用。

### 3.2 正确方向运动占比

相邻样本的弧长增量为

$$
\Delta s_t=s_t-s_{t-1}.
$$

程序分别计算正向路程和全部弧长变化：

$$
D_+=\sum_t\max(\Delta s_t,0),\qquad
D_{\mathrm{abs}}=\sum_t|\Delta s_t|,
$$

并定义

$$
\boxed{
\mathrm{DirectionRatio}=
\begin{cases}
D_+/D_{\mathrm{abs}}, & D_{\mathrm{abs}}>0,\\
0, & D_{\mathrm{abs}}=0.
\end{cases}
}
$$

对应字段为 `completion.direction_ratio`。完全正向运动时等于 1，完全反向时等于 0；频繁
回退会使其下降。默认要求 `DirectionRatio >= 0.90`。完成条件要求 PCR 和方向占比同时
通过，避免模型走到较远位置后又长距离反向运动仍被判定为完成。当前计算没有额外死区或
平滑，小幅定位噪声也会形成正、负增量，因此不同实验必须保持相近的采样率和滤波口径。

### 3.3 横向焊缝跟踪误差 CTE

CTE（Cross-Track Error）衡量 TCP 偏离焊缝中心线的程度，是焊缝跟踪的核心位置指标。
位置误差向量为

$$
\mathbf d_t=\mathbf p_t-\hat{\mathbf c}_t.
$$

去掉沿焊缝切向的分量，只保留垂直于焊缝方向的误差：

$$
\mathbf P_{\perp,t}=\mathbf I-\mathbf t_t\mathbf t_t^\top,
\qquad
e_{\perp,t}=\left\lVert\mathbf P_{\perp,t}\mathbf d_t\right\rVert_2.
$$

单个 episode 固定报告三个统计量：

$$
\boxed{
\mathrm{CTE}_{\mathrm{RMSE}}=
\sqrt{\frac{1}{N}\sum_{t=1}^{N}e_{\perp,t}^2}
}
$$

$$
\boxed{\mathrm{CTE}_{95}=Q_{0.95}(\{e_{\perp,t}\})},\qquad
\boxed{\mathrm{CTE}_{\max}=\max_t e_{\perp,t}}.
$$

对应字段分别为 `tracking.cte_rmse_m`、`tracking.cte_p95_m` 和
`tracking.cte_max_m`，越小越好：

- RMSE 描述整段轨迹的典型误差，并对较大偏差施加平方惩罚；
- 95% 分位数描述绝大多数时刻的最差水平，较少被单个异常采样点支配；
- 最大值用于捕捉短暂但可能影响焊接质量的严重偏离。

位置条件要求三项同时不超过阈值。当前默认阈值依次为 1.5 mm、2 mm 和 5 mm。
由于 CTE 有意移除了切向分量，沿焊缝方向的超前或滞后不会计入 CTE；这部分由 PCR、方向性、
速度和终止条件约束。特别地，越过折线端点后继续沿末段切向移动也可能保持很小的 CTE，不能
仅凭 CTE 判断 episode 成功。

### 3.4 焊枪姿态误差 OE

参考焊缝的每个离散点同时保存期望四元数。实际投影位于线段内部时，程序先处理四元数
$\mathbf q$ 与 $-\mathbf q$ 的等价性，选择同一半球上的表示，再执行归一化线性插值
（NLERP）：

$$
\tilde{\mathbf q}_{i+1}=
\begin{cases}
-\mathbf q_{i+1}, & \mathbf q_i^\top\mathbf q_{i+1}<0,\\
\mathbf q_{i+1}, & \text{otherwise},
\end{cases}
$$

$$
\mathbf q_{d,t}=\operatorname{normalize}\left(
(1-\alpha_t)\mathbf q_{i_t}+\alpha_t\tilde{\mathbf q}_{i_t+1}
\right).
$$

设期望和实际旋转矩阵分别为 $\mathbf R_{d,t}$ 与 $\mathbf R_t$，姿态误差取 SO(3) 上的
最短测地角：

$$
\boxed{
e_{R,t}=\left\lVert\log(\mathbf R_{d,t}\mathbf R_t^\top)\right\rVert_2
=\cos^{-1}\left(
\frac{\operatorname{tr}(\mathbf R_{d,t}^\top\mathbf R_t)-1}{2}
\right)
}.
$$

代码内部以弧度计算，报告时转换为度。单个 episode 输出

$$
\boxed{\mathrm{OE}_{95}=Q_{0.95}(\{e_{R,t}\})},\qquad
\boxed{\mathrm{OE}_{\max}=\max_t e_{R,t}}.
$$

对应字段为 `tracking.orientation_p95_deg` 和
`tracking.orientation_max_deg`，越小越好。默认 ESR 只要求 `OE95 <= 2°`，最大值用于
诊断短时姿态突变，当前不直接参与成功判定。

### 3.5 速度跟踪误差 SpeedMAPE

首先根据真实时间戳对 TCP 位置进行数值微分：

$$
\dot{\mathbf p}_t\approx\frac{d\mathbf p(t)}{dt}.
$$

实现使用 `numpy.gradient(values, timestamps)`，因此支持非等间隔采样。只保留沿局部焊缝
切向的有符号速度：

$$
v_{\parallel,t}=\mathbf t_t^\top\dot{\mathbf p}_t.
$$

期望速度可以是整条焊缝的标量，也可以在每个参考点给出；后一种情况按弧长进行一维线性
插值，得到 $v_d(s_t)$。最终定义

$$
\boxed{
\mathrm{SpeedMAPE}=\frac{1}{N}\sum_{t=1}^{N}
\frac{|v_{\parallel,t}-v_d(s_t)|}{v_d(s_t)+10^{-6}}
}.
$$

对应字段 `tracking.speed_mape` 保存比例而不是百分数，例如 `0.20` 表示 20%，越小越好。
默认要求 `SpeedMAPE <= 0.20`。因为 $v_{\parallel,t}$ 带符号，反向运动不仅影响方向占比，
也会产生较大的速度误差。该指标假设期望速度为正且明显大于 $10^{-6}$ m/s；接近零速的
停顿任务不适合直接使用 MAPE。

### 3.6 轨迹平滑性与 jerk ratio

jerk 是位置对时间的三阶导数：

$$
\mathbf j_t=\frac{d^3\mathbf p(t)}{dt^3}.
$$

程序对位置连续执行三次数值微分，再报告 jerk 向量模长的均方根：

$$
\boxed{
J_{\mathrm{model}}=
\sqrt{\frac{1}{N}\sum_{t=1}^{N}\lVert\mathbf j_t\rVert_2^2}
}.
$$

若提供同任务的专家 TCP 轨迹，则用完全相同的方法得到 $J_{\mathrm{expert}}$，并计算

$$
\boxed{
J_{\mathrm{ratio}}=\frac{J_{\mathrm{model}}}{J_{\mathrm{expert}}}
}.
$$

对应字段为 `tracking.jerk_rms_m_s3`、`tracking.expert_jerk_rms_m_s3` 和
`tracking.jerk_ratio`。`jerk_ratio ≈ 1` 表示与专家平滑性接近，大于 1 表示比专家更抖。

jerk 的数值微分对采样率和噪声非常敏感，因此实现有三项有效性约束：

1. track 阶段少于 4 个样本时，jerk 为 `null`；
2. 估计采样率低于 `jerk_min_sample_rate_hz`（默认 80 Hz）时不计算 jerk；
3. 专家 jerk 小于 `jerk_reference_floor_m_s3`（默认 0.001 m/s³）时不计算比值，避免
   近零分母制造极大的无意义数值。

平滑性默认只报告，不计入 ESR。一方面当前指标尚未内置滤波，另一方面把 jerk 强制加入成功
条件可能鼓励模型通过降低速度获得更小的 jerk。只有显式设置
`require_smoothness_for_success: true` 时，才要求有效的 `jerk_ratio <= jerk_ratio_max`；
此时缺少有效 jerk ratio 也会判定平滑性失败。

### 3.7 安全与正常终止

设任务要求记录的安全信号集合为 $\mathcal K$。每个信号 $k$ 是逐帧布尔数组，任意一帧为
真即表示该类违规：

$$
V_k=\bigvee_t \mathrm{signal}_{k,t}.
$$

安全条件采用 fail-closed 设计：

$$
\boxed{
C_{\mathrm{safety}}=
\left(\bigwedge_{k\in\mathcal K}\neg V_k\right)
\land (\text{没有缺失的必需信号})
}.
$$

`safety.violations` 保存发生过违规的信号名称，`safety.missing_signals` 保存缺失名称；两者
任一非空都会令 `safety.safe=false`。默认必需信号包括：

- `collision`：发生不允许的接触；
- `joint_limit`：关节位置越界；
- `joint_velocity`：关节速度超过限制；
- `joint_acceleration`：关节加速度超过限制；
- `action_increment`：单步 TCP 位移对应的速度超过限制。

对于项目 raw episode，`joint_velocity` 由反馈速度与
`safety.joint_velocity_limit_rad_s` 比较；`joint_acceleration` 由反馈速度数值微分后与
`evaluation.joint_acceleration_limit_rad_s2` 比较；`action_increment` 由每步已执行 TCP
平移量除以步长后，与 `safety.tcp_speed_limit_m_s` 比较。当前 raw adapter 尚无独立的
关节限位事件，`joint_limit` 暂以全 false 占位；正式实验若要把该项作为可靠结论，必须记录
控制器或安全监控器给出的真实限位信号。

正常终止条件为

$$
\boxed{
C_{\mathrm{termination}}=\mathrm{completed}
\land\neg\mathrm{timed\_out}
\land\neg\mathrm{operator\_stopped}
}.
$$

即轨迹必须由任务状态机正常完成，超时或人工中止都不能算成功。

## 4. ESR 成功判定与数据集聚合

### 4.1 单个 episode 的成功条件

Episode Success Rate（ESR）首先为每条 episode 定义布尔成功值。当前默认条件为

$$
\boxed{
\mathrm{Success}=
C_{\mathrm{instruction}}
\land C_{\mathrm{completion}}
\land C_{\mathrm{position}}
\land C_{\mathrm{orientation}}
\land C_{\mathrm{speed}}
\land C_{\mathrm{safety}}
\land C_{\mathrm{termination}}
}.
$$

其中连续指标到布尔条件的映射为：

| `conditions` 字段 | 通过条件 |
| --- | --- |
| `completion` | `PCR >= pcr_min` 且 `DirectionRatio >= direction_ratio_min` |
| `position` | CTE RMSE、CTE95、CTE max 三者都不超过各自阈值 |
| `orientation` | `OE95 <= orientation_p95_deg` |
| `speed` | `SpeedMAPE <= speed_mape_max` |
| `smoothness` | jerk ratio 有效且不超过阈值；默认不要求其通过 |
| `safety` | 没有安全违规且没有缺失的必需信号 |
| `termination` | 正常完成、未超时、未人工终止 |

平滑性无论是否纳入 ESR 都会写入 `conditions.smoothness`，便于错误分析；仅当
`require_smoothness_for_success=true` 时，它才加入上面的逻辑合取。

### 4.2 默认阈值

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

这些是初始工程阈值，不应直接当作论文最终阈值。正式阈值应根据焊接工艺允许偏差、机器人
重复定位精度、标定误差和传感器噪声预注册。`evaluation.tracking_band_m` 当前是保留字段，
尚未参与论文级指标或 ESR 判定。

### 4.3 数据集聚合

设数据集包含 $E$ 条 episode，第 $e$ 条的成功指示量为 $S_e\in\{0,1\}$，则

$$
\boxed{
\mathrm{ESR}=\frac{1}{E}\sum_{e=1}^{E}S_e
}.
$$

连续指标同样先在每条 episode 内计算，再对 episode 做算术平均。例如

$$
\mathrm{PCR}_{\mathrm{mean}}=\frac{1}{E}\sum_e\mathrm{PCR}_e,
\qquad
\overline{\mathrm{CTE}_{95}}=\frac{1}{E}\sum_e\mathrm{CTE}_{95,e}.
$$

这种方式使长、短 episode 权重相同，不会让帧数更多的长轨迹支配结果。当前汇总字段包括
`ESR`、`PCR_mean`、CTE RMSE 均值、CTE95 均值、OE95 均值、SpeedMAPE 均值，以及所有
有效 jerk ratio 的均值。`condition_failures` 统计每类条件失败的 episode 数，用于区分完成、
精度、速度和安全等不同失败原因。若没有任何有效 jerk ratio，汇总值为 `null`。

## 5. 仿真 episode 评估

专家数据用于验证指标链路时，可显式声明任务选择正确：

```bash
pixi run -e dev evaluate-episode \
  --episode=datasets/weldpath_raw_v2/episodes/episode_000000 \
  --config_path=configs/default.yaml \
  --assume_reference_task=true
```

评估完整数据集：

```bash
pixi run -e dev evaluate-dataset \
  --collection.dataset_root=datasets/weldpath_raw_v2 \
  --config_path=configs/default.yaml \
  --assume_reference_task=true \
  --output=outputs/evaluation/simulation-summary.json
```

评估真实策略输出时不要使用 `--assume_reference_task=true`。应在 episode 元数据中写入：

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

## 6. 真机应如何记录

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
  --source=real \
  --episode=outputs/real_eval/real_eval_000001 \
  --config_path=configs/default.yaml \
  --output=outputs/real_eval/real_eval_000001/report.json
```

缺少任意 `required_safety_signals` 时安全条件按失败处理，而不是默认无违规。

## 7. 真机测量来源

- TCP 位姿：机器人控制器反馈，统一变换到项目 world frame；评估前完成基座外参与 TCP 标定。
- 焊缝真值：优先使用 CAD + 夹具标定，其次使用独立三维测量；不能直接使用被评估模型自己的焊缝预测作为真值。
- 碰撞：安全 PLC、保护停止、力/力矩阈值或经验证的电流阈值组合。
- 关节限位/速度/加速度：使用控制器状态和安全门事件，不只依赖命令已被下层裁剪这一事实。
- 动作增量：记录安全投影前后的命令，超限或被裁剪都应留下事件。
- 超时/人工终止：由状态机和操作员面板写入，不从轨迹长度猜测。

真机控制日志建议保持 100 Hz，图像至少保持 30 Hz，通过时间戳关联。低于
`jerk_min_sample_rate_hz` 时，jerk 三项输出为 `null`。若专家轨迹近似直线匀速，
其 jerk 小于 `jerk_reference_floor_m_s3`，分母退化，此时只报告双方 jerk，
`jerk_ratio` 输出为 `null`，而不是给出误导性的巨大比值。

100 Hz TCP 数值三阶微分会放大编码器和标定噪声；正式 jerk 对比必须对模型轨迹和
专家轨迹使用同一滤波器、截止频率和边界处理，并同时保存未滤波原始数据。当前首版
指标不内置滤波，避免隐藏实验处理。

## 8. ICR 与辅助任务头

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
