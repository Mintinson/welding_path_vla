# 仿真数据采集原理与数据格式

本文解释当前项目如何用几何专家生成焊接演示、如何在 robosuite / MuJoCo 中执行与验收，
以及原始 episode 和 LeRobot Dataset v3 分别保存什么。日常命令、训练和 Hub 上传见
[数据采集与训练工作流](data-training-workflow.md)，工件几何见
[工件与焊缝任务](simulation/workpieces.md)。

本文以当前代码为准，主要实现位于：

- [`simulation/task_sampling.py`](../src/welding_path_vla/simulation/task_sampling.py)：任务随机化、
  staging、连续 IK 与碰撞预检；
- [`simulation/expert_generator.py`](../src/welding_path_vla/simulation/expert_generator.py)：几何专家
  参考轨迹；
- [`simulation/collector.py`](../src/welding_path_vla/simulation/collector.py)：执行、逐步诊断和质量验收；
- [`dataset/recorder.py`](../src/welding_path_vla/dataset/recorder.py)：原始 episode 写入；
- [`dataset/export_lerobot.py`](../src/welding_path_vla/dataset/export_lerobot.py)：有效 episode 到
  LeRobot Dataset 的转换。

## 1. 两层数据的职责

项目有意保留“原始事实”和“训练表示”两层，而不是采集时直接写成某个模型专用格式：

```text
任务配置与随机种子
  → 工件和焊缝几何
  → 几何专家参考轨迹
  → 连续 IK / 限位 / 碰撞预检
  → robosuite 闭环执行 + 黑色焊缝逐段变白
  → 原始 episode：N 条动作 + N+1 个状态和图像
  → 质量门：有效 / 无效
  → LeRobot：N 行训练样本 + 双相机视频
  → 训练 processor：absolute EE target → relative action chunk
```

两层数据分别解决不同问题：

| 层 | 主要用途 | 是否绑定策略 |
|---|---|---|
| 原始 episode | 回放、故障诊断、重新评价、改变动作标签来源、重新导出 | 否 |
| LeRobot Dataset | 视频解码、采样、归一化、训练划分和 Hub 分发 | 绑定统一 feature schema，但不绑定具体模型 |

原始层同时保存参考、命令和执行结果，因此更接近实验日志；LeRobot 层只保留训练真正需要的输入和
监督目标，因此更紧凑。

## 2. 一条 episode 如何产生

### 2.1 配置、编号与随机种子

`sim-collect` 首先读取模块化 YAML。采集器扫描已有的 `episode_*` 和 `dataset.json`，从下一个
未使用的全局编号继续，因此可以向同一个 raw 目录增量采集。

随机性分成两类：

- 任务参数按 `episode_index // randomization.task_group_size` 分组；默认每 10 个编号共享一组
  方向、姿态、速度和焊缝参数；
- 工件位姿、初始关节、初始 TCP 偏移和恢复扰动使用 episode 自己的种子，逐条变化。

任务组种子为 `collection.seed + group_index`，episode 种子为
`collection.seed + episode_index`。因此相同配置和全局编号可以复现采样结果；改变 worker 数量
不会改变某个编号对应的随机样本。

### 2.2 构造有向焊缝

`WorkpieceObject.seam()` 根据工件最终位姿和任务参数返回统一的 `SeamPath`。当前有四种几何
实现：

| 路径 | 用途 | `sample(progress)` 的行为 |
|---|---|---|
| `StraightSeamPath` | L 型和三面角直线焊缝 | 解析插值位置，切向和法向固定 |
| `CircularSeamPath` | 圆管上、下圆弧 | 按圆心角计算位置，切向和法向随圆周旋转 |
| `SinusoidalSeamPath` | 平板正弦 / 余弦焊缝 | 用稠密查找表按真实弧长采样，不按曲线参数等间隔采样 |
| `RoundedCornerSeamPath` | 三面角水平双焊缝 | 两条直线通过连续圆角一次连接 |

每个 `SeamFrame` 都包含世界系位置、沿任务方向的单位切向和焊枪接近侧法向。副法向由
`normal × tangent` 得到，三者构成焊缝局部标架。`direction=reverse` 会交换直线起终点或改变
圆弧、曲线的参数方向，所以 `progress=0 → 1` 始终表示任务要求的执行方向。

对 L 型、圆管和三面角任务，工作角决定焊枪相对工件表面的方向；曲线平板任务当前使用固定平板
法向。行走角在法向与切向之间产生倾斜，`tool_roll_deg` 再绕焊枪轴旋转。圆弧的局部标架会沿
管壁转动；`orientation_follow_ratio` 决定输出姿态跟随这种变化的比例。

工件还把所有候选任务焊缝离散为黑色视觉分段。每个策略 step 后，环境计算 TCP 到各段中心线
的距离，小于 `task.weld_success_distance_m` 的分段变为白色且不会回黑。相机观测因此同时包含
“还有哪些位置待焊”和“当前任务已经完成到哪里”；新 episode 或 `reset()` 会恢复全部黑线。

### 2.3 先证明轨迹可执行

正式渲染和录制前会进行分层拒绝采样：

1. 随机化工件在桌面上的位置和偏航角；
2. 在焊缝起点上方求解 staging 位姿，拒绝 IK 六维残差超过 0.005 或发生碰撞的场景；
3. 在 staging 构型附近随机化六个关节，并拒绝不可行构型；
4. 尝试施加初始 TCP 平移偏差，模拟初态和标定误差；
5. 从最终初态重新生成完整专家轨迹；
6. 以上一帧关节解为下一帧 IK 初值，对所有参考帧执行连续预检。

完整预检会拒绝以下情况：

- 任一帧 IK 六维残差超过 0.005；
- 任一关节进入配置的软限位余量；
- 相邻关节解超过 `joint_velocity_limit / policy_hz`，说明发生跳解或无法按时执行；
- 任一预检姿态出现有效碰撞。

预检得到的关节轨迹只用于证明参考轨迹连续可行。正式执行仍由在线控制链重新求解 IK，避免把
演示退化为直接回放预计算的 `qpos`。

### 2.4 几何专家如何生成轨迹

几何专家不是学习模型。它使用仿真中已知的焊缝中心线、法向和任务参数，生成世界系绝对 TCP
参考。完整轨迹由六段组成：

```text
当前 TCP
  ├─ 1. lift：竖直抬升到安全高度
  ├─ 2. transfer：在空中平移到焊缝起点上方
  ├─ 3. lower：下降到预接近点
  ├─ 4. descend：沿接近方向到达焊缝起点
  ├─ 5. track：沿有向焊缝执行
  └─ 6. retreat：从焊缝终点沿退出方向离开
```

直线段的离散帧数为：

\[
N_{segment}=\max\left(1,\left\lceil
\frac{L\,f_{policy}}{v_{segment}}
\right\rceil\right),
\]

跟踪段用焊缝真实弧长计算：

\[
N_{track}=\max\left(1,\left\lceil
\frac{L_{seam}\,f_{policy}}{v_{weld}}
\right\rceil\right).
\]

因此相邻参考点的平移距离不超过 `speed / policy_hz`。过渡段姿态使用 SLERP；跟踪圆弧时，专家
逐帧累积相邻焊缝标架的小旋转，避免整圆超过 180° 后 SLERP 选择错误的最短旋转分支。

默认接近速度高于焊接速度：空中运动用于快速到位，`track` 则按任务指定的工艺速度执行。退出段
使用独立速度。所有速度、姿态和几何实参都会写入 `task_parameters`，而英文指令在同一任务内
保持稳定。

### 2.5 30 Hz 动作如何变成 600 Hz 仿真

默认时钟为：

| 层级 | 频率 | 每个上层周期的次数 |
|---|---:|---:|
| MuJoCo 物理 | 600 Hz | 每个策略周期 20 步 |
| IK / 关节位置控制 | 120 Hz | 每个策略周期 4 次，每次 5 个物理步 |
| 专家、图像和数据行 | 30 Hz | 1 次 |

对第 `t` 个专家参考 `r_t`，环境接收绝对 TCP 位姿
`[x, y, z, qw, qx, qy, qz]`。在一个策略周期内，控制器分四次向目标插值，每次重新求解阻尼
最小二乘 IK，并按 120 Hz 关节速度上限裁剪命令，最后把关节位置目标送入 MuJoCo 执行器。

碰撞不是只在 30 Hz 周期末检查。环境会汇总该周期内所有物理子步观察到的接触，避免短暂碰撞
在采样时刻之间消失。焊丝尖端与目标工件之间亚毫米、低接触力的正常擦碰会被过滤；其他碰撞或
超过深度和力阈值的尖端接触仍会记录。

### 2.6 恢复样本

当 episode 被采样为恢复任务时，采集器在参考轨迹中点给 TCP 施加一个小的位置和姿态扰动，
后续仍跟踪同一几何参考。`recovery_window` 标出扰动后的短窗口；质量验收计算跟踪和姿态统计时
排除该窗口，但碰撞仍按完整 episode 计算。

这种数据让策略看到“偏离后回到焊缝”的状态，但不应把不可恢复的大幅碰撞伪装成恢复样本。

## 3. 时间对齐：为什么是 N 条动作和 N+1 个状态

一条含 `N` 个专家目标的 episode 按下面的顺序记录：

```text
state[0], image[0]
  -- action[0] --> state[1], image[1]
  -- action[1] --> state[2], image[2]
  ...
  -- action[N-1] --> state[N], image[N]
```

第 `t` 条训练关系是：

\[
(I_t, s_t, instruction) \longrightarrow a_t,
\qquad s_{t+1}\text{ 是执行结果。}
\]

保留终态有三个原因：

1. 可以直接计算 `tcp[t+1] - tcp[t]` 和实际速度；
2. 可以区分“发出了什么命令”和“机器人实际走到了哪里”；
3. 发生动力学滞后、限速或碰撞时仍能还原最后一步。

LeRobot 转换时只生成 `N` 行：第 `t` 行使用原始 `state[t]`、`image[t]` 和 `action[t]`。终态
`state[N]` 与最后一帧图像保留在 raw 数据中，但因为没有对应的下一条监督动作，不进入训练行。

## 4. 原始 episode 保存了什么

### 4.1 目录结构

```text
datasets/<raw_dataset>/
├── dataset.json
├── .incomplete/                   # 仅在写入过程中存在
└── episodes/
    └── episode_000000/
        ├── metadata.json
        ├── trajectory.npz
        ├── global.mp4
        └── wrist.mp4
```

录制先写入 `.incomplete/episode_xxxxxx`。视频、数组和元数据全部完成后，目录通过原子重命名进入
`episodes/`；异常时暂存目录会清理，从而避免半条数据被误当成完整 episode。

`metadata.json` 中当前 raw schema 标识是 `weldpath_raw_v1`；常用目录名里的 `raw_v2` 表示第二代
采集数据版本，而不是另一套数组 schema。判断兼容性应读取 metadata，不能只看目录名。

双相机 raw 视频默认是 640×480、30 FPS、H.264 MP4，使用相同调用时序写帧。H.264 便于在
VS Code 和浏览器中直接预览；训练归档阶段再由 LeRobot 转码为更节省空间的 AV1。

### 4.2 `trajectory.npz`

`trajectory.npz` 是若干同长度 NumPy 数组组成的压缩容器，不是逐帧 Python 字典，也不允许
pickle。下表中 `N` 是动作数，状态类数组为 `N+1`，动作和诊断类数组为 `N`。

| 字段 | 形状 | 含义 |
|---|---:|---|
| `timestamp` | `[N+1]` | 状态时间戳，秒，起点为 0 |
| `joint_position` | `[N+1, 6]` | 六轴实际关节角，rad |
| `joint_velocity` | `[N+1, 6]` | 六轴实际关节速度，rad/s |
| `tcp_position` | `[N+1, 3]` | 实际 TCP 世界坐标，m |
| `tcp_quaternion_wxyz` | `[N+1, 4]` | 实际 TCP 世界姿态，四元数 `wxyz` |
| `command_delta_pose_world` | `[N, 6]` | 从 `state[t]` 到专家参考的世界系位置差和旋转向量 |
| `command_delta_pose_base` | `[N, 6]` | 同一参考增量旋转到机器人基坐标系 |
| `command_delta_pose_seam` | `[N, 6]` | 同一参考增量旋转到当前焊缝局部坐标系 |
| `joint_position_command` | `[N, 6]` | 本策略周期最终送给位置执行器的关节目标，rad |
| `reference_position` | `[N, 3]` | 几何专家的理想世界系位置 |
| `reference_quaternion_wxyz` | `[N, 4]` | 几何专家的理想世界系姿态 |
| `safe_command_position` | `[N, 3]` | 对限速后关节命令做 FK 得到的世界系 TCP 目标 |
| `safe_command_quaternion_wxyz` | `[N, 4]` | 上述安全命令对应姿态 |
| `executed_delta_pose_world` | `[N, 6]` | 从实际 `state[t]` 到 `state[t+1]` 的世界系执行增量 |
| `executed_delta_pose_base` | `[N, 6]` | 实际执行增量在机器人基坐标系中的表达 |
| `phase` | `[N]` | `approach`、`track` 或 `retreat` |
| `seam_progress` | `[N]` | 当前专家参考在有向焊缝上的规划进度 `[0, 1]` |
| `cross_track_error` | `[N]` | 执行后 TCP 到有限焊缝的距离，m |
| `orientation_error_deg` | `[N]` | 执行后 TCP 与专家参考的轴角误差，deg |
| `ik_residual` | `[N]` | 在线 IK 最终六维位姿误差范数 |
| `collision` | `[N]` | 当前策略周期内是否出现有效碰撞 |
| `collision_pairs` | `[N]` | 碰撞 geom 对；同一步多个碰撞用 `|` 连接 |
| `recovery_window` | `[N]` | 是否位于恢复扰动后的排除窗口 |
| `episode_done` | `[N]` | 仅最后一个动作位置为真 |

三个“目标”容易混淆：

| 名称 | 它是什么 | 适合做什么 |
|---|---|---|
| `reference` | 几何专家希望到达的理想 TCP | 分析规划误差、专家消融 |
| `safe_command` | IK、关节速度限制后真正交给执行器的目标的 FK | 默认行为克隆标签 |
| `executed` | 动力学执行后的下一状态 | 分析控制滞后；无显式命令的真机示教标签 |

默认选择 `safe_command`，因为它保留专家意图，同时是当前机器人约束下实际可发送的命令；直接用
`reference` 可能要求策略模仿无法瞬时达到的目标，直接用 `executed` 又会把控制滞后和仿真动力学
误差当成期望行为。

当前 IK residual 将三维位置误差（m）与三维旋转向量（rad）直接拼接后取二范数。代码中部分
历史字段带有 `_m` 后缀，但它不是纯位置误差，不能直接解释为毫米；比较实验应保持同一 IK 定义
和阈值。

### 4.3 `metadata.json`

元数据保存不能自然表示为逐帧定长数组的信息：

| 类别 | 代表字段 | 目的 |
|---|---|---|
| 身份 | `episode_index`、`seed`、`robot_model`、`asset_id`、`seam_id` | 可追溯和复现 |
| 任务 | `instruction`、`direction`、`task_parameters` | 恢复该条数据实际执行的任务 |
| 初态采样 | `initial_joint_offset_deg`、`initial_tcp_offset_m`、各类 attempts | 分析随机覆盖和拒绝率 |
| 规划 | `staging_ik_residual`、`planning_max_ik_residual` | 区分场景采样问题与执行问题 |
| 坐标契约 | `coordinate_frames`、`quaternion_order`、单位字段 | 防止坐标系和单位误用 |
| 场景 | `workpiece_position`、`workpiece_quaternion_wxyz` | 复现场景和研究域随机化 |
| 质量 | `quality`、`recovery` | 决定能否进入训练集并保留失败原因 |
| 配置快照 | `resolved_config` | 即使 YAML 后续修改，也能还原采集参数 |

`resolved_config` 是完成 includes 和命令行覆盖后的最终配置，不是入口 YAML 的文件名。这对于长期
实验很重要：同名配置文件后来发生变化时，历史数据的真实参数仍在 episode 内。

### 4.4 `dataset.json`

数据集根目录的摘要记录下一 episode 编号、有效数、已落盘尝试数、预采样错误数、worker 数和
质量状态分布。它用于增量采集和快速查看健康度，不替代每条 episode 的 `metadata.json`。

预采样阶段完全找不到可行轨迹时，只增加 `collection_error`，不会生成 episode 目录；已经完成
录制但未通过质量门的数据仍会保留，便于排查失败，而不会被 LeRobot exporter 选中。

## 5. 质量门如何工作

正式执行后，`report_from_arrays()` 只在 `phase == track` 的帧上计算跟踪与姿态统计；恢复任务还
会排除 `recovery_window`。当前检查项为：

- 规划进度达到 `quality.minimum_progress`；
- CTE mean、P95 和 max 分别低于阈值；
- 姿态误差 P95 和 max 分别低于阈值；
- 所有在线 IK 残差不超过 5 mm；
- 整条 episode 没有有效碰撞。

全部通过时状态是 `valid_success`；带恢复扰动且通过时是 `valid_recovery`。否则根据主要原因标成
`collision_failure`、`invalid_planning` 或 `invalid_simulation`，具体检查项写入
`quality.failure_reasons`。

需要注意，raw 的 `seam_progress` 是专家参考进度，因此这里的 minimum-progress 检查确认采集器
执行到了规划末端，并不是从实际 TCP 独立估计的论文 PCR。实际轨迹是否跟住焊缝主要由 CTE 和
姿态门约束；策略 rollout 的 PCR、方向和自然退出由[评估规范](evaluation.md)单独计算。

LeRobot exporter 只接受 `valid_success` 和 `valid_recovery`。失败 episode 保留在 raw 层，但不会
污染训练数据。

## 6. LeRobot Dataset 保存了什么

### 6.1 转换时的帧映射

对每个有效 raw episode，exporter 依次执行：

1. 取前 `N` 个关节状态、TCP 状态和双相机帧；
2. 从 `resolved_config.policy.action_source` 选择 `safe_command`、`reference` 或 `executed`；
3. 把目标四元数转换为 rotation-6D；
4. 调用 `LeRobotDataset.add_frame()` 写入 `N` 行；
5. 调用 `save_episode()` 生成 episode 索引、统计和视频片段信息。

每一行的项目自定义 feature 为：

| Feature | dtype / shape | 顺序 |
|---|---|---|
| `observation.images.global` | video `[480, 640, 3]` | RGB，全局相机 |
| `observation.images.wrist` | video `[480, 640, 3]` | RGB，腕部相机 |
| `observation.state` | `float32 [13]` | `joint_1…joint_6, tcp_x, tcp_y, tcp_z, tcp_qw, tcp_qx, tcp_qy, tcp_qz` |
| `action` | `float32 [9]` | `x, y, z, r1x, r1y, r1z, r2x, r2y, r2z` |

`action` 的前三维是世界系绝对位置，后六维是目标旋转矩阵的前两行。rotation-6D 避免了四元数
`q` 与 `-q` 表示同一姿态造成的不连续，也可以通过 Gram–Schmidt 恢复合法旋转矩阵。

LeRobot 自动增加以下索引列：

| 列 | 含义 |
|---|---|
| `timestamp` | episode 内时间，当前为 `frame_index / 30` |
| `frame_index` | episode 内从 0 开始的帧编号 |
| `episode_index` | LeRobot 目标数据集内连续编号，不等于 raw 目录编号 |
| `index` | 整个数据集内连续的全局行号 |
| `task_index` | 指向 `meta/tasks.parquet` 中英文任务字符串的整数 ID |

任务文本不会在每行重复保存。`meta/tasks.parquet` 保存 `task_index → task` 映射，读取样本时由
LeRobot 恢复 `task` 字段，这样可以减少重复字符串占用。

当前 exporter 不把 raw `task_parameters` 展开成 LeRobot feature；精确速度、角度、圆弧范围和
工件位姿仍只存在 raw `metadata.json`。因此现有模型可以按稳定英文任务区分任务类别，但不能从
LeRobot 行中显式读取某条 episode 的目标速度。如果后续要研究技术参数条件控制，应新增明确的
结构化 feature 或文本模板，并建立新的数据集 schema 版本。

以上名称描述当前 exporter。历史数据集如果仍显示 `state_0…state_12` 或 `dx, dy, dz`，只代表
旧版 metadata 名称，不应据此推断当前字段顺序或动作存储语义。

### 6.2 目录结构

默认视频模式的目录为：

```text
<lerobot_dataset>/
├── data/
│   └── chunk-000/file-000.parquet
├── videos/
│   ├── observation.images.global/chunk-000/file-000.mp4
│   └── observation.images.wrist/chunk-000/file-000.mp4
└── meta/
    ├── info.json
    ├── stats.json
    ├── tasks.parquet
    ├── episodes/chunk-000/file-000.parquet
    └── welding_path_vla_export.json
```

这些 `file-xxx` 是按 LeRobot 大小上限切分的 shard，不保证一个文件只包含一个 episode：

- `data/*.parquet` 保存定长数值列和索引，不保存 Python 字典；`observation.state` 和 `action`
  分别是 Arrow `fixed_size_list<float>[13]` 与 `fixed_size_list<float>[9]`；
- `videos/.../*.mp4` 可以连续容纳多个 episode 的帧；
- `meta/episodes/*.parquet` 保存每个 episode 的长度、数据 shard、视频 shard 和起止时间，因此
  LeRobot 能从共享视频文件中定位正确片段；
- `meta/info.json` 保存 FPS、feature schema、总 episode / frame / task 数和路径模板；
- `meta/stats.json` 保存归一化需要的统计量；
- `meta/welding_path_vla_export.json` 是本项目增加的增量清单，记录 raw 源、已导出目录名和动作契约。

默认 `save_images=false`，因此没有逐帧图片目录。设为 `true` 时视觉 feature 改为图片，主要用于
调试，不建议用于大规模训练集。

### 6.3 为什么 Parquet 存 absolute，训练却叫 relative action

当前 LeRobot Parquet 有意保存世界系 absolute EE target。训练 preprocessor 取预测时刻的实际
TCP 作为共同锚点，把 future action chunk 转为：

\[
p^{rel}_{t,k}=R_t^\top(p^{target}_{t+k}-p_t),
\qquad
R^{rel}_{t,k}=R_t^\top R^{target}_{t+k}.
\]

块内所有未来目标共享同一个 `(p_t, R_t)`，所以这是 relative trajectory，不是每一步相对前一步
的 sequential delta。模型输出后，postprocessor 用同一次观测缓存的 TCP 锚点恢复世界系绝对
目标，再交给安全门和 IK。

这样设计的原因是：

- raw 和 Parquet 保留不依赖预测时刻的绝对事实；
- 不需要为每个 horizon 和 stride 复制一份动作数组；
- underlying absolute 行无需因动作块变化而重写；但更换 horizon 或 stride 时必须重算统计和
  manifest，当前增量契约要求使用新的目标数据集版本；
- 部署时相对轨迹对工件和机器人绝对位置变化更稳健。

一个容易忽略的细节是：Parquet 的 `action` 值为 absolute，但本项目会按训练期转换后的 relative
action 重新计算 `meta/stats.json` 中的 action 统计量。项目 processor 在归一化之前完成转换，
因此二者一致。绕过项目 processor 直接用 absolute action 配合该统计文件会产生错误结果。
`meta/episodes/*.parquet` 中由 LeRobot 在 `save_episode()` 时生成的单 episode action 统计仍对应
存储的 absolute 值；当前训练只使用重算后的全局 `meta/stats.json`，不要把两者混用。

`meta/welding_path_vla_export.json` 明确记录：

```text
type: relative_action
storage: absolute_ee_world
frame: prediction_tcp
rotation: rotation_6d_rows
horizon: policy.action_horizon
stride: policy.action_stride
```

增量转换会校验该契约；horizon、stride、FPS、视觉类型或 action schema 不一致时，不应写入同一
目标数据集。

## 7. 采集、检查和导出

先用单 worker 采一小批：

```bash
pixi run -e sim sim-collect \
  --config_path=configs/default.yaml \
  --collection.episodes=5 \
  --collection.workers=1
```

检查质量摘要并回放：

```bash
pixi run -e sim data-validate \
  --config_path=configs/default.yaml \
  --collection.dataset_root=datasets/weldpath_raw_v2

pixi run -e sim sim-replay \
  --episode=datasets/weldpath_raw_v2/episodes/episode_000000
```

确认画面、碰撞和失败原因合理后，再提高 `collection.workers`。每个 worker 都持有独立的 MuJoCo、
EGL 和 H.264 编码器；并行数受 GPU 渲染上下文、CPU 编码和内存共同限制。

把一个或多个 raw 数据集导出为统一 LeRobot Dataset：

```bash
pixi run -e data export-lerobot \
  --config_path=configs/smolvla.yaml \
  --dataset_glob='datasets/*_raw_v2' \
  --output=datasets/weldpath_lerobot_relative_v1 \
  --repo_id=USER/weldpath_relative_v1
```

默认转换双相机视频，使用 AV1、`yuv420p`、CRF 30 和 preset 12。多个 raw 源顺序写入同一个
LeRobot writer，双相机可在单 episode 内并行编码，从而避免并发修改 metadata 或过量占用内存。

## 8. 直接读取数据

读取 raw episode：

```python
from welding_path_vla.dataset.raw_schema import EpisodeReader

episode = EpisodeReader("datasets/weldpath_raw_v2/episodes/episode_000000")
print(episode.action_count, episode.state_count)
print(episode.metadata["quality"])
print(episode.trajectory["safe_command_position"].shape)
```

读取 LeRobot Dataset：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="USER/weldpath_relative_v1",
    root="datasets/weldpath_lerobot_relative_v1",
)
sample = dataset[0]
print(sample["observation.state"].shape)         # torch.Size([13])
print(sample["action"].shape)                    # torch.Size([9])
print(sample["observation.images.global"].shape) # LeRobot 解码后的图像张量
print(sample["task"])
```

这里直接读取到的单帧 `action` 仍是 absolute storage。训练 pipeline 会先由 LeRobot 按时间索引取
future chunk，再由项目 relative-action processor 转换；不要在数据文件上手工覆盖该列。

## 9. 修改采集器时必须保持的契约

新增工件、真机 recorder 或新的专家时，至少保持以下约束：

1. raw 层始终保留绝对状态、绝对目标和执行结果；
2. 每条 episode 严格满足 `state_count = action_count + 1`；
3. 图像、状态和动作共享同一个策略时钟，并明确记录单位和坐标系；
4. 任务变化写入结构化参数，指令只描述稳定语义；
5. 失败数据保留诊断原因，但只有明确有效的数据进入训练集；
6. LeRobot feature 名称、维数和顺序变化时新建数据集版本，不能静默追加；
7. 修改 relative action 定义时同步更新 processor、manifest 和 action statistics。

这组边界让数据可以重复评价、重新转换，并让 ACT、SmolVLA、π0、Trajectory-VLA 等策略使用同一
份训练事实进行公平对比。
