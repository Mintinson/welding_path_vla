# 仿真策略部署与闭环 Rollout

`deploy_simulation_policy.py` 将训练完成的 LeRobot checkpoint 放入项目的 robosuite / MuJoCo
焊接环境中执行闭环 rollout。它用于观察策略能否在自身预测造成的状态分布上完成任务，并同步
保存视频、逐步控制日志、诊断摘要和论文级轨迹指标。

该入口不是数据集离线评估，也不是几何专家数据采集：

| 流程 | 输入 | 执行者 | 主要输出 |
| --- | --- | --- | --- |
| `policy-evaluate` | LeRobot 留出数据 | 模型前向 | loss、动作 MAE |
| `sim-collect` | 随机仿真任务 | 几何专家 | 训练数据 |
| `policy-sim-deploy` | 模型 checkpoint | 训练后的策略 | 闭环轨迹、视频、成功与安全指标 |

## 1. 快速开始

优先使用已经组合好“基础场景 + 策略 checkpoint + 任务”的部署配置：

```bash
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla_l_joint.yaml \
  --deployment.episodes=5
```

临时更换 checkpoint：

```bash
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla_pipe_top.yaml \
  --policy.checkpoint=outputs/train/RUN/checkpoints/last/pretrained_model \
  --deployment.episodes=5
```

输出目录会自动生成为 `outputs/deploy/smolvla_pipe_top`。一次执行全部任务时，模型只加载一次：

```bash
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla.yaml \
  --deployment.run_all_tasks=true \
  --deployment.episodes=5
```

进入环境后也可以直接运行脚本：

```bash
pixi shell -e policy-sim
python scripts/deploy_simulation_policy.py \
  --config_path=configs/deploy/smolvla_pipe_top.yaml
```

服务器已经缓存完整模型、但无法访问 Hugging Face 时，可关闭远端探测：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/deploy_simulation_policy.py \
  --config_path=configs/deploy/smolvla_pipe_top.yaml
```

checkpoint 参数可以指向以下任一层级，程序会定位其中包含 `config.json` 的
`pretrained_model`：

```text
outputs/train/RUN/
outputs/train/RUN/checkpoints/last/
outputs/train/RUN/checkpoints/last/pretrained_model/
```

checkpoint 必须同时保存模型配置、权重、preprocessor 和 postprocessor。当前项目要求其中
包含 `relative_action` processor；旧的 delta-action checkpoint 不能直接部署。

## 2. 配置如何组合

通用模型部署配置先组合基础场景与策略，单任务入口只需在其上追加任务：

```yaml
includes:
  - smolvla.yaml              # 已包含基础场景、策略类型及 checkpoint
  - ../tasks/pipe_top.yaml    # 工件、焊缝、指令和稳定 task_id
```

合并优先级为“较早 include < 较晚 include < 当前文件 < 命令行覆盖”。常用部署参数如下：

| 参数 | 含义 |
| --- | --- |
| `policy.family` | 从公共注册表选择 ACT、SmolVLA、Trajectory-VLA、Qwen 或 π 系列 pipeline |
| `policy.checkpoint` | 待加载的 LeRobot checkpoint；部署时必须提供 |
| `policy.device` | 首选推理设备，例如 `cuda`；不可用时自动选择可用设备 |
| `deployment.output_root` | 自动命名的公共输出根目录，默认 `outputs/deploy` |
| `deployment.auto_log_dir` | 是否自动使用 `{policy.family}_{task.task_id}` 子目录 |
| `deployment.log_dir` | 关闭自动命名后的兼容完整路径 |
| `deployment.run_all_tasks` | 是否遍历 `task_config_dir` 下的全部任务 |
| `deployment.task_config_dir` | 批量部署使用的任务 YAML 目录 |
| `deployment.episodes` | 连续执行的 episode 数量 |
| `deployment.max_steps` | 每条 episode 的最大策略步数 |
| `deployment.seed` | 第 0 条 episode 的随机种子 |
| `deployment.record_video` | 是否录制全局与腕部相机视频 |
| `deployment.completion_progress_min` | 自然退出所需的最小焊缝进度 |
| `deployment.completion_distance_m` | 自然退出时允许的最大焊缝距离 |

`deployment.dry_run` 是保留字段，当前仿真部署入口不会读取它；即使该值为 `true` 也会实际执行
rollout。若只想检查配置，应使用 `policy-config`，不要把 `dry_run` 当作部署预览开关。

模型结构、归一化统计、`chunk_size` 和 `n_action_steps` 从 checkpoint 自身配置恢复。外层 YAML
中的 `policy.action_horizon`、`policy.action_steps` 和 `policy.parameters` 主要用于训练构造，不能
在部署时覆盖一个已经训练好的模型结构。部署 YAML 主要决定策略家族、checkpoint、设备、仿真
任务和 rollout 参数。

默认输出路径为：

```text
{deployment.output_root}/{policy.family}_{task.task_id}
```

例如 SmolVLA 的管口任务自动写入 `outputs/deploy/smolvla_pipe_top`，不再需要在每个任务配置中
同步维护目录名。若必须使用完整自定义路径，应同时设置
`--deployment.auto_log_dir=false --deployment.log_dir=PATH`。

## 3. 完整调用链

```text
scripts/deploy_simulation_policy.py
        │ 组合 YAML，设置 MUJOCO_GL
        ▼
policies/deployment.py
        │ 检查 checkpoint，按 policy.family 查找 pipeline
        ▼
LeRobotRuntime.from_pretrained()
        │ 加载模型、processor、归一化统计和可选 PEFT adapter
        ▼
deploy_episodes()
        │ 为每个 episode 创建独立 WeldingEnv 和输出目录
        ▼
rollout_episode()
        │ 任务采样 → 观测 → 策略动作块 → 安全门 → IK → MuJoCo step
        ▼
rollout.npz + 双相机 MP4 + episode summary.json
        │
        └── evaluate_trace() → PCR、CTE、姿态、速度、安全和 ESR 条件
```

入口只负责解析配置和输出最终 JSON；所有策略共享同一个 `LeRobotRuntime` 和
`simulation_rollout.py`，因此不同模型的闭环控制、安全门和评价口径保持一致。

## 4. Episode 初始化

第 $e$ 条 episode 使用确定性种子

$$
\mathrm{seed}_e=\mathrm{deployment.seed}+e.
$$

初始化依次执行：

1. 创建 `WeldingEnv`，加载 Elfin5-Pro、焊枪、桌面、双相机和任务工件；
2. 在配置范围内随机化工件位置和偏航角，并重采样不可达或碰撞的 staging pose；
3. 在 staging 构型附近采样无碰撞的初始关节位置；
4. 对初始 TCP 施加配置允许的小幅扰动；
5. 根据当前工件位姿建立 `SeamPath` 和 `ExpertTrajectory` 参考；
6. 创建 `SafetyMonitor`，并清空模型、processor 与动作队列的跨 episode 状态。

这里的 `ExpertTrajectory` 只提供参考焊缝位姿和自动评估基准，不会向机器人发送动作。闭环中的
每一个动作都来自待评估策略。

当前部署流程会随机化工件位姿、初始关节和初始 TCP，但不会调用采集流程中的
`sample_task_config()`；因此速度、圆弧范围、姿态角等任务参数使用任务 YAML 中的标称值。
这使同一部署配置的任务定义稳定，同时仍保留场景和初始状态变化。

## 5. 策略看到什么

每个策略周期构造一条统一 `Observation`：

| 输入 | 内容 |
| --- | --- |
| `observation.state` | 13D：6 个关节角 + TCP 世界位置 3D + TCP `wxyz` 四元数 4D |
| `observation.images.global` | 全局 RGB 图像 |
| `observation.images.wrist` | 腕部 RGB 图像 |
| `task` | 当前任务的英文指令；ACT 等无语言策略不会接收该字段 |

图像从 `uint8 HWC` 转为 `[0,1]` 范围的 `float32 CHW`，再交给 checkpoint 保存的
preprocessor。是否由 processor 添加 batch 维、图像 resize 和语言 tokenization 均由策略
规格和 checkpoint 决定。

部署使用动作块，而不是每一步重新运行大模型：

1. 动作队列为空时，preprocessor 缓存当前 TCP 作为本次预测的共享锚点；
2. `predict_action_chunk()` 输出归一化的 relative-action chunk；
3. postprocessor 先反归一化，再把整段相对动作恢复为世界系绝对末端目标；
4. 最多将 checkpoint 配置中的 `n_action_steps` 个目标加入队列；
5. 后续策略周期依次弹出目标，队列耗尽后再读取最新观测并重新预测。

因此，同一动作块中的目标共享“预测该块时”的 TCP 锚点，但每次新预测都会以最新 TCP 重新
闭环。保存到 `rollout.npz` 的 `action` 已经是 postprocessor 输出的 9D 世界系绝对目标：

```text
[x, y, z, rotation_6d(6)]
```

随后 `absolute_ee_action_to_pose()` 将其解码为仿真环境使用的 7D 位姿：

```text
[x, y, z, qw, qx, qy, qz]
```

## 6. 动作执行与多速率控制

默认时序为：

```text
策略 / RGB：30 Hz
关节控制：120 Hz
MuJoCo 物理：600 Hz
```

每个 30 Hz 策略目标进入环境前依次经过：

1. **动作解码**：检查 6D rotation 能否转换为有效姿态；
2. **平移增量限制**：单步最大平移量为
   `safety.tcp_speed_limit_m_s / timing.policy_hz`；
3. **IK**：把绝对 TCP 位姿变为六关节目标；
4. **安全门**：检查当前状态和关节命令是否越过带余量的关节范围；
5. **控制插值**：一个策略周期内以 120 Hz 更新中间 TCP/关节目标；
6. **物理步进**：MuJoCo 以 600 Hz 执行动力学和接触计算。

平移目标过大时不会立即终止。程序会把目标裁剪到单步上限，设置
`action_increment=true`，在 `step_error` 中记录裁剪前后数值，然后继续闭环。这能避免模型
偶发的大动作直接破坏仿真，但该事件仍属于论文评估中的安全违规。

IK 残差超过 `10 × robot.ik_tolerance`、关节命令越界、动作无法解码或出现不允许的 MuJoCo
接触时，episode 会立即停止。环境会汇总整个策略周期内出现过的碰撞对，而不是只检查周期末
最后一个物理帧。

## 7. 终止原因与“成功”的区别

rollout 的自然完成条件是

$$
\mathrm{progress}\geq\mathrm{completion\_progress\_min}
\quad\land\quad
\mathrm{seam\_distance}\leq\mathrm{completion\_distance\_m}.
$$

`termination_reason` 可能为：

| 值 | 含义 |
| --- | --- |
| `completed` | 到达配置定义的焊缝末端区域，自然退出 |
| `timeout` | 执行到 `deployment.max_steps` 仍未自然完成 |
| `collision` | 当前策略周期出现不允许的接触 |
| `action_violation` | 9D 动作无法转换为有效位姿等动作错误 |
| `safety_violation: ...` | IK 残差过大、关节命令越界或其他安全门异常 |

必须区分三个字段：

- `completed`：是否以 `termination_reason == completed` 自然退出；
- `diagnostics.termination.natural`：同一自然退出状态的诊断表示；
- `evaluation.success`：是否同时满足完成度、方向、位置、姿态、速度、安全和正常终止条件。

因此视频看起来顺利且 `completed=true`，也可能因为 CTE、速度误差、动作裁剪、关节速度或
加速度越界而得到 `evaluation.success=false`。论文主表应使用 `evaluation.success`，自然退出
只用于判断策略是否走到了任务末端。完整指标定义见[评估规范](evaluation.md)。

## 8. 自动评价何时触发

每步执行后，程序把实际 TCP 投影到当前有向焊缝，并记录：

```text
seam_progress
seam_distance_m
track_mask = seam_distance_m <= evaluation.tracking_band_m
```

episode 结束后，只有 `track_mask` 至少包含 2 帧时才调用 `evaluate_trace()`；否则
`evaluation` 为 `null`。自动评价使用几何专家的 TRACK 段作为参考焊缝，并在仿真中将指令
选择标记为正确，因为工件、焊缝和方向由当前配置明确给出。

当前 rollout 没有把专家时间序列传入 jerk 对比，因此 `jerk_ratio` 通常为 `null`；默认
`require_smoothness_for_success=false` 时不影响 ESR。PCR、CTE、姿态、速度和安全仍会正常
计算。

## 9. 输出目录与字段

单任务默认目录由模型和任务自动生成：

```text
outputs/deploy/smolvla_pipe_top/
├── summary.json                    # 所有 episode 的报告列表
├── episode_000000/
│   ├── config.json                 # 本次执行的完整合并配置
│   ├── rollout.npz                 # 逐策略步轨迹和诊断原始量
│   ├── global.mp4                  # 全局相机 H.264 视频
│   ├── wrist.mp4                   # 腕部相机 H.264 视频
│   └── summary.json                # 本 episode 的结构化报告
└── episode_000001/
    └── ...
```

每次调用都从 `episode_000000` 开始，且 episode 目录使用 `exist_ok=false`。因此不能直接复用
含有旧 episode 的自动目录；新一轮实验可覆盖 `deployment.output_root`，或先把旧结果移动到
归档位置。
若中途异常退出，已经完成的 episode 会保留，但根目录 `summary.json` 可能尚未生成。

批量模式在同一个 `output_root` 中建立六个模型—任务子目录，并额外写入跨任务摘要：

```text
outputs/deploy/
├── smolvla_l_joint/
├── smolvla_pipe_bottom/
├── smolvla_pipe_top/
├── smolvla_curve_plate/
├── smolvla_trihedral_horizontal/
├── smolvla_trihedral_vertical/
└── smolvla_all_tasks_summary.json
```

每条 report 新增 `task` 字段，根摘要可以直接按任务分组。各任务仍拥有自己的 `summary.json`、
episode 目录和视频。

### 9.1 `rollout.npz`

同一数组索引描述一次完整的 $state_t\rightarrow action_t\rightarrow state_{t+1}$ 转移：

| 类别 | 主要字段 |
| --- | --- |
| 策略前观测 | `observation_tcp_position`、`observation_tcp_quaternion_wxyz`、`observation_joint_position` |
| 模型与控制命令 | `action`、`command_tcp_position`、`command_tcp_quaternion_wxyz`、`joint_command` |
| 执行后状态 | `tcp_position`、`tcp_quaternion_wxyz`、`joint_position`、`joint_velocity` |
| 跟踪 | `seam_progress`、`seam_distance_m`、`track_mask` |
| 控制诊断 | `ik_residual_m`、`step_error` |
| 安全 | `collision`、`collision_pairs`、`joint_limit`、`joint_velocity_limit`、`joint_acceleration`、`action_increment` |

`command_tcp_position` 是经过单步平移限制后送给 IK 的目标；`action` 是限制前的模型绝对目标；
`tcp_position` 是执行一个策略周期后的真实反馈。三者同时保存是为了区分“模型预测有误”、
“安全门修改了命令”和“仿真执行没有跟上命令”。

### 9.2 `summary.json`

episode 报告包含：

- 基本信息：episode、seed、steps、completed、termination reason；
- 文件位置：rollout 轨迹和视频路径；
- `diagnostics.tracking`：最近/最终焊缝距离、最大/最终进度、track 帧数；
- `diagnostics.control`：最大命令增量、最大 IK 残差、最大 TCP 速度和错误文本；
- `diagnostics.safety`：各类违规帧数及具体碰撞对；
- `evaluation`：论文级单 episode 指标，未进入跟踪带时为 `null`。

视频帧率等于 `timing.policy_hz`，默认使用 H.264、CRF 23、`veryfast`。视频包含初始状态和
最后一次动作后的终止状态，因此帧数为 `steps + 1`；`frame_0` 是初始观测，`frame_n` 是
执行第 $n$ 个动作后的状态。

## 10. 切换策略与任务

现有 SmolVLA、Trajectory-VLA、π0 和 π0.5 均提供模块化部署入口。任务后缀包括：

```text
l_joint
pipe_bottom
pipe_top
curve_plate
trihedral_horizontal
trihedral_vertical
```

例如：

```bash
# 同一个 SmolVLA checkpoint 切换任务
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla_curve_plate.yaml

# 同一个任务切换策略
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/trajectory_vla_curve_plate.yaml
```

不想逐个切换任务时，直接使用模型的通用部署配置并开启批量模式：

```bash
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/trajectory_vla.yaml \
  --deployment.run_all_tasks=true
```

程序按文件名排序读取 `configs/tasks/*.yaml`，保留相同的模型、checkpoint、episode 数和安全
配置，同时应用每个任务自己的工件、焊缝、随机化范围及 `max_steps`。任务通过 YAML 中稳定的
`task.task_id` 命名输出目录。

若某个新策略尚无对应部署文件，只需增加一个薄组合配置，不要复制基础配置：

```yaml
includes:
  - ../base.yaml
  - ../policies/traj_vla_qwen.yaml
  - ../tasks/curve_plate.yaml

policy:
  checkpoint: outputs/train/RUN/checkpoints/last/pretrained_model

deployment:
  episodes: 5
```

## 11. 常见问题定位

### 输出目录已存在

若出现 `FileExistsError`，说明目标 `episode_XXXXXX` 已存在。为新实验指定新的
`--deployment.output_root`，避免覆盖可复查的旧结果。若已关闭自动命名，则改用新的
`--deployment.log_dir`。

### checkpoint 加载失败

确认路径下能解析到 `config.json`、权重及两个 processor 文件，并确认 `policy.family` 与
checkpoint 的配置类型一致。报错提示缺少 `relative_action processor` 时，该 checkpoint
使用旧动作语义，需要重新训练或执行明确的 checkpoint 迁移，不能只复制权重文件。

### 视频正常但 summary 显示失败

依次检查：

1. `termination_reason` 是否为 `completed`；
2. `diagnostics.safety` 是否存在动作裁剪、速度、加速度或接触事件；
3. `evaluation.conditions` 中具体失败的是 completion、position、orientation、speed 还是 safety；
4. `rollout.npz` 中模型 `action`、安全门 `command_tcp_position` 和真实 `tcp_position` 的差异。

视频只能显示肉眼可见的运动，无法可靠判断毫米级 CTE、瞬时接触或单周期速度/加速度越界。

### `evaluation` 为 `null`

这表示 TCP 进入 `evaluation.tracking_band_m` 的帧数少于 2。通常说明策略没有靠近目标焊缝，
应先检查初始画面、动作尺度、任务配置和 `diagnostics.tracking.closest_seam_distance_m`，而不是
直接放宽论文成功阈值。

### rollout 总是 timeout

比较 `deployment.max_steps / timing.policy_hz` 与任务所需实际时长，并查看最终进度。如果最大
进度接近 1 但最终距离过大，说明策略越过末端或没有进入完成距离；如果进度长期不增长，应检查
相对动作方向、processor、相机输入和任务指令。

### 服务器无显示或 OpenGL 初始化失败

仿真部署使用离屏渲染，默认应设置 `camera.offscreen_backend=egl`。脚本会在导入仿真环境前
设置 `MUJOCO_GL`；服务器需提供可用的 NVIDIA 驱动和 EGL。仅 CPU 环境可以尝试 OSMesa，
但大型策略推理和双相机渲染会很慢。

## 12. 推荐复现实验流程

1. 用独立 `output_root` 运行 1 条 episode，确认 checkpoint、processor、相机和动作语义正确；
2. 检查双相机视频以及 episode `summary.json`；
3. 检查 `rollout.npz` 中动作、命令、反馈和安全字段；
4. 固定策略、任务、随机种子、阈值和 episode 数执行正式评估；
5. 用根目录 `summary.json` 汇总结果，并保留每条 episode 的配置和轨迹以便复查。

不同策略对比时，应保持任务配置、seed、episode 数、相机、`relative_action` 语义、自然退出
条件和论文评估阈值一致。模型特有的图像尺寸、tokenizer 和去噪步数由各自 checkpoint 保持。
