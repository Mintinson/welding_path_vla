# 真机被动数据采集设计

状态：已实现并通过模拟设备测试，尚未进行真实设备验收。以下源码结论核对至 2026-10-19 本仓库
`.pixi/envs/train/lib/python3.12/site-packages/lerobot/`，不假设它与在线 main 完全相同。

## 本机操作流程

本机使用统一的 Pixi 默认环境，已声明本地 CPS SDK、LeRobot 自动发现插件和训练部署依赖。
首次准备环境时执行：

```bash
pixi install
```

2026-10-19 已使用 `192.168.56.101` 虚拟控制器与两路全黑占位相机完成实际 SDK 读取、
录制和项目 Trajectory-VLA 的三步训练；见[验收报告](validation/vm-capture-2026-10-19.md)。
这证明数据与训练链路可用，但不代表物理实机、真实视觉或严格 30 Hz 验收通过。

该安装步骤可能下载尚未缓存的依赖。日常使用 `pixi run` 执行任务，无需切换用途环境。
历史验收报告中的 `dev`、`real` 和 `train` 是迁移前的环境名称，复现时使用当前默认环境。

先编辑 `configs/real_record.yaml`：填写机器人 IP、选择两路 USB 相机，并设定数据位置和任务文本。
正式采集仅须按真实任务填写 `task_conditions.direction`；`parameters` 和 `extra` 均可省略。
`require_task_labels: true` 会在接机前拒绝未填写方向的正式采集。
相机使用系统已有的稳定设备路径，确认 global 和 wrist 没有互换，输出颜色为 RGB。
这只修改 YAML 中的设备选择，不修改系统权限、不添加 udev 规则。
探测输出的 `Camera #1` 只是结果列表序号，不代表设备编号 1；应使用其 `Id` 或稳定路径。
2026-10-21 用户将 global 改为 RealSense D435i，wrist 仍为 LRCP U3-JX02。
RealSense 使用 `/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6:1.3-video-index0`，
固定当前 USB 口的彩色接口 1.3；本机 RealSense 的 by-id 名称缺少唯一序列号，
不同接口的 video-index0 别名会重名，不适合作为彩色接口的可靠选择。
by-path 不随 video 编号变化，但移动到其他 USB 口时要更新。
LRCP 使用 `/dev/v4l/by-id/usb-LRCP_ZWAK_LRCP_U3-JX02_SN0001-video-index0`，按设备身份识别。
两者均使用 V4L2（backend 200）、640×480、30 FPS；RealSense 使用 YUYV，LRCP 使用 MJPG。
若需 RealSense 跨 USB 口按序列号绑定，可用原生 `type: intelrealsense` 与
`serial_number_or_name`，并为默认环境另行声明 `pyrealsense2`；当前配置使用 OpenCV 后端。
`fourcc` 是相机采集传输格式，保存到数据集的视频仍按 `dataset.rgb_encoder` 编码为 AV1。
仅使用 `dataset.root` 指定数据目录；录制封装在连接前自动绑定适配器的日志目录，
将采集契约、标定和时序日志保存到该目录的 `meta/` 下，不再配置 `robot.capture_root`。
省略 `dataset.root` 时沿用原生 `$HF_LEROBOT_HOME/<repo_id>` 默认目录，包含原生创建时追加的时间戳。
如果修改 FPS，须同时修改
`robot.fps`、`dataset.fps` 和两个相机的 FPS。
正式采集的 `dataset.rgb_encoder` 使用与参考训练集相同的 AV1、yuv420p、GOP 2、
CRF 30、preset 12；参考集校验同时核对保存的视频编码信息。旧 H.264 联调数据保留原样。

录制前可先运行独立连接检查。默认连接机器人与两路相机，连接和相机预热完成后观测
5 秒，每 0.5 秒读取并打印一次；不保存数据。机器人输出与采集一致的关节角（弧度）、
world 系 TCP 位置（米）和 wxyz 四元数；相机输出帧尺寸与平均通道值，并显示预览画面。
移动机器人或改变相机视野即可观察数值和画面是否更新。`robot` 与 `cameras` 可独立开关：

```bash
pixi run check-real-devices
# 只检查机器人；临时 IP 覆盖也可省略，默认读取 real_record.yaml。
pixi run check-real-devices --cameras=false --host=192.168.10.10
# 只检查相机；无图形界面时附加 --preview=false，仅打印帧信息。
pixi run check-real-devices --robot=false
```

默认环境已声明本地 CPS SDK，无需设置 `PYTHONPATH`。

### 使用官方 teleoperate 循环持续预览

确认相机视角时，可以直接复用当前录制 YAML：

```bash
pixi run preview-real --record_config=configs/real_record.yaml
# 预览 30 秒后自动断开：
pixi run preview-real --record_config=configs/real_record.yaml --duration_s=30
```

本机 2026-10-26 的验证使用当时已有的 `dev` 环境；当前统一默认环境已包含这些依赖。

不指定 `duration_s` 时持续显示，按 Ctrl+C 退出。该入口强制开启显示，因此无需修改录制配置的
`display_data: false`；频率使用 `dataset.fps`，设备和可视化后端参数沿用同一 YAML。
预览仍须连接机械臂读取状态，但不发送任何 SDK 运动指令，不创建 episode、不编码视频、不写采集
日志，也不触碰已有数据集、标定记录或 relative-action manifest。
若只需相机且不连接机械臂，使用上面的 `check-real-devices --robot=false`。

本机 LeRobot 0.6.0 的 `lerobot-teleoperate` 并不是纯观察程序：每周期依次获取观测、获取
teleop action、处理 action、调用 `robot.send_action()`，然后可视化。原生
`TeleoperateConfig` 不接受 `dataset`、`task_conditions` 等录制字段，默认 60 FPS；其入口
先连接 teleoperator 再连接 robot，也与本项目被动标签提供器需要先注册 robot 的顺序不同。
因此不能把完整 `real_record.yaml` 直接传给原生命令。

`preview-real` 只提取 `TeleoperateConfig` 支持的字段，先连接只读 robot 再连接 passive
teleoperator，并初始化相邻观测配对，随后调用官方 `teleop_loop()`，没有复制该循环。
此入口只允许 `elfin_welding + elfin_passive`，其 `send_action()` 返回标签且不调用 SDK。
退出、异常或 Ctrl+C 时关闭可视化、相机和 SDK 连接；不会停止其他程序正在驱动的运动。
录制入口仍会自动绑定 `dataset.root` 并保存完整额外日志，两种用途相互独立。

没有物理相机时可在测试配置中将相机设为 `type: black`，显式使用全黑 RGB 占位图像。
原生相机工厂会创建 `BlackCamera`，不会打开笔记本摄像头；不得将这类数据当作正式视觉示范。
现成配置为 `configs/real_record_vm_smoke.yaml`，对应训练配置为 `configs/real_training_vm_smoke.yaml`。

可用 `--record_config=其他录制配置.yaml` 切换设备配置，使用 `--duration_s=10 --interval_s=0.5`
调整时长与间隔。Ctrl+C 或预览窗口中的 Esc 可提前结束，退出时关闭连接与窗口。
轮询使用主机软件计时，设备查询阻塞可能导致实际间隔变长；此检查不代表 30 Hz 采集验收。

```bash
# 先在 YAML 中填写真实方向，其余任务条件按需提供，再调用原生录制的薄封装。
pixi run record-real

# 保存后整理：horizon/stride 必须与将来训练使用的策略一致。
pixi run prepare-real-data --dataset=datasets/real/elfin_demo --policy.action_horizon=30
pixi run policy-data-check --config_path=configs/real_training.yaml

# 可选：逐字段对照实际服务器训练子集的 dtype、shape、names 和 FPS。
pixi run prepare-real-data \
  --dataset=datasets/real/elfin_demo --reference_dataset=datasets/weldpath_lerobot_relative_small
```

运行前由外部程序或示教器准备机械臂运动；命令连接完成后自动录制完整路径。
右箭头结束当前录制阶段，左箭头丢弃当前条并重录，Esc 正常结束整次采集且保存当前条。
回合间可手动恢复场景，右箭头可提前结束重置等待。

恢复采集使用同一数据目录和 `--resume=true`；录制后重新运行整理命令：

```bash
pixi run record-real --resume=true --task_conditions.direction=reverse
```

默认关闭上传；录制与整理均只保存在本地。需要切换工具 TCP/UCS 时，先结束该采集会话再重新启动。
`scene_config` 中的基座到 project world 外参仍须符合实际安装；工具 TCP 标定不能替代这组外参。

### 数据与时序记录

- 最终训练特征沿用仿真：双相机、13D 状态、9D 世界系绝对 TCP 动作；仅任务文本时不生成虚构参数或阶段标签。
- `meta/real_capture_contract.json` 标记实测后继动作来源；恢复录制会使旧训练清单失效，必须重新整理。
- `meta/real_sessions/*.jsonl` 记录每次连接时的有效 TCP/UCS、world 外参和采集时序。sample 记录包含 source 状态查询起止、主线程取图完成和 target 状态查询起止时间，以及相应状态/动作。
- 会话日志同时包含重置和作废周期，不能直接按行与最终 Parquet 对齐；后续同步研究还需完善与已保存样本索引的关联。所有这些时间都是主机时序，不是曝光时间。
- 整理只扫描数值数据并更新动作统计与清单，不修改 Parquet、视频或图像，也不触碰机械臂。

### 标签生成与速率

每段原生录制或重置循环开始前预采一个观测，并等待一次目标采样间隔。
循环内每周期只调用一次 `HRIF_ReadActPos` 并取双相机快照；返回上一周期的观测，
以本次状态作为其后继实测动作。后续节拍由原生 `record_loop` 控制，不再逐帧额外等待。
相邻样本复用同一次实测状态，首帧不补零，最后一帧也有真实后继目标。
图像在取帧时复制，避免缓存跨周期时被相机复用的缓冲区覆盖。
每次循环退出（包括异常）均清空未配对观测，录制、重置、重录和保存间不携带旧缓存。
`record-real` 只在原生 `record_loop` 调用前后增加这一配对上下文，不复制循环或修改安装文件。
每次状态查询直接读取 Base 系实测 TCP 位姿，不额外查询正运动学。
实际速率仍需结合原生慢周期提示和会话日志验收，配置的 30 FPS 本身不是实测速率。

2026-10-20 修改前真机两个回合分别约 26.74/26.57 Hz，
见[慢周期告警诊断](validation/real-capture-warnings-2026-10-20.md)。
连续配对改动及验证见[采样调度修正](validation/consecutive-capture-2026-10-20.md)。
修改后实机两段各 120 帧，平均约 29.81/29.89 Hz，慢周期及编码队列满告警均为 0；
仍有少量调度抖动，本次机械臂基本静止，尚不构成动态轨迹质量验收。
开启 `display_data` 时，每回合首帧仍可能因流式编码器惰性初始化占用 GIL 而超时。
已实测原生 `dataset.streaming_encoding=false` 可将编码移到保存阶段，保留 AV1 并消除本次首帧告警，
代价是临时图片及保存等待；详见[显示启动告警实测](validation/display-startup-2026-10-20.md)。

### Rerun 预览缓存上限

`Exceeded gRPC proxy server memory limit (1.0 GiB)` 表示预览代理达到历史日志缓存上限，
开始丢弃旧预览消息；后连接的客户端无法获得完整历史。
`Dropping old static log messages` 表示部分旧静态消息也被裁剪，本机 LeRobot 的图像日志
使用 `static=True`，因此这一提示也可能出现。它们不表示磁盘上的 Parquet/MP4 被删除，
是否存在采集丢帧仍需独立查看采集时序、编码队列告警和文件帧数。

本机 Rerun 的 gRPC 代理默认 `server_memory_limit=1GiB`，
与 LeRobot 通过 `LEROBOT_RERUN_MEMORY_LIMIT` 设置的 viewer 内存限制是两个独立限制。
对实时预览而言，这是限制内存占用的正常裁剪行为；若需要完整历史回放，不能依赖有界预览缓存。
原生 `display_compressed_images=true` 可压缩发送到预览的图像，降低传输和缓存开销，
但会增加 CPU 工作，也不能保证长时间运行后永不触顶；不改变保存的数据集视频编码。
行为与配置见 [Rerun CLI 官方说明](https://github.com/rerun-io/rerun/blob/main/docs/content/reference/cli.md)。

测试命令：

```bash
pixi run pytest -q tests/test_real_capture.py
```

测试使用模拟 CPS 与双 RGB 图像，验证单位/坐标转换、工具参数变化、只读错误路径、原生回合缓冲、重录、恢复和训练入口读取；不代表已完成实机速率或相机验收。

### 本次验证记录

2026-10-19 条件标签扩展：已有 Pixi `dev` 环境中 **64 项相关测试通过**（未运行 5 项既有
VLM 默认配置用例），包括调用原生 `record()` 保存/恢复不同方向回合、完整四项参数、
自定义数值/布尔/文本字段读回、训练提示生成、字段不兼容时接机前拒绝，以及公开 CLI 覆盖。
Ruff、相关文件 BasedPyright 与锁文件检查通过；未连接真实设备。
仅为原生恢复校验通过 Pixi 补装 `deepdiff`/`orderly-set` 两个纯 Python 包，未新建环境。

此前 2026-10-18 的基础采集验证记录如下：

在已有 Pixi 环境中完成以下验证，未创建或安装新的 `real` 环境：

- 新采集测试、相关几何/动作测试及整理脚本帮助入口共 **16 项通过**；包含整理后用 LeRobot 重新打开并读取状态、动作和双相机 Tensor。
- 变更文件的 Ruff 和 BasedPyright 检查通过，`git diff --check` 通过。
- 扩展配置保留原生 `RecordConfig` 字段并增加 `task_conditions`，官方设备工厂成功创建 Robot/Teleoperator，未连接硬件。
- Pixi 锁文件已更新为本地 SDK/插件依赖；检查时锁文件保持最新。

扩展检查为 33 项通过、5 项失败。失败点如下，未在本次修改训练配置或图形环境来处理：

| 范围 | 当前失败原因 |
|---|---|
| `test_interfaces.py` 的训练命令用例（1 项） | `dev` 环境缺少 `accelerate` |
| `test_welding_prompt.py` 的 Qwen 配置用例（1 项） | 用例要求五个提示参数，现有 Qwen YAML 只启用方向与速度 |
| raw 重渲染用例（2 项） | 当前 EGL 无法初始化无头渲染上下文 |
| LeRobot 重渲染用例（1 项） | 旧测试 fixture 未提供导出器要求的 `phase` 数组 |

真实设备连接、实际 TCP/安装坐标核对、30 Hz 持续采集和相机图像质量尚待接机验证。

## 已确认的需求

- 整个项目全程不开弧，研究焊枪轨迹的学习与复现；真实焊接不属于后续发展目标。
- Elfin5-Pro 通过以太网连接采集电脑，全局和腕部两个相机通过 USB 连接。
- 运动来自外部程序、示教器或人工拖动；采集端只读取，不改变机械臂运动和控制模式。
- 使用 `lerobot-record` 直接保存 LeRobot 数据，复用其 episode 管理。
- 动作标签采用下一采样时刻的实测 TCP，详见 [ADR 0007](adr/0007-passive-executed-action-targets.md)。
- 没有预设速度、焊枪角度等工艺参数的示范允许只提供任务文本，不强制填写未知参数。
- 真机工具标定以控制器当前生效配置为准，连接后通过 API 获取，不要求人工确认或沿用仿真中的固定数值。
- 一次采集会话内保持 TCP/UCS 不变；需要切换时结束当前采集命令，切换后重新启动录制。
- 首版接受软件时间对齐，目标沿用仿真的 30 Hz 和双路 640×480；更精确的传感器同步列为后续关注事项。
- 每条 episode 记录完整路径，覆盖接近焊缝、沿缝运动和退出，不能只截取沿缝段。
- 接受录制后的轻量整理步骤，用于动作时序校验、动作契约清单和 relative action 统计，使数据兼容当前训练入口。

不开弧是项目长期边界，并非仅限首版采集。文档中的焊接任务和参数指沿焊缝的运动任务及其速度、姿态要求；不引入焊机控制、焊接电流电压或实际焊缝成形质量标签。

### 必填方向与可选任务标签

参数仅在明确给定时作为任务条件记录；未知参数不使用仿真默认值或零值冒充真实标签。
2026-10-20 已对照 `datasets/weldpath_lerobot_relative_small`：正式格式仅要求方向标签，
`record-real` 和 `prepare-real-data` 默认启用 `require_task_labels: true`，
`configs/real_training.yaml` 默认仅启用方向提示。工艺参数及自定义字段均为可选；
即使参考数据集含这些字段，也不要求真机补齐。已提供字段仍按相同格式校验。
阶段标签按用户要求暂不提供，
`training.phase_sampling.enabled` 必须保持 false。
字段和 Parquet 类型的逐项结论见[服务器训练子集对照](validation/real-schema-2026-10-20.md)。

仅联调时允许显式设置 `--require_task_labels=false` 录制或整理没有方向标签的数据，
此时训练的 `policy.welding_prompt_fields` 应设为 `[]`。整理不会补造缺失列。
`dataset.single_task` 保存基本任务文本，`task_conditions` 保存用户在启动前明确填写的附加标签。
这些标签不会控制机械臂，也不会根据实际运动自动推断或验证；由用户保证示范与标签一致。

```yaml
require_task_labels: true  # 只有方向也满足正式采集要求。
task_conditions:
  direction: forward
  parameters: {}  # 不知道四项参数时保持为空。
  extra:
    fixture: jig_a
    trial: 1
    dry_run: true
    gap_mm: 0.5
```

| 配置 | 保存字段 | 含义 |
|---|---|---|
| `direction` | `task.direction`，int64，形状 `[1]` | 必填，`forward=0`、`reverse=1`；仅联调关闭检查后允许 null |
| `parameters` | `task.parameters`，float32，形状 `[4]` | 仿真已有四项参数，空字典时省略 |
| `extra.<名称>` | `task.extra.<名称>` | 自定义整数、浮点数、布尔或文本标量；数值形状为 `[1]` |

`parameters` 非空时必须完整指定以下四项，顺序与仿真相同，不对缺失项补零。
若只知道某个参数，可以先在 `extra` 中记录；该字段不会自动进入现有参数提示构造器。

```yaml
task_conditions:
  direction: reverse
  parameters:
    welding_speed_mps: 0.02
    work_angle_deg: 45.0
    travel_angle_deg: 10.0
    tool_roll_deg: -20.0
  extra: {}
```

命令行可覆盖方向和整个自定义参数字典：

```bash
pixi run record-real \
  --task_conditions.direction=forward \
  --task_conditions.parameters='{welding_speed_mps: 0.02, work_angle_deg: 45.0, travel_angle_deg: 10.0, tool_roll_deg: -20.0}' \
  --task_conditions.extra='{fixture: jig_a, trial: 1, dry_run: true, gap_mm: 0.5}'
```

上面的数值只是配置用法示例，执行前须替换为实际已知参数。

一次运行内这些参数固定，覆盖该运行保存的全部 episode。切换参数后可用 `--resume=true`
继续录制新的 episode；同一数据集必须保持字段集合、类型和形状一致。
例如已有方向列的数据可以追加 reverse 回合，但不能对已有纯文本数据直接追加方向列，
也不能在恢复时新增/删除自定义字段或把整数改为浮点数；原生兼容性检查会在接机前拒绝。

仅包含方向的正式数据可直接使用默认训练配置检查：

```bash
pixi run policy-data-check \
  --config_path=configs/real_training.yaml
```

若数据已包含四项参数，训练时可按实验需要扩展 `welding_prompt_fields`，
包括 `direction`、`welding_speed`、`work_angle`、`travel_angle`、`tool_roll`。
自定义 `extra` 字段会随数据保存和读回，但不会自动添加到模型提示；需要解释时可将其语义
写入 `dataset.single_task`，或以后单独扩展提示构造器。

本机 LeRobot 0.6.0 的 `LeRobotDataset` 支持自定义 schema，但原生 `DatasetRecordConfig`
没有任意附加标签入口，录制循环仅组装 observation/action/task。
`scripts/record_real.py` 因此只扩展配置并临时适配录制模块引用的 schema/帧组装函数，
完整调用原生 `lerobot_record.record()`，在每段 `record_loop` 外初始化及清理连续观测配对，
退出后恢复适配；不修改安装文件、不复制回合管理逻辑。
增强后的 YAML 应通过 `record-real` 运行；直接运行未扩展的 `lerobot-record` 会拒绝
`task_conditions`/`require_task_labels` 配置字段。普通原生配置不含任务标签时，
须显式使用 `--require_task_labels=false` 才能通过薄封装运行。

## CPS 读取链路与工具标定

已阅读旧仓库 `src_py/cps_client/cps_client/cps.py` 以及
`lerobot_packages/lerobot_robot_huayan_v2/lerobot_robot_huayan_v2/`：

- `HRIF_Connect(box_id, ip, port)` 由 SDK 建立连接，并处理心跳与快速命令端口；项目直接复用 SDK，不另写 socket 协议。
- `HRIF_ReadActPos(box_id, rbt_id, result)` 一次返回 24 个值，旧适配器在 `read_robot_state()` 中按下表拆分。
- `HRIF_ReadCurTCP` 和 `HRIF_ReadCurUCS` 也可分别读取当前生效的工具和用户坐标系参数。

| 范围 | 旧适配器字段 | 含义 |
|---|---|---|
| `result[0:6]` | `joints` | 六个关节的实际角度 |
| `result[6:12]` | `pose` | 相对机器人 Base 的实际 TCP 笛卡尔位姿，用于观测与动作派生 |
| `result[12:18]` | `tcp` | 当前 TCP 工具标定参数，不是实时工具位置 |
| `result[18:24]` | `ucs` | 当前用户坐标系参数 |

旧实现中，`observation_from_state()` 使用 `state.pose` 生成位置和姿态，关节角与位置保留度和毫米；
姿态可转换为 wxyz 四元数。主动控制路径还将同一份 TCP/UCS 传给逆运动学，说明这些参数用于解释实际位姿。
当前项目需将关节角转换为弧度、位置转换为米，并按坐标约定转换到项目 world；复用已有几何函数。
不能把 `result[12:18]` 当作实时位置，也不能对已包含工具标定的实际 TCP 位姿再次叠加工具偏置。
用户依据手册确认 `result[6:12]` 相对机器人 Base 表达。当前实现直接解析该段，完成毫米到米、
欧拉角到四元数及 Base 到 project world 的转换。TCP/UCS 偏移仅作为标定元信息并检查会话内不变，
不重复应用偏移，也不通过正运动学重新计算已由控制器提供的实际位姿。

旧适配器的被动连接分支在连接机械臂与相机后直接返回，不启动 Servo、不设置 override；
被动断开也不调用控制器停止。SDK 本身的查询、连接和心跳属于读取所需通信。

旧 `get_observation()` 缓存上一组完整观测，通过 bridge 把相邻实测位姿的 delta 提供给
`HuayanDragTeleop.get_action()`；被动 `send_action()` 不调用运动 API。
本项目可参考这一接入方式，但动作应为已确认的下一时刻绝对 TCP 目标，且需处理首帧和回合边界；
本次实现独立采样当前观测及其后继状态，不迁移旧缓存或 delta 逻辑。
高频 ServoController、轨迹规划和主动控制限幅不属于本次采集实现。

采集结果应记录实际读到的 TCP/UCS 和使用的 world 安装外参，支持追溯数据坐标定义。
同一采集会话使用固定的 TCP/UCS；重新启动录制时重新读取有效标定。

## 相机与时间对齐的源码约束

当前仿真配置使用 30 Hz 策略频率和两路 640×480 图像。
LeRobot 的 OpenCV 相机在后台线程读图，主线程通过 `read_latest()` 获取缓存帧，
或通过 `async_read()` 等待可用帧。其 `latest_timestamp` 是主机读取并处理图像后的时间，
并非相机曝光时间；`read_latest()` 本身仅返回图像。

本地 `DatasetWriter.add_frame()` 默认按 `frame_index / fps` 生成数据集时间戳。
因此统一 FPS 和逐行保存不等于双相机与机械臂硬件同步，真实采集时间与标准训练时间轴应明确区分。
首版复用相机后台采集，在每个采集周期读取机械臂状态和双相机最新帧，接受各传感器采集时刻存在偏差。
主机采集时序记录在会话 JSONL 中供后续分析；不能将其标记为曝光时间或硬件同步时间。
更精确的时间对齐作为后续研究事项，不作为首版采集上线的阻塞条件。

## 原生命令的实际行为

`RecordConfig.__post_init__` 要求配置 Teleoperator。主入口自动发现已安装的第三方设备插件；
官方 [硬件接入文档](https://huggingface.co/docs/lerobot/main/en/integrate_hardware)
说明了 Robot、Teleoperator 的接口和插件约定。

每个录制周期依次执行：

1. `robot.get_observation()`，然后经过 observation processor。
2. `teleop.get_action()`，然后经过 teleop action processor。
3. robot action processor，再调用 `robot.send_action()`。
4. 将处理后的 observation、teleop action、`single_task` 和显式填写的任务条件交给 `dataset.add_frame()`。
5. 按目标 FPS 等待；运行超时会警告，不保证设备已经硬件同步。

实际保存的是 teleop action processor 的输出，`send_action()` 的返回值没有用于数据写入。
因此仅在 Robot 中关闭命令发送或返回派生动作，并不能自动得到所需标签。

### Episode 与按键

| 操作或阶段 | 本地版本行为 |
|---|---|
| 启动命令 | 设备连接完成后进入录制；没有独立的等待开始按键 |
| 录制时按右箭头 | 提前结束本次录制循环，随后按正常流程保存 |
| 重置等待时按右箭头 | 提前结束等待，进入后续保存或重录流程 |
| 按左箭头 | 请求重录；主流程清空当前未保存 episode 的缓冲后重新录制 |
| 录制时按 Esc | 结束采集；正常流程仍会保存当前 episode，不等于丢弃 |
| 达到 `episode_time_s` | 自动结束该回合 |
| 回合之间 | 运行最多 `reset_time_s` 的不写数据循环；结束后自动进入下一回合 |

原生命令已有“丢弃当前回合并重录”，没有必要另做一套作废操作。
重置等待仍会调用 Robot 和 Teleoperator，包含 `send_action()`；只读边界必须覆盖整个生命周期。

## 当前设计汇总

首版使用 CPS SDK 读取实际状态和有效工具标定，通过薄的 LeRobot Robot/Teleoperator 适配接入
原生 `lerobot-record`；复用相机、键盘、回合管理和数据写入，运动由外部产生。
每条 episode 记录完整路径，最终数据使用当前项目的双相机、13D 状态与 9D 绝对 TCP 动作定义，
由下一采样时刻实测 TCP 提供监督，训练时复用 relative action 转换。

数据兼容性包含本仓库要求的动作契约清单和 relative action 统计，而不仅是 LeRobot 标准格式。
已确认接受录制后的轻量整理步骤，用于动作时序校验和训练元信息整理，不引入 raw 格式中转。
真机数据存储范围见 [ADR 0008](adr/0008-passive-lerobot-storage.md)。

### 实现范围与复用

1. 接入旧仓库的 CPS SDK，增加精简的只读状态适配；连接后读取有效 TCP/UCS，统一单位与项目 world 坐标定义。
2. 提供可被 LeRobot 发现的 Robot/Teleoperator 接入层，复用官方 OpenCV 相机、录制循环和 episode 管理。
3. 构造与完整观测配对的下一采样时刻实测绝对动作，处理首尾样本及回合边界；保留可获取的主机时序和标定元信息。
4. 提供轻量数据整理入口，复用现有动作表示、relative action 统计和训练校验，确保最终数据可供当前训练入口读取。
5. 补齐安装、相机选择、原生录制、重录、恢复采集、整理和数据检查的操作说明。

### 验收标准

- 用模拟 CPS 与图像输入验证连接、采集、重录和断开路径；采集端没有调用运动或控制模式修改 API。
- 最终双相机 feature 为 `observation.images.global` 和 `observation.images.wrist`，RGB 图像形状与配置一致。
- `observation.state` 沿用现有 13D 命名和顺序：六个关节角、TCP 位置和 wxyz 四元数；单位为弧度和米。
- `action` 沿用现有 9D 世界系绝对位姿表示，标签来自下一采样时刻实测 TCP；无伪造首帧动作、跨回合配对或无后继目标的训练样本。
- 无工艺参数的示范以任务文本训练，不伪造参数或阶段标签。
- LeRobot 可重新打开整理后的数据集，现有 `policy-data-check` 在匹配的任务文本训练配置下通过；relative action 转换和统计与采集动作来源一致。
- 未连接真实设备时，仅报告模拟验证结果；实机连接、坐标核对、采集速率和图像质量在接机验收时单独确认。

## 适配时必须解决的问题

- 原生命令要求 Teleoperator 且每帧调用 `send_action()`；被动适配必须提供动作标签，且不向机械臂发送控制命令。
- 当前观测必须配对下一采样时刻的实测目标；首帧不得填入伪造动作，也不能跨回合配对。
- 状态、动作和双相机命名沿用现有训练契约；直接保存后还需提供训练要求的 relative action 元信息和统计量。
- 参数增强训练使用 `task.direction` 与 `task.parameters`；通过 `task_conditions` 显式录入，避免复用仿真参数默认值。任意自定义标量保存在 `task.extra.*`。
- 按 [ADR 0008](adr/0008-passive-lerobot-storage.md) 将真机数据直接保存为 LeRobot，并明确区分仿真命令标签与真机实测标签的来源。

以上需求已由用户确认并完成代码实现；真实设备验收仍需在接机后进行。

## 后续关注事项（备忘）

### 双相机与机械臂状态的时间对齐

当前接受软件对齐，但该问题需要持续关注，尤其是运动速度提高时图像与实测动作标签的错位。

- 区分相机曝光时间、主机收到图像的时间、SDK 请求与响应时间，以及 LeRobot 固定 FPS 的训练时间轴。
- 使用可获取的主机时间检查双相机接收时间差、图像相对状态读取的时间差，以及重复帧、缺帧和采集周期超时；主机时间差只能用于诊断，不能直接视为曝光同步误差。
- 后续评估相机缓冲延迟、USB 传输延迟、SDK 查询延迟，以及长时间采集中的时钟漂移。
- 根据实测结果再决定是否需要时间插值、驱动或硬件时间戳、公共时钟或硬件触发。

本备忘仅记录后续研究方向，不提前引入同步服务、复杂缓冲框架或新的硬件要求。
