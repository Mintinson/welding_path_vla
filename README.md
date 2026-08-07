# welding_path_vla

面向 Huayan Elfin5-Pro 焊接 VLA 研究的可复现工程。当前仿真环境由 robosuite
管理 episode、`reset()`、`step()` 和多模态观测，底层仍使用 MuJoCo 与真实 Elfin5
mesh/惯量模型；数据链路包含参数化工件、专家轨迹、双相机原始数据、质量验证、回放和
LeRobot Dataset v3 导出。

## 环境

```bash
pixi install -e sim
pixi run -e sim sim-view --config_path=configs/default.yaml
pixi run -e sim sim-collect --config_path=configs/default.yaml --collection.episodes=1
pixi run -e sim sim-replay --episode=datasets/weldpath_raw_v2/episodes/episode_000000
pixi run -e dev check
```

Pixi 环境按用途拆分为 `sim`、`data`、`real`、`train`、`deploy` 和 `dev`；项目代码不依赖
后期会删除的 `repo/` 目录。

机器人 URDF、原始 MJCF 和真实 STL 集中在 `src/welding_path_vla/assets/elfin5/`。运行时由
`Elfin5ProRobotModel`、`WeldingArena` 和 `WorkpieceObject` 分别构建机器人、桌面场景和
工件，再组合成 robosuite `Task`；机器人资产由 YAML 的 `robot.model_asset` 选择
`elfin5pro_robot.xml`。

默认场景按真实实验台照片布置：机械臂底座位于桌面，并按真实安装绕底座竖直向上的局部 Z 轴旋转 `-90°`；该轴与 world Z 同向。L 型工件位于前方工作区，全局相机位于机械臂正前上方。腕部相机安装架与 link6 负 Y 侧安装螺钉对齐，镜头光心位于安装架前方；焊枪从末端法兰中心伸出。`configs/pipe_bottom.yaml` 和 `configs/pipe_top.yaml` 提供空心圆管加方形底板工件的上下圆弧任务；工件和任务切换说明见[工件与焊缝任务](docs/simulation/workpieces.md)。

`WeldingEnv` 使用 robosuite 标准接口，动作是世界系绝对 TCP 位姿：

```python
import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.simulation import WeldingEnv

env = WeldingEnv(AppConfig.load("configs/default.yaml"), seed=0)
observation = env.reset()
pose = env.tcp_pose()
action = np.concatenate([pose.position, pose.quaternion_wxyz])
observation, reward, done, info = env.step(action)
env.close()
```

观测键包括 `global_image`、`wrist_image`、`joint_position`、`joint_velocity`、
`tcp_position` 和 `tcp_quaternion_wxyz`。需要接触距离或 MuJoCo 底层状态时使用
`env.mj_model` 与 `env.mj_data`。

## 工程边界

```text
src/welding_path_vla/
├── core/        # 配置、领域对象和坐标几何工具
├── robot/       # Elfin5-Pro 驱动协议、实时控制与安全门
├── dataset/     # 原始数据协议、录制、动作构造与 LeRobot 导出
├── simulation/  # robosuite 环境、models、tasks、专家轨迹与采集
├── policies/    # 策略规格、公共训练/评估/部署流程及本地模型实现
└── evaluation/  # 轨迹、碰撞与真机评估

scripts/
├── collect_simulation_data.py  # 仿真数据采集
├── view_simulation.py          # 场景检查
├── replay_episode.py           # 双相机回放
├── validate_dataset.py         # 原始数据质量校验
├── export_lerobot.py           # LeRobot 导出
├── evaluate.py                 # episode/数据集论文指标
├── show_robot_config.py        # 机器人配置检查
├── show_policy_config.py       # 策略配置检查
├── train_policy.py             # 模型训练
├── evaluate_policy.py          # 留出数据离线评估
└── deploy_simulation_policy.py # robosuite 闭环部署
```

原始 episode 是唯一事实源，使用 `trajectory.npz + global.mp4 + wrist.mp4 + metadata.json`。实际 TCP、专家参考和安全命令均保存为绝对轨迹；动作同时保存 seam、robot base 和 world frame 表达。`build_relative_actions` 可构造相对预测时刻 TCP、共享同一锚点的 9D future chunk 与有效 mask。每条记录包含 N 个命令与 N+1 个同步状态/图像。

`sim view` 显示 YAML 中的真实 home 关节姿态；`sim collect` 默认每 10 个 episode 编号采样一组焊枪姿态、执行方向、速度以及圆管起点和扫掠角，工件位姿和机器人初始状态仍逐条变化。录制前会对接近、跟踪和退出的完整轨迹执行连续 IK、关节限位、速度连续性和碰撞预检；不可行时在 `randomization.max_sampling_attempts` 内重采场景与初态，而不是录制一条已知失败的长视频。实际任务参数仅写入 `task_parameters`，instruction 保持任务 YAML 中的固定文本。

## 常用命令

```bash
pixi run -e sim sim-view --config_path=configs/default.yaml
pixi run -e sim sim-collect --config_path=configs/default.yaml --collection.episodes=50
pixi run -e sim sim-view --config_path=configs/pipe_bottom.yaml
pixi run -e sim sim-collect --config_path=configs/pipe_top.yaml --collection.episodes=50
pixi run -e sim sim-view --config_path=configs/curve_plate.yaml
pixi run -e sim sim-collect --config_path=configs/curve_plate.yaml --collection.episodes=50
pixi run -e sim sim-replay --episode=PATH
pixi run -e sim data-validate --collection.dataset_root=datasets/weldpath_raw_v2
pixi run -e data export-lerobot --dataset_glob='datasets/*_raw_v2' --output=datasets/weldpath_lerobot_relative_v1
# 在上述命令后添加 --repo_id=USER/REPO --lerobot_export.push_to_hub=true 可自动上传
pixi run -e data upload-lerobot --dataset=datasets/weldpath_lerobot_relative_v1 --repo_id=USER/REPO
pixi run -e dev evaluate-episode --episode=PATH --source=raw --config_path=configs/default.yaml
pixi run -e dev evaluate-dataset --collection.dataset_root=datasets/weldpath_raw_v2 --config_path=configs/default.yaml
pixi run -e dev robot-config --config_path=configs/default.yaml
pixi run -e dev policy-config --config_path=configs/default.yaml
pixi run -e train train-policy --config_path=configs/default.yaml --dry_run=true
```

无头采集默认使用 4 个独立进程。可通过 `--collection.workers=2` 调整；单条调试或显存不足时使用 `--collection.workers=1`。

ACT 基线使用 `configs/act.yaml`，提供 LeRobot 原生训练、留出集测试和 robosuite 闭环部署：

```bash
pixi run -e train train-policy --config_path=configs/act.yaml --training.steps=4 --training.output_dir=outputs/train/act_smoke_01
pixi run -e train policy-evaluate --config_path=configs/act.yaml --policy.checkpoint=outputs/train/act_smoke_01
pixi run -e policy-sim policy-sim-deploy --config_path=configs/act.yaml --policy.checkpoint=outputs/train/act_smoke_01 --deployment.episodes=1
```

完整的数据约定、checkpoint、日志和评估说明见 [ACT pipeline](docs/act-pipeline.md)。

SmolVLA 基线可使用统一的多任务 LeRobot 数据集和相同入口：

```bash
pixi run -e train policy-data-check --config_path=configs/smolvla.yaml
pixi run -e train train-policy --config_path=configs/smolvla.yaml
pixi run -e train policy-evaluate --config_path=configs/smolvla.yaml --policy.checkpoint=CHECKPOINT
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/smolvla_l_joint.yaml
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/smolvla_pipe_bottom.yaml
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/smolvla_pipe_top.yaml
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/smolvla_curve_plate.yaml
```

配置由 `base.yaml`、`tasks/`、`policies/` 和 `deploy/` 分层组合；部署时更换一个入口
YAML 即可同时切换工件、焊缝、指令、最大步数和输出目录。公共 checkpoint 只需在
`configs/deploy/smolvla.yaml` 修改一次。

实际采集数量、增量合并、RTX 4060 训练参数和最终产物见
[SmolVLA 三任务基线](docs/smolvla-pipeline.md)。

π0 与 π0.5 复用相同入口，并分别提供本机低显存和 2×A100 配置：

```bash
pixi run -e train train-policy --config_path=configs/pi0.yaml
pixi run -e train train-policy --config_path=configs/pi0_5.yaml
pixi run -e train train-policy-2gpu --config_path=configs/pi0_a100.yaml
pixi run -e train train-policy-2gpu --config_path=configs/pi0_5_a100.yaml
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/pi0_5_pipe_top.yaml
```

配置层次、显存取舍、恢复训练、离线评估和三任务部署见
[π0 / π0.5 pipeline](docs/pi-pipeline.md)。

Trajectory-VLA 将官方 SmolVLA 的网络与 flow matching 完整重写到项目内，并公开视觉、
语言、状态、动作 token 和逐步去噪接口：

```bash
pixi run -e train train-policy --config_path=configs/trajectory_vla.yaml
pixi run -e train policy-evaluate --config_path=configs/trajectory_vla.yaml --policy.checkpoint=CHECKPOINT
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/trajectory_vla_pipe_top.yaml
```

模块边界、可修改接口、官方权重兼容性和三任务部署见
[Trajectory-VLA 本地实现](docs/trajectory-vla.md)。

Prismatic-Qwen 变体使用 DINOv2 + SigLIP + Qwen2.5-0.5B，并以 16 个成对层
逐层交织轻量动作专家：

```bash
pixi run -e train train-policy --config_path=configs/traj_vla_qwen.yaml
```

架构、解冻开关与训练/评估/部署命令见
[Trajectory-VLA Qwen](src/welding_path_vla/policies/traj_vla_qwen/README.md)。

`export-lerobot` 默认采用 LeRobot 官方 AV1 配置，只保存视频 feature。可通过 `--lerobot_export.incremental=true` 追加数据，用 `--lerobot_export.start_episode/--lerobot_export.end_episode` 选择源 episode 闭区间；逐帧图片是显式选项 `--lerobot_export.save_images=true`。完整配置与示例见[数据采集、训练与多样性扩展报告](docs/data-training-workflow.md#5-导出-lerobot-数据)。

Pixi 任务直接调用 `scripts/` 下的单一职责脚本；也可以在对应 Pixi shell 中直接运行，例如 `python scripts/evaluate.py --help`。`src/welding_path_vla` 只提供可复用包代码，不再承载命令行入口。任务与脚本的完整映射见 [脚本入口与模块边界](docs/script-entrypoints.md)。

所有运行参数来自同一个类型化 YAML；更换 `--config_path` 即可切换场景、真机、安全、策略、训练和部署配置，任意字段都可用 `--section.field=value` 覆盖。`collection.episodes` 表示目标有效回合数，质量不合格的回合仍会分类保留，并在达到 `max_attempt_multiplier` 上限时报错。每个 episode 都保存解析后的配置、seed、任务和质量结果。原始数据与 rollout 由同一个 LeRobot / PyAV 接口写入可预览的 H.264；最终训练数据默认转为体积更小的 AV1。完整流程见 [数据采集、训练与多样性扩展报告](docs/data-training-workflow.md)；论文指标与真机日志规范见 [焊接短时轨迹评估规范](docs/evaluation.md)。
