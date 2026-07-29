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

Pixi 环境按用途拆分：`sim`、`data`、`real`、`train`、`deploy` 和 `dev`。真机驱动和模型训练将在后续里程碑实现，当前不会从 `repo/` 迁移旧控制代码。

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
├── policies/    # ACT / Diffusion / SmolVLA / Trajectory-VLA 及训练部署契约
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
└── train_policy.py             # 模型训练
```

原始 episode 是唯一事实源，使用 `trajectory.npz + global.mp4 + wrist.mp4 + metadata.json`。实际 TCP、专家参考和安全命令均保存为绝对轨迹；动作同时保存 seam、robot base 和 world frame 表达。`build_relative_action_chunk` 可在训练期构造相对当前末端坐标系的 9D future chunk 与有效 mask。每条记录包含 N 个命令与 N+1 个同步状态/图像。

`sim view` 显示 YAML 中的真实 home 关节姿态；`sim collect` 会在录制前进入经过 IK 与碰撞检查的工件上方 staging pose。随机工件位姿不可达或 staging 发生碰撞时，采集器会在 `randomization.max_sampling_attempts` 上限内确定性重采样，而不会降低 IK 精度；实际尝试次数记录为 episode 元数据中的 `scene_sampling_attempts`。home 到 staging 的运动不属于焊接专家 episode，未来应由独立的关节空间规划器负责。

## 常用命令

```bash
pixi run -e sim sim-view --config_path=configs/default.yaml
pixi run -e sim sim-collect --config_path=configs/default.yaml --collection.episodes=50
pixi run -e sim sim-view --config_path=configs/pipe_bottom.yaml
pixi run -e sim sim-collect --config_path=configs/pipe_top.yaml --collection.episodes=50
pixi run -e sim sim-replay --episode=PATH
pixi run -e sim data-validate --collection.dataset_root=datasets/weldpath_raw_v2
pixi run -e data export-lerobot --dataset=datasets/weldpath_raw_v2 --output=datasets/weldpath_lerobot_v2
pixi run -e dev evaluate-episode --episode=PATH --source=raw --config_path=configs/default.yaml
pixi run -e dev evaluate-dataset --collection.dataset_root=datasets/weldpath_raw_v2 --config_path=configs/default.yaml
pixi run -e dev robot-config --config_path=configs/default.yaml
pixi run -e dev policy-config --config_path=configs/default.yaml
pixi run -e train train-policy --config_path=configs/default.yaml --dry_run=true
```

ACT 基线使用 `configs/act.yaml`，提供 LeRobot 原生训练、留出集测试和 robosuite 闭环部署：

```bash
pixi run -e train train-policy --config_path=configs/act.yaml --training.steps=4 --training.output_dir=outputs/train/act_smoke_01
pixi run -e train policy-evaluate --config_path=configs/act.yaml --policy.checkpoint=outputs/train/act_smoke_01
pixi run -e policy-sim policy-sim-deploy --config_path=configs/act.yaml --policy.checkpoint=outputs/train/act_smoke_01 --deployment.episodes=1
```

完整的数据约定、checkpoint、日志和评估说明见 [ACT pipeline](docs/act-pipeline.md)。

`export-lerobot` 默认采用 LeRobot 官方 AV1 配置，只保存视频 feature。可通过 `--lerobot_export.incremental=true` 追加数据，用 `--lerobot_export.start_episode/--lerobot_export.end_episode` 选择源 episode 闭区间；逐帧图片是显式选项 `--lerobot_export.save_images=true`。完整配置与示例见[数据采集、训练与多样性扩展报告](docs/data-training-workflow.md#5-导出-lerobot-数据)。

Pixi 任务直接调用 `scripts/` 下的单一职责脚本；也可以在对应 Pixi shell 中直接运行，例如 `python scripts/evaluate.py --help`。`src/welding_path_vla` 只提供可复用包代码，不再承载命令行入口。任务与脚本的完整映射见 [脚本入口与模块边界](docs/script-entrypoints.md)。

所有运行参数来自同一个类型化 YAML；更换 `--config_path` 即可切换场景、真机、安全、策略、训练和部署配置，任意字段都可用 `--section.field=value` 覆盖。`collection.episodes` 表示目标有效回合数，质量不合格的回合仍会分类保留，并在达到 `max_attempt_multiplier` 上限时报错。每个 episode 都保存解析后的配置、seed、任务和质量结果。原始数据与 rollout 由同一个 LeRobot / PyAV 接口写入可预览的 H.264；最终训练数据默认转为体积更小的 AV1。完整流程见 [数据采集、训练与多样性扩展报告](docs/data-training-workflow.md)；论文指标与真机日志规范见 [焊接短时轨迹评估规范](docs/evaluation.md)。
