# welding_path_vla

面向 Huayan Elfin5-Pro 焊接 VLA 研究的可复现工程。当前里程碑用纯 MuJoCo 和真实 Elfin5 mesh/惯量模型跑通参数化工件、专家轨迹、双相机原始数据、质量验证、回放和 LeRobot Dataset v3 导出。

## 环境

```bash
pixi install -e sim
pixi run -e sim sim-view --config configs/default.yaml
pixi run -e sim sim-collect --config configs/default.yaml --episodes 1
pixi run -e sim sim-replay --episode datasets/weldpath_raw_v1/episodes/episode_000000
pixi run -e dev check
```

Pixi 环境按用途拆分：`sim`、`data`、`real`、`train`、`deploy` 和 `dev`。真机驱动和模型训练将在后续里程碑实现，当前不会从 `repo/` 迁移旧控制代码。

机器人 URDF、原始 MJCF、真实 STL 和当前焊接场景集中在 `src/welding_path_vla/assets/elfin5/`，由 `simulation/elfin5pro_mujoco` 暴露稳定的模型入口。原始 LIBERO 场景仅作参考，运行时由 YAML 中的 `robot.model_asset` 选择 `elfin5_welding.xml`。

默认场景按真实实验台照片布置：机械臂底座位于桌面，并按真实安装绕底座竖直向上的局部 Z 轴旋转 `-90°`；该轴与 world Z 同向。L 型工件位于前方工作区，全局相机位于机械臂正前上方。腕部相机与 link6 负 Y 侧安装螺钉同轴；焊枪从末端法兰中心伸出。桌面、底座、工件和相机坐标均可在 `configs/default.yaml` 中修改。

## 工程边界

```text
src/welding_path_vla/
├── robot/       # Elfin5-Pro 驱动协议、实时控制与安全门
├── dataset/     # 原始数据协议、录制、动作构造与 LeRobot 导出
├── simulation/  # MuJoCo 环境、专家轨迹、采集与模型入口
├── policies/    # ACT / Diffusion / SmolVLA / Trajectory-VLA 及训练部署契约
└── evaluation/  # 轨迹、碰撞与真机评估
```

原始 episode 是唯一事实源，使用 `trajectory.npz + global.mp4 + wrist.mp4 + metadata.json`。实际 TCP、专家参考和安全命令均保存为绝对轨迹；动作同时保存 seam、robot base 和 world frame 表达。`build_relative_action_chunk` 可在训练期构造相对当前末端坐标系的 9D future chunk 与有效 mask。每条记录包含 N 个命令与 N+1 个同步状态/图像。

`sim view` 显示 YAML 中的真实 home 关节姿态；`sim collect` 会在录制前进入经过 IK 与碰撞检查的工件上方 staging pose。随机工件位姿不可达或 staging 发生碰撞时，采集器会在 `randomization.max_sampling_attempts` 上限内确定性重采样，而不会降低 IK 精度；实际尝试次数记录为 episode 元数据中的 `scene_sampling_attempts`。home 到 staging 的运动不属于焊接专家 episode，未来应由独立的关节空间规划器负责。

## 常用命令

```bash
welding-vla sim view --config configs/default.yaml
welding-vla sim collect --config configs/default.yaml --episodes 50
welding-vla sim replay --episode PATH
welding-vla data validate --dataset datasets/weldpath_raw_v1
welding-vla data export-lerobot --dataset datasets/weldpath_raw_v1 --output datasets/weldpath_lerobot_v1
welding-vla evaluation episode --episode PATH --source raw --config configs/default.yaml
welding-vla evaluation dataset --dataset datasets/weldpath_raw_v1 --config configs/default.yaml
welding-vla robot show-config --config configs/default.yaml
welding-vla policy show-config --config configs/default.yaml
welding-vla policy train --config configs/default.yaml --dry-run
```

所有运行参数来自同一个类型化 YAML；更换 `--config` 即可切换场景、真机、安全、策略、训练和部署配置。CLI 只覆盖常用选项。`--episodes` 表示目标有效回合数，质量不合格的回合仍会分类保留，并在达到 `max_attempt_multiplier` 上限时报错。每个 episode 都保存解析后的配置、seed、任务和质量结果。视频优先使用 H.264，系统 OpenCV 不支持时自动回退为 MPEG-4。完整流程见 [数据采集、训练与多样性扩展报告](docs/data-training-workflow.md)；论文指标与真机日志规范见 [焊接短时轨迹评估规范](docs/evaluation.md)。
