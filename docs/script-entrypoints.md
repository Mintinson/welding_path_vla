# 脚本入口与模块边界

`src/welding_path_vla` 只包含可复用的包代码，不执行参数解析、打开窗口、启动采集或
调用训练进程。所有人工直接运行的入口位于仓库根目录的 `scripts/`。

## 入口映射

| Pixi 任务 | 直接脚本 | 职责 |
|---|---|---|
| `sim-view` | `scripts/view_simulation.py` | 检查 robosuite / MuJoCo 场景 |
| `sim-collect` | `scripts/collect_simulation_data.py` | 采集仿真原始 episode |
| `sim-replay` | `scripts/replay_episode.py` | 回放双相机视频 |
| `data-validate` | `scripts/validate_dataset.py` | 校验原始数据质量 |
| `export-lerobot` | `scripts/export_lerobot.py` | 导出 LeRobot 数据集 |
| `policy-data-check` | `scripts/check_policy_data.py` | 检查策略训练数据契约 |
| `policy-evaluate` | `scripts/evaluate_policy.py` | 离线评估策略 checkpoint |
| `policy-sim-deploy` | `scripts/deploy_simulation_policy.py` | 策略 robosuite 闭环部署 |
| `evaluate-episode` | `scripts/evaluate.py --mode=episode` | 评估单条仿真/真机 episode |
| `evaluate-dataset` | `scripts/evaluate.py --mode=dataset` | 聚合仿真数据集指标 |
| `robot-config` | `scripts/show_robot_config.py` | 检查机器人和安全配置 |
| `policy-config` | `scripts/show_policy_config.py` | 检查策略、训练和部署配置 |
| `train-policy` | `scripts/train_policy.py` | 启动或预览训练 |

## 统一参数规则

配置按职责拆分为基础场景、任务、策略和部署入口：

```text
configs/
├── base.yaml
├── tasks/{l_joint,pipe_bottom,pipe_top}.yaml
├── policies/{act,smolvla}.yaml
└── deploy/smolvla_{l_joint,pipe_bottom,pipe_top}.yaml
```

入口 YAML 使用 `includes` 按顺序组合模块，后面的模块覆盖前面的同名字段，入口自身
优先级最高。组合完成后仍由 Draccus 解析类型化 dataclass，并应用命令行点号覆盖：

```bash
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla_pipe_top.yaml \
  --deployment.episodes=10
```

`configs/default.yaml`、`pipe_bottom.yaml`、`pipe_top.yaml`、`act.yaml` 和
`smolvla.yaml` 是兼容旧命令的组合入口。参数名称仍与 dataclass/YAML 完全一致，
布尔值使用 `--field=true/false`。

推荐通过 Pixi 运行，以确保使用正确的依赖环境：

```bash
pixi run -e sim sim-collect --config_path=configs/default.yaml --collection.episodes=50
pixi run -e dev evaluate-dataset --collection.dataset_root=datasets/weldpath_raw_v2
pixi run -e train train-policy --config_path=configs/default.yaml --dry_run=true
```

进入对应 Pixi shell 后，也可以直接执行脚本：

```bash
pixi shell -e sim
python scripts/collect_simulation_data.py \
  --config_path=configs/default.yaml \
  --collection.episodes=50
```

## 包内边界

- `core/`：类型化配置、领域对象和坐标几何函数；
- `simulation/models/`：Elfin5-Pro、Arena 和可替换工件模型；
- `simulation/tasks/`：直线/圆弧焊缝的采样、投影和局部标架；
- `simulation/`：robosuite 环境、专家轨迹和采集业务逻辑；
- `dataset/`：原始 episode、录制、动作构造和数据导出；
- `evaluation/`：轨迹指标、聚合规则和日志适配器；
- `robot/`：真机驱动、实时控制和安全门；
- `policies/`：策略接口、训练与部署请求。

脚本只负责参数解析和调用这些模块。新增可执行流程时，应在对应包中实现业务逻辑，
再在 `scripts/` 增加薄入口；不要重新建立集中式总 CLI。

ACT 与 SmolVLA 共用 `train-policy`、`policy-evaluate` 和 `policy-sim-deploy`，
由 `policy.family` 选择实现。SmolVLA 三任务流程见
[`smolvla-pipeline.md`](smolvla-pipeline.md)。
