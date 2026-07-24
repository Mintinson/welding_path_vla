# 脚本入口与模块边界

`src/welding_path_vla` 只包含可复用的包代码，不执行参数解析、打开窗口、启动采集或
调用训练进程。所有人工直接运行的入口位于仓库根目录的 `scripts/`。

## 入口映射

| Pixi 任务 | 直接脚本 | 职责 |
|---|---|---|
| `sim-view` | `scripts/view_simulation.py` | 检查 MuJoCo 场景 |
| `sim-collect` | `scripts/collect_simulation_data.py` | 采集仿真原始 episode |
| `sim-replay` | `scripts/replay_episode.py` | 回放双相机视频 |
| `data-validate` | `scripts/validate_dataset.py` | 校验原始数据质量 |
| `export-lerobot` | `scripts/export_lerobot.py` | 导出 LeRobot 数据集 |
| `evaluate-episode` | `scripts/evaluate.py episode` | 评估单条仿真/真机 episode |
| `evaluate-dataset` | `scripts/evaluate.py dataset` | 聚合仿真数据集指标 |
| `robot-config` | `scripts/show_robot_config.py` | 检查机器人和安全配置 |
| `policy-config` | `scripts/show_policy_config.py` | 检查策略、训练和部署配置 |
| `train-policy` | `scripts/train_policy.py` | 启动或预览训练 |

推荐通过 Pixi 运行，以确保使用正确的依赖环境：

```bash
pixi run -e sim sim-collect --config configs/default.yaml --episodes 50
pixi run -e dev evaluate-dataset --dataset datasets/weldpath_raw_v1
pixi run -e train train-policy --config configs/default.yaml --dry-run
```

进入对应 Pixi shell 后，也可以直接执行脚本：

```bash
pixi shell -e sim
python scripts/collect_simulation_data.py --config configs/default.yaml --episodes 50
```

## 包内边界

- `core/`：类型化配置、领域对象和坐标几何函数；
- `simulation/`：MuJoCo 环境、专家轨迹和采集业务逻辑；
- `dataset/`：原始 episode、录制、动作构造和数据导出；
- `evaluation/`：轨迹指标、聚合规则和日志适配器；
- `robot/`：真机驱动、实时控制和安全门；
- `policies/`：策略接口、训练与部署请求。

脚本只负责参数解析和调用这些模块。新增可执行流程时，应在对应包中实现业务逻辑，
再在 `scripts/` 增加薄入口；不要重新建立集中式总 CLI。
