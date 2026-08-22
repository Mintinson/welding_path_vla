# 脚本入口与配置规则

所有人工执行的入口位于仓库根目录 `scripts/`；`src/welding_path_vla/` 只包含可复用代码。
推荐通过 Pixi 任务运行，以避免误用系统 Python 或错误的 CUDA、TorchCodec、MuJoCo 依赖。

## Pixi 环境

| 环境 | 用途 |
|---|---|
| `sim` | robosuite / MuJoCo 场景、采集、回放与原始数据验证 |
| `data` | LeRobot Dataset 转换、视频编码和 Hub 上传 |
| `train` | CUDA 策略训练与离线策略评估 |
| `policy-sim` | CUDA 策略与 robosuite 闭环部署 |
| `real` | 真机接口的 CPU 运行环境 |
| `deploy` | CUDA 策略与真机部署依赖 |
| `dev` | 格式化、类型检查、测试和论文指标脚本 |

首次使用某个环境时执行 `pixi install -e ENV`。不要复制 `.pixi/` 到服务器；同步源码、
`pyproject.toml` 和 `pixi.lock` 后在目标机器重新安装。

## 任务与脚本映射

| Pixi 任务 | 脚本 | 职责 |
|---|---|---|
| `sim-view` | `view_simulation.py` | 打开场景并检查机器人、工件和相机 |
| `sim-collect` | `collect_simulation_data.py` | 采集通过质量门的仿真 episode |
| `sim-replay` | `replay_episode.py` | 同步回放双相机视频 |
| `data-validate` | `validate_dataset.py` | 汇总原始数据质量状态 |
| `export-lerobot` | `export_lerobot.py` | 单源或多源导出 LeRobot Dataset v3 |
| `upload-lerobot` | `upload_lerobot_dataset.py` | 上传已经完成的 LeRobot 数据集 |
| `policy-data-check` | `check_policy_data.py` | 检查数据 schema、规模和策略输入契约 |
| `train-policy` | `train_policy.py` | 单进程训练、恢复训练或命令预览 |
| `train-policy-2gpu` | `train_policy.py` | 用 Accelerate 启动两个 DDP 进程 |
| `policy-evaluate` | `evaluate_policy.py` | 在 LeRobot 留出数据上评估 checkpoint |
| `policy-sim-deploy` | `deploy_simulation_policy.py` | 在 robosuite 中执行[闭环 rollout](simulation-deployment.md) |
| `evaluate-episode` | `evaluate.py --mode=episode` | 计算单条仿真或真机 episode 指标 |
| `evaluate-dataset` | `evaluate.py --mode=dataset` | 聚合原始数据集指标 |
| `robot-config` | `show_robot_config.py` | 输出机器人、安装外参与安全配置 |
| `policy-config` | `show_policy_config.py` | 输出策略、训练和部署配置 |

`fix_lerobot_state_names.py` 和 `fix_task_language.py` 是历史数据的一次性迁移脚本，
不属于日常 pipeline；新数据已经直接使用正确的状态名称和英文任务指令。

## 模块化 YAML

配置按职责组织：

```text
configs/
├── base.yaml                       # 频率、相机、场景、机器人、安全及公共默认值
├── tasks/                          # 工件、焊缝、任务随机化和数据目录
├── policies/                       # 策略及本机 / A100 训练参数
├── deploy/                         # checkpoint + policy + task 的部署入口
└── *.yaml                          # 常用组合入口和兼容入口
```

入口文件通过 `includes` 递归组合 YAML。相对路径以当前 YAML 所在目录为基准；合并优先级为：

```text
较早 include < 较晚 include < 入口文件本身 < 命令行覆盖
```

例如 `configs/deploy/smolvla_pipe_top.yaml` 依次组合基础配置、SmolVLA checkpoint 和
管口任务。临时改变 episode 数不需要复制配置：

```bash
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla_pipe_top.yaml \
  --deployment.episodes=10
```

目录默认自动命名为 `outputs/deploy/{policy.family}_{task.task_id}`。同一 checkpoint 批量运行
全部任务时增加 `--deployment.run_all_tasks=true`，无需逐个切换配置和输出目录。

项目的组合层只识别 `--config_path=PATH` 形式。其他字段由 Draccus 直接解析，使用
`--section.field=value` 覆盖；布尔值显式写成 `true` 或 `false`。

## 常用调用

```bash
# 检查场景并采集 5 条小样本
pixi run -e sim sim-view --config_path=configs/default.yaml
pixi run -e sim sim-collect \
  --config_path=configs/default.yaml \
  --collection.episodes=5 \
  --collection.workers=1

# 验证原始数据
pixi run -e sim data-validate \
  --config_path=configs/default.yaml \
  --collection.dataset_root=datasets/weldpath_raw_v2

# 预览训练计划，不创建正式训练
pixi run -e train train-policy \
  --config_path=configs/smolvla.yaml \
  --dry_run=true

# 运行仓库检查
pixi run -e dev check
```

进入对应环境后可以直接执行脚本，参数语义不变：

```bash
pixi shell -e sim
python scripts/collect_simulation_data.py \
  --config_path=configs/default.yaml \
  --collection.episodes=5
```

## 包内边界

- `core/`：类型化配置、领域对象、配置组合和坐标几何；
- `simulation/models/`：Elfin5-Pro、Arena 和可替换工件；
- `simulation/tasks/`：焊缝采样、投影和局部标架；
- `simulation/`：robosuite 环境、专家轨迹和采集业务；
- `dataset/`：原始 episode、录制、动作构造和 LeRobot 导出；
- `evaluation/`：轨迹指标、聚合规则和仿真/真机适配；
- `robot/`：真机驱动、实时控制和安全门；
- `policies/`：共享训练、评估、部署流程及本地策略实现。

新增可执行流程时，应先在对应包中实现业务逻辑，再在 `scripts/` 增加薄入口；不要重新建立
集中式总 CLI。策略公共接口见[策略扩展方式](policy-extension.md)。
