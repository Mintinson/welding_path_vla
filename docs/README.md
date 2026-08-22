# Welding Path VLA 文档

本文档目录描述当前代码和配置。命令示例默认从仓库根目录执行，并优先使用 Pixi 任务；
`src/welding_path_vla/` 中的包代码不作为命令行入口。

## 推荐阅读顺序

1. [脚本入口与配置规则](script-entrypoints.md)：环境、Pixi 任务、模块化 YAML 和命令行覆盖。
2. [仿真数据采集原理与格式](data-collection.md)：几何专家、时间对齐、raw schema 和 LeRobot 结构。
3. [数据采集与训练工作流](data-training-workflow.md)：从仿真采集到 LeRobot 导出、训练和上传。
4. [工件与焊缝任务](simulation/workpieces.md)：五类现有任务及新增任务的边界。
5. [仿真策略部署](simulation-deployment.md)：checkpoint 加载、闭环控制、安全门、输出和排错。
6. [评估规范](evaluation.md)：ESR、ICR、PCR、CTE、姿态、速度、平滑性与真机日志。

## 按主题查找

| 主题 | 文档 | 内容 |
|---|---|---|
| 快速入口 | [脚本入口与配置规则](script-entrypoints.md) | Pixi 环境、脚本映射、配置组合和覆盖优先级 |
| 采集原理 | [仿真数据采集原理与格式](data-collection.md) | 几何专家、执行时序、raw 字段、质量门和 LeRobot 目录结构 |
| 数据全流程 | [数据采集与训练工作流](data-training-workflow.md) | 采集、验证、回放、导出、Hub 上传、训练和恢复 |
| 训练规模 | [Batch、Step 与训练时间](training-scale-guide.md) | DDP 全局 batch、epoch、step 和 wall time 计算 |
| 仿真任务 | [工件与焊缝任务](simulation/workpieces.md) | L 型、管底、管口、曲线平板和三面角工件 |
| 场景布局 | [真实照片对齐布局](simulation/photo-layout.md) | 桌面、机器人、相机和坐标近似 |
| 仿真部署 | [闭环 Rollout](simulation-deployment.md) | checkpoint、动作处理、安全门、终止、输出与排错 |
| ACT | [ACT 基线](act-pipeline.md) | 数据契约、训练、离线评估和仿真部署 |
| SmolVLA | [SmolVLA 基线](smolvla-pipeline.md) | 官方模型接入、恢复训练和多任务部署 |
| π0 / π0.5 | [π0 系列](pi-pipeline.md) | LoRA、A100 配置、训练和部署 |
| Trajectory-VLA | [Trajectory-VLA](trajectory-vla.md) | 本地 SmolVLA 重写及可修改接口 |
| 新策略 | [策略扩展方式](policy-extension.md) | 公共 pipeline 与最小注册接口 |
| 评估 | [焊接短时轨迹评估](evaluation.md) | 仿真和真机统一指标 |
| 标定 | [基座外参](calibration/base-frame.md) / [焊枪 TCP](calibration/elfin5-tool.md) | 坐标约定与当前工具近似 |
| 架构记录 | [ADR](adr/README.md) | 已接受、已完成和被替代的工程决策 |

Trajectory-VLA Qwen 的模型结构、显存估算和解冻接口与源码放在
[`src/welding_path_vla/policies/traj_vla_qwen/README.md`](../src/welding_path_vla/policies/traj_vla_qwen/README.md)，
避免在两处维护同一份模型细节。

## 信息来源与维护规则

- 参数默认值以 `configs/base.yaml` 和 `configs/policies/*.yaml` 为准。
- 可执行任务以 `pyproject.toml` 的 `[tool.pixi.tasks]` 为准。
- 数据字段以 `src/welding_path_vla/dataset/` 和实际 LeRobot `meta/info.json` 为准。
- 论文指标以 `src/welding_path_vla/evaluation/` 为准。
- 文档中的数据规模是带日期的快照，不应替代 `policy-data-check` 的实时输出。
- 新增入口、任务或策略时，应同时更新本索引和对应专题文档；历史决策只在 ADR 中更新状态，
  不删除原始背景。
