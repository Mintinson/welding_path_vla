# 架构决策记录

ADR 记录当时的约束、选择和后果。它们不是操作手册；当前运行方式请以
[文档索引](../README.md)和代码配置为准。

| ADR | 状态 | 决策 |
|---|---|---|
| [0001](0001-pixi-multi-environment.md) | 已由 0010 替代 | 使用一个 Pixi workspace 和多个用途环境 |
| [0002](0002-framework-independent-episodes.md) | 仿真部分有效；真机被动采集由 0008 替代 | 原始 episode 是框架无关的唯一事实源 |
| [0003](0003-pure-mujoco-first.md) | 已完成，后续被 0005 接续 | 先用纯 MuJoCo 验证底层模型和数据链路 |
| [0004](0004-package-elfin5-assets.md) | 已接受 | Elfin5 资产随 Python 包分发 |
| [0005](0005-robosuite-environment.md) | 已接受 | 用 robosuite 管理机器人学习环境生命周期 |
| [0006](0006-prismatic-qwen-paired-layers.md) | 已接受 | Prismatic-Qwen 使用成对层双流交织 |
| [0007](0007-passive-executed-action-targets.md) | 已接受 | 被动采集使用下一时刻实测 TCP 作为动作标签 |
| [0008](0008-passive-lerobot-storage.md) | 已实现，待实机验收 | 真机被动采集直接保存 LeRobot 数据，录制后轻量整理 |
| [0009](0009-separate-elfin-servo-from-policy.md) | 已接受 | LeRobot 策略与 Elfin ServoP 分层，首版串行执行动作块 |
| [0010](0010-single-pixi-environment.md) | 已接受 | 仿真、数采、训练、部署和开发共用一个 Pixi 默认环境 |
