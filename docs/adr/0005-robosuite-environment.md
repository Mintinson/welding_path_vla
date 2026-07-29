# 使用 robosuite 管理机器人学习环境

当前 Elfin5-Pro 模型、焊枪 TCP、碰撞几何、专家轨迹和数据协议已经通过直接 MuJoCo
测试，因此完成 ADR 0003 所要求的前置验证，并将环境生命周期迁移到 robosuite。

## 决策

- `WeldingEnv` 继承 robosuite `MujocoEnv`；
- robosuite 负责 `reset()`、`step(action)`、episode 时钟和 observation dictionary；
- 动作统一为世界坐标系绝对 TCP 位姿 `[x, y, z, qw, qx, qy, qz]`；
- 一个 30 Hz `step` 内保持 600 Hz MuJoCo 物理和 120 Hz IK/位置控制；
- 使用 `Elfin5ProRobotModel`、`WeldingArena` 和 `WorkpieceObject` 组合 robosuite `Task`；
- 机器人模型继续使用项目阻尼 IK，不接入与真机 `PushServoP` 语义不一致的通用 OSC；
- 直线和圆弧通过同一个 `SeamPath` 接口提供采样、投影、局部切向与法向；
- 采集和 ACT rollout 继续使用同一套原始 episode 与评价协议。

## 版本约束

项目固定 `robosuite==1.5.1`。`1.5.2` 通过旧版 Mink 把 NumPy 限制到 1.x，与
LeRobot 0.6 所需的 NumPy 2.x 冲突。当前环境不使用 Mink 全身 IK、GR1 或外部
`robosuite_models`，这些可选组件的导入提示由项目兼容层定向静默；真正的导入异常不会
被吞掉。

## 模型边界

- `Elfin5ProRobotModel`：本体 mesh、碰撞代理、焊枪、TCP、腕部相机和执行器；
- `WeldingArena`：地面、桌面、灯光和固定于桌面的全局相机；
- `WorkpieceObject`：可替换工件几何及其焊缝定义；
- `WeldingEnv`：生命周期、观测、IK/控制、碰撞和稀疏成功条件。

该拆分不改变现有动作语义、数据字段和评价接口。新增工件不应修改机器人或 Arena。
