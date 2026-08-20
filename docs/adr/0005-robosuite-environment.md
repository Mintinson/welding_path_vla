# ADR 0005：使用 robosuite 管理机器人学习环境

状态：已接受

## 背景

[ADR 0003](0003-pure-mujoco-first.md) 已验证 Elfin5-Pro、焊枪 TCP、碰撞、专家轨迹和原始数据
协议。下一阶段需要统一 reset、step、多模态 observation、任务组合和批量 episode 生命周期。

## 决策

- `WeldingEnv` 继承 robosuite `MujocoEnv`；
- robosuite 管理 `reset()`、`step()`、episode 时钟和 observation dictionary；
- 底层仍使用 MuJoCo、项目阻尼 IK 和位置控制，不强行套用与实机 `PushServoP` 不一致的 OSC；
- 一个 30 Hz 策略 step 内执行 600 Hz 物理和 120 Hz 控制；
- 环境由 `Elfin5ProRobotModel`、`WeldingArena` 和 `WorkpieceObject` 组合；
- 工件焊缝统一实现 `SeamPath`，策略动作统一为世界系绝对 TCP 位姿。

项目固定 `robosuite==1.5.1`。当前不使用 Mink、GR1 或外部 `robosuite_models`；兼容层只抑制
这些可选组件的提示，不吞掉真正的运行异常。

## 后果

新增工件和任务不修改机器人、Arena、录制器或策略接口，并可继续直接访问 `mj_model`、`mj_data`
完成接触和底层状态查询。代价是调试时需要区分 robosuite 生命周期与 MuJoCo 控制层。
