# ADR 0003：先完成纯 MuJoCo 验证

状态：已完成，由 [ADR 0005](0005-robosuite-environment.md) 接续

## 背景

自定义 Elfin5-Pro、焊枪 TCP、执行器和碰撞代理需要先排除 MJCF 与底层控制问题，再引入机器人
学习框架的额外抽象。

## 决策

第一阶段直接使用 MuJoCo 验证模型加载、运动学、相机、碰撞、专家轨迹、录制和质量门；不采用
PyBullet，也不立即接入 robosuite 控制器。

## 结果

底层链路验证完成后，环境生命周期迁移到 robosuite。该 ADR 保留为迁移背景，不再描述当前运行
入口。
