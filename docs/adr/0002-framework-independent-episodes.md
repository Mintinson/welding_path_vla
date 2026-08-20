# ADR 0002：原始 episode 作为唯一事实源

状态：已接受

## 背景

策略可能使用不同动作 horizon、坐标表示、训练框架和图像编码。若采集时直接写入某个模型格式，
后续实验会被迫重新采集。

## 决策

仿真和真机先保存框架无关的 `trajectory.npz + metadata.json + MP4` episode。LeRobot 数据集和
relative action chunk 都从原始 episode 派生。

## 后果

可在不重新采集的情况下改变动作表示或训练框架，并保留命令与执行轨迹用于诊断；代价是需要维护
明确的导出契约和数据迁移工具。
