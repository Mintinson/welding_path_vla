# 独立运行 LeRobot 策略与 Elfin ServoP

Elfin 控制采用两层。LeRobot 每次输出一个共享观测 TCP 锚点的相对动作块，经 checkpoint processor 解码后，独立的高频 ServoP 循环负责按训练时间间隔插值并执行世界系路点。首版确认整块到达后才取新观测和推理，以简化两块间的时序和故障判定。执行器与调度分开，后续可接入 RTC 的提前预测和块对齐。
