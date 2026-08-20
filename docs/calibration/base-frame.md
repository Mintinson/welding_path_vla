# 机器人基座安装外参

真实 Elfin5-Pro 基座相对项目 world frame 绕底座竖直向上的局部 Z 轴旋转 `-90 deg`，安装
位姿由 `scene.robot_base_position_m` 和 `scene.robot_base_yaw_deg` 统一配置。MuJoCo 使用
位于底座中心的 `robot_mount` 父节点承载 base visual 和整条运动链，yaw 只施加在该节点上。

项目约定 TCP、参考轨迹和 `*_world` 字段在 world frame 中表达；`*_base` 字段在机器人基座
坐标系中表达。对平移和旋转向量都使用同一变换：
`delta_base = R_world_from_base.T @ delta_world`。episode 的 `coordinate_frames` 元数据会
记录这些语义。

基座安装外参不会改变机器人编码器关节读数，因此当前 YAML 初始关节角保持
`[90.9411, -72.1133, 22.0613, 45.5546, 128.4704, -51.3849] deg`，不再用 joint1
抵消 base yaw。修改基座 yaw、工件位置或初态后，必须重新运行整条焊缝 IK、碰撞和 episode
质量验证。

由于真实安装后的 home pose 位于工件另一侧，专家 approach 不允许直接用一条斜线连接焊缝
前置点。当前路径先沿 world Z 抬升到 `task.staging_clearance_m` 指定的安全高度，再水平移动
到焊缝外侧，最后下降到 pre-weld 位姿。
