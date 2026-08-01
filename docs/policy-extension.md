# 策略公共层与扩展方式

## 模块边界

`src/welding_path_vla/policies/` 只保留一套 LeRobot 胶水流程：

```text
spec.py                 声明策略类、预训练来源和输入差异
lerobot_pipeline.py     统一训练、加载、评估和仿真部署入口
lerobot_training.py     配置构造、日志和断点恢复
runtime.py              项目 Observation 到 LeRobot processor
offline_evaluation.py   留出数据 loss 与动作 MAE
simulation_rollout.py   共用 robosuite 闭环 rollout
checkpoint.py           checkpoint 与 resume 路径解析
data.py                 LeRobotDataset 查询和数据检查
trajectory_vla/         项目真正维护的本地模型实现
```

ACT、SmolVLA、π0 和 π0.5 直接使用 LeRobot 官方模型，因此各目录仅保留实验说明，不再
复制 `training.py`、`runtime.py`、`evaluation.py` 和 `rollout.py`。

## 接入一个 LeRobot 策略

在 `spec.py` 新增一个 `LeRobotPolicySpec`，再加入 `POLICY_SPECS`。最常用字段如下：

- `family`：项目 YAML 中的 `policy.family`；
- `policy_type`：LeRobot 配置注册名；
- `config_class_path`、`policy_class_path`：官方或本地实现的完整类路径；
- `config_mode`：从零训练、读取官方预训练配置或本地模型承接预训练权重；
- `processor_adds_batch`：官方 processor 是否自行添加 batch 维；
- `language`：运行时是否传入 `task`；
- `return_uint8`：视频帧是否保持 uint8 交给 processor。

配置类和模型类采用字符串路径惰性导入，因此查看 YAML 或使用无 GPU 的数据工具时不会加载
大模型依赖。若新策略遵循 LeRobot 的 `PreTrainedPolicy`、processor 和 checkpoint 契约，
不需要再新增 pipeline 文件。

只有下列情况才应增加算法目录：模型结构由本项目维护、processor 语义不同，或确实需要公共
接口无法表达的新训练目标。此时仍优先扩展一项明确的规格或 hook，避免复制整套生命周期代码。
