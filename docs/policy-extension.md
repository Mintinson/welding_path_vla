# 策略公共层与扩展方式

新策略应先复用公共数据、relative-action processor、训练、评估和 rollout。只有模型结构或
processor 语义确实不同，才增加策略专属目录。

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
welding_prompt.py       VLM 无关的结构化焊接参数到任务文本转换
trajectory_vla/         项目真正维护的本地模型实现
traj_vla_qwen/          Prismatic-Qwen 与成对层动作专家
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

语言策略默认复用 `WeldingPromptBuilder`：共享训练工厂在 `AddBatchDimension` 之后插入它，再进入
模型自己的换行、chat template、状态文本化和 tokenizer。新增 VLM 时只要官方 processor 遵循
这一公共 batch 边界，就不应在算法目录复制 prompt builder。字段配置统一使用
`policy.welding_prompt_fields`，不要塞入模型专属 `policy.parameters`。

只有下列情况才应增加算法目录：模型结构由本项目维护、processor 语义不同，或确实需要公共
接口无法表达的新训练目标。此时仍优先扩展一项明确的规格或 hook，避免复制整套生命周期代码。

## 接入检查清单

1. 在 `spec.py` 注册策略，并为惰性类路径增加导入测试；
2. 在 `configs/policies/` 增加策略配置，必要时增加独立 A100 覆盖；
3. 使用 `policy-data-check` 验证双相机、13D state、9D action 和任务字段；
4. 运行短程训练，确认 loss、日志、processor 和 checkpoint 均落盘；
5. 从 `checkpoints/last/pretrained_model` 做离线评估和单 episode 动作预测；
6. 复用 `configs/deploy/` 任务入口执行闭环，不在模型目录增加第二套 rollout；
7. 为缺少顶层 `type` 的旧官方 checkpoint 使用公共 `load_policy_config()`，不要手工把
   `input_features` / `output_features` 字典拼成 `PolicyFeature`。

策略对比必须保持数据划分、relative-action 语义、horizon、执行步数和评价阈值一致；模型特有的
图像分辨率、tokenizer、flow 步数和冻结范围单独记录。
