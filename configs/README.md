# 配置模块

配置按职责拆分，部署入口只负责组合：

```text
base.yaml                         机器人、相机、场景、安全和公共默认值
tasks/*.yaml                      工件、焊缝、指令和采集目录
policies/{model}.yaml             本机策略、训练和离线评估参数
policies/{model}_a100.yaml        2×A100 正式训练覆盖项
deploy/{model}.yaml               部署共用 checkpoint 和 episode 数量
deploy/{model}_{task}.yaml        可以直接运行的完整任务入口
```

入口文件使用 `includes` 按顺序加载模块。后面的模块覆盖前面的同名字段，入口文件自身
优先级最高。例如：

```yaml
includes:
  - smolvla.yaml
  - ../tasks/pipe_top.yaml
```

输出目录由程序自动生成为 `outputs/deploy/{model}_{task_id}`，任务组合文件不需要再同步维护
目录名。一次运行全部任务：

```bash
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/smolvla.yaml \
  --deployment.run_all_tasks=true
```

策略和任务是两个正交模块。例如 π0.5 训练、双 A100 训练和三个任务部署分别为：

```bash
pixi run -e train train-policy --config_path=configs/pi0_5.yaml
pixi run -e train train-policy-2gpu --config_path=configs/pi0_5_a100.yaml
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/pi0_5_l_joint.yaml
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/pi0_5_pipe_bottom.yaml
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/pi0_5_pipe_top.yaml
```

所有训练策略都有对应的双 A100 入口：

```text
act_a100.yaml
smolvla_a100.yaml
trajectory_vla_a100.yaml
traj_vla_qwen_a100.yaml
pi0_a100.yaml
pi0_5_a100.yaml
```

A100 profile 使用较大的每卡 batch，双卡全局 batch 为其两倍，再按当前 3,674,023 个训练帧
重新计算约一轮数据所需 step。支持的 VLA 同时启用 `torch.compile`；π0 / π0.5 默认保留
梯度检查点，确保 40 GiB 型号也有合理的显存余量。需要固定 optimizer update 数量的实验，
可以单独覆盖 `--training.steps`，但这时增大 batch 会增加总训练样本和计算量。
详细计算公式和硬件调优流程见
[多 GPU Batch、Step 与训练时间计算](../docs/training-scale-guide.md)。

`{model}` 当前可取 `smolvla`、`trajectory_vla`、`pi0` 和 `pi0_5`；`{task}` 可取 `l_joint`、
`pipe_bottom`、`pipe_top`、`curve_plate` 和 `trihedral_vertical`。更换 checkpoint 只修改对应的 `deploy/{model}.yaml`，
调整工件或焊缝只修改 `tasks/*.yaml`。临时参数仍可通过 Draccus 覆盖，例如
`--deployment.episodes=10`，无需复制整份配置。
