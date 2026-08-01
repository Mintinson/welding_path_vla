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
  - ../base.yaml
  - smolvla.yaml
  - ../tasks/pipe_top.yaml

deployment:
  log_dir: outputs/deploy/smolvla_pipe_top
```

策略和任务是两个正交模块。例如 π0.5 训练、双 A100 训练和三个任务部署分别为：

```bash
pixi run -e train train-policy --config_path=configs/pi0_5.yaml
pixi run -e train train-policy-2gpu --config_path=configs/pi0_5_a100.yaml
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/pi0_5_l_joint.yaml
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/pi0_5_pipe_bottom.yaml
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/pi0_5_pipe_top.yaml
```

`{model}` 当前可取 `smolvla`、`trajectory_vla`、`pi0` 和 `pi0_5`；`{task}` 可取 `l_joint`、
`pipe_bottom` 和 `pipe_top`。更换 checkpoint 只修改对应的 `deploy/{model}.yaml`，
调整工件或焊缝只修改 `tasks/*.yaml`。临时参数仍可通过 Draccus 覆盖，例如
`--deployment.episodes=10`，无需复制整份配置。
