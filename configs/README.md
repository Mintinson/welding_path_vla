# 配置模块

配置按职责拆分，部署入口只负责组合：

```text
base.yaml                         机器人、相机、场景、安全和公共默认值
tasks/*.yaml                      工件、焊缝、指令和采集目录
policies/*.yaml                   策略、训练和离线评估参数
deploy/smolvla.yaml               部署共用 checkpoint 和 episode 数量
deploy/smolvla_{task}.yaml        可以直接运行的完整任务入口
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

三个 SmolVLA 部署入口：

```bash
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/smolvla_l_joint.yaml
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/smolvla_pipe_bottom.yaml
pixi run -e policy-sim policy-sim-deploy --config_path=configs/deploy/smolvla_pipe_top.yaml
```

更换模型只修改 `deploy/smolvla.yaml` 的 `policy.checkpoint`。调整工件或焊缝只修改对应
`tasks/*.yaml`。临时参数仍可通过 Draccus 覆盖，例如
`--deployment.episodes=10`。
