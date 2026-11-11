# Elfin 动作块测试与部署

在仓库根目录使用 `welding_vla` 环境。默认生成一圈立体八字，控制器地址、安装外参、相机和任务文本取自 `configs/real_record.yaml`。

```bash
python scripts/test_elfin_control.py
python scripts/test_elfin_control.py --host=192.168.56.101
```

两条命令均为只读。现场确认后，加 `--execute=true` 启动运动。录制配置中的相机逐路连接，失败的通道使用黑帧。

同一入口通过 `--checkpoint` 切换到已训练的 LeRobot 策略。也可以在项目配置中设置 `policy.checkpoint`。

```bash
python scripts/test_elfin_control.py \
  --app_config=configs/real_training_vm_smoke.yaml \
  --checkpoint=outputs/train/vm_black_smoke_20261019/checkpoints/last/pretrained_model \
  --duration_s=0.2
```

`--app_config` 默认取录制配置的 `robot.scene_config`。显式指定时从所选项目配置读取策略和任务参数，机器人仍采用录制配置里的安装外参。八字用 `policy.action_horizon` 构造动作块，checkpoint 由模型自身结构决定块长。两者每块均执行 `policy.action_steps` 个点，动作频率为录制的 `robot.fps` 除以 `policy.action_stride`。可用 `--task` 覆盖任务文本。

两种策略都输出世界系绝对 TCP 动作块。checkpoint 由现有 postprocessor 解码相对动作。测试入口提前预测下一块，按动作点频率提交目标。测试和官方 rollout 共用机器人的执行线程，以 100 Hz 插值位置和姿态，推理晚到时保持上一块末点。测试结束前等待最后一个目标下发完成。主动控制只在发送前检查示教器 Base 系目标，X 为 −400～200 mm，Y 为 200～450 mm，Z 不低于 −15 mm。越界时拒发该目标并停止 ServoP。

原有 LeRobot 采集默认使用被动模式。

## 官方 `lerobot-rollout`

在当前锁定的 LeRobot 0.6.0 环境中，Elfin 插件支持 `base` 策略和 `sync` 推理。把 `ELFIN_HOST` 和
`CHECKPOINT` 设为本次使用的控制器地址和模型路径。以下命令会在收到第一个动作后启动 ServoP。

```bash
cameras=$(python -c 'import json, yaml; from pathlib import Path; print(json.dumps(yaml.safe_load(Path("configs/real_record.yaml").read_text())["robot"]["cameras"]))')
lerobot-rollout \
  --strategy.type=base --inference.type=sync \
  --robot.type=elfin_welding --robot.active=true \
  --robot.host="$ELFIN_HOST" --robot.cameras="$cameras" \
  --policy.path="$CHECKPOINT" \
  --task="Weld along the straight butt seam between the two flat plates." \
  --fps=30 --duration=10 --play_sounds=false
```

`--fps` 按训练数据的动作点频率设置。`--interpolation_multiplier` 使用默认值 1，Elfin 在机器人侧独立以
100 Hz 插值和下发。焊接 checkpoint 仍按 `n_action_steps` 执行前缀，同一块共享一个 TCP 锚点。
退出时停止 ServoP 并断开连接。可先用上文的 `test_elfin_control.py` 做只读检查。
