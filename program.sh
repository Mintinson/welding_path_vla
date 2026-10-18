rclone copy \
    datasets/pipe_top_raw_v2 \
    gdrive:welding_path_vla/datasets/pipe_top_raw_v2 \
    -P \
    --transfers 4 \
    --checkers 8

python scripts/deploy_simulation_policy.py --config_path=configs/traj_vla_qwen_a100.yaml \
    --policy.checkpoint=outputs/train_v2/traj_vla_qwen_self_without_dense/checkpoints/last/pretrained_model \
    --deployment.episodes=5 \
    --deployment.output_root=outputs/deploy/traj_vla_qwen_self_without_dense

accelerate launch --num_processes=2 scripts/train_policy.py --config_path=configs/traj_vla_qwen_a100.yaml --training.output_dir=outputs/train_v2/trajvla_qwen_self_withx_dense_prompt

python scripts/deploy_simulation_policy.py \
  --config_path=configs/traj_vla_qwen_a100.yaml \
  --policy.checkpoint=outputs/train_v2/traj_vla_qwen_l_joint/checkpoints/020000/pretrained_model \
  --deployment.include_tasks='["l_joint"]'

python scripts/export_lerobot.py \
  --config_path=configs/base.yaml \
  --dataset=datasets/raw_v3/pipe_top \
  --output=datasets/lerobot/weldpath_lerobot_relative_v4 \
  --lerobot_export.video_passthrough=false \
  --lerobot_export.video_codec=libsvtav1 \
  --lerobot_export.video_quality=30 \
  --lerobot_export.video_preset=12 \
  --lerobot_export.streaming_encoding=true \
  --lerobot_export.parallel_video_encoding=true \
  --lerobot_export.encoder_threads=8 \
  --lerobot_export.encoder_queue_maxsize=256 \
  --lerobot_export.incremental=true

python scripts/export_lerobot.py \
  --config_path=configs/base.yaml \
  --dataset=datasets/raw_v3/trihedral_vertical \
  --output=datasets/lerobot/weldpath_lerobot_relative_v5 \
  --lerobot_export.video_passthrough=false \
  --lerobot_export.video_codec=libsvtav1 \
  --lerobot_export.video_quality=30 \
  --lerobot_export.video_preset=12 \
  --lerobot_export.streaming_encoding=true \
  --lerobot_export.parallel_video_encoding=true \
  --lerobot_export.encoder_threads=8 \
  --lerobot_export.encoder_queue_maxsize=1024 \
  --lerobot_export.incremental=true


要中断训练，优先在启动训练的终端按 Ctrl+C。如果无法访问那个终端，可以向 accelerate 主进程发送中断信号：
```kill -INT 1155229```
不要用 kill -9，它会强制终止。注意：当前最近保存的 checkpoint 是 040001，下一次定期保存预计在 60000 步；如果在此之前中断，恢复时会从最近已保存的 checkpoint 继续，40001 之后尚未保存的进度不会保留

accelerate launch --num_processes=2 scripts/train_policy.py \
  --config_path=configs/vla_adapter_flow.yaml \
  --training.output_dir=outputs/train_v2/vla_adapter_flow_curve_plate  \
  --training.include_tasks='["curve_plate"]'


accelerate launch --num_processes=2 scripts/train_policy.py \
  --config_path=configs/traj_vla_qwen_a100.yaml \
  --training.output_dir=outputs/train_v2/traj_vla_qwen_aug_curve_plate  \
  --training.image_aug=true \
  --training.include_tasks='["curve_plate"]'