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
  --policy.checkpoint=outputs/train_v2/trajvla_qwen_self_withx_dense_prompt/checkpoints/last/pretrained_model \
  --deployment.output_root=outputs/deploy/trajvla_qwen_self_withx_dense_prompt_scene

pixi run -e data export-lerobot \
  --config_path=configs/base.yaml \
  --datasets='[
    "datasets/raw_v3/curve_plate",
    "datasets/raw_v3/l_joint",
    "datasets/raw_v3/pipe_bottom",
    "datasets/raw_v3/pipe_top",
    "datasets/raw_v3/trihedral_horizontal",
    "datasets/raw_v3/trihedral_vertical"
  ]' \
  --output=datasets/lerobot/weldpath_lerobot_relative_v3_av1 \
  --repo_id=local/weldpath_relative_v3_av1 \
  --lerobot_export.video_passthrough=false \
  --lerobot_export.video_codec=libsvtav1 \
  --lerobot_export.video_quality=30 \
  --lerobot_export.video_preset=12 \
  --lerobot_export.streaming_encoding=true \
  --lerobot_export.parallel_video_encoding=true \
  --lerobot_export.encoder_threads=8 \
  --lerobot_export.encoder_queue_maxsize=256 \