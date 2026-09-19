"""从 raw 视频原地修复直通导出的相机统计。"""
import argparse
from glob import glob
from pathlib import Path

from welding_path_vla.dataset.repair_video_stats import repair_video_stats


def main() -> None:
    """默认仅计算、验证和保存检查点，显式 --apply 才发布元数据。"""
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        '默认解码 raw 并保存 meta/video-stats-checkpoints，不修改最终元数据。'
        ' --apply 在全部校验后备份并原子替换 episode parquet 和 stats.json。'
        ' 同一命令可中断续跑；原备份 *.before-video-stats 不覆盖。'
        ' 不修改 data parquet 或 MP4。请勿与导出或另一修复进程同时运行。'
        ' 跨文件发布中断后请重跑 --apply，或从备份恢复全部元数据。'))
    parser.add_argument('--dataset', type=Path, default=Path('datasets/lerobot/weldpath_lerobot_relative_v3'))
    parser.add_argument('--raw-glob', default='datasets/raw_v3/*', help='用 source_id 匹配的 raw 根目录 glob')
    parser.add_argument('--workers', type=int, default=1, help='解码进程数（默认 1）')
    parser.add_argument('--apply', action='store_true', help='完整校验后备份并发布最终元数据')
    args = parser.parse_args()
    repair_video_stats(args.dataset, [Path(p) for p in sorted(glob(args.raw_glob)) if Path(p).is_dir()],
                       workers=args.workers, apply=args.apply)


if __name__ == '__main__':
    main()
