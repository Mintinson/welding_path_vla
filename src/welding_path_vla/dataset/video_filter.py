"""在 episode 关键帧边界原样过滤视频，避免清理数据时二次有损编码。"""

from fractions import Fraction
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

import av


def remux_video_ranges(
    source: Path,
    destination: Path,
    ranges: list[tuple[int, int]],
    fps: float,
    video_encoder=None,
) -> None:
    """原样保留半开帧区间，兼容官方 delete_episodes 的视频编辑回调。

    Args:
        source: MP4 输入；每个保留区间必须从关键帧开始，无 B 帧/时间戳重排。
        destination: 新 MP4，视频参数和 packet payload 均保持不变。
        ranges: 按原视频时间排序、互不重叠的完整 episode 帧区间。
        fps: 数据集帧率。
        video_encoder: 官方回调参数；原样复制不使用编码器。

    Raises:
        ValueError: 关键帧、时间戳、帧数量或 payload 不满足原样剪辑契约。
    """
    ranges = sorted(ranges)
    if not ranges or any(start < 0 or end <= start for start, end in ranges):
        raise ValueError("invalid video ranges")
    if any(end > next_start for (_, end), (next_start, _) in pairwise(ranges)):
        raise ValueError("overlapping video ranges")
    expected = sum(end - start for start, end in ranges)
    input_digest, written, range_index = sha256(), 0, 0
    with av.open(str(source)) as incoming, av.open(str(destination), "w") as outgoing:
        stream = incoming.streams.video[0]
        copied = outgoing.add_stream_from_template(stream, opaque=True)
        time_base = stream.time_base
        if time_base is None:
            raise ValueError(f"video has no time base: {source}")
        copied.time_base = time_base
        step = Fraction(1, 1) / Fraction(fps).limit_denominator(1000) / time_base
        if step.denominator != 1:
            raise ValueError(f"nonintegral frame time base: {source}")
        source_index = 0
        for packet in incoming.demux(stream):
            if packet.dts is None:
                continue
            if packet.pts != packet.dts or packet.pts != source_index * int(step):
                raise ValueError(f"video requires decoding/reordering: {source}")
            while range_index < len(ranges) and source_index >= ranges[range_index][1]:
                range_index += 1
            if range_index == len(ranges):
                break
            start, end = ranges[range_index]
            if start <= source_index < end:
                if source_index == start and not packet.is_keyframe:
                    raise ValueError(f"episode does not begin with a keyframe: {source}/{start}")
                input_digest.update(bytes(packet))
                packet.pts = packet.dts = written * int(step)
                packet.duration = int(step)
                packet.stream = copied
                outgoing.mux(packet)
                written += 1
            source_index += 1
    output_digest, output_frames = sha256(), 0
    with av.open(str(destination)) as output:
        for packet in output.demux(output.streams.video[0]):
            if packet.dts is not None:
                output_digest.update(bytes(packet))
                output_frames += 1
    if written != expected or output_frames != expected:
        raise ValueError(f"filtered packet count mismatch: {written}/{output_frames}/{expected}")
    if input_digest.digest() != output_digest.digest():
        raise ValueError(f"video payload changed during remux: {destination}")
