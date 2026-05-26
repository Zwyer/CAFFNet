#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 ROS bag 中 PointCloud2 的 intensity 分布，辅助判断是否需要 --intensity_clip_01。

读取后端优先级：
  1) rosbags (纯 Python, 不依赖 ROS 环境)
  2) rosbag  (ROS1 环境)

示例：
python tools/check_rosbag_intensity_stats.py \
  --bag /data/mid360.bag \
  --topic /lslidar_point_cloud \
  --sample_every 10 \
  --max_frames 200
"""

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _load_backend():
    """返回 (backend_name, iterator_factory)。"""
    try:
        from rosbags.highlevel import AnyReader  # type: ignore

        def _iter_messages(bag_path: str, topic: str):
            with AnyReader([Path(bag_path)]) as reader:
                conns = [c for c in reader.connections if c.topic == topic]
                if not conns:
                    topics = sorted({c.topic for c in reader.connections})
                    raise ValueError(f"bag 中不存在话题: {topic}，可用话题: {topics}")
                for conn, t_ns, raw in reader.messages(connections=conns):
                    msg = reader.deserialize(raw, conn.msgtype)
                    yield msg, int(t_ns)

        return "rosbags", _iter_messages
    except Exception:
        pass

    try:
        import rosbag  # type: ignore

        def _iter_messages(bag_path: str, topic: str):
            with rosbag.Bag(bag_path, "r") as bag:
                info = bag.get_type_and_topic_info()
                if topic not in info.topics:
                    topics = sorted(info.topics.keys())
                    raise ValueError(f"bag 中不存在话题: {topic}，可用话题: {topics}")
                for _topic, msg, t in bag.read_messages(topics=[topic]):
                    t_ns = int(getattr(t, "secs", 0)) * 1_000_000_000 + int(getattr(t, "nsecs", 0))
                    yield msg, t_ns

        return "rosbag", _iter_messages
    except Exception:
        pass

    raise RuntimeError(
        "未找到可用 bag 读取后端。\n"
        "推荐安装（无需 ROS 环境）：pip install rosbags\n"
        "或使用 ROS1 环境中的 rosbag。"
    )


def _dtype_for_field(datatype: int):
    # PointField: INT8=1 UINT8=2 INT16=3 UINT16=4 INT32=5 UINT32=6 FLOAT32=7 FLOAT64=8
    if datatype == 1:
        return np.int8
    if datatype == 2:
        return np.uint8
    if datatype == 3:
        return np.int16
    if datatype == 4:
        return np.uint16
    if datatype == 5:
        return np.int32
    if datatype == 6:
        return np.uint32
    if datatype == 7:
        return np.float32
    if datatype == 8:
        return np.float64
    return None


def _build_struct_dtype(msg) -> np.dtype:
    names, formats, offsets = [], [], []
    fields = sorted(msg.fields, key=lambda x: x.offset)
    for f in fields:
        base = _dtype_for_field(int(f.datatype))
        if base is None:
            continue
        dt = np.dtype(base) if int(f.count) == 1 else np.dtype((base, int(f.count)))
        names.append(str(f.name))
        formats.append(dt)
        offsets.append(int(f.offset))

    if not names:
        raise ValueError("PointCloud2 fields 为空或类型不支持")

    return np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": int(msg.point_step),
    })


def _pick_intensity_field(dtype_names):
    for name in ("intensity", "reflectivity", "i"):
        if name in dtype_names:
            return name
    return None


def _msg_data_as_bytes(msg) -> bytes:
    data = msg.data
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    try:
        return bytes(data)
    except Exception as e:
        raise ValueError(f"无法读取 PointCloud2.data: {type(data)}") from e


def parse_args():
    p = argparse.ArgumentParser(description="Check intensity distribution from PointCloud2 topic")
    p.add_argument("--bag", required=True, help=".bag 文件路径")
    p.add_argument("--topic", default="/lslidar_point_cloud", help="点云话题")
    p.add_argument("--sample_every", type=int, default=1, help="每 N 帧采样 1 帧")
    p.add_argument("--max_frames", type=int, default=300, help="最多统计多少帧，-1=全部")
    p.add_argument("--log_every", type=int, default=20, help="每多少帧打印一次进度")
    return p.parse_args()


def fmt(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "nan"
    return f"{v:.6f}"


def main():
    args = parse_args()
    backend, itermsg = _load_backend()
    print(f"[INFO] backend={backend}")

    all_vals = []

    n_seen = 0
    n_used = 0

    for msg, _t_ns in itermsg(args.bag, args.topic):
        n_seen += 1
        if args.sample_every > 1 and ((n_seen - 1) % args.sample_every != 0):
            continue

        n_points = int(msg.width) * int(msg.height)
        if n_points <= 0:
            continue

        dt = _build_struct_dtype(msg)
        raw = _msg_data_as_bytes(msg)
        arr = np.frombuffer(raw, dtype=dt, count=n_points)

        int_name = _pick_intensity_field(arr.dtype.names)
        if int_name is None:
            raise ValueError(f"未找到 intensity 字段，可用字段: {arr.dtype.names}")

        val = arr[int_name].astype(np.float64, copy=False)
        finite = np.isfinite(val)
        val = val[finite]
        if val.size == 0:
            continue

        all_vals.append(val)
        n_used += 1

        if args.log_every > 0 and (n_used % args.log_every == 0):
            p50 = float(np.percentile(val, 50))
            p99 = float(np.percentile(val, 99))
            print(f"[INFO] seen={n_seen} used={n_used} frame_p50={p50:.4f} frame_p99={p99:.4f}")

        if args.max_frames > 0 and n_used >= args.max_frames:
            break

    if not all_vals:
        print("[WARN] 没有统计到有效 intensity")
        return

    cat = np.concatenate(all_vals, axis=0)
    gmin = float(np.min(cat))
    gmax = float(np.max(cat))
    gmean = float(np.mean(cat))
    gstd = float(np.std(cat))
    gp001 = float(np.percentile(cat, 0.1))
    gp01 = float(np.percentile(cat, 1))
    gp50 = float(np.percentile(cat, 50))
    gp99 = float(np.percentile(cat, 99))
    gp999 = float(np.percentile(cat, 99.9))

    print("\n===== Intensity Global Stats =====")
    print(f"topic:      {args.topic}")
    print(f"frames:     used={n_used}, seen={n_seen}")
    print(f"points:     {cat.size}")
    print(f"min/max:    {fmt(gmin)} / {fmt(gmax)}")
    print(f"mean/std:   {fmt(gmean)} / {fmt(gstd)}")
    print(f"p0.1/p1:    {fmt(gp001)} / {fmt(gp01)}")
    print(f"p50:        {fmt(gp50)}")
    print(f"p99/p99.9:  {fmt(gp99)} / {fmt(gp999)}")

    gt1 = float(np.mean(cat > 1.0) * 100.0)
    lt0 = float(np.mean(cat < 0.0) * 100.0)
    print(f">1 ratio:   {gt1:.4f}%")
    print(f"<0 ratio:   {lt0:.4f}%")

    print("\n===== Suggestion =====")
    if gmin >= 0.0 and gmax <= 1.05 and gt1 < 0.1:
        print("强度基本在 [0,1]，--intensity_clip_01 可开可不开。")
    elif gmin >= 0.0 and gmax <= 255.0:
        print("强度看起来像 [0,255] 量纲，建议先除以 255，再按 [0,1] clip。")
    else:
        print("强度存在越界/异常值，建议启用 --intensity_clip_01，并考虑做分位数归一化。")


if __name__ == "__main__":
    main()
