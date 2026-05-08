#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ROS bag 自动提取 PointCloud2 点云并保存为 SemanticKITTI 风格 .bin。

读取后端优先级：
  1) rosbags (纯 Python, 不依赖 ROS 环境)
  2) rosbag  (ROS1 环境)

输出格式：
  <out_root>/sequences/<seq>/velodyne/000000.bin

每帧 .bin 内容：
  float32[N, 4]，列为 [x, y, z, intensity]

示例：
  python tools/extract_rosbag_lidar.py \
    --bag /data/mid360.bag \
    --topic /lslidar_point_cloud \
    --out_root /data/mid360_semkitti \
    --seq 00 \
    --z_max 1.0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

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
    # PointField constants:
    # INT8=1 UINT8=2 INT16=3 UINT16=4 INT32=5 UINT32=6 FLOAT32=7 FLOAT64=8
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
    names = []
    formats = []
    offsets = []

    for f in sorted(msg.fields, key=lambda x: x.offset):
        base = _dtype_for_field(int(f.datatype))
        if base is None:
            continue
        if int(f.count) == 1:
            dt = np.dtype(base)
        else:
            dt = np.dtype((base, int(f.count)))
        names.append(str(f.name))
        formats.append(dt)
        offsets.append(int(f.offset))

    if not names:
        raise ValueError("PointCloud2.fields 为空或字段类型不支持")

    return np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": int(msg.point_step),
    })


def _pick_intensity_field(field_names) -> Optional[str]:
    candidates = ["intensity", "reflectivity", "i"]
    s = set(field_names)
    for c in candidates:
        if c in s:
            return c
    return None


def _msg_data_as_bytes(msg) -> bytes:
    data = msg.data
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    try:
        return bytes(data)
    except Exception as e:
        raise ValueError(f"无法读取 PointCloud2.data: {type(data)}") from e


def pointcloud2_to_xyzi(msg) -> np.ndarray:
    n_points = int(msg.width) * int(msg.height)
    if n_points <= 0:
        return np.zeros((0, 4), dtype=np.float32)

    dt = _build_struct_dtype(msg)
    raw = _msg_data_as_bytes(msg)
    arr = np.frombuffer(raw, dtype=dt, count=n_points)

    required = ["x", "y", "z"]
    for k in required:
        if k not in arr.dtype.names:
            raise ValueError(f"PointCloud2 缺少字段: {k}")

    x = arr["x"].astype(np.float32, copy=False)
    y = arr["y"].astype(np.float32, copy=False)
    z = arr["z"].astype(np.float32, copy=False)

    int_name = _pick_intensity_field(arr.dtype.names)
    if int_name is None:
        intensity = np.zeros_like(x, dtype=np.float32)
    else:
        intensity = arr[int_name].astype(np.float32, copy=False)

    pts = np.stack([x, y, z, intensity], axis=1).astype(np.float32, copy=False)
    valid = np.isfinite(pts).all(axis=1)
    if not np.any(valid):
        return np.zeros((0, 4), dtype=np.float32)
    return pts[valid]


def apply_filters(
    pts: np.ndarray,
    z_min: Optional[float],
    z_max: Optional[float],
    r_min: Optional[float],
    r_max: Optional[float],
    intensity_divisor: float,
    intensity_clip_01: bool,
) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts

    mask = np.ones((pts.shape[0],), dtype=bool)

    if z_min is not None:
        mask &= pts[:, 2] >= z_min
    if z_max is not None:
        mask &= pts[:, 2] <= z_max

    if r_min is not None or r_max is not None:
        r = np.linalg.norm(pts[:, :3], axis=1)
        if r_min is not None:
            mask &= r >= r_min
        if r_max is not None:
            mask &= r <= r_max

    out = pts[mask]
    if out.shape[0] > 0 and intensity_divisor > 0 and abs(intensity_divisor - 1.0) > 1e-12:
        out[:, 3] = out[:, 3] / float(intensity_divisor)
    if intensity_clip_01 and out.shape[0] > 0:
        out[:, 3] = np.clip(out[:, 3], 0.0, 1.0)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract PointCloud2 from bag to SemanticKITTI bins")
    p.add_argument("--bag", required=True, help=".bag 文件路径")
    p.add_argument("--topic", default="/lslidar_point_cloud", help="PointCloud2 话题")
    p.add_argument("--out_root", required=True, help="输出根目录（将创建 sequences/<seq>/velodyne）")
    p.add_argument("--seq", default="00", help="序列名（默认 00）")
    p.add_argument("--start_index", type=int, default=0, help="输出帧起始编号")
    p.add_argument("--every_n", type=int, default=1, help="每 N 帧保留 1 帧")
    p.add_argument("--max_frames", type=int, default=-1, help="最多导出帧数，-1 表示全部")

    p.add_argument("--z_min", type=float, default=None, help="高度下限（米），按 z 轴过滤")
    p.add_argument("--z_max", type=float, default=None, help="高度上限（米），按 z 轴过滤")
    p.add_argument("--r_min", type=float, default=0.5, help="距离下限（米），默认 0.5")
    p.add_argument("--r_max", type=float, default=None, help="距离上限（米），可选")
    p.add_argument("--intensity_divisor", type=float, default=1.0,
                   help="强度缩放除数，例如 255 表示 intensity/=255")
    p.add_argument("--intensity_clip_01", action="store_true", help="将 intensity 裁剪到 [0,1]")

    p.add_argument("--overwrite", action="store_true", help="若输出已存在，允许覆盖")
    p.add_argument("--log_every", type=int, default=100, help="每导出多少帧打印一次日志")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    backend, itermsg = _load_backend()
    print(f"[INFO] backend={backend}")

    out_dir = os.path.join(args.out_root, "sequences", args.seq, "velodyne")
    os.makedirs(out_dir, exist_ok=True)

    ts_path = os.path.join(args.out_root, "sequences", args.seq, "timestamps.txt")
    if os.path.exists(ts_path) and not args.overwrite:
        print(f"[ERROR] timestamps 已存在: {ts_path}。如需覆盖请加 --overwrite", file=sys.stderr)
        return 2

    n_seen = 0
    n_saved = 0
    frame_id = int(args.start_index)

    with open(ts_path, "w", encoding="utf-8") as fts:
        for msg, t_ns in itermsg(args.bag, args.topic):
            n_seen += 1

            # 抽帧
            if args.every_n > 1 and ((n_seen - 1) % args.every_n != 0):
                continue

            try:
                pts = pointcloud2_to_xyzi(msg)
            except Exception as e:
                print(f"[WARN] 第 {n_seen} 帧解析失败，跳过: {e}")
                continue

            pts = apply_filters(
                pts,
                z_min=args.z_min,
                z_max=args.z_max,
                r_min=args.r_min,
                r_max=args.r_max,
                intensity_divisor=args.intensity_divisor,
                intensity_clip_01=args.intensity_clip_01,
            )

            out_bin = os.path.join(out_dir, f"{frame_id:06d}.bin")
            if os.path.exists(out_bin) and not args.overwrite:
                print(f"[ERROR] 文件已存在: {out_bin}。如需覆盖请加 --overwrite", file=sys.stderr)
                return 2

            pts.astype(np.float32, copy=False).tofile(out_bin)

            sec = t_ns // 1_000_000_000
            nsec = t_ns % 1_000_000_000
            fts.write(f"{sec}.{nsec:09d}\n")

            n_saved += 1
            frame_id += 1

            if args.log_every > 0 and (n_saved % args.log_every == 0):
                print(f"[INFO] seen={n_seen} saved={n_saved} last_points={pts.shape[0]}")

            if args.max_frames > 0 and n_saved >= args.max_frames:
                break

    print("[DONE] 导出完成")
    print(f"[DONE] bag:      {args.bag}")
    print(f"[DONE] topic:    {args.topic}")
    print(f"[DONE] out_dir:  {out_dir}")
    print(f"[DONE] seen:     {n_seen}")
    print(f"[DONE] saved:    {n_saved}")
    print(f"[DONE] z_range:  [{args.z_min}, {args.z_max}]")
    print(f"[DONE] r_range:  [{args.r_min}, {args.r_max}]")
    print(f"[DONE] intensity_divisor: {args.intensity_divisor}")
    print(f"[DONE] intensity_clip_01: {args.intensity_clip_01}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
