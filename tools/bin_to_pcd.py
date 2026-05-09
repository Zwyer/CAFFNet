#!/usr/bin/env python3
"""
bin -> pcd 批量转换（基于 selected_frames_select.txt）

读取 SELECT 选出的帧列表，把对应的 .bin（KITTI 格式：x,y,z,intensity float32）
转换为 ASCII PCD（x,y,z,intensity），输出到指定目录，文件名保留 seq/stem 结构。

Usage:
    python tools/bin_to_pcd.py \\
        --list selected_frames_select.txt \\
        --root /root/autodl-tmp/16lidar_v1.0 \\
        --out  /root/autodl-tmp/16lidar_v1.0/pcd_for_labeling

输出结构:
    out/
      00/000123.pcd
      00/000456.pcd
      01/000789.pcd
      ...
"""

import argparse
import os
import sys
import numpy as np


def read_bin(path: str) -> np.ndarray:
    pts = np.fromfile(path, dtype=np.float32)
    if pts.size % 4 != 0:
        raise ValueError(f"{path}: size {pts.size} not multiple of 4")
    return pts.reshape(-1, 4)


def write_pcd_ascii(path: str, pts: np.ndarray) -> None:
    n = pts.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA ascii\n"
    )
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, pts, fmt="%.6f %.6f %.6f %.6f")


def parse_list(path: str):
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "/" not in line:
                raise ValueError(f"unexpected line format (expect seq/stem): {line}")
            seq, stem = line.split("/", 1)
            out.append((seq, stem))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="selected_frames_select.txt")
    ap.add_argument("--root", required=True, help="dataset root containing sequences/")
    ap.add_argument("--out",  required=True, help="output dir for PCD files")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing pcd")
    args = ap.parse_args()

    frames = parse_list(args.list)
    print(f"[bin->pcd] frames in list: {len(frames)}")

    os.makedirs(args.out, exist_ok=True)
    n_done = 0
    n_skip = 0
    n_miss = 0

    for seq, stem in frames:
        bin_path = os.path.join(args.root, "sequences", seq, "velodyne", f"{stem}.bin")
        out_dir  = os.path.join(args.out, seq)
        out_path = os.path.join(out_dir, f"{stem}.pcd")

        if not os.path.isfile(bin_path):
            print(f"[MISS] {bin_path}", file=sys.stderr)
            n_miss += 1
            continue

        if os.path.isfile(out_path) and not args.overwrite:
            n_skip += 1
            continue

        os.makedirs(out_dir, exist_ok=True)
        pts = read_bin(bin_path)
        write_pcd_ascii(out_path, pts)
        n_done += 1
        if n_done % 50 == 0:
            print(f"  ... done {n_done}/{len(frames)}")

    print(f"[bin->pcd] done={n_done} skip(existing)={n_skip} miss={n_miss} total={len(frames)}")


if __name__ == "__main__":
    main()
