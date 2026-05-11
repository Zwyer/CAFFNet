#!/usr/bin/env python3
"""
按 selected_frames_select.txt 把 SELECT 选出的 .bin 复制到指定目录。

输入列表每行格式：seq/stem（与 bin_to_pcd / pcd_to_label 一致）。
镜像 SemanticKITTI 目录结构，输出：
    <out_dir>/sequences/<seq>/velodyne/<stem>.bin

如需平铺或软链，请单独再写脚本——这里只做"复制 + 镜像目录"。

Usage:
    python tools/copy_selected_bins.py \\
        --list selected_frames_select.txt \\
        --root /root/autodl-tmp/16lidar_v1.0 \\
        --out  /root/autodl-tmp/16lidar_v1.0/bin_selected
"""

import argparse
import os
import shutil
import sys


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="selected_frames_select.txt")
    ap.add_argument("--root", required=True, help="dataset root containing sequences/")
    ap.add_argument("--out", required=True, help="output dir for bin copies (mirrors sequences/<seq>/velodyne)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing bin in destination")
    args = ap.parse_args()

    frames = parse_list(args.list)
    print(f"[copy-bin] frames in list: {len(frames)}")

    n_done = n_skip = n_miss = 0
    for seq, stem in frames:
        src = os.path.join(args.root, "sequences", seq, "velodyne", f"{stem}.bin")
        dst_dir = os.path.join(args.out, "sequences", seq, "velodyne")
        dst = os.path.join(dst_dir, f"{stem}.bin")

        if not os.path.isfile(src):
            print(f"[MISS] {src}", file=sys.stderr)
            n_miss += 1
            continue
        if os.path.isfile(dst) and not args.overwrite:
            n_skip += 1
            continue

        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, dst)
        n_done += 1
        if n_done % 50 == 0:
            print(f"  ... done {n_done}/{len(frames)}")

    print(f"[copy-bin] done={n_done} skip(existing)={n_skip} miss={n_miss} total={len(frames)}")


if __name__ == "__main__":
    main()
