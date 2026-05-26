#!/usr/bin/env python3
"""
标注后的 PCD -> SemanticKITTI .label 批量转换（基于 selected_frames_select.txt）

读取每个标注 PCD：要求包含 fields x y z intensity label（label 为 uint32 自定义 ID）。
通过 mapping.yaml 把自定义 ID 映射为 SemanticKITTI raw ID（10/40/50...），
按 KITTI 约定写入 <root>/sequences/<seq>/labels/<stem>.label（uint32，每点一个）。

支持 ASCII 和 binary 两种 PCD 编码。

校验：原始 .bin 点数 == PCD 点数 == label 点数；不一致直接报错。

Usage:
    python tools/pcd_to_label.py \\
        --list selected_frames_select.txt \\
        --root /root/autodl-tmp/16lidar_v1.0 \\
        --pcd_dir /root/autodl-tmp/16lidar_v1.0/pcd_labeled \\
        --mapping tools/label_mapping_16lidar.yaml
"""

import argparse
import os
import struct
import sys
import numpy as np
import yaml


# ────────────────────────────────────────────────────────────
# PCD parser (header + ASCII or binary body)
# ────────────────────────────────────────────────────────────

def _parse_header(lines):
    h = {"fields": [], "size": [], "type": [], "count": [],
         "width": 0, "height": 0, "points": 0, "data": "ascii"}
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        key = toks[0].upper()
        if key == "FIELDS":
            h["fields"] = toks[1:]
        elif key == "SIZE":
            h["size"] = [int(x) for x in toks[1:]]
        elif key == "TYPE":
            h["type"] = toks[1:]
        elif key == "COUNT":
            h["count"] = [int(x) for x in toks[1:]]
        elif key == "WIDTH":
            h["width"] = int(toks[1])
        elif key == "HEIGHT":
            h["height"] = int(toks[1])
        elif key == "POINTS":
            h["points"] = int(toks[1])
        elif key == "DATA":
            h["data"] = toks[1].lower()
            return h
    raise ValueError("PCD header missing DATA line")


def _np_dtype(t: str, sz: int):
    t = t.upper()
    if t == "F":
        return {4: np.float32, 8: np.float64}[sz]
    if t == "U":
        return {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[sz]
    if t == "I":
        return {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[sz]
    raise ValueError(f"unknown PCD type {t}{sz}")


def read_pcd(path: str) -> dict:
    """Returns dict {field_name: ndarray(N,)}."""
    with open(path, "rb") as f:
        # header is text; read line by line until DATA
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF in header")
            try:
                txt = line.decode("ascii", errors="strict")
            except UnicodeDecodeError:
                raise ValueError(f"{path}: non-ascii in header")
            header_lines.append(txt)
            if txt.strip().upper().startswith("DATA"):
                break

        h = _parse_header(header_lines)
        n = h["points"] if h["points"] > 0 else h["width"] * max(1, h["height"])
        if not (len(h["fields"]) == len(h["size"]) == len(h["type"]) == len(h["count"])):
            raise ValueError(f"{path}: inconsistent header lengths")
        if any(c != 1 for c in h["count"]):
            raise NotImplementedError(f"{path}: COUNT > 1 fields not supported")

        out = {}
        if h["data"] == "ascii":
            arr = np.loadtxt(f, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[0] != n:
                raise ValueError(f"{path}: ASCII rows {arr.shape[0]} != POINTS {n}")
            for i, name in enumerate(h["fields"]):
                dt = _np_dtype(h["type"][i], h["size"][i])
                out[name] = arr[:, i].astype(dt, copy=False)
        elif h["data"] == "binary":
            dtypes = [(name, _np_dtype(h["type"][i], h["size"][i]))
                      for i, name in enumerate(h["fields"])]
            structured = np.frombuffer(f.read(), dtype=np.dtype(dtypes), count=n)
            for name, _ in dtypes:
                out[name] = np.array(structured[name])
        elif h["data"] == "binary_compressed":
            raise NotImplementedError("binary_compressed PCD not supported; "
                                      "ask annotator to export as ascii or binary")
        else:
            raise ValueError(f"{path}: unknown DATA mode {h['data']}")

        return out


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def parse_list(path: str):
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seq, stem = line.split("/", 1)
            out.append((seq, stem))
    return out


def build_lut(mapping_path: str) -> np.ndarray:
    """Build a uint32 LUT [custom_id -> raw_id]; unmapped IDs default to 0."""
    with open(mapping_path, "r") as f:
        cfg = yaml.safe_load(f)
    table = cfg.get("custom_to_raw", {})
    if not isinstance(table, dict) or len(table) == 0:
        raise ValueError(f"{mapping_path}: 'custom_to_raw' is empty or missing")
    max_id = int(max(int(k) for k in table.keys()))
    lut = np.zeros(max_id + 1, dtype=np.uint32)
    for k, v in table.items():
        ki = int(k)
        vi = int(v)
        if ki < 0 or vi < 0 or vi > 0xFFFF:
            raise ValueError(f"mapping entry out of range: {k}->{v}")
        lut[ki] = vi
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="selected_frames_select.txt")
    ap.add_argument("--root", required=True, help="dataset root containing sequences/")
    ap.add_argument("--pcd_dir", required=True, help="dir of labeled PCD (mirrors seq/stem)")
    ap.add_argument("--mapping", required=True, help="custom_to_raw mapping yaml")
    ap.add_argument("--label_field", default="label", help="PCD field name holding the class id")
    ap.add_argument("--strict_count", action="store_true",
                    help="fail if PCD point count != bin point count")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing .label")
    args = ap.parse_args()

    frames = parse_list(args.list)
    lut = build_lut(args.mapping)
    max_custom = lut.shape[0] - 1
    print(f"[pcd->label] frames={len(frames)}  max_custom_id={max_custom}")

    n_done = 0
    n_skip = 0
    n_miss = 0
    n_oor  = 0

    for seq, stem in frames:
        pcd_path = os.path.join(args.pcd_dir, seq, f"{stem}.pcd")
        bin_path = os.path.join(args.root, "sequences", seq, "velodyne", f"{stem}.bin")
        lbl_dir  = os.path.join(args.root, "sequences", seq, "labels")
        lbl_path = os.path.join(lbl_dir, f"{stem}.label")

        if not os.path.isfile(pcd_path):
            print(f"[MISS-PCD] {pcd_path}", file=sys.stderr)
            n_miss += 1
            continue
        if not os.path.isfile(bin_path):
            print(f"[MISS-BIN] {bin_path}", file=sys.stderr)
            n_miss += 1
            continue
        if os.path.isfile(lbl_path) and not args.overwrite:
            n_skip += 1
            continue

        # validate point count
        bin_n = os.path.getsize(bin_path) // (4 * 4)
        fields = read_pcd(pcd_path)
        if args.label_field not in fields:
            raise ValueError(f"{pcd_path}: missing field '{args.label_field}'; "
                             f"available={list(fields.keys())}")
        pcd_n = fields[args.label_field].shape[0]
        if bin_n != pcd_n:
            msg = f"{pcd_path}: PCD points {pcd_n} != BIN points {bin_n}"
            if args.strict_count:
                raise ValueError(msg)
            else:
                print(f"[WARN] {msg} (use --strict_count to fail)", file=sys.stderr)

        custom = fields[args.label_field].astype(np.int64, copy=False)
        oor = (custom < 0) | (custom > max_custom)
        if oor.any():
            n_oor += int(oor.sum())
            custom = np.where(oor, 0, custom)
        raw = lut[custom].astype(np.uint32, copy=False)

        os.makedirs(lbl_dir, exist_ok=True)
        raw.tofile(lbl_path)
        n_done += 1
        if n_done % 50 == 0:
            print(f"  ... done {n_done}/{len(frames)}")

    print(f"[pcd->label] done={n_done} skip(existing)={n_skip} "
          f"miss={n_miss} out_of_range_pts={n_oor} total={len(frames)}")


if __name__ == "__main__":
    main()
