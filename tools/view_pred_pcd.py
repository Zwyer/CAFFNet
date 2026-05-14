#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看 infer_semantickitti.py 导出的预测 PCD。

支持：
  - ASCII / binary PCD
  - 按字段着色：pred / gt / conf / intensity / z
  - 从单文件或目录自动选择

示例：
  python tools/view_pred_pcd.py --pcd /root/autodl-tmp/pred_pcd_val/01/000123.pcd
  python tools/view_pred_pcd.py --pcd_dir /root/autodl-tmp/pred_pcd_val --seq 01 --index 0 --color_by pred
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np


def _parse_header(lines: List[str]) -> Dict:
    h = {
        "fields": [],
        "size": [],
        "type": [],
        "count": [],
        "width": 0,
        "height": 0,
        "points": 0,
        "data": "ascii",
    }
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
    raise ValueError("PCD header missing DATA")


def _np_dtype(t: str, sz: int):
    t = t.upper()
    if t == "F":
        return {4: np.float32, 8: np.float64}[sz]
    if t == "U":
        return {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[sz]
    if t == "I":
        return {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[sz]
    raise ValueError(f"unknown dtype: {t}{sz}")


def read_pcd(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: EOF in header")
            txt = line.decode("ascii", errors="strict")
            header_lines.append(txt)
            if txt.strip().upper().startswith("DATA"):
                break

        h = _parse_header(header_lines)
        if not (len(h["fields"]) == len(h["size"]) == len(h["type"]) == len(h["count"])):
            raise ValueError(f"{path}: inconsistent header lengths")
        if any(c != 1 for c in h["count"]):
            raise NotImplementedError(f"{path}: COUNT > 1 not supported")

        n = h["points"] if h["points"] > 0 else h["width"] * max(1, h["height"])
        out: Dict[str, np.ndarray] = {}
        if h["data"] == "ascii":
            arr = np.loadtxt(f, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[0] != n:
                raise ValueError(f"{path}: rows {arr.shape[0]} != points {n}")
            for i, name in enumerate(h["fields"]):
                out[name] = arr[:, i].astype(_np_dtype(h["type"][i], h["size"][i]), copy=False)
        elif h["data"] == "binary":
            dtypes = [(name, _np_dtype(h["type"][i], h["size"][i])) for i, name in enumerate(h["fields"])]
            rec = np.frombuffer(f.read(), dtype=np.dtype(dtypes), count=n)
            for name, _ in dtypes:
                out[name] = np.array(rec[name])
        else:
            raise NotImplementedError(f"{path}: DATA={h['data']} not supported")
        return out


def _palette_u8(n: int = 256) -> np.ndarray:
    # High-contrast base palette (maximally separated hues), then repeat.
    base = np.array([
        [230,  25,  75],  # red
        [ 60, 180,  75],  # green
        [255, 225,  25],  # yellow
        [  0, 130, 200],  # blue
        [245, 130,  48],  # orange
        [145,  30, 180],  # purple
        [ 70, 240, 240],  # cyan
        [240,  50, 230],  # magenta
        [210, 245,  60],  # lime
        [250, 190, 190],  # pink
        [  0, 128, 128],  # teal
        [230, 190, 255],  # lavender
        [170, 110,  40],  # brown
        [255, 250, 200],  # beige
        [128,   0,   0],  # maroon
        [170, 255, 195],  # mint
        [128, 128,   0],  # olive
        [255, 215, 180],  # apricot
        [  0,   0, 128],  # navy
        [128, 128, 128],  # gray
    ], dtype=np.uint8)
    pal = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        pal[i] = base[i % len(base)]
    # Common background/ignore class
    pal[0] = np.array([30, 30, 30], dtype=np.uint8)
    return pal


def _norm01(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return np.zeros_like(v, dtype=np.float32)
    return (v - vmin) / (vmax - vmin)


def _colorize_scalar(v01: np.ndarray) -> np.ndarray:
    """
    High-contrast pseudo-color map (blue -> cyan -> green -> yellow -> red).
    v01: float in [0, 1], shape (N,)
    """
    anchors = np.array([
        [0.00,  20,  30, 160],
        [0.25,  30, 180, 235],
        [0.50,  40, 220, 120],
        [0.75, 250, 220,  60],
        [1.00, 220,  30,  30],
    ], dtype=np.float32)
    x = np.clip(v01.astype(np.float32, copy=False), 0.0, 1.0)
    rgb = np.zeros((x.shape[0], 3), dtype=np.float32)
    for c in range(3):
        rgb[:, c] = np.interp(x, anchors[:, 0], anchors[:, c + 1])
    return rgb / 255.0


def build_colors(fields: Dict[str, np.ndarray], color_by: str) -> np.ndarray:
    n = fields["x"].shape[0]
    if color_by in ("pred", "gt") and color_by in fields:
        cls = fields[color_by].astype(np.int64, copy=False)
        cls = np.clip(cls, 0, 255)
        pal = _palette_u8(256)
        return (pal[cls] / 255.0).astype(np.float32)

    if color_by in ("conf", "intensity") and color_by in fields:
        x = _norm01(fields[color_by])
        return _colorize_scalar(x).astype(np.float32)

    if color_by == "z":
        z = _norm01(fields["z"])
        return _colorize_scalar(z).astype(np.float32)

    # fallback
    return np.full((n, 3), 0.8, dtype=np.float32)


def choose_pcd(args) -> str:
    if args.pcd:
        return args.pcd
    if not args.pcd_dir:
        raise ValueError("need --pcd or --pcd_dir")
    root = Path(args.pcd_dir)
    seq_dir = root / args.seq if args.seq else root
    if not seq_dir.exists():
        raise FileNotFoundError(f"not found: {seq_dir}")
    files = sorted(str(p) for p in seq_dir.rglob("*.pcd"))
    if not files:
        raise FileNotFoundError(f"no .pcd under: {seq_dir}")
    idx = max(0, min(args.index, len(files) - 1))
    return files[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", default=None, help="single pcd file")
    ap.add_argument("--pcd_dir", default=None, help="pcd root dir")
    ap.add_argument("--seq", default=None, help="optional seq filter, e.g. 01")
    ap.add_argument("--index", type=int, default=0, help="index in sorted pcd list")
    ap.add_argument("--color_by", default="pred",
                    choices=["pred", "gt", "conf", "intensity", "z"],
                    help="color source")
    ap.add_argument("--point_size", type=float, default=2.0)
    ap.add_argument("--bg", default="black", choices=["black", "white"])
    args = ap.parse_args()

    pcd_path = choose_pcd(args)
    fields = read_pcd(pcd_path)
    for k in ("x", "y", "z"):
        if k not in fields:
            raise ValueError(f"{pcd_path}: missing field '{k}'")

    xyz = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float32, copy=False)
    colors = build_colors(fields, args.color_by)
    if colors.shape[0] != xyz.shape[0]:
        colors = np.full((xyz.shape[0], 3), 0.8, dtype=np.float32)

    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("open3d not installed. install with: pip install open3d") from e

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"PCD Viewer - {os.path.basename(pcd_path)}", width=1600, height=900)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    if args.bg == "black":
        opt.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    vis.run()
    vis.destroy_window()

    print(f"[VIEW] file: {pcd_path}")
    print(f"[VIEW] points: {xyz.shape[0]}")
    print(f"[VIEW] color_by: {args.color_by}")
    print(f"[VIEW] fields: {sorted(fields.keys())}")


if __name__ == "__main__":
    main()
