#!/usr/bin/env python3
"""
NPU 全流程推理 + 保存上色 PCD（在 RK3588 板端运行）

用法:
  python3 save_colored_pcd_npu.py

输出:
  rknn_npu_colored.pcd
"""

import os, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, '/home/techinao/ros1/catkin_ws/src/RoboNeo_S142/modules/localization/caf_node/scripts')

from preprocessor import CPUHead, Preprocessor
import yaml

CFG_PATH     = os.path.join(SCRIPT_DIR, 'config.yaml')
RKNN_PATH    = os.path.join(SCRIPT_DIR, 'backbone.rknn')
BIN_PATH     = os.path.join(SCRIPT_DIR, '000000.bin')
HEAD_NPZ     = os.path.join(SCRIPT_DIR, 'head_weights.npz')
OUT_PCD      = os.path.join(SCRIPT_DIR, 'rknn_npu_colored.pcd')

CLASS_COLORS = np.array([
    [  0,   0, 142], [128,  64, 128], [ 70,  70,  70],
    [255, 128,   0], [255, 255,   0], [ 70, 130, 180],
    [ 64,  50,  38], [152,  16, 152], [255,   0,   0],
    [153, 153, 153],
], dtype=np.uint8)

CLASS_NAMES = ['car', 'road', 'building', 'fence', 'curb',
               'vegetation', 'trunk', 'terrain', 'pole', 'other-object']

# ── 1. 加载数据 ──────────────────────────────────────────────────
print("Step 1: 加载数据")
cfg = yaml.safe_load(open(CFG_PATH))
pre = Preprocessor(cfg)

pts = np.fromfile(BIN_PATH, dtype=np.float32).reshape(-1, 4)
pts = pre.filter_points(pts)
print(f"  点云: {len(pts)} 点")

(rv_img, pb_img,
 rv_row, rv_col,
 pb_row, pb_col,
 xyz, intensity) = pre.prepare(pts)

# ── 2. NPU Backbone ─────────────────────────────────────────────
print("Step 2: NPU Backbone")

import logging
_saved = dict(logging._levelToName), dict(logging._nameToLevel), \
         logging.root.handlers[:], logging.root.level

from rknnlite.api import RKNNLite

logging._levelToName.clear(); logging._levelToName.update(_saved[0])
logging._nameToLevel.clear(); logging._nameToLevel.update(_saved[1])
logging.root.handlers = _saved[2]; logging.root.level = _saved[3]

rknn = RKNNLite()
assert rknn.load_rknn(RKNN_PATH) == 0
assert rknn.init_runtime() == 0
print("  NPU 就绪")

outputs = rknn.inference(inputs=[rv_img, pb_img], data_format='nchw')
rv_feat = np.array(outputs[0], dtype=np.float32)
pb_feat = np.array(outputs[1], dtype=np.float32)
rknn.release()
print(f"  rv_feat: {rv_feat.shape}  [{rv_feat.min():.4f}, {rv_feat.max():.4f}]")
print(f"  pb_feat: {pb_feat.shape}  [{pb_feat.min():.4f}, {pb_feat.max():.4f}]")

# ── 3. CPU Head ────────────────────────────────────────────────
print("Step 3: CPU Head")
head = CPUHead(HEAD_NPZ)
pred_labels = head.infer(
    rv_feat[0], pb_feat[0],
    rv_row, rv_col,
    pb_row, pb_col,
    xyz, intensity,
)
for cls_id in range(10):
    count = (pred_labels == cls_id).sum()
    print(f"    {CLASS_NAMES[cls_id]:>12s}: {count:6d}  ({count/len(pred_labels)*100:5.1f}%)")

# ── 4. 保存 PCD ────────────────────────────────────────────────
print("Step 4: 保存 PCD")
colors = CLASS_COLORS[pred_labels]
with open(OUT_PCD, 'w') as f:
    f.write("# .PCD v0.7 - Point Cloud Data file format\n")
    f.write("VERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n")
    f.write(f"WIDTH {len(xyz)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n")
    f.write(f"POINTS {len(xyz)}\nDATA ascii\n")
    for i in range(len(xyz)):
        x, y, z = xyz[i]
        r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
        f.write(f"{x:.6f} {y:.6f} {z:.6f} {(r<<16)|(g<<8)|b}\n")

print(f"  保存: {OUT_PCD}")
print("=== DONE ===")
