#!/usr/bin/env python3
"""
RKNN Simulator 全流程推理 + 保存上色 PCD

用法（PC with rknn-toolkit2）:
  conda run -n rknn python3 save_colored_pcd.py
输出:
  rknn_sim_colored.pcd — 可用 pcl_viewer 或 CloudCompare 查看
"""

import os, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(SCRIPT_DIR)  # CAFFNet/
LOCALIZATION_DIR = os.path.dirname(MODULE_DIR)  # modules/localization/
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(LOCALIZATION_DIR, 'caf_node', 'scripts'))

from convert_to_rknn import _project_range_image, _project_polar_bev
from preprocessor import CPUHead, Preprocessor
import yaml
import onnx

ONNX_PATH    = os.path.join(SCRIPT_DIR, 'backbone.onnx')
BIN_PATH     = os.path.join(SCRIPT_DIR, '000000.bin')
CFG_PATH     = os.path.join(SCRIPT_DIR, 'config.yaml')
HEAD_NPZ     = os.path.join(SCRIPT_DIR, 'head_weights.npz')
OUT_PCD      = os.path.join(SCRIPT_DIR, 'rknn_sim_colored.pcd')
TEMP_DIR     = os.path.join(SCRIPT_DIR, 'temp_analysis')

os.makedirs(TEMP_DIR, exist_ok=True)

# 10 类颜色（SemanticKITTI 风格）
CLASS_COLORS = np.array([
    [  0,   0, 142],   # 0: car - blue
    [128,  64, 128],   # 1: road - purple
    [ 70,  70,  70],   # 2: building - gray
    [255, 128,   0],   # 3: fence - orange
    [255, 255,   0],   # 4: curb - yellow
    [ 70, 130, 180],   # 5: vegetation - steel blue
    [ 64,  50,  38],   # 6: trunk - dark brown
    [152,  16, 152],   # 7: terrain - magenta
    [255,   0,   0],   # 8: pole - red
    [153, 153, 153],   # 9: other-object - light gray
], dtype=np.uint8)

CLASS_NAMES = ['car', 'road', 'building', 'fence', 'curb',
               'vegetation', 'trunk', 'terrain', 'pole', 'other-object']

# ── 1. 加载配置和数据 ──────────────────────────────────────────────
print("=" * 60)
print("Step 1: 加载数据")
print("=" * 60)

cfg = yaml.safe_load(open(CFG_PATH))
pre = Preprocessor(cfg)

pts = np.fromfile(BIN_PATH, dtype=np.float32).reshape(-1, 4)
pts = pre.filter_points(pts)
print(f"  点云: {len(pts)} 点 (过滤后)")

# 用 Preprocessor 获取完整投影（包含 row/col 索引）
(rv_img, pb_img,
 rv_row, rv_col,
 pb_row, pb_col,
 xyz, intensity) = pre.prepare(pts)

print(f"  rv_img: {rv_img.shape}")
print(f"  pb_img: {pb_img.shape}")
print(f"  xyz:    {xyz.shape}")

# ── 2. RKNN Simulator Backbone ────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: RKNN Simulator Backbone")
print("=" * 60)

# Resize 修复
m_onnx = onnx.load(ONNX_PATH)
n_fixed = 0
for node in m_onnx.graph.node:
    if node.op_type != 'Resize':
        continue
    for attr in node.attribute:
        if attr.name == 'mode':
            attr.s = b'nearest'
        elif attr.name == 'coordinate_transformation_mode':
            attr.s = b'asymmetric'
        elif attr.name == 'nearest_mode':
            attr.s = b'floor'
    n_fixed += 1
FIXED_ONNX = os.path.join(TEMP_DIR, '_pcd_fixed.onnx')
onnx.save(m_onnx, FIXED_ONNX)

from rknn.api import RKNN
rknn = RKNN(verbose=False)
assert rknn.config(target_platform='rk3588', optimization_level=0,
                   float_dtype='float16') == 0
assert rknn.load_onnx(model=FIXED_ONNX) == 0
assert rknn.build(do_quantization=False) == 0
assert rknn.init_runtime() == 0
print("  RKNN Simulator 就绪")

outputs = rknn.inference(inputs=[rv_img, pb_img], data_format='nchw')
rv_feat = np.array(outputs[0], dtype=np.float32)
pb_feat = np.array(outputs[1], dtype=np.float32)
rknn.release()
print(f"  rv_feat: {rv_feat.shape}  [{rv_feat.min():.4f}, {rv_feat.max():.4f}]")
print(f"  pb_feat: {pb_feat.shape}  [{pb_feat.min():.4f}, {pb_feat.max():.4f}]")

# ── 3. CPU Head ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: CPU Head (grid_sample + MLP)")
print("=" * 60)

head = CPUHead(HEAD_NPZ)
pred_labels = head.infer(
    rv_feat[0], pb_feat[0],
    rv_row, rv_col,
    pb_row, pb_col,
    xyz, intensity,
)
print(f"  pred_labels: {pred_labels.shape}")
for cls_id in range(10):
    count = (pred_labels == cls_id).sum()
    ratio = count / len(pred_labels) * 100
    print(f"    {CLASS_NAMES[cls_id]:>12s} (class {cls_id}): {count:6d}  ({ratio:5.1f}%)")

# ── 4. 保存上色 PCD ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: 保存 PCD")
print("=" * 60)

# RGB 颜色映射
colors = CLASS_COLORS[pred_labels]  # (N, 3)

# 写入 ASCII PCD 格式
with open(OUT_PCD, 'w') as f:
    f.write("# .PCD v0.7 - Point Cloud Data file format\n")
    f.write("VERSION 0.7\n")
    f.write("FIELDS x y z rgb\n")
    f.write("SIZE 4 4 4 4\n")
    f.write("TYPE F F F U\n")
    f.write("COUNT 1 1 1 1\n")
    f.write(f"WIDTH {len(xyz)}\n")
    f.write("HEIGHT 1\n")
    f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
    f.write(f"POINTS {len(xyz)}\n")
    f.write("DATA ascii\n")

    for i in range(len(xyz)):
        x, y, z = xyz[i]
        r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
        rgb_packed = (r << 16) | (g << 8) | b
        f.write(f"{x:.6f} {y:.6f} {z:.6f} {rgb_packed}\n")

print(f"  保存: {OUT_PCD}")
print(f"  点数: {len(xyz)}")
print("\n=== DONE ===")
print(f"  用 pcl_viewer {OUT_PCD}  或 CloudCompare 查看")
