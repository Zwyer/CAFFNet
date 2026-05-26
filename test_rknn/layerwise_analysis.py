#!/usr/bin/env python3
"""
逐层精度分析：用 RKNN Toolkit2 的 accuracy_analysis 对比 ONNX vs RKNN
每一层的输出差异，定位问题算子。

用法（PC with rknn-toolkit2）:
  conda run -n rknn python3 layerwise_analysis.py

输出：
  snapshot/ 目录下包含每层的误差报告和 CSV 汇总
"""

import os, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from convert_to_rknn import _project_range_image, _project_polar_bev
import yaml
import onnx

ONNX_PATH = os.path.join(SCRIPT_DIR, 'backbone.onnx')
BIN_PATH  = os.path.join(SCRIPT_DIR, '000000.bin')
CFG_PATH  = os.path.join(SCRIPT_DIR, 'config.yaml')
SNAPSHOT_DIR = os.path.join(SCRIPT_DIR, 'snapshot')
TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp_analysis')

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ── 1. 准备真实输入 ──────────────────────────────────────────────
print("=" * 72)
print("Step 1: 准备输入数据")
print("=" * 72)

cfg = yaml.safe_load(open(CFG_PATH))
d = cfg['data']
pts = np.fromfile(BIN_PATH, dtype=np.float32).reshape(-1, 4)
dist = np.linalg.norm(pts[:, :3], axis=1)
pts = pts[(dist > 0.5) & (dist < d['R_max'])]
xyz = pts[:, :3]
rem = np.zeros(len(pts), dtype=np.float32)
rv_in = _project_range_image(xyz, rem, d['rv_H'], d['rv_W'],
                              d['fov_up'], d['fov_down'], d['R_max'], True)
pb_in = _project_polar_bev(xyz, rem, d['pb_H'], d['pb_W'], d['R_max'])

print(f"  rv: {rv_in.shape}  [{rv_in.min():.4f}, {rv_in.max():.4f}]")
print(f"  pb: {pb_in.shape}  [{pb_in.min():.4f}, {pb_in.max():.4f}]")

# 保存为 npy（accuracy_analysis 需要文件路径）
rv_file = os.path.join(TEMP_DIR, 'real_rv.npy')
pb_file = os.path.join(TEMP_DIR, 'real_pb.npy')
np.save(rv_file, rv_in)
np.save(pb_file, pb_in)
print(f"  输入已保存: {rv_file}, {pb_file}")

# ── 2. 修复 Resize（bilinear→nearest） ──────────────────────────
print("\n" + "=" * 72)
print("Step 2: Resize 修复 + 创建临时 ONNX")
print("=" * 72)

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

FIXED_ONNX = os.path.join(TEMP_DIR, 'backbone_fixed_resize.onnx')
onnx.save(m_onnx, FIXED_ONNX)
print(f"  Resize 修复: {n_fixed} 节点")
print(f"  临时 ONNX: {FIXED_ONNX}")

# ── 3. 逐层精度分析 ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("Step 3: RKNN 逐层精度分析 (accuracy_analysis)")
print("=" * 72)

from rknn.api import RKNN

rknn = RKNN(verbose=False)

# config
ret = rknn.config(
    target_platform='rk3588',
    optimization_level=0,
    float_dtype='float16',
    mean_values=[[0]*9, [0]*9],
    std_values=[[1]*9, [1]*9],
)
assert ret == 0, f"config 失败: {ret}"
print("  config OK")

# load ONNX
ret = rknn.load_onnx(model=FIXED_ONNX)
assert ret == 0, f"load_onnx 失败: {ret}"
print("  load_onnx OK")

# build
ret = rknn.build(do_quantization=False)
assert ret == 0, f"build 失败: {ret}"
print("  build OK")

# accuracy_analysis: 逐层对比 ONNX vs RKNN
print("\n  运行 accuracy_analysis (simulator mode)...")
print("  这可能需要几分钟...")
ret = rknn.accuracy_analysis(
    inputs=[rv_file, pb_file],
    output_dir=SNAPSHOT_DIR,
    target=None,  # None = simulator
)
if ret != 0:
    print(f"  [ERROR] accuracy_analysis 失败 ret={ret}")
    rknn.release()
    sys.exit(1)

rknn.release()
print(f"  结果已保存到: {SNAPSHOT_DIR}")

# ── 4. 解析结果 ──────────────────────────────────────────────────
print("\n" + "=" * 72)
print("Step 4: 解析分析结果")
print("=" * 72)

# 查找 snapshot 目录下的文件
snapshot_files = sorted(os.listdir(SNAPSHOT_DIR))
print(f"  文件数: {len(snapshot_files)}")

# 尝试读取 CSV 汇总（如果存在）
csv_files = [f for f in snapshot_files if f.endswith('.csv')]
if csv_files:
    import csv
    for cf in csv_files:
        csv_path = os.path.join(SNAPSHOT_DIR, cf)
        print(f"\n  CSV 汇总: {cf}")
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        if rows:
            print(f"  列: {rows[0]}")
            # 找 Cosine 最小的前 20 行
            print(f"\n  误差最大的前 20 层:")
            print(f"  {'Layer':<35s} {'Cosine':>10s} {'MAE':>12s} {'MaxAE':>12s}")
            print(f"  {'─'*35} {'─'*10} {'─'*12} {'─'*12}")
            # 简单解析（假设列顺序固定）
            for row in rows[1:21]:
                if len(row) >= 3:
                    print(f"  {row[0]:<35s} {row[1]:>10s} {row[2]:>12s}" if len(row) >= 3 else row)
else:
    print("\n  未找到 CSV 汇总文件")
    # 列出所有文件
    for f in snapshot_files[:30]:
        fpath = os.path.join(SNAPSHOT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"    {f}  ({size} bytes)")
    if len(snapshot_files) > 30:
        print(f"    ... 共 {len(snapshot_files)} 个文件")

# ── 5. 重点关注算子类型 ─────────────────────────────────────────
print("\n" + "=" * 72)
print("Step 5: 重点排查")
print("=" * 72)
print("""
以下算子类型在 RK3588 NPU 上精度风险最高：
  1. HardSigmoid (alpha=0.16667) — 查 AAFF gate
  2. ReduceMean / ReduceSum — 跨大维度归约，FP16 累加误差大
  3. Conv 大通道数 — weight 范围检查
  4. Softmax / Sigmoid — 指数运算精度
  5. Resize (nearest) — 我们已改为 nearest，但仍需关注

请在 snapshot 目录中查找这些算子对应的输出层，重点关注 Cosine < 0.99 的层。
""")

print("=== DONE ===")
print(f"详细结果: {SNAPSHOT_DIR}/")
