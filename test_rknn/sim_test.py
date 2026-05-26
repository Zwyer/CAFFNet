#!/usr/bin/env python3
"""
RKNN Simulator 对比：定位问题在编译器还是 NPU 硬件。

Simulator 在 CPU 上运行 RKNN 编译后的 IR，不涉及 NPU 硬件。
- 如果 simulator 输出 Cos~0.67 → 问题在编译器/IR 转换
- 如果 simulator 输出 Cos~0.9999 → 问题在 NPU 硬件/驱动

用法（PC with rknn-toolkit2）:
  conda run -n rknn python3 sim_test.py
"""

import os, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from convert_to_rknn import _project_range_image, _project_polar_bev
import onnxruntime as ort
import yaml

ONNX_PATH = os.path.join(SCRIPT_DIR, 'backbone.onnx')
BIN_PATH  = os.path.join(SCRIPT_DIR, '000000.bin')
CFG_PATH  = os.path.join(SCRIPT_DIR, 'config.yaml')


def load_inputs():
    cfg = yaml.safe_load(open(CFG_PATH))
    d = cfg['data']
    pts = np.fromfile(BIN_PATH, dtype=np.float32).reshape(-1, 4)
    dist = np.linalg.norm(pts[:, :3], axis=1)
    pts = pts[(dist > 0.5) & (dist < d['R_max'])]
    xyz = pts[:, :3]
    rem = np.zeros(len(pts), dtype=np.float32)
    rv = _project_range_image(xyz, rem, d['rv_H'], d['rv_W'],
                               d['fov_up'], d['fov_down'], d['R_max'], True)
    pb = _project_polar_bev(xyz, rem, d['pb_H'], d['pb_W'], d['R_max'])
    return rv, pb


def compare(name, ref, other):
    d = np.abs(ref - other)
    cos = np.sum(ref * other) / (np.sqrt(np.sum(ref**2)) * np.sqrt(np.sum(other**2)) + 1e-8)
    rmse = np.sqrt(np.mean(d**2))
    snr = 10 * np.log10(np.mean(ref**2) / np.mean((ref - other)**2))
    return {'MAE': np.mean(d), 'MaxAE': np.max(d), 'RMSE': rmse, 'SNR_dB': snr, 'Cosine': cos}


print("=" * 72)
print("RKNN Simulator 诊断")
print("=" * 72)

# 1. 加载输入
rv_in, pb_in = load_inputs()
print(f"输入: rv={rv_in.shape}  pb={pb_in.shape}")

# 2. FP32 ONNX 参考
print("\n[1/3] FP32 ONNX 参考推理...")
sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
rv_fp32, pb_fp32 = sess.run(None, {'rv_img': rv_in, 'pb_img': pb_in})
print(f"  rv_feat: [{rv_fp32.min():.4f}, {rv_fp32.max():.4f}] mean={rv_fp32.mean():.4f}")
print(f"  pb_feat: [{pb_fp32.min():.4f}, {pb_fp32.max():.4f}] mean={pb_fp32.mean():.4f}")

# 3. RKNN Simulator (CPU, 不上板)
print("\n[2/3] RKNN Simulator (CPU, 不涉及 NPU)...")

# 先加载 ONNX 并修复 Resize (bilinear→nearest, 避免 "Unkown op target: 0")
import onnx
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
FIXED_ONNX = os.path.join(SCRIPT_DIR, '_sim_fixed.onnx')
onnx.save(m_onnx, FIXED_ONNX)
print(f"  Resize fix: {n_fixed} nodes → {FIXED_ONNX}")

from rknn.api import RKNN

rknn = RKNN(verbose=False)

# 必须: config 在 load_onnx 之前
ret = rknn.config(target_platform='rk3588', optimization_level=0, float_dtype='float16')
if ret != 0:
    print(f"  config 失败: {ret}")
    sys.exit(1)
print("  config OK")

ret = rknn.load_onnx(model=FIXED_ONNX)
if ret != 0:
    print(f"  load_onnx 失败: {ret}")
    sys.exit(1)
print("  load_onnx OK")

ret = rknn.build(do_quantization=False)
if ret != 0:
    print(f"  build 失败: {ret}")
    sys.exit(1)
print("  build OK")

# init_runtime: 无板端时自动走 simulator
print("  init_runtime (simulator)...")
ret = rknn.init_runtime()
if ret != 0:
    print(f"  init_runtime 失败: {ret}")
    sys.exit(1)
print("  init_runtime OK")

print("  running inference (data_format='nchw')...")
outputs = rknn.inference(inputs=[rv_in, pb_in], data_format='nchw')
rv_sim = np.array(outputs[0], dtype=np.float32)
pb_sim = np.array(outputs[1], dtype=np.float32)
rknn.release()

# 清理临时文件
os.unlink(FIXED_ONNX)

print(f"  rv_feat: [{rv_sim.min():.4f}, {rv_sim.max():.4f}] mean={rv_sim.mean():.4f}")
print(f"  pb_feat: [{pb_sim.min():.4f}, {pb_sim.max():.4f}] mean={pb_sim.mean():.4f}")

# 4. 对比
print("\n[3/3] 对比分析")
print("=" * 72)

for name, ref, sim in [('rv_feat', rv_fp32, rv_sim), ('pb_feat', pb_fp32, pb_sim)]:
    m = compare(name, ref[0], sim[0])
    print(f"\n  {name}:")
    print(f"    MAE   : {m['MAE']:.4e}")
    print(f"    MaxAE : {m['MaxAE']:.4e}")
    print(f"    RMSE  : {m['RMSE']:.4e}")
    print(f"    SNR   : {m['SNR_dB']:.1f} dB")
    print(f"    Cosine: {m['Cosine']:.10f}")

    # 逐通道
    print(f"\n    逐通道 Cosine (Cos<0.9 标记):")
    bad_chs = []
    for c in range(ref.shape[1]):
        rc = ref[0, c].ravel()
        sc = sim[0, c].ravel()
        cos_c = np.dot(rc, sc) / (np.linalg.norm(rc) * np.linalg.norm(sc) + 1e-8)
        if cos_c < 0.9:
            bad_chs.append(c)
            print(f"      ch[{c:2d}]: cos={cos_c:.6f}  *** BAD ***")
        elif c < 8:
            print(f"      ch[{c:2d}]: cos={cos_c:.6f}")
    if len(bad_chs) > 0:
        print(f"    异常通道 ({len(bad_chs)}/{ref.shape[1]}): {bad_chs}")

# 5. 结论
print(f"\n{'='*72}")
print("结论:")
print("  如果 Simulator Cosine ~0.9999: 问题在 NPU 硬件/驱动")
print("  如果 Simulator Cosine ~0.67  : 问题在 RKNN 编译器/IR 转换")
print(f"{'='*72}")
