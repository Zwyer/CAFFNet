#!/usr/bin/env python3
"""
在 RK3588 板端运行，对比 RKNN 输出与 ONNX 参考输出的数值差异。
用法：
  python3 diagnose_rknn.py --rknn backbone_xxx.rknn --onnx backbone_fp32.onnx --bin 000000.bin
"""
import argparse, sys, os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

# ── 预处理（复用 convert_to_rknn 里的投影函数）─────────────────────
from convert_to_rknn import _project_range_image, _project_polar_bev
import yaml

def load_inputs(bin_path, cfg):
    d = cfg['data']
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    dist = np.linalg.norm(pts[:, :3], axis=1)
    pts = pts[(dist > 0.5) & (dist < d['R_max'])]
    xyz = pts[:, :3]
    rem = np.zeros(len(pts), dtype=np.float32)
    rv = _project_range_image(xyz, rem, d['rv_H'], d['rv_W'],
                               d['fov_up'], d['fov_down'], d['R_max'], True)
    pb = _project_polar_bev(xyz, rem, d['pb_H'], d['pb_W'], d['R_max'])
    return rv, pb   # (1,9,H,W)

def stat(name, arr):
    a = arr.ravel().astype(np.float32)
    print(f"  {name}: min={a.min():.4f}  max={a.max():.4f}  "
          f"mean={a.mean():.4f}  std={a.std():.4f}  "
          f"|>3|={(np.abs(a)>3).mean()*100:.1f}%  "
          f"nan={np.isnan(a).sum()}  inf={np.isinf(a).sum()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rknn',   required=True)
    ap.add_argument('--onnx',   default=None, help='可选：ONNX 参考模型路径')
    ap.add_argument('--bin',    default=os.path.join(SCRIPT_DIR, '000000.bin'))
    ap.add_argument('--config', default=os.path.join(SCRIPT_DIR, 'config.yaml'))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rv_np, pb_np = load_inputs(args.bin, cfg)
    print(f"输入数据: rv={rv_np.shape} pb={pb_np.shape}")
    stat('rv_input', rv_np)
    stat('pb_input', pb_np)
    print()

    # ── RKNN 推理 ─────────────────────────────────────────────────
    print(f"[RKNN] 加载: {args.rknn}")
    try:
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
        assert rknn.load_rknn(args.rknn) == 0
        assert rknn.init_runtime() == 0
        outputs = rknn.inference(inputs=[rv_np, pb_np])
        rv_rknn = np.array(outputs[0], dtype=np.float32)
        pb_rknn = np.array(outputs[1], dtype=np.float32)
        rknn.release()
        print("[RKNN] 推理成功")
        stat('rv_feat (RKNN)', rv_rknn)
        stat('pb_feat (RKNN)', pb_rknn)
    except Exception as e:
        print(f"[RKNN] 失败: {e}")
        rv_rknn = pb_rknn = None
    print()

    # ── ONNX 参考推理 ─────────────────────────────────────────────
    if args.onnx and os.path.exists(args.onnx):
        print(f"[ONNX] 加载: {args.onnx}")
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
            rv_onnx, pb_onnx = sess.run(None, {'rv_img': rv_np, 'pb_img': pb_np})
            print("[ONNX] 推理成功")
            stat('rv_feat (ONNX)', rv_onnx)
            stat('pb_feat (ONNX)', pb_onnx)

            if rv_rknn is not None:
                print()
                print("[对比] RKNN vs ONNX 差异:")
                diff_rv = np.abs(rv_rknn.astype(np.float32) - rv_onnx.astype(np.float32))
                diff_pb = np.abs(pb_rknn.astype(np.float32) - pb_onnx.astype(np.float32))
                stat('rv_diff', diff_rv)
                stat('pb_diff', diff_pb)

                # 找差异最大的位置
                flat = diff_rv.ravel()
                top_idx = np.argsort(flat)[-5:][::-1]
                print(f"  rv 最大差异的5个像素位置及值:")
                shape = rv_rknn.shape  # (1, C, H, W)
                for idx in top_idx:
                    b = idx // (shape[1]*shape[2]*shape[3])
                    c = (idx // (shape[2]*shape[3])) % shape[1]
                    h = (idx // shape[3]) % shape[2]
                    w = idx % shape[3]
                    print(f"    [b={b},c={c},h={h},w={w}] rknn={rv_rknn[b,c,h,w]:.4f} onnx={rv_onnx[b,c,h,w]:.4f} diff={flat[idx]:.4f}")

                # 检查 RKNN 输出是否有结构性异常（如固定值、全零通道）
                print()
                print("[诊断] RKNN 各通道统计（rv_feat）:")
                for c in range(rv_rknn.shape[1]):
                    ch = rv_rknn[0, c].ravel()
                    is_const = ch.std() < 1e-4
                    flag = " ← 常数通道!" if is_const else ""
                    print(f"  ch{c:2d}: mean={ch.mean():7.4f} std={ch.std():.4f} min={ch.min():.4f} max={ch.max():.4f}{flag}")
        except Exception as e:
            import traceback; traceback.print_exc()
    else:
        print("[ONNX] 未提供或文件不存在，跳过对比")

if __name__ == '__main__':
    main()
