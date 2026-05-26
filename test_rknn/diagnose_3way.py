#!/usr/bin/env python3
"""
3-way 对比：FP32 ONNX vs FP16 ONNX vs RKNN NPU
精确定位 RKNN 额外引入的误差（在 FP16 基础之上）。

用法（板端 or PC with rknn-toolkit2）：
  python3 diagnose_3way.py --onnx_fp32 backbone.onnx --onnx_fp16 backbone_fp16.onnx --bin 000000.bin
  python3 diagnose_3way.py --onnx_fp32 backbone.onnx --rknn backbone.rknn --bin 000000.bin
"""
import argparse, sys, os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

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
    return rv, pb


def run_onnx(onnx_path, rv_in, pb_in):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    return sess.run(None, {'rv_img': rv_in, 'pb_img': pb_in})


def run_rknn(rknn_path, rv_in, pb_in):
    try:
        from rknnlite.api import RKNNLite
        rk = RKNNLite()
        ok = (rk.load_rknn(rknn_path) == 0 and rk.init_runtime() == 0)
    except ImportError:
        from rknn.api import RKNN
        rk = RKNN(verbose=False)
        ok = (rk.load_rknn(rknn_path) == 0 and rk.init_runtime() == 0)
    if not ok:
        rk.release() if hasattr(rk, 'release') else None
        raise RuntimeError('RKNN 初始化失败')
    outs = rk.inference(inputs=[rv_in, pb_in], data_format='nchw')
    rk.release()
    return [np.array(o, dtype=np.float32) for o in outs]


def stats(name, arr):
    a = arr.ravel()
    return (f"{name:>24s}: min={a.min():+.4f}  max={a.max():+.4f}  "
            f"mean={a.mean():+.4f}  std={a.std():+.4f}")


def compare(name, ref, other):
    """返回差异指标字典"""
    d = np.abs(ref - other)
    cos = np.sum(ref * other) / (np.sqrt(np.sum(ref**2)) * np.sqrt(np.sum(other**2)) + 1e-8)
    rmse = np.sqrt(np.mean(d**2))
    snr = 10 * np.log10(np.mean(ref**2) / np.mean((ref - other)**2))
    mae = np.mean(d)
    maxe = np.max(d)
    return {
        'name': name,
        'MAE': mae, 'MaxAE': maxe, 'RMSE': rmse,
        'SNR_dB': snr, 'Cosine': cos,
    }


def main():
    ap = argparse.ArgumentParser(description='3-way FP32/FP16/RKNN 对比')
    ap.add_argument('--onnx_fp32', default=None, help='FP32 ONNX 参考模型')
    ap.add_argument('--onnx_fp16', default=None, help='FP16 ONNX 模型')
    ap.add_argument('--rknn',      default=None, help='RKNN 模型')
    ap.add_argument('--bin',       default=os.path.join(SCRIPT_DIR, '000000.bin'))
    ap.add_argument('--config',    default=os.path.join(SCRIPT_DIR, 'config.yaml'))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rv_in, pb_in = load_inputs(args.bin, cfg)
    print(f"输入: rv={rv_in.shape}  pb={pb_in.shape}")
    print(f"     {stats('rv_input', rv_in)}")
    print(f"     {stats('pb_input', pb_in)}")

    results = {}  # name -> (rv_feat, pb_feat)

    # ---- FP32 ONNX ----
    if args.onnx_fp32 and os.path.exists(args.onnx_fp32):
        print(f"\n[FP32 ONNX] {args.onnx_fp32}")
        rv_f32, pb_f32 = run_onnx(args.onnx_fp32, rv_in, pb_in)
        results['FP32'] = (rv_f32, pb_f32)
        print(f"  {stats('rv_feat', rv_f32)}")
        print(f"  {stats('pb_feat', pb_f32)}")
    else:
        print("\n[FP32 ONNX] 跳过")

    # ---- FP16 ONNX ----
    if args.onnx_fp16 and os.path.exists(args.onnx_fp16):
        print(f"\n[FP16 ONNX] {args.onnx_fp16}")
        rv_f16, pb_f16 = run_onnx(args.onnx_fp16, rv_in, pb_in)
        results['FP16'] = (rv_f16, pb_f16)
        print(f"  {stats('rv_feat', rv_f16)}")
        print(f"  {stats('pb_feat', pb_f16)}")
    else:
        print("\n[FP16 ONNX] 跳过")

    # ---- RKNN ----
    if args.rknn and os.path.exists(args.rknn):
        print(f"\n[RKNN] {args.rknn}")
        try:
            rv_rk, pb_rk = run_rknn(args.rknn, rv_in, pb_in)
            results['RKNN'] = (rv_rk, pb_rk)
            print(f"  {stats('rv_feat', rv_rk)}")
            print(f"  {stats('pb_feat', pb_rk)}")

            # 检查 RKNN 是否有异常通道
            print("\n  [RKNN 通道诊断] 检查常数通道/异常值:")
            for name, feat in [('rv', rv_rk), ('pb', pb_rk)]:
                for c in range(feat.shape[1]):
                    ch = feat[0, c].ravel()
                    const = ch.std() < 1e-5
                    dead = np.all(ch == 0)
                    if const or dead:
                        flag = '常数' if const else '全零'
                        print(f"    {name} ch[{c:2d}]: {flag}! mean={ch.mean():.4f} std={ch.std():.6f}")
        except Exception as e:
            print(f"  [RKNN] 失败: {e}")
    else:
        print("\n[RKNN] 跳过")

    # ---- 交叉对比 ----
    if len(results) < 2:
        print("\n至少需要2个模型才能对比")
        return

    names = list(results.keys())
    print(f"\n{'='*80}")
    print(f"交叉对比: {' vs '.join(names)}")
    print(f"{'='*80}")

    for feat_name in ['rv_feat', 'pb_feat']:
        print(f"\n{'─'*80}")
        print(f"  {feat_name}")
        print(f"{'─'*80}")

        ref_name = names[0]
        ref_val = results[ref_name][0] if feat_name == 'rv_feat' else results[ref_name][1]

        for other_name in names[1:]:
            other_val = results[other_name][0] if feat_name == 'rv_feat' else results[other_name][1]
            m = compare(f'{ref_name}→{other_name}', ref_val[0], other_val[0])

            print(f"\n  {ref_name} vs {other_name}:")
            print(f"    MAE   : {m['MAE']:.4e}")
            print(f"    MaxAE : {m['MaxAE']:.4e}")
            print(f"    RMSE  : {m['RMSE']:.4e}")
            print(f"    SNR   : {m['SNR_dB']:.1f} dB")
            print(f"    Cosine: {m['Cosine']:.10f}")

        # 如果有3个模型，对比 FP16→RKNN 的额外误差
        if len(results) == 3 and 'FP16' in results and 'RKNN' in results:
            f16_val = results['FP16'][0] if feat_name == 'rv_feat' else results['FP16'][1]
            rk_val = results['RKNN'][0] if feat_name == 'rv_feat' else results['RKNN'][1]
            f32_val = results['FP32'][0] if feat_name == 'rv_feat' else results['FP32'][1]

            m_fp16 = compare('FP32→FP16', f32_val[0], f16_val[0])
            m_rknn = compare('FP32→RKNN', f32_val[0], rk_val[0])

            extra_rmse = m_rknn['RMSE'] - m_fp16['RMSE']
            if m_fp16['RMSE'] > 0:
                ratio = m_rknn['RMSE'] / m_fp16['RMSE']
                print(f"\n  [1;33mRKNN额外误差: RMSE增加 {extra_rmse:.4e} (×{ratio:.1f} 倍于FP16)[0m")


if __name__ == '__main__':
    main()
