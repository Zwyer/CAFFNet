#!/usr/bin/env python3
"""
PRFNet 参数量与 FLOPs 统计脚本

用法：
    python tools/model_stats.py
    python tools/model_stats.py --cfg prfnet/configs/prfnet_semantickitti.yaml
    python tools/model_stats.py --N 16384      # 用小点数加速（FLOPs 与 N 线性）
    python tools/model_stats.py --full-N       # 使用配置的 max_points（131072）

FLOPs 依赖（二选一）：
    pip install thop     # 轻量，推荐
    pip install fvcore   # Facebook，更精确
"""

import argparse
import os
import sys

import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prfnet.models.prfnet import PRFNet


# ─────────────────────────────────────────────────────────────

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg, device='cpu'):
    dc, mc = cfg['data'], cfg['model']
    _use_normals = mc.get('use_surface_normals', False)
    _use_angle   = mc.get('use_angle_encoding',  False)
    rv_in = 6 + 3 * int(_use_normals) + 3 * int(_use_angle)

    model = PRFNet(
        rv_in=rv_in, pb_in=mc['pb_in'],
        num_classes=dc['num_classes'],
        enc_channels=mc['enc_channels'],
        dec_out_c=mc['dec_out_c'],
        expand_ratios=mc['expand_ratios'],
        rv_strides=mc.get('rv_strides', [[1,2],[2,2],[2,2],[2,2]]),
        pb_strides=mc.get('pb_strides', [[2,2],[2,2],[2,2],[2,2]]),
        aspp_rates=mc.get('aspp_rates', [1,3,6,9]),
        rv_H=dc['rv_H'], pb_H=dc['pb_H'],
        use_ds_aaff=mc.get('use_ds_aaff', True),
        ds_aaff_K=mc.get('ds_aaff_K', 4),
        use_vcg=mc.get('use_vcg', True),
        use_proto=mc.get('use_proto', True),
        proto_dim=mc.get('proto_dim', 64),
    ).to(device)
    return model, rv_in


def make_inputs(cfg, rv_in, N, device='cpu'):
    dc, mc = cfg['data'], cfg['model']
    B = 1
    return (
        torch.zeros(B, rv_in,      dc['rv_H'], dc['rv_W'], device=device),
        torch.zeros(B, mc['pb_in'], dc['pb_H'], dc['pb_W'], device=device),
        torch.zeros(B, N, 1, 2, device=device),
        torch.zeros(B, N, 1, 2, device=device),
        torch.zeros(B, N, 4,    device=device),
    )


def param_table(model):
    """打印顶层子模块参数量"""
    total   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buf   = sum(b.numel() for b in model.buffers())

    # 汇总到顶层模块
    top = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = name.split('.')[0]
        top[key] = top.get(key, 0) + p.numel()

    W = 46
    print(f"\n  {'Module':<{W}} {'Params':>10}   {'%':>5}")
    print('  ' + '─' * (W + 20))
    for name, n in sorted(top.items(), key=lambda x: -x[1]):
        print(f"  {name:<{W}} {n:>10,d}   {100*n/total:>4.1f}%")
    print('  ' + '─' * (W + 20))
    print(f"  {'TOTAL (trainable)':<{W}} {total:>10,d}")
    if n_buf:
        print(f"  {'buffers (non-trainable, e.g. prototypes)':<{W}} {n_buf:>10,d}")
    return total


def _fmt(n):
    """将大数格式化为 G/M/K 单位"""
    if n >= 1e9:  return f'{n/1e9:.3f} G'
    if n >= 1e6:  return f'{n/1e6:.3f} M'
    if n >= 1e3:  return f'{n/1e3:.3f} K'
    return str(n)


def flops_thop(model, inputs, N, max_N):
    from thop import profile
    with torch.no_grad():
        macs, _ = profile(model, inputs=inputs, verbose=False)
    flops = macs * 2
    print(f"\n  {'[thop]':<18} MACs = {_fmt(macs)}   FLOPs = {_fmt(flops)}   (N={N:,d})")
    if N != max_N:
        scale = max_N / N
        print(f"  {'':18} ≈ FLOPs @ N={max_N:,d}: {_fmt(flops*scale)}  (线性外推 ×{scale:.0f})")


def flops_fvcore(model, inputs, N, max_N):
    from fvcore.nn import FlopCountAnalysis
    fa = FlopCountAnalysis(model, inputs)
    fa.unsupported_ops_settings(raise_on_error=False)
    flops = fa.total()
    print(f"\n  {'[fvcore]':<18} FLOPs = {flops/1e9:.3f} G   (N={N:,d})")
    if N != max_N:
        scale = max_N / N
        print(f"  {'':18} ≈ FLOPs @ N={max_N:,d}: {flops*scale/1e9:.3f} G  (线性外推)")


# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg',    default='prfnet/configs/prfnet_semantickitti.yaml')
    ap.add_argument('--N',      type=int, default=16384,
                    help='统计用点数（默认 16384，与 max_points 线性关系）')
    ap.add_argument('--full-N', action='store_true',
                    help='使用 max_points（131072）作为 N（较慢）')
    ap.add_argument('--cpu',    action='store_true', help='强制 CPU 运行')
    args = ap.parse_args()

    cfg   = load_cfg(args.cfg)
    max_N = cfg['data']['max_points']
    N     = max_N if args.full_N else args.N
    dev   = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')

    model, rv_in = build_model(cfg, dev)
    model.eval()

    dc, mc = cfg['data'], cfg['model']
    _use_normals = mc.get('use_surface_normals', False)
    _use_angle   = mc.get('use_angle_encoding',  False)

    print(f"\n{'═'*68}")
    print(f"  PRFNet 模型统计")
    print(f"  Config : {args.cfg}")
    print(f"  Device : {dev}")
    print(f"  RV     : {dc['rv_H']}×{dc['rv_W']}  ch={rv_in}"
          f"  (normals={_use_normals}, angle={_use_angle})")
    print(f"  PB     : {dc['pb_H']}×{dc['pb_W']}  ch={mc['pb_in']}")
    print(f"  N      : {N:,d} 点  (max_points={max_N:,d})")
    print(f"  VCG    : {mc.get('use_vcg',True)}   SPM: {mc.get('use_proto',True)}"
          f"   DS-AAFF: {mc.get('use_ds_aaff',True)}")
    print(f"{'═'*68}")

    total = param_table(model)

    # ── FLOPs ────────────────────────────────────────────────
    inputs = make_inputs(cfg, rv_in, N, dev)

    tried = False
    try:
        flops_thop(model, inputs, N, max_N)
        tried = True
    except ImportError:
        pass

    try:
        flops_fvcore(model, inputs, N, max_N)
        tried = True
    except ImportError:
        pass

    if not tried:
        print("\n  未安装 FLOPs 统计库，请运行：")
        print("    pip install thop      # 推荐，轻量")
        print("    pip install fvcore    # 备选，精确")

    print(f"\n{'═'*68}\n")


if __name__ == '__main__':
    main()
