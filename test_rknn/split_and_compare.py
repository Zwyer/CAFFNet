#!/usr/bin/env python3
"""
将 backbone 切成两段 ONNX（encoder+aaff / decoder），
分别用 RKNN 和 ONNX 推理，找出误差主要来自哪一段。
在 x86 + rknn-toolkit2 的机器上运行。

用法：
  python3 split_and_compare.py --pth best.pth --config config.yaml
"""
import sys, os, argparse, yaml
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CAFFNET_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, CAFFNET_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from prfnet.models.prfnet import PRFNet
from convert_to_rknn import (
    load_cfg, build_model, load_checkpoint,
    _project_range_image, _project_polar_bev,
)


# ── 切分包装 ──────────────────────────────────────────────────────

class EncoderWrapper(nn.Module):
    """stem + encoder + AAFF → 输出 4 个融合特征图 + 2 个 stem"""
    def __init__(self, model: PRFNet):
        super().__init__()
        self.m = model
    def forward(self, rv_img, pb_img):
        fused_rv, fused_pb, rv_stem, pb_stem = self.m._encode(rv_img, pb_img)
        # fused_rv/pb: list of 4 tensors; 展平为独立输出
        return (*fused_rv, *fused_pb, rv_stem, pb_stem)  # 10 outputs


class DecoderWrapper(nn.Module):
    """decoder 单侧（rv 或 pb）"""
    def __init__(self, dec, name='rv'):
        super().__init__()
        self.dec  = dec
        self.name = name
    def forward(self, s1, s2, s3, s4, stem):
        return self.dec([s1, s2, s3, s4], stem)


# ── 工具 ──────────────────────────────────────────────────────────

def stat(name, arr):
    a = np.asarray(arr, dtype=np.float32).ravel()
    return (f"{name}: min={a.min():.4f} max={a.max():.4f} "
            f"mean={a.mean():.4f} std={a.std():.4f}")


def export_onnx(model, dummy_inputs, path, input_names, output_names):
    with torch.no_grad():
        torch.onnx.export(
            model, dummy_inputs, path,
            input_names=input_names, output_names=output_names,
            opset_version=16, do_constant_folding=True,
        )
    # 修复 Resize: bilinear+half_pixel -> nearest+asymmetric
    import onnx
    m_onnx = onnx.load(path)
    for node in m_onnx.graph.node:
        if node.op_type != 'Resize':
            continue
        new_attrs = []
        for attr in node.attribute:
            if attr.name == 'mode':              attr.s = b'nearest';    new_attrs.append(attr)
            elif attr.name == 'coordinate_transformation_mode': attr.s = b'asymmetric'; new_attrs.append(attr)
            elif attr.name == 'nearest_mode':    attr.s = b'floor';      new_attrs.append(attr)
            elif attr.name == 'cubic_coeff_a':   pass
            else:                                new_attrs.append(attr)
        del node.attribute[:]
        node.attribute.extend(new_attrs)
    onnx.save(m_onnx, path)
    print(f'  [ONNX] 导出: {path}')


def to_rknn(onnx_path, rknn_path, n_inputs):
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    rknn.config(target_platform='rk3588', optimization_level=3,
                mean_values=[[0]*9]*n_inputs if n_inputs <= 2 else None,
                std_values =[[1]*9]*n_inputs if n_inputs <= 2 else None,
                float_dtype='float16')
    assert rknn.load_onnx(onnx_path) == 0
    assert rknn.build(do_quantization=False) == 0
    assert rknn.export_rknn(rknn_path) == 0
    rknn.release()
    print(f'  [RKNN] 导出: {rknn_path}')


def rknn_infer(rknn_path, inputs):
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    assert rknn.load_rknn(rknn_path) == 0
    assert rknn.init_runtime(target=None) == 0  # PC 模拟
    out = rknn.inference(inputs=[np.asarray(x, dtype=np.float32) for x in inputs])
    rknn.release()
    return [np.asarray(o, dtype=np.float32) for o in out]


def onnx_infer(onnx_path, input_dict):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    return sess.run(None, {k: np.asarray(v, dtype=np.float32) for k, v in input_dict.items()})


# ── 主流程 ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pth',    default='best.pth')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--bin',    default='000000.bin')
    ap.add_argument('--out_dir', default='split_debug')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg   = load_cfg(os.path.join(SCRIPT_DIR, args.config))
    d, m  = cfg['data'], cfg['model']
    model = build_model(cfg)
    model = load_checkpoint(model, os.path.join(SCRIPT_DIR, args.pth))
    model.eval()

    # 准备输入
    pts = np.fromfile(os.path.join(SCRIPT_DIR, args.bin), dtype=np.float32).reshape(-1,4)
    dist = np.linalg.norm(pts[:,:3], axis=1)
    pts = pts[(dist>0.5) & (dist<d['R_max'])]
    xyz, rem = pts[:,:3], np.zeros(len(pts), dtype=np.float32)
    rv_np = _project_range_image(xyz, rem, d['rv_H'], d['rv_W'],
                                  d['fov_up'], d['fov_down'], d['R_max'], True)
    pb_np = _project_polar_bev(xyz, rem, d['pb_H'], d['pb_W'], d['R_max'])
    rv_t  = torch.from_numpy(rv_np)
    pb_t  = torch.from_numpy(pb_np)

    print('=== Step 1: 导出 Encoder ONNX + RKNN ===')
    enc_onnx = os.path.join(args.out_dir, 'encoder.onnx')
    enc_rknn = os.path.join(args.out_dir, 'encoder.rknn')
    enc = EncoderWrapper(model).eval()
    with torch.no_grad():
        enc_out_ref = enc(rv_t, pb_t)
    out_names = [f'frv_{i}' for i in range(4)] + [f'fpb_{i}' for i in range(4)] + ['rv_stem','pb_stem']
    export_onnx(enc, (rv_t, pb_t), enc_onnx, ['rv_img','pb_img'], out_names)
    to_rknn(enc_onnx, enc_rknn, 2)

    print()
    print('=== Step 2: 对比 Encoder 输出 ===')
    enc_onnx_out = onnx_infer(enc_onnx, {'rv_img': rv_np, 'pb_img': pb_np})
    enc_rknn_out = rknn_infer(enc_rknn, [rv_np, pb_np])

    total_diff_enc = []
    for i, (oo, ro, name) in enumerate(zip(enc_onnx_out, enc_rknn_out, out_names)):
        diff = np.abs(oo - ro)
        total_diff_enc.append(diff.mean())
        flag = ' ← 高误差!' if diff.mean() > 0.5 else ''
        print(f'  {name:10s}: onnx={stat("",oo)} | rknn_diff mean={diff.mean():.4f} max={diff.max():.4f}{flag}')

    print(f'\nEncoder 平均误差: {np.mean(total_diff_enc):.4f}')

    print()
    print('=== Step 3: 导出 RV Decoder ONNX + RKNN ===')
    # 用 ONNX encoder 输出作为 decoder 输入（隔离 encoder 误差）
    frv = [torch.from_numpy(enc_onnx_out[i]) for i in range(4)]
    rv_stem_t = torch.from_numpy(enc_onnx_out[8])
    dec_rv_onnx = os.path.join(args.out_dir, 'decoder_rv.onnx')
    dec_rv_rknn = os.path.join(args.out_dir, 'decoder_rv.rknn')
    dec_rv_w = DecoderWrapper(model.rv_dec).eval()
    dec_input_names = ['s1','s2','s3','s4','stem']
    export_onnx(dec_rv_w, tuple(frv + [rv_stem_t]), dec_rv_onnx, dec_input_names, ['rv_feat'])
    to_rknn(dec_rv_onnx, dec_rv_rknn, 5)

    dec_inputs = {n: np.asarray(enc_onnx_out[i]) for i, n in enumerate(['s1','s2','s3','s4'])}
    dec_inputs['stem'] = enc_onnx_out[8]

    dec_onnx_out = onnx_infer(dec_rv_onnx, dec_inputs)
    dec_rknn_out = rknn_infer(dec_rv_rknn, list(dec_inputs.values()))

    diff_dec = np.abs(dec_onnx_out[0] - dec_rknn_out[0])
    print(f'  rv_feat ONNX: {stat("", dec_onnx_out[0])}')
    print(f'  rv_feat RKNN diff: mean={diff_dec.mean():.4f} max={diff_dec.max():.4f}')
    print(f'\nDecoder RV 平均误差: {diff_dec.mean():.4f}')

    print()
    print('=== 结论 ===')
    enc_mean = np.mean(total_diff_enc)
    dec_mean = diff_dec.mean()
    print(f'Encoder 误差: {enc_mean:.4f}')
    print(f'Decoder 误差: {dec_mean:.4f}')
    if enc_mean > dec_mean * 2:
        print('→ 误差主要来自 Encoder/AAFF')
    elif dec_mean > enc_mean * 2:
        print('→ 误差主要来自 Decoder')
    else:
        print('→ Encoder 和 Decoder 误差相当')


if __name__ == '__main__':
    main()
