#!/usr/bin/env python3
"""
逐层二分定位：找出 RKNN 推理中第一个出问题的算子层。
在 RK3588 板端运行。

用法：
  python3 diagnose_layerwise.py \
      --rknn ../../caf_node/models/backbone_phalf.rknn \
      --bin 000000.bin --config config.yaml
"""
import argparse, sys, os, re
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from convert_to_rknn import _project_range_image, _project_polar_bev
import yaml

def load_inputs(bin_path, cfg):
    d = cfg['data']
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    dist = np.linalg.norm(pts[:, :3], axis=1)
    pts = pts[(dist > 0.5) & (dist < d['R_max'])]
    xyz, rem = pts[:, :3], np.zeros(len(pts), dtype=np.float32)
    rv = _project_range_image(xyz, rem, d['rv_H'], d['rv_W'],
                               d['fov_up'], d['fov_down'], d['R_max'], True)
    pb = _project_polar_bev(xyz, rem, d['pb_H'], d['pb_W'], d['R_max'])
    return rv, pb

def stat_arr(arr):
    a = arr.ravel().astype(np.float32)
    return (f"min={a.min():8.4f} max={a.max():8.4f} "
            f"mean={a.mean():7.4f} std={a.std():.4f} "
            f"|>3|={(np.abs(a)>3).mean()*100:.1f}%")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rknn',   required=True)
    ap.add_argument('--bin',    default=os.path.join(SCRIPT_DIR, '000000.bin'))
    ap.add_argument('--config', default=os.path.join(SCRIPT_DIR, 'config.yaml'))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rv_np, pb_np = load_inputs(args.bin, cfg)

    import logging
    _saved = dict(logging._levelToName), dict(logging._nameToLevel), logging.root.handlers[:], logging.root.level
    from rknnlite.api import RKNNLite
    logging._levelToName.clear(); logging._levelToName.update(_saved[0])
    logging._nameToLevel.clear(); logging._nameToLevel.update(_saved[1])
    logging.root.handlers = _saved[2]; logging.root.level = _saved[3]

    # ── 1. 先获取完整推理结果（不带中间输出）─────────────────────
    rknn = RKNNLite()
    assert rknn.load_rknn(args.rknn) == 0
    assert rknn.init_runtime() == 0
    full_out = rknn.inference(inputs=[rv_np, pb_np])
    rv_full = np.array(full_out[0], dtype=np.float32)
    pb_full = np.array(full_out[1], dtype=np.float32)
    rknn.release()
    print(f"完整推理:")
    print(f"  rv_feat: {stat_arr(rv_full)}")
    print(f"  pb_feat: {stat_arr(pb_full)}")
    print()

    # ── 2. 带中间输出的推理 ────────────────────────────────────
    rknn2 = RKNNLite()
    assert rknn2.load_rknn(args.rknn) == 0
    # 获取所有可以输出的中间层
    # RKNNLite 的 inference 支持 intermediate_outputs 参数（toolkit2 >= 1.6）
    try:
        outputs = rknn2.inference(inputs=[rv_np, pb_np],
                                   intermediate_outputs=True)
        assert rknn2.init_runtime() == 0
        print(f"共 {len(outputs)} 个中间输出层")
        for i, o in enumerate(outputs):
            arr = np.array(o, dtype=np.float32)
            has_nan = np.isnan(arr).any()
            has_inf = np.isinf(arr).any()
            flag = ""
            if has_nan: flag += " ← NaN!"
            if has_inf: flag += " ← Inf!"
            if arr.std() < 1e-4: flag += " ← 常数!"
            if np.abs(arr).max() > 20: flag += f" ← 异常大值({np.abs(arr).max():.1f})!"
            print(f"  [{i:3d}] shape={tuple(arr.shape)} {stat_arr(arr)}{flag}")
    except TypeError:
        print("[INFO] 当前 rknnlite 版本不支持 intermediate_outputs，改用逐输出节点测试")
        rknn2.release()
        _probe_by_custom_outputs(args, rv_np, pb_np, RKNNLite)
        return
    rknn2.release()


def _probe_by_custom_outputs(args, rv_np, pb_np, RKNNLite):
    """fallback: 用 rknn-toolkit2 的 get_intermediate_tensor 或逐段对比"""
    print("请在 x86 机器上用 rknn-toolkit2 的 rknn.eval_perf() 或")
    print("rknn.accuracy_analysis() 定位问题层。")
    print()
    print("板端可用方案：将模型用 rknn-toolkit2 切分为两段（encoder/decoder），")
    print("分别导出后对比中间特征。")
    print()
    print("快速验证命令（需要 onnxruntime）：")
    print("  python3 diagnose_rknn.py \\")
    print("      --rknn", args.rknn)
    print("      --onnx backbone_fp32_phalf.onnx \\")
    print("      --bin", args.bin)


if __name__ == '__main__':
    main()
