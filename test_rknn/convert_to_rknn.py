#!/usr/bin/env python3
"""
CAFFNet PTH -> ONNX -> RKNN 转换工具（RK3588，grid_sample CPU 绕过版）

由于 RK3588 RKNN NPU 不支持 grid_sample 算子，模型被切分为两部分：
  1. backbone.rknn  — Encoder + DS-AAFF + Decoder（全卷积，NPU 运行）
     输入：rv_img (1, C_rv, rv_H, rv_W), pb_img (1, 9, pb_H, pb_W)
     输出：rv_feat (1, dec_out_c, rv_H, rv_W), pb_feat (1, dec_out_c, pb_H, pb_W)

  2. head_weights.npz — PointSampleAggregator 权重（CPU numpy 推理）
     grid_sample 及后续 MLP 在 CPU 上用 numpy 实现

新增：--accuracy_analysis 可对转换后的模型进行逐层精度分析，定位误差来源。
自动根据 config.yaml 设置正确的输入尺寸。
"""

import sys
import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CAFFNET_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, CAFFNET_ROOT)

from prfnet.models.prfnet import PRFNet
from prfnet.models.modules import PointSampleAggregator


# ──────────────────────────────────────────────────────────────────
# Backbone 导出包装（Encoder + AAFF + Decoder，无 grid_sample）
# ──────────────────────────────────────────────────────────────────

class BackboneWrapper(nn.Module):
    def __init__(self, model: PRFNet):
        super().__init__()
        self.model = model

    def forward(self, rv_img: torch.Tensor, pb_img: torch.Tensor):
        fused_rv, fused_pb, rv_stem, pb_stem = self.model._encode(rv_img, pb_img)
        rv_feat = self.model.rv_dec(fused_rv, rv_stem)
        pb_feat = self.model.pb_dec(fused_pb, pb_stem)
        return rv_feat, pb_feat


# ──────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> PRFNet:
    d, m = cfg['data'], cfg['model']
    return PRFNet(
        rv_in         = m.get('rv_in', 9),
        pb_in         = m.get('pb_in', 9),
        num_classes   = d['num_classes'],
        enc_channels  = m['enc_channels'],
        dec_out_c     = m['dec_out_c'],
        rv_H          = d['rv_H'],
        pb_H          = d['pb_H'],
        rv_strides    = m.get('rv_strides', [[1,2],[2,2],[2,2],[2,2]]),
        pb_strides    = m.get('pb_strides', [[2,2],[2,2],[2,2],[2,2]]),
        expand_ratios = m.get('expand_ratios', [2,2,4,4]),
        aspp_rates    = m.get('aspp_rates', [1,3,6,9]),
        use_ds_aaff   = m.get('use_ds_aaff', True),
        ds_aaff_K     = m.get('ds_aaff_K', 2),
        head_dropout  = 0.0,
        use_vcg       = m.get('use_vcg', False),
        use_proto     = m.get('use_proto', False),
        proto_dim     = m.get('proto_dim', 64),
    )


def load_checkpoint(model: PRFNet, pth_path: str) -> PRFNet:
    ckpt = torch.load(pth_path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'ema_state_dict' in ckpt:
            state = ckpt['ema_state_dict']
            print('[INFO] 使用 EMA weights')
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'model' in ckpt:
            state = ckpt['model']
        else:
            state = ckpt
    else:
        state = ckpt

    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'[WARN] Missing  ({len(missing)}): {missing[:5]}{"..." if len(missing)>5 else ""}')
    if unexpected:
        print(f'[WARN] Unexpected ({len(unexpected)}): {unexpected[:5]}{"..." if len(unexpected)>5 else ""}')
    return model


# ──────────────────────────────────────────────────────────────────
# 导出 Backbone ONNX
# ──────────────────────────────────────────────────────────────────

def export_backbone_onnx(model: PRFNet, cfg: dict, onnx_path: str):
    d, m   = cfg['data'], cfg['model']
    rv_in  = m.get('rv_in', 9)
    pb_in  = m.get('pb_in', 9)
    rv_H, rv_W = d['rv_H'], d['rv_W']
    pb_H, pb_W = d['pb_H'], d['pb_W']

    rv_dummy = torch.zeros(1, rv_in, rv_H, rv_W, dtype=torch.float32)
    pb_dummy = torch.zeros(1, pb_in, pb_H, pb_W, dtype=torch.float32)

    wrapper = BackboneWrapper(model).eval()
    with torch.no_grad():
        rv_feat, pb_feat = wrapper(rv_dummy, pb_dummy)
    print(f'[INFO] Backbone forward 验证通过')
    print(f'       rv_feat: {rv_feat.shape}  pb_feat: {pb_feat.shape}')

    print(f'[INFO] 导出 Backbone ONNX (opset 16): {onnx_path}')
    torch.onnx.export(
        wrapper,
        (rv_dummy, pb_dummy),
        onnx_path,
        input_names  = ['rv_img', 'pb_img'],
        output_names = ['rv_feat', 'pb_feat'],
        opset_version       = 16,
        do_constant_folding = True,
        verbose             = False,
    )
    print('[INFO] Backbone ONNX 导出完成')

    # ONNX 后处理：Resize模式改为nearest（避免RKNN上采样bug）
    # 注意：不再做 BN folding，因为 PyTorch export 时已通过 do_constant_folding=True
    # 和训练时的 BN 融合（deploy mode）完成，重复 folding 会破坏权重
    try:
        import onnx
        from onnxsim import simplify as onnxsim

        m_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(m_onnx)

        simplified, ok = onnxsim(m_onnx)
        if ok:
            m_onnx = simplified
            print('[INFO] onnxsim 简化完成')
        else:
            print('[WARN] onnxsim 简化失败，保留原始 ONNX')

        # 将 bilinear + half_pixel → nearest + asymmetric
        n_fixed = 0
        for node in m_onnx.graph.node:
            if node.op_type != 'Resize':
                continue
            new_attrs = []
            for attr in node.attribute:
                if attr.name == 'mode':
                    attr.s = b'nearest'
                    new_attrs.append(attr)
                elif attr.name == 'coordinate_transformation_mode':
                    attr.s = b'asymmetric'
                    new_attrs.append(attr)
                elif attr.name == 'nearest_mode':
                    attr.s = b'floor'
                    new_attrs.append(attr)
                elif attr.name == 'cubic_coeff_a':
                    pass
                else:
                    new_attrs.append(attr)
            del node.attribute[:]
            node.attribute.extend(new_attrs)
            n_fixed += 1
        if n_fixed:
            print(f'[INFO] Resize: bilinear+half_pixel -> nearest+asymmetric ({n_fixed} 节点)')

        onnx.save(m_onnx, onnx_path)
    except ImportError as e:
        print(f'[WARN] {e}，跳过优化（pip install onnx onnxsim）')
    except Exception as e:
        print(f'[WARN] ONNX 优化异常: {e}，跳过')


# ──────────────────────────────────────────────────────────────────
# 导出 Head 权重（CPU numpy 推理用）
# ──────────────────────────────────────────────────────────────────

def export_head_weights(model: PRFNet, cfg: dict, npz_path: str):
    agg: PointSampleAggregator = model.aggregator
    d = cfg['data']
    m_cfg = cfg['model']

    weights = {}
    weights['pt_enc_fc_weight'] = agg.pt_enc[0].weight.detach().numpy()
    weights['pt_enc_bn_weight'] = agg.pt_enc[1].weight.detach().numpy()
    weights['pt_enc_bn_bias']   = agg.pt_enc[1].bias.detach().numpy()
    weights['pt_enc_bn_mean']   = agg.pt_enc[1].running_mean.detach().numpy()
    weights['pt_enc_bn_var']    = agg.pt_enc[1].running_var.detach().numpy()
    weights['pt_enc_bn_eps']    = np.float32(agg.pt_enc[1].eps)

    weights['head_fc0_weight']  = agg.head[0].weight.detach().numpy()
    weights['head_bn0_weight']  = agg.head[1].weight.detach().numpy()
    weights['head_bn0_bias']    = agg.head[1].bias.detach().numpy()
    weights['head_bn0_mean']    = agg.head[1].running_mean.detach().numpy()
    weights['head_bn0_var']     = agg.head[1].running_var.detach().numpy()
    weights['head_bn0_eps']     = np.float32(agg.head[1].eps)
    weights['head_fc1_weight']  = agg.head[4].weight.detach().numpy()
    weights['head_fc1_bias']    = agg.head[4].bias.detach().numpy()

    weights['rv_c']          = np.int32(agg.rv_c)
    weights['num_classes']   = np.int32(d['num_classes'])
    weights['use_vcg']       = np.bool_(m_cfg.get('use_vcg', False))
    weights['use_proto']     = np.bool_(m_cfg.get('use_proto', False))

    np.savez_compressed(npz_path, **weights)
    print(f'[INFO] Head 权重已保存: {npz_path}')
    print(f'       rv_c={agg.rv_c}, num_classes={d["num_classes"]}, use_vcg={m_cfg.get("use_vcg")}, use_proto={m_cfg.get("use_proto")}')


# ──────────────────────────────────────────────────────────────────
# INT8 校准数据集准备
# ──────────────────────────────────────────────────────────────────

def _project_range_image(xyz, remissions, H, W, fov_up, fov_down, R_max, use_angle_encoding):
    fov_up_r   = np.deg2rad(fov_up)
    fov_down_r = np.deg2rad(fov_down)
    fov_r      = fov_up_r - fov_down_r
    n_ch = 9 if use_angle_encoding else 6

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2).clip(1e-5, R_max)
    phi   = np.arcsin(np.clip(z / r, -1.0, 1.0))
    theta = np.arctan2(y, x)

    row = np.clip(np.round((fov_up_r - phi) / fov_r * (H - 1)).astype(np.int32), 0, H - 1)
    col = np.clip(np.round((theta + np.pi) / (2 * np.pi) * (W - 1)).astype(np.int32), 0, W - 1)

    order = np.argsort(-r)
    point_idx = np.full((H, W), -1, dtype=np.int32)
    point_idx[row[order], col[order]] = order

    phi_grid   = (fov_up_r - np.arange(H, dtype=np.float32) / max(H-1,1) * fov_r)[:, None]
    theta_grid = np.linspace(-np.pi, np.pi, W, dtype=np.float32)[None, :]

    rv = np.zeros((n_ch, H, W), dtype=np.float32)
    valid = point_idx >= 0
    vi    = point_idx[valid]
    rv[0][valid] = x[vi] / R_max
    rv[1][valid] = y[vi] / R_max
    rv[2][valid] = z[vi] / R_max
    rv[3][valid] = r[vi] / R_max
    rv[4][valid] = remissions[vi]
    rv[5]        = np.cos(phi_grid) * np.ones((1, W), dtype=np.float32)
    if use_angle_encoding:
        rv[6] = np.sin(phi_grid) * np.ones((1, W), dtype=np.float32)
        rv[7] = np.ones((H, 1), dtype=np.float32) * np.cos(theta_grid)
        rv[8] = np.ones((H, 1), dtype=np.float32) * np.sin(theta_grid)
    return rv[np.newaxis]


def _project_polar_bev(xyz, remissions, H_p, W_p, R_max):
    rho_edges   = np.linspace(0, np.sqrt(R_max), H_p + 1) ** 2
    theta_edges = np.linspace(-np.pi, np.pi, W_p + 1)

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r     = np.sqrt(x**2 + y**2 + z**2)
    rho   = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    row = np.clip(np.searchsorted(rho_edges[1:],   rho,   side='left'), 0, H_p - 1)
    col = np.clip(np.searchsorted(theta_edges[1:], theta, side='left'), 0, W_p - 1)
    flat = row * W_p + col
    HW   = H_p * W_p

    bev   = np.zeros((9, HW), dtype=np.float32)
    count = np.bincount(flat, minlength=HW).astype(np.float32)
    bev[0] = np.bincount(flat, weights=x,          minlength=HW)
    bev[1] = np.bincount(flat, weights=y,          minlength=HW)
    bev[2] = np.bincount(flat, weights=z,          minlength=HW)
    bev[5] = np.bincount(flat, weights=remissions,  minlength=HW)
    bev[7] = np.bincount(flat, weights=r,          minlength=HW)
    z_min = np.full(HW,  np.inf, dtype=np.float32)
    z_max = np.full(HW, -np.inf, dtype=np.float32)
    np.minimum.at(z_min, flat, z)
    np.maximum.at(z_max, flat, z)
    v = count > 0
    bev[0][v] /= count[v]; bev[0][v] /= R_max
    bev[1][v] /= count[v]; bev[1][v] /= R_max
    bev[2][v] /= count[v]; bev[2][v] /= R_max
    bev[3]     = np.where(v, z_min, 0.0) / R_max
    bev[4]     = np.where(v, z_max, 0.0) / R_max
    bev[5][v] /= count[v]
    bev[6]     = np.log1p(count) / np.log1p(64)
    bev[7][v] /= (count[v] * R_max)
    bev[8]     = v.astype(np.float32)
    return bev.reshape(1, 9, H_p, W_p)


def prepare_calibration_dataset(cfg, bin_dir, out_dir, n_frames=100):
    dc = cfg['data']
    R_max      = dc['R_max']
    use_angle  = dc.get('use_angle_encoding', True)
    use_intens = dc.get('use_intensity', False)

    bin_files = []
    for root, _, files in os.walk(bin_dir):
        for f in sorted(files):
            if f.endswith('.bin'):
                bin_files.append(os.path.join(root, f))
    if not bin_files:
        raise FileNotFoundError(f'在 {bin_dir} 下未找到任何 .bin 文件')

    if len(bin_files) > n_frames:
        step = len(bin_files) / n_frames
        bin_files = [bin_files[int(i * step)] for i in range(n_frames)]

    os.makedirs(out_dir, exist_ok=True)
    rv_paths, pb_paths = [], []

    print(f'[INFO] 准备校准数据：共 {len(bin_files)} 帧 → {out_dir}')
    for i, bp in enumerate(bin_files):
        pts = np.fromfile(bp, dtype=np.float32).reshape(-1, 4)
        dist = np.linalg.norm(pts[:, :3], axis=1)
        pts = pts[(dist > 0.5) & (dist < R_max)].copy()
        if not use_intens:
            pts[:, 3] = 0.0
        if pts.shape[0] == 0:
            continue

        xyz = pts[:, :3]
        rem = pts[:, 3]
        rv_img = _project_range_image(xyz, rem,
                                      dc['rv_H'], dc['rv_W'],
                                      dc['fov_up'], dc['fov_down'],
                                      R_max, use_angle)
        pb_img = _project_polar_bev(xyz, rem, dc['pb_H'], dc['pb_W'], R_max)

        rv_p = os.path.join(out_dir, f'rv_{i:04d}.npy')
        pb_p = os.path.join(out_dir, f'pb_{i:04d}.npy')
        np.save(rv_p, rv_img)
        np.save(pb_p, pb_img)
        rv_paths.append(rv_p)
        pb_paths.append(pb_p)

        if (i + 1) % 20 == 0 or i == len(bin_files) - 1:
            print(f'  [{i+1}/{len(bin_files)}] {os.path.basename(bp)}')

    dataset_txt = os.path.join(out_dir, 'dataset.txt')
    with open(dataset_txt, 'w') as f:
        for rv_p, pb_p in zip(rv_paths, pb_paths):
            f.write(f'{rv_p} {pb_p}\n')

    print(f'[INFO] 校准集列表已生成: {dataset_txt}  ({len(rv_paths)} 帧有效)')
    return dataset_txt


# ──────────────────────────────────────────────────────────────────
# RKNN 转换（支持精度分析，自动适配输入尺寸）
# ──────────────────────────────────────────────────────────────────

def convert_to_rknn(cfg, onnx_path, rknn_path, quantize=False, dataset_path=None,
                    do_accuracy_analysis=False, analysis_inputs=None,
                    target='rk3588', device_id=None):
    """
    cfg: 配置字典，用于获取正确的输入尺寸
    """
    try:
        from rknn.api import RKNN
    except ImportError:
        print('[ERROR] rknn-toolkit2 未安装（pip install rknn-toolkit2，仅 x86）')
        sys.exit(1)

    # 从配置中提取输入尺寸
    d = cfg['data']
    m = cfg['model']
    rv_H, rv_W = d['rv_H'], d['rv_W']
    pb_H, pb_W = d['pb_H'], d['pb_W']
    rv_C = m.get('rv_in', 9)
    pb_C = m.get('pb_in', 9)

    rknn = RKNN(verbose=False)

    print('[INFO] 配置 RKNN (target=rk3588)...')
    ret = rknn.config(
        target_platform    = 'rk3588',
        optimization_level = 0,
        mean_values        = [[0]*rv_C, [0]*pb_C],
        std_values         = [[1]*rv_C, [1]*pb_C],
        float_dtype        = 'float16',
        remove_weight      = False,
        compress_weight    = False,
    )
    if ret != 0:
        print(f'[ERROR] rknn.config 失败 ret={ret}')
        sys.exit(1)

    print(f'[INFO] 加载 ONNX: {onnx_path}')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f'[ERROR] load_onnx 失败 ret={ret}')
        sys.exit(1)

    print('[INFO] 构建 RKNN 模型...')
    if quantize and dataset_path:
        print(f'[INFO] INT8 量化，校准集: {dataset_path}')
        ret = rknn.build(do_quantization=True, dataset=dataset_path)
    else:
        if quantize:
            print('[WARN] --quantize 需配合 --dataset，退回 FP16')
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        print(f'[ERROR] 构建失败 ret={ret}')
        sys.exit(1)

    # ----- 精度分析（在导出前进行，可及时发现量化问题）-----
    if do_accuracy_analysis:
        print('[INFO] 开始精度分析...')
        # 准备输入文件
        if analysis_inputs is not None and len(analysis_inputs) == 2:
            rv_file, pb_file = analysis_inputs
            if not os.path.exists(rv_file) or not os.path.exists(pb_file):
                print(f'[WARN] 提供的分析输入文件不存在，改用随机数据')
                analysis_inputs = None

        if analysis_inputs is None:
            # 根据 cfg 中的尺寸生成随机输入
            temp_dir = os.path.join(os.path.dirname(rknn_path), 'temp_analysis')
            os.makedirs(temp_dir, exist_ok=True)
            rv_file = os.path.join(temp_dir, 'analysis_rv.npy')
            pb_file = os.path.join(temp_dir, 'analysis_pb.npy')
            np.save(rv_file, np.random.randn(1, rv_C, rv_H, rv_W).astype(np.float32))
            np.save(pb_file, np.random.randn(1, pb_C, pb_H, pb_W).astype(np.float32))
            print(f'[INFO] 使用随机输入: {rv_file} ({1, rv_C, rv_H, rv_W}), {pb_file} ({1, pb_C, pb_H, pb_W})')
        else:
            rv_file, pb_file = analysis_inputs
            print(f'[INFO] 使用指定输入: {rv_file}, {pb_file}')

        # 调用精度分析
        ret = rknn.accuracy_analysis(
            inputs=[rv_file, pb_file],
            output_dir='./snapshot',   # 结果输出目录
            target=target,             # 若为 None 则使用仿真器
            device_id=device_id
        )
        if ret != 0:
            print(f'[ERROR] 精度分析失败 ret={ret}，可能设备未连接或存储不足')
        else:
            print('[INFO] 精度分析完成，结果保存在 ./snapshot 目录')
            print('       可查看各层误差报告，定位问题算子。')

    # 导出模型
    print(f'[INFO] 导出 RKNN: {rknn_path}')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f'[ERROR] export_rknn 失败 ret={ret}')
        sys.exit(1)

    rknn.release()
    print('[INFO] RKNN 转换完成')


# ──────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='CAFFNet PTH → ONNX → RKNN 转换工具（RK3588，grid_sample CPU 绕过版）'
    )
    parser.add_argument('--config',    default='config.yaml')
    parser.add_argument('--pth',       default='best.pth')
    parser.add_argument('--out',       default='backbone.rknn')
    parser.add_argument('--n_fixed',   type=int, default=32768)
    parser.add_argument('--onnx_only', action='store_true')
    parser.add_argument('--quantize',  action='store_true')
    parser.add_argument('--bin_dir',   default=None)
    parser.add_argument('--calib_frames', type=int, default=100)
    parser.add_argument('--dataset',   default=None)

    # 精度分析相关参数
    parser.add_argument('--accuracy_analysis', action='store_true',
                        help='启用逐层精度分析（需 rknn-toolkit2）')
    parser.add_argument('--analysis_target', default=None,
                        help='精度分析目标平台，如 rk3588，默认 None 使用仿真器')
    parser.add_argument('--device_id', default=None)
    parser.add_argument('--analysis_input_rv', default=None)
    parser.add_argument('--analysis_input_pb', default=None)
    parser.add_argument('--analysis_bin', default=None,
                        help='SemanticKITTI 格式 .bin 点云文件，自动投影为 rv/pb 用于精度分析')

    args = parser.parse_args()

    def _abs(p):
        return p if (p is None or os.path.isabs(p)) else os.path.join(SCRIPT_DIR, p)

    cfg_path  = _abs(args.config)
    pth_path  = _abs(args.pth)
    rknn_path = _abs(args.out)
    onnx_path = rknn_path.replace('.rknn', '.onnx')
    npz_path  = os.path.join(os.path.dirname(rknn_path), 'head_weights.npz')

    print('=' * 56)
    print('  CAFFNet PTH → RKNN 转换（grid_sample CPU 绕过版）')
    print('=' * 56)
    print(f'Config  : {cfg_path}')
    print(f'PTH     : {pth_path}')
    print(f'N_fixed : {args.n_fixed}（运行时可通过 ROS param 调整）')

    cfg   = load_cfg(cfg_path)
    model = build_model(cfg)
    model = load_checkpoint(model, pth_path)
    model.eval()

    # 1. 导出 Backbone ONNX
    export_backbone_onnx(model, cfg, onnx_path)

    # 2. 导出 Head 权重
    export_head_weights(model, cfg, npz_path)

    # 3. 准备 INT8 校准数据（如需要）
    dataset_path = _abs(args.dataset)
    if args.quantize and dataset_path is None:
        if args.bin_dir is None:
            print('[ERROR] INT8 量化需提供 --bin_dir 或 --dataset')
            sys.exit(1)
        calib_dir = os.path.join(os.path.dirname(rknn_path), 'calib_data')
        dataset_path = prepare_calibration_dataset(
            cfg, _abs(args.bin_dir), calib_dir, args.calib_frames)

    # 4. 收集精度分析输入文件（若启用）
    analysis_inputs = None
    if args.accuracy_analysis:
        if args.analysis_input_rv and args.analysis_input_pb:
            analysis_inputs = [_abs(args.analysis_input_rv), _abs(args.analysis_input_pb)]
        elif args.analysis_bin:
            # 从 SemanticKITTI .bin 点云投影生成 rv/pb 输入
            bin_path = _abs(args.analysis_bin)
            print(f'[INFO] 从点云生成精度分析输入: {bin_path}')
            pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
            d_cfg = cfg['data']
            R_max = d_cfg['R_max']
            use_angle = d_cfg.get('use_angle_encoding', True)
            use_intens = d_cfg.get('use_intensity', False)

            # 距离滤波
            dist = np.linalg.norm(pts[:, :3], axis=1)
            pts = pts[(dist > 0.5) & (dist < R_max)].copy()
            if not use_intens:
                pts[:, 3] = 0.0

            xyz = pts[:, :3]
            rem = pts[:, 3]
            rv_img = _project_range_image(xyz, rem,
                                          d_cfg['rv_H'], d_cfg['rv_W'],
                                          d_cfg['fov_up'], d_cfg['fov_down'],
                                          R_max, use_angle)
            pb_img = _project_polar_bev(xyz, rem, d_cfg['pb_H'], d_cfg['pb_W'], R_max)

            temp_dir = os.path.join(os.path.dirname(rknn_path), 'temp_analysis')
            os.makedirs(temp_dir, exist_ok=True)
            rv_file = os.path.join(temp_dir, 'analysis_rv.npy')
            pb_file = os.path.join(temp_dir, 'analysis_pb.npy')
            np.save(rv_file, rv_img)
            np.save(pb_file, pb_img)
            analysis_inputs = [rv_file, pb_file]
            print(f'[INFO] 已生成: {rv_file}  {pb_file}')
        else:
            # 尝试从校准数据目录中取第一帧（如果存在）
            calib_dir = os.path.join(os.path.dirname(rknn_path), 'calib_data')
            rv_sample = os.path.join(calib_dir, 'rv_0000.npy')
            pb_sample = os.path.join(calib_dir, 'pb_0000.npy')
            if os.path.exists(rv_sample) and os.path.exists(pb_sample):
                print('[INFO] 未指定分析输入，使用校准数据第一帧')
                analysis_inputs = [rv_sample, pb_sample]
            else:
                print('[INFO] 未指定分析输入，将根据 config 自动生成随机数据')

    # 5. 转换 RKNN（包含精度分析）
    if not args.onnx_only:
        convert_to_rknn(
            cfg, onnx_path, rknn_path, args.quantize, dataset_path,
            do_accuracy_analysis=args.accuracy_analysis,
            analysis_inputs=analysis_inputs,
            target=args.analysis_target,
            device_id=args.device_id
        )

    print()
    print('=== 转换完成 ===')
    print(f'  Backbone ONNX : {onnx_path}')
    if not args.onnx_only:
        print(f'  Backbone RKNN : {rknn_path}')
    print(f'  Head weights  : {npz_path}')
    print()
    print('  请将以下文件复制到 caf_node/models/ 目录：')
    print(f'    {os.path.basename(rknn_path if not args.onnx_only else onnx_path)}')
    print(f'    {os.path.basename(npz_path)}')
    print(f'    {os.path.basename(cfg_path)}')


if __name__ == '__main__':
    main()