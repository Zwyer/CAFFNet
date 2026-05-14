# -*- coding: utf-8 -*-
"""
tools/infer_semantickitti.py
SemanticKITTI 推理脚本 — 生成官方格式预测文件 / 计算 val mIoU

用法：
  # 在 val 集（Seq 08）上计算 mIoU
  python tools/infer_semantickitti.py \\
      --cfg prfnet/configs/prfnet_semantickitti.yaml \\
      --ckpts runs/exp_a/checkpoints/best.pth runs/exp_b/checkpoints/best.pth runs/exp_c/checkpoints/best.pth \\
      --split val

  # 对 test 集（Seq 11-21）生成预测文件，用于提交官方榜单
  python tools/infer_semantickitti.py \\
      --cfg prfnet/configs/prfnet_semantickitti.yaml \\
      --ckpts runs/exp_a/checkpoints/best.pth runs/exp_b/checkpoints/best.pth runs/exp_c/checkpoints/best.pth \\
      --split test \\
      --output_dir predictions/

  # 从目录中按 val mIoU 自动选择 top-k checkpoint 做集成
  python tools/infer_semantickitti.py ... --ckpt_dir runs/xxx/checkpoints --topk 5

  # 开启 TTA（val/test 均可，命令行优先于 config）
  python tools/infer_semantickitti.py ... --use_tta --tta_augs 4 --tta_workers 2

  # 覆盖 config 里的 KNN 开关（test 模式常用）
  python tools/infer_semantickitti.py ... --split test --use_knn --knn_dual

官方提交格式：
  predictions/sequences/{seq}/predictions/{frame_id}.label
  每个 .label 文件：原始帧点数 × uint32，低 16 位为语义标签（原始 SemanticKITTI ID）
  未落在有效距离范围内的点统一赋 0（unlabeled）

评测指标：
  SemanticKITTI 官方榜单使用各序列的逐帧 mIoU 聚合，最终以 19 类平均 IoU 排名
"""

import os
import sys
import argparse
import glob
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

# 将项目根目录加入搜索路径（tools/ 子目录中执行时需要）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prfnet.models.prfnet import PRFNet
from prfnet.datasets.semantickitti import (
    CLASS_NAMES,
    LABEL_MAPPING, TRAIN_SEQS, VAL_SEQS, TEST_SEQS,
)
from prfnet.utils.projection import RangeImageProjector, PolarBEVProjector


# ─────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────

def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _write_pred_pcd_binary(path: str,
                           pts_xyzi: np.ndarray,
                           pred_train: np.ndarray,
                           conf: np.ndarray = None,
                           gt_train: np.ndarray = None):
    """
    写带预测字段的 binary PCD，便于可视化实际预测效果。

    基础字段：
      x y z intensity pred
    可选字段：
      conf, gt
    """
    n = int(pts_xyzi.shape[0])
    if pred_train.shape[0] != n:
        raise ValueError(f'pred length {pred_train.shape[0]} != points {n}')
    if conf is not None and conf.shape[0] != n:
        raise ValueError(f'conf length {conf.shape[0]} != points {n}')
    if gt_train is not None and gt_train.shape[0] != n:
        raise ValueError(f'gt length {gt_train.shape[0]} != points {n}')

    fields = ['x', 'y', 'z', 'intensity', 'pred']
    sizes = [4, 4, 4, 4, 4]
    types = ['F', 'F', 'F', 'F', 'U']
    counts = [1, 1, 1, 1, 1]
    dtype_items = [
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
        ('pred', np.uint32),
    ]

    if conf is not None:
        fields.append('conf')
        sizes.append(4)
        types.append('F')
        counts.append(1)
        dtype_items.append(('conf', np.float32))
    if gt_train is not None:
        fields.append('gt')
        sizes.append(4)
        types.append('U')
        counts.append(1)
        dtype_items.append(('gt', np.uint32))

    arr = np.empty(n, dtype=np.dtype(dtype_items))
    arr['x'] = pts_xyzi[:, 0].astype(np.float32, copy=False)
    arr['y'] = pts_xyzi[:, 1].astype(np.float32, copy=False)
    arr['z'] = pts_xyzi[:, 2].astype(np.float32, copy=False)
    arr['intensity'] = pts_xyzi[:, 3].astype(np.float32, copy=False)
    arr['pred'] = pred_train.astype(np.uint32, copy=False)
    if conf is not None:
        arr['conf'] = conf.astype(np.float32, copy=False)
    if gt_train is not None:
        arr['gt'] = gt_train.astype(np.uint32, copy=False)

    header = (
        '# .PCD v0.7 - Point Cloud Data file format\n'
        'VERSION 0.7\n'
        f'FIELDS {" ".join(fields)}\n'
        f'SIZE {" ".join(str(x) for x in sizes)}\n'
        f'TYPE {" ".join(types)}\n'
        f'COUNT {" ".join(str(x) for x in counts)}\n'
        f'WIDTH {n}\n'
        'HEIGHT 1\n'
        'VIEWPOINT 0 0 0 1 0 0 0\n'
        f'POINTS {n}\n'
        'DATA binary\n'
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        arr.tofile(f)


# ─────────────────────────────────────────────────────────────
# train_id → 官方 SemanticKITTI 原始标签 ID
# ─────────────────────────────────────────────────────────────
TRAIN_TO_RAW = {
    0:  10,   # car
    1:  11,   # bicycle
    2:  15,   # motorcycle
    3:  18,   # truck
    4:  20,   # other-vehicle
    5:  30,   # person
    6:  31,   # bicyclist
    7:  32,   # motorcyclist
    8:  40,   # road
    9:  44,   # parking
    10: 48,   # sidewalk
    11: 49,   # other-ground
    12: 50,   # building
    13: 51,   # fence
    14: 70,   # vegetation
    15: 71,   # trunk
    16: 72,   # terrain
    17: 80,   # pole
    18: 81,   # traffic-sign
    255: 0,   # ignore → unlabeled
}

_TRAIN2RAW_LUT = np.zeros(256, dtype=np.uint32)
for t, r in TRAIN_TO_RAW.items():
    _TRAIN2RAW_LUT[t] = r


# ─────────────────────────────────────────────────────────────
# TTA 变换函数（作用于原始点云 ndarray，in-place）
# ─────────────────────────────────────────────────────────────

def _tta_identity(pts: np.ndarray) -> np.ndarray:
    return pts

def _tta_flip_x(pts: np.ndarray) -> np.ndarray:
    out = pts.copy(); out[:, 0] *= -1; return out

def _tta_flip_y(pts: np.ndarray) -> np.ndarray:
    out = pts.copy(); out[:, 1] *= -1; return out

def _tta_flip_xy(pts: np.ndarray) -> np.ndarray:
    out = pts.copy(); out[:, 0] *= -1; out[:, 1] *= -1; return out

_TTA_FNS = [_tta_identity, _tta_flip_x, _tta_flip_y, _tta_flip_xy]


# ─────────────────────────────────────────────────────────────
# 几何点特征提取（与训练 SemanticKITTIDataset 保持一致）
# ─────────────────────────────────────────────────────────────

def _compute_geo_features(pts: np.ndarray, rv_img: np.ndarray, pb_img: np.ndarray,
                          rv_proj: RangeImageProjector,
                          pb_proj: PolarBEVProjector,
                          R_max: float) -> np.ndarray:
    """
    计算 5 维几何点特征，拼接到 (x,y,z,intensity) 后得到 9 维。
    逻辑与 SemanticKITTIDataset.__getitem__ 中的 use_geometric_pt_features 完全一致。
    """
    # --- Column-to-Point Lifting (PB voxel stats → per-point) ---
    pb_row, pb_col = pb_proj._point_to_grid(pts[:, :3])
    z_min_col = pb_img[3, pb_row, pb_col] * R_max
    z_max_col = pb_img[4, pb_row, pb_col] * R_max
    z_min_col = np.clip(z_min_col, -5.0, 5.0)
    h_above  = np.clip(pts[:, 2] - z_min_col, 0.0, 8.0)
    col_h    = np.maximum(z_max_col - z_min_col, 0.1)
    h_norm   = np.clip(h_above / col_h, 0.0, 1.0)
    log_dens = pb_img[6, pb_row, pb_col]

    # --- Scan-line Curvature (RV depth gradient) ---
    rv_row, rv_col = rv_proj.compute_pixel_coords(pts[:, :3])
    depth_map = rv_img[3].copy()                          # r/R_max
    valid_rv  = depth_map > 0
    # 方位角方向梯度（周期循环）
    grad_w = np.abs(np.roll(depth_map, -1, axis=1)
                    - np.roll(depth_map,  1, axis=1)) / 2.0
    grad_w *= (np.roll(valid_rv, -1, axis=1)
               & np.roll(valid_rv,  1, axis=1)).astype(np.float32)
    # 仰角方向梯度（边界 edge padding）
    dp = np.pad(depth_map, ((1, 1), (0, 0)), mode='edge')
    vp = np.pad(valid_rv,  ((1, 1), (0, 0)), mode='constant',
                constant_values=False)
    grad_h = np.abs(dp[2:] - dp[:-2]) / 2.0
    grad_h *= (vp[2:] & vp[:-2]).astype(np.float32)
    norm = 0.3
    grad_w_pt = np.clip(grad_w / norm, 0.0, 1.0)[rv_row, rv_col]
    grad_h_pt = np.clip(grad_h / norm, 0.0, 1.0)[rv_row, rv_col]

    return np.stack([
        pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3],
        h_above / 8.0, h_norm, log_dens, grad_w_pt, grad_h_pt,
    ], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# TTA 预处理 worker（在 ThreadPoolExecutor 线程中运行）
# ─────────────────────────────────────────────────────────────

def _prepare_aug(aug_fn, rv_proj: RangeImageProjector,
                 pb_proj: PolarBEVProjector,
                 raw_pts: np.ndarray, R_max: float,
                 use_geo_pt: bool = False,
                 pb_continuous_coords: bool = False) -> dict:
    """
    CPU 密集型：对 raw_pts 施加变换，重新投影，返回 CPU 张量字典。
    在独立线程中运行，与 GPU 推理流水线并行。
    距离在旋转/翻转下保持不变，valid_mask 逻辑等价于原始帧。
    """
    pts = aug_fn(raw_pts)                               # (N_raw, 4)
    dist = np.linalg.norm(pts[:, :3], axis=1)
    valid = (dist > 0.5) & (dist < R_max)
    pts_v = pts[valid]
    intensity = pts_v[:, 3].copy()

    rv_img, _  = rv_proj.project(pts_v[:, :3], intensity)
    rv_coords  = rv_proj.compute_sample_coords(pts_v[:, :3])
    pb_img     = pb_proj.project(pts_v[:, :3], intensity)
    pb_coords  = pb_proj.compute_sample_coords(pts_v[:, :3])

    return dict(
        rv_img    = torch.from_numpy(rv_img),
        pb_img    = torch.from_numpy(pb_img),
        rv_coords = torch.from_numpy(rv_coords).unsqueeze(1),   # (N, 1, 2)
        pb_coords = torch.from_numpy(pb_coords).unsqueeze(1),
        points    = torch.from_numpy(
            _compute_geo_features(pts_v, rv_img, pb_img, rv_proj, pb_proj, R_max)
            if use_geo_pt else pts_v.astype(np.float32)),
        valid     = valid,
    )


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    mc = cfg['model']
    dc = cfg['data']
    _use_normals = mc.get('use_surface_normals', False)
    _use_angle   = mc.get('use_angle_encoding',  False)
    _use_geo_pt  = mc.get('use_geometric_pt_features', False)
    _use_pb_z_hist = mc.get('use_pb_z_hist', False)
    rv_in  = 6 + 3 * int(_use_normals) + 3 * int(_use_angle)
    pb_in  = mc.get('pb_in', 9 + 3 * int(_use_pb_z_hist))
    pt_dim = 4 + 5 * int(_use_geo_pt)

    model = PRFNet(
        rv_in=rv_in, pb_in=pb_in,
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
        head_dropout=mc.get('head_dropout', 0.1),
        use_vcg=mc.get('use_vcg', True),
        use_proto=mc.get('use_proto', True),
        proto_dim=mc.get('proto_dim', 64),
#        use_multiscale_agg=mc.get('use_multiscale_agg', False),
#        use_w_aspp=mc.get('use_w_aspp', False),
#        use_circular_padding=mc.get('use_circular_padding', False),
#        use_se=mc.get('use_se', False),
#        se_reduction=mc.get('se_reduction', 8),
#        use_assertion_gate=mc.get('use_assertion_gate', False),
#        pt_dim=pt_dim,
    ).to(device)
    return model


def load_checkpoint(model: torch.nn.Module,
                    ckpt_path: str,
                    device: torch.device,
                    use_ema: bool = True) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if use_ema and 'ema_state_dict' in ckpt:
        state = ckpt['ema_state_dict']
        key = 'ema_state_dict'
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
        key = 'state_dict'
    else:
        state = ckpt
        key = 'raw_state_dict'

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'[警告] 缺少权重 ({key}): {missing}')
    if unexpected:
        print(f'[警告] 多余权重 ({key}): {unexpected}')

    epoch = ckpt.get('epoch', '?')
    try:
        miou = float(ckpt.get('miou', float('nan')))
    except Exception:
        miou = float('nan')
    if not np.isfinite(miou):
        miou = float('nan')
    print(f'已加载 checkpoint: {os.path.abspath(ckpt_path)}')
    print(f'  epoch={epoch}  val_mIoU={miou:.2f}%  (来源: {key})')
    return {
        'path': os.path.abspath(ckpt_path),
        'epoch': epoch,
        'miou': miou,
        'key': key,
    }


def _dedup_paths(paths: List[str]) -> List[str]:
    out = []
    seen = set()
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)
    return out


def _rank_checkpoints_by_miou(ckpt_paths: List[str], topk: int) -> List[str]:
    scored = []
    valid_paths = []
    for p in ckpt_paths:
        if not os.path.isfile(p):
            print(f'[警告] checkpoint 不存在，已跳过: {p}')
            continue
        valid_paths.append(p)
        try:
            ckpt = torch.load(p, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f'[警告] 读取 checkpoint 失败，已跳过: {p}  ({e})')
            continue

        has_state = ('ema_state_dict' in ckpt or 'state_dict' in ckpt)
        if not has_state:
            continue

        try:
            miou_val = float(ckpt.get('miou', float('nan')))
        except Exception:
            miou_val = float('nan')
        if not np.isfinite(miou_val):
            continue

        epoch = ckpt.get('epoch', -1)
        if isinstance(epoch, (int, np.integer)):
            epoch_i = int(epoch)
        else:
            try:
                epoch_i = int(epoch)
            except Exception:
                epoch_i = -1
        scored.append((float(miou_val), epoch_i, p))

    if scored:
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected = [p for _, _, p in scored[:topk]]

        print(f'按 val mIoU 选择 Top-{len(selected)} checkpoint：')
        for i, p in enumerate(selected, 1):
            miou, epoch = next((m, e) for m, e, pp in scored if pp == p)
            print(f'  [{i}] epoch={epoch:<4d}  mIoU={miou:>6.2f}%  {p}')
        return selected

    if valid_paths:
        print('[警告] 未在 checkpoint 中找到可用 miou，回退到输入顺序。')
        return valid_paths[:topk]
    return []


def _enable_mc_dropout(model: torch.nn.Module) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d,
                          torch.nn.Dropout3d, torch.nn.AlphaDropout)):
            m.train()
            n += 1
    return n


def _prepare_models_for_infer(models: List[torch.nn.Module],
                              mc_dropout_passes: int = 1) -> int:
    drop_layers = 0
    for model in models:
        model.eval()
        if mc_dropout_passes > 1:
            drop_layers += _enable_mc_dropout(model)
    return drop_layers


@torch.no_grad()
def _forward_ensemble_logits(models: List[torch.nn.Module],
                             rv_img: torch.Tensor,
                             pb_img: torch.Tensor,
                             rv_coords: torch.Tensor,
                             pb_coords: torch.Tensor,
                             points: torch.Tensor,
                             mc_dropout_passes: int = 1) -> torch.Tensor:
    logits_sum = None
    total_passes = 0

    for model in models:
        passes = max(1, mc_dropout_passes)
        for _ in range(passes):
            outputs = model(rv_img, pb_img, rv_coords, pb_coords, points)
            logits = outputs['logits'].squeeze(0).float()
            logits_sum = logits if logits_sum is None else (logits_sum + logits)
            total_passes += 1

    return logits_sum / max(total_passes, 1)


# ─────────────────────────────────────────────────────────────
# KNN 后处理
# ─────────────────────────────────────────────────────────────

def _grid_votes(coords_norm, preds, grid_H, grid_W, num_classes, radius,
                pb_mode=False, fix_vote_accum=False):
    N = len(preds)

    # 阶段 A：几何对齐修复
    if pb_mode:
        cols = np.clip(
            np.floor((coords_norm[:, 0] + 1.0) * 0.5 * grid_W).astype(np.int32),
            0, grid_W - 1,
        )
        rows = np.clip(
            np.floor((coords_norm[:, 1] + 1.0) * 0.5 * grid_H).astype(np.int32),
            0, grid_H - 1,
        )
    else:
        cols = np.clip(
            np.round((coords_norm[:, 0] + 1.0) * 0.5 * (grid_W - 1)).astype(np.int32),
            0, grid_W - 1,
        )
        rows = np.clip(
            np.round((coords_norm[:, 1] + 1.0) * 0.5 * (grid_H - 1)).astype(np.int32),
            0, grid_H - 1,
        )

    # 阶段 B：投票累计修复
    if fix_vote_accum:
        cell_votes = np.zeros((grid_H, grid_W, num_classes), dtype=np.int32)
        np.add.at(cell_votes, (rows, cols, preds), 1)

        padded = np.pad(cell_votes, ((radius, radius), (radius, radius), (0, 0)), mode='constant')
        padded[:, :radius, :] = padded[:, -2*radius:-radius, :]
        padded[:, -radius:, :] = padded[:, radius:2*radius, :]

        pr = rows + radius
        pc = cols + radius
        votes = np.zeros((N, num_classes), dtype=np.int32)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                votes += padded[pr + dr, pc + dc]
    else:
        pred_map = np.full((grid_H, grid_W), -1, dtype=np.int32)
        pred_map[rows, cols] = preds

        padded = np.pad(pred_map, pad_width=radius, mode='wrap')
        padded[:radius, :]  = -1
        padded[-radius:, :] = -1

        pr = rows + radius
        pc = cols + radius
        votes = np.zeros((N, num_classes), dtype=np.int32)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                neighbor = padded[pr + dr, pc + dc]
                valid = neighbor >= 0
                if valid.any():
                    votes[valid, neighbor[valid]] += 1
    return votes


def _apply_votes(preds, votes, confs=None, knn_cfg=None):
    """阶段 C：门控投票应用"""
    if knn_cfg is None:
        knn_cfg = {}

    has_votes = votes.sum(axis=1) > 0
    replace_mask = has_votes.copy()

    if knn_cfg.get('knn_use_conf_gate', False) and confs is not None:
        low_conf = confs < knn_cfg.get('knn_conf_thresh', 0.7)
        replace_mask &= low_conf

    if knn_cfg.get('knn_use_vote_ratio_gate', False):
        vote_max = votes.max(axis=1)
        vote_sum = votes.sum(axis=1)
        vote_ratio = np.where(vote_sum > 0, vote_max / vote_sum, 0.0)
        strong_consensus = vote_ratio >= knn_cfg.get('knn_vote_ratio_thresh', 0.6)
        replace_mask &= strong_consensus

    if knn_cfg.get('knn_use_protected_classes', False):
        protected = knn_cfg.get('knn_protected_classes', [])
        is_protected = np.isin(preds, protected)
        replace_mask &= ~is_protected

    refined = preds.copy()
    if replace_mask.any():
        vote_winners = votes[replace_mask].argmax(axis=1)
        if knn_cfg.get('knn_tie_break', 'keep_pred') == 'keep_pred':
            vote_max_vals = votes[replace_mask].max(axis=1)
            vote_counts = (votes[replace_mask] == vote_max_vals[:, None]).sum(axis=1)
            tied = vote_counts > 1
            vote_winners[tied] = preds[replace_mask][tied]
        refined[replace_mask] = vote_winners.astype(preds.dtype)

    return refined


def knn_refine(rv_coords_norm, pb_coords_norm, preds,
               rv_H, rv_W, pb_H, pb_W, num_classes, radius, dual,
               confs=None, knn_cfg=None):
    if knn_cfg is None:
        knn_cfg = {}

    if dual:
        votes_rv = _grid_votes(rv_coords_norm, preds, rv_H, rv_W, num_classes, radius,
                               pb_mode=False, fix_vote_accum=knn_cfg.get('knn_fix_vote_accum', False))
        votes_pb = _grid_votes(pb_coords_norm, preds, pb_H, pb_W, num_classes, radius,
                               pb_mode=knn_cfg.get('knn_fix_pb_mapping', False),
                               fix_vote_accum=knn_cfg.get('knn_fix_vote_accum', False))

        # 归一化融合：RV(64×1024) 和 PB(480×1024) 邻域密度差异 ~7.5×
        # 直接相加导致 RV 占 88%，PB 仅 12%，双视图融合失效
        # 归一化到概率后等权融合，保证语义权重平衡
        rv_sum = votes_rv.sum(axis=1, keepdims=True)
        pb_sum = votes_pb.sum(axis=1, keepdims=True)
        votes_rv_norm = np.where(rv_sum > 0, votes_rv / rv_sum, 0)
        votes_pb_norm = np.where(pb_sum > 0, votes_pb / pb_sum, 0)

        # 等权融合后恢复整数票数
        avg_neighbors = (2 * radius + 1) ** 2
        votes = ((votes_rv_norm + votes_pb_norm) * avg_neighbors).astype(np.int32)
    else:
        votes = _grid_votes(rv_coords_norm, preds, rv_H, rv_W, num_classes, radius,
                            pb_mode=False, fix_vote_accum=knn_cfg.get('knn_fix_vote_accum', False))

    return _apply_votes(preds, votes, confs, knn_cfg)


# ─────────────────────────────────────────────────────────────
# mIoU 评估
# ─────────────────────────────────────────────────────────────

class IoUMetric:
    def __init__(self, num_classes, ignore_index=255):
        self.C  = num_classes
        self.ig = ignore_index
        self.reset()

    def reset(self):
        self.intersect = np.zeros(self.C, dtype=np.float64)
        self.union     = np.zeros(self.C, dtype=np.float64)
        self.cm        = np.zeros((self.C, self.C), dtype=np.float64)  # cm[gt, pred]

    def update(self, preds: np.ndarray, labels: np.ndarray):
        p = preds.flatten().astype(np.int64)
        g = labels.flatten().astype(np.int64)
        valid = g != self.ig
        p, g = p[valid], g[valid]
        if len(p) == 0:
            return
        idx = g * self.C + p
        cm  = np.bincount(idx, minlength=self.C * self.C).reshape(self.C, self.C)
        self.cm        += cm
        self.intersect += np.diag(cm)
        self.union     += cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)

    def per_class_iou(self):
        valid = self.union > 0
        iou = np.where(valid, self.intersect / (self.union + 1e-10), np.nan)
        return iou * 100.

    def miou(self):
        return float(np.nanmean(self.per_class_iou()))

    def per_class_precision(self):
        """每类 precision = TP / (TP + FP) = cm[c,c] / cm[:,c].sum()"""
        tp = np.diag(self.cm)
        denom = self.cm.sum(axis=0)  # predicted as class c (col sum)
        return np.where(denom > 0, tp / (denom + 1e-10), np.nan) * 100.

    def per_class_recall(self):
        """每类 recall = TP / (TP + FN) = cm[c,c] / cm[c,:].sum()"""
        tp = np.diag(self.cm)
        denom = self.cm.sum(axis=1)  # actual class c (row sum)
        return np.where(denom > 0, tp / (denom + 1e-10), np.nan) * 100.


# ─────────────────────────────────────────────────────────────
# 推理数据集（逐帧，保留原始点云用于 TTA 重投影）
# ─────────────────────────────────────────────────────────────

class InferenceDataset:
    """
    按帧加载原始点云，保留 raw_pts 用于 TTA 重投影。

    返回字典：
      raw_pts:     (N_raw, 4)   float32 ndarray — 完整原始点云
      rv_img:      (C_rv, H, W) float32 tensor  — 标准投影（identity aug）
      pb_img:      (9, H_p, W_p)
      rv_coords:   (N_valid, 1, 2)
      pb_coords:   (N_valid, 1, 2)
      points:      (N_valid, 4)
      valid_mask:  (N_raw,) bool ndarray
      raw_N:       int
      seq:         str
      stem:        str
      labels:      (N_valid,) int64 tensor（仅 val split 有效）
    """

    def __init__(self, root: str, split: str, dc: dict, mc: dict):
        self.R_max   = dc.get('R_max', 80.0)
        self.rv_proj = RangeImageProjector(
            dc['rv_H'], dc['rv_W'],
            dc['fov_up'], dc['fov_down'], self.R_max,
            use_surface_normals=mc.get('use_surface_normals', False),
            use_angle_encoding=mc.get('use_angle_encoding',  False),
        )
        self.pb_proj = PolarBEVProjector(dc['pb_H'], dc['pb_W'], self.R_max
                                        )
        self.use_geo_pt = mc.get('use_geometric_pt_features', False)
        self.pb_continuous_coords = mc.get('pb_continuous_coords', False)

        lut = np.full(65536, 255, dtype=np.int32)
        for raw, train in LABEL_MAPPING.items():
            if raw < 65536:
                lut[raw] = train
        self._lut = lut

        seqs = {'train': TRAIN_SEQS, 'val': VAL_SEQS, 'test': TEST_SEQS}[split]
        self.frames = []
        for seq in seqs:
            velo_dir  = os.path.join(root, 'sequences', seq, 'velodyne')
            label_dir = os.path.join(root, 'sequences', seq, 'labels')
            for b in sorted(os.listdir(velo_dir)):
                stem = b.replace('.bin', '')
                entry = {'velo': os.path.join(velo_dir, b), 'seq': seq, 'stem': stem}
                if split != 'test' and os.path.isdir(label_dir):
                    entry['label'] = os.path.join(label_dir, stem + '.label')
                self.frames.append(entry)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int):
        entry   = self.frames[idx]
        raw_pts = np.fromfile(entry['velo'], dtype=np.float32).reshape(-1, 4)
        raw_N   = len(raw_pts)

        dist       = np.linalg.norm(raw_pts[:, :3], axis=1)
        valid_mask = (dist > 0.5) & (dist < self.R_max)
        pts        = raw_pts[valid_mask]
        intensity  = pts[:, 3].copy()

        rv_img, _  = self.rv_proj.project(pts[:, :3], intensity)
        rv_coords  = self.rv_proj.compute_sample_coords(pts[:, :3])
        pb_img     = self.pb_proj.project(pts[:, :3], intensity)
        pb_coords  = self.pb_proj.compute_sample_coords(pts[:, :3])

        if 'label' in entry:
            raw_labels = np.fromfile(entry['label'], dtype=np.uint32)
            raw_labels = (raw_labels & 0xFFFF).astype(np.int32)
            labels     = self._lut[raw_labels[valid_mask]]
        else:
            labels = np.zeros(len(pts), dtype=np.int32)

        # 几何点特征
        if self.use_geo_pt:
            pts_aug = _compute_geo_features(pts, rv_img, pb_img,
                                            self.rv_proj, self.pb_proj, self.R_max)
        else:
            pts_aug = pts.astype(np.float32)

        return {
            'raw_pts':    raw_pts,          # ndarray，TTA 重投影用
            'rv_img':     torch.from_numpy(rv_img),
            'pb_img':     torch.from_numpy(pb_img),
            'rv_coords':  torch.from_numpy(rv_coords).unsqueeze(1),
            'pb_coords':  torch.from_numpy(pb_coords).unsqueeze(1),
            'points':     torch.from_numpy(pts_aug),
            'labels':     torch.from_numpy(labels.astype(np.int64)),
            'valid_mask': valid_mask,
            'raw_N':      raw_N,
            'seq':        entry['seq'],
            'stem':       entry['stem'],
        }


# ─────────────────────────────────────────────────────────────
# 单帧 TTA 推理（val/test 共用）
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def _infer_one_frame(models, sample: dict, device: torch.device,
                     rv_proj: RangeImageProjector,
                     pb_proj: PolarBEVProjector,
                     tta_fns: list, tta_workers: int,
                     R_max: float,
                     use_geo_pt: bool = False,
                     pb_continuous_coords: bool = False,
                     mc_dropout_passes: int = 1,
                     return_logits: bool = False):
    """
    对单帧执行（可选 TTA 的）推理，返回 (preds, confs) 或 (preds, confs, logits)。

    Args:
        return_logits: 若为 True，返回 (preds, confs, logits)，用于 CRF 后处理
    """
    # 非 TTA
    if len(tta_fns) == 1:
        rv_img    = sample['rv_img'].unsqueeze(0).to(device)
        pb_img    = sample['pb_img'].unsqueeze(0).to(device)
        rv_coords = sample['rv_coords'].unsqueeze(0).to(device)
        pb_coords = sample['pb_coords'].unsqueeze(0).to(device)
        points    = sample['points'].unsqueeze(0).to(device)
        logits = _forward_ensemble_logits(
            models, rv_img, pb_img, rv_coords, pb_coords, points,
            mc_dropout_passes=mc_dropout_passes
        )
        preds     = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
        confs     = torch.softmax(logits, dim=-1).max(dim=-1)[0].cpu().numpy().astype(np.float32)
        if return_logits:
            return preds, confs, logits.cpu().numpy().astype(np.float32)
        return preds, confs

    # TTA：累积 logits，最后统一计算 confs
    raw_pts = sample['raw_pts']
    logits_sum = None
    executor   = ThreadPoolExecutor(max_workers=tta_workers)

    future_list = []
    for fn in tta_fns[1:]:
        future_list.append(
            executor.submit(_prepare_aug, fn, rv_proj, pb_proj, raw_pts, R_max, use_geo_pt, pb_continuous_coords)
        )

    rv_img    = sample['rv_img'].unsqueeze(0).to(device)
    pb_img    = sample['pb_img'].unsqueeze(0).to(device)
    rv_coords = sample['rv_coords'].unsqueeze(0).to(device)
    pb_coords = sample['pb_coords'].unsqueeze(0).to(device)
    points    = sample['points'].unsqueeze(0).to(device)
    logits_sum = _forward_ensemble_logits(
        models, rv_img, pb_img, rv_coords, pb_coords, points,
        mc_dropout_passes=mc_dropout_passes
    )

    for future in future_list:
        data = future.result()
        rv_img    = data['rv_img'].unsqueeze(0).to(device)
        pb_img    = data['pb_img'].unsqueeze(0).to(device)
        rv_coords = data['rv_coords'].unsqueeze(0).to(device)
        pb_coords = data['pb_coords'].unsqueeze(0).to(device)
        points    = data['points'].unsqueeze(0).to(device)
        logits_sum = logits_sum + _forward_ensemble_logits(
            models, rv_img, pb_img, rv_coords, pb_coords, points,
            mc_dropout_passes=mc_dropout_passes
        )

    executor.shutdown(wait=False)

    # 阶段 E：TTA 时从 logits_mean 计算 confs（避免虚高）
    logits_mean = logits_sum / len(tta_fns)
    preds = logits_mean.argmax(dim=-1).cpu().numpy().astype(np.int32)
    confs = torch.softmax(logits_mean, dim=-1).max(dim=-1)[0].cpu().numpy().astype(np.float32)
    if return_logits:
        return preds, confs, logits_mean.cpu().numpy().astype(np.float32)
    return preds, confs


# ─────────────────────────────────────────────────────────────
# val 模式：计算 mIoU
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_val(models, cfg, device, knn_cfg, tta_cfg, infer_cfg, crf_cfg=None):
    dc = cfg['data']
    mc = cfg['model']

    use_tta     = tta_cfg['use_tta']
    tta_fns     = _TTA_FNS[:tta_cfg['tta_augs']]
    tta_workers = tta_cfg['tta_workers']
    mc_dropout_passes = infer_cfg.get('mc_dropout_passes', 1)

    use_knn  = knn_cfg['use_knn']
    knn_rad  = knn_cfg['knn_radius']
    knn_dual = knn_cfg['knn_dual']
    num_cls  = dc['num_classes']
    ignore   = dc['ignore_index']
    R_max    = dc.get('R_max', 80.0)
    save_pcd_cfg = infer_cfg.get('save_val_pcd', {}) or {}
    save_pcd_enable = bool(save_pcd_cfg.get('enable', False))
    save_pcd_dir = str(save_pcd_cfg.get('dir', ''))
    save_pcd_every = max(1, int(save_pcd_cfg.get('every', 1)))
    save_pcd_max_frames = int(save_pcd_cfg.get('max_frames', 0))
    save_pcd_with_gt = bool(save_pcd_cfg.get('with_gt', True))
    save_pcd_with_conf = bool(save_pcd_cfg.get('with_conf', False))

    # CRF 配置
    use_crf = crf_cfg.get('use_crf', False) if crf_cfg else False
    use_adaptive = crf_cfg.get('crf_distance_adaptive', True) if crf_cfg else True
    if use_crf:
        try:
            if use_adaptive:
                from prfnet.postprocess.crf import crf_refine_adaptive as crf_refine
            else:
                from prfnet.postprocess.crf import crf_refine
        except ImportError:
            print("警告：CRF 未安装，跳过 CRF 后处理")
            use_crf = False

    # TTA 或非 TTA 均使用 InferenceDataset（保留 raw_pts）
    dataset = InferenceDataset(root=dc['root'], split='val', dc=dc, mc=mc)

    rv_proj = dataset.rv_proj
    pb_proj = dataset.pb_proj

    n_workers = dc.get('val_num_workers', dc.get('num_workers', 4))
    loader = DataLoader(dataset, batch_size=1, num_workers=n_workers,
                        collate_fn=lambda b: b[0])

    metric = IoUMetric(num_cls, ignore)
    _prepare_models_for_infer(models, mc_dropout_passes=mc_dropout_passes)
    t0 = time.time()

    tta_desc = f'TTA×{len(tta_fns)}' if use_tta else 'no-TTA'
    ens_desc = f'ens×{len(models)}'
    mcdo_desc = f'+MCDO×{mc_dropout_passes}' if mc_dropout_passes > 1 else ''
    crf_desc = '+CRF(adaptive)' if (use_crf and use_adaptive) else ('+CRF' if use_crf else '')
    seen_count = 0
    saved_count = 0
    for sample in tqdm(loader, desc=f'val [{ens_desc}{mcdo_desc}|{tta_desc}{crf_desc}]', dynamic_ncols=True):
        seen_count += 1
        labels = sample['labels'].numpy()

        fns = tta_fns if use_tta else [_tta_identity]
        result = _infer_one_frame(models, sample, device,
                                  rv_proj, pb_proj, fns, tta_workers, R_max,
                                  mc.get('use_geometric_pt_features', False),
                                  mc.get('pb_continuous_coords', False),
                                  mc_dropout_passes=mc_dropout_passes,
                                  return_logits=use_crf)

        if use_crf:
            preds, confs, logits = result
        else:
            preds, confs = result
            logits = None

        # 阶段 D：ignore 过滤后再 KNN
        if use_knn:
            valid = labels != ignore
            if valid.any():
                rv_c = sample['rv_coords'][:, 0, :].numpy()
                pb_c = sample['pb_coords'][:, 0, :].numpy()
                preds[valid] = knn_refine(
                    rv_c[valid], pb_c[valid], preds[valid],
                    dc['rv_H'], dc['rv_W'], dc['pb_H'], dc['pb_W'],
                    num_cls, knn_rad, knn_dual,
                    confs=confs[valid] if confs is not None else None,
                    knn_cfg=knn_cfg
                )

        # CRF 后处理（在 KNN 之后）
        if use_crf and logits is not None:
            valid = labels != ignore
            if valid.any():
                raw_pts = _as_numpy(sample['raw_pts'])
                # 提取 intensity（raw_pts 第4列）
                intensity = raw_pts[valid, 3] if raw_pts.shape[1] > 3 else None

                crf_params = {
                    'points': raw_pts[valid, :3],
                    'logits': logits[valid],
                    'num_classes': num_cls,
                    'intensity': intensity,
                    'theta_alpha': crf_cfg.get('crf_theta_alpha', 0.3),
                    'theta_beta': crf_cfg.get('crf_theta_beta', 0.5),
                    'theta_gamma': crf_cfg.get('crf_theta_gamma', 0.15),
                    'compat_spatial': crf_cfg.get('crf_compat_spatial', 3.0),
                    'compat_bilateral': crf_cfg.get('crf_compat_bilateral', 10.0),
                    'num_iter': crf_cfg.get('crf_num_iter', 5)
                }

                if use_adaptive:
                    crf_params['distance_adaptive'] = True

                preds[valid] = crf_refine(**crf_params)

        metric.update(preds, labels)

        if save_pcd_enable and save_pcd_dir:
            allow_by_every = ((seen_count - 1) % save_pcd_every == 0)
            allow_by_max = (save_pcd_max_frames <= 0) or (saved_count < save_pcd_max_frames)
            if allow_by_every and allow_by_max:
                raw_pts = _as_numpy(sample['raw_pts'])
                valid_mask = _as_numpy(sample['valid_mask']).astype(bool)
                pts_valid = raw_pts[valid_mask]
                if pts_valid.shape[0] == preds.shape[0]:
                    gt_vals = labels.astype(np.uint32) if save_pcd_with_gt else None
                    conf_vals = confs.astype(np.float32) if (save_pcd_with_conf and confs is not None) else None
                    seq = str(sample['seq'])
                    stem = str(sample['stem'])
                    out_path = os.path.join(save_pcd_dir, seq, f'{stem}.pcd')
                    _write_pred_pcd_binary(
                        out_path,
                        pts_xyzi=pts_valid,
                        pred_train=preds.astype(np.uint32),
                        conf=conf_vals,
                        gt_train=gt_vals,
                    )
                    saved_count += 1
                else:
                    print(f'[WARN] skip PCD save due to shape mismatch: pts={pts_valid.shape[0]} pred={preds.shape[0]}')

    total_seen = int(seen_count)
    total_saved = int(saved_count)

    elapsed = time.time() - t0
    per_cls = metric.per_class_iou()
    miou    = metric.miou()

    print(f'\n{"─"*60}')
    print(f'Val mIoU: {miou:.2f}%   (耗时 {elapsed:.1f}s  [{ens_desc}{mcdo_desc}|{tta_desc}{crf_desc}])')
    print(f'{"─"*60}')
    header = f'{"Class":<18}  {"IoU":>6}'
    print(header)
    print('─' * len(header))
    for name, iou in zip(CLASS_NAMES, per_cls):
        if not np.isnan(iou):
            print(f'  {name:<16}  {iou:>6.2f}')
        else:
            print(f'  {name:<16}  {"--":>6}  (no GT)')
    print(f'{"─"*60}')
    print(f'  {"mIoU":<16}  {miou:>6.2f}')

    # ── 诊断：每类 Precision / Recall ──────────────────────────
    prec = metric.per_class_precision()
    rec  = metric.per_class_recall()
    print(f'\n{"─"*60}')
    print(f'Per-class Precision / Recall')
    print(f'{"─"*60}')
    hdr = f'{"Class":<18}  {"Prec":>6}  {"Rec":>6}  {"F1":>6}'
    print(hdr)
    print('─' * len(hdr))
    for name, p_, r_ in zip(CLASS_NAMES, prec, rec):
        if np.isnan(p_) and np.isnan(r_):
            print(f'  {name:<16}  {"--":>6}  {"--":>6}  {"--":>6}  (no GT)')
        else:
            f1 = (2 * p_ * r_ / (p_ + r_ + 1e-10)) if (not np.isnan(p_) and not np.isnan(r_)) else float('nan')
            print(f'  {name:<16}  {p_:>6.2f}  {r_:>6.2f}  {f1:>6.2f}')

    # ── 诊断：motorcyclist 混淆行（motorcyclist GT 被预测为哪些类）──
    moto_idx = CLASS_NAMES.index('motorcyclist') if 'motorcyclist' in CLASS_NAMES else None
    if moto_idx is not None:
        print(f'\n{"─"*60}')
        print(f'Confusion row — GT=motorcyclist (idx {moto_idx}):')
        print(f'  {"Predicted as":<18}  {"Count":>8}  {"Frac%":>7}')
        print('─' * 42)
        row = metric.cm[moto_idx]
        total = row.sum()
        if total > 0:
            order = np.argsort(row)[::-1]
            for c in order:
                if row[c] == 0:
                    continue
                pname = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'cls{c}'
                print(f'  {pname:<18}  {int(row[c]):>8}  {row[c]/total*100:>7.2f}')
        else:
            print('  (no motorcyclist GT points in val set)')
    print(f'{"─"*60}\n')
    if save_pcd_enable and save_pcd_dir:
        print(f'PCD 导出: saved={total_saved}/{total_seen}  dir={os.path.abspath(save_pcd_dir)}')

    return miou, per_cls


# ─────────────────────────────────────────────────────────────
# test 模式：生成官方格式 .label 文件
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_test(models, cfg, device, output_dir, knn_cfg, tta_cfg, infer_cfg):
    dc = cfg['data']
    mc = cfg['model']

    use_tta     = tta_cfg['use_tta']
    tta_fns     = _TTA_FNS[:tta_cfg['tta_augs']]
    tta_workers = tta_cfg['tta_workers']
    mc_dropout_passes = infer_cfg.get('mc_dropout_passes', 1)

    use_knn  = knn_cfg['use_knn']
    knn_rad  = knn_cfg['knn_radius']
    knn_dual = knn_cfg['knn_dual']
    num_cls  = dc['num_classes']
    R_max    = dc.get('R_max', 80.0)

    dataset = InferenceDataset(root=dc['root'], split='test', dc=dc, mc=mc)
    rv_proj = dataset.rv_proj
    pb_proj = dataset.pb_proj

    n_workers = dc.get('val_num_workers', dc.get('num_workers', 4))
    loader = DataLoader(dataset, batch_size=1, num_workers=n_workers,
                        collate_fn=lambda b: b[0])

    _prepare_models_for_infer(models, mc_dropout_passes=mc_dropout_passes)
    t0 = time.time()
    label_files_written = 0

    tta_desc = f'TTA×{len(tta_fns)}' if use_tta else 'no-TTA'
    ens_desc = f'ens×{len(models)}'
    mcdo_desc = f'+MCDO×{mc_dropout_passes}' if mc_dropout_passes > 1 else ''
    for sample in tqdm(loader, desc=f'test [{ens_desc}{mcdo_desc}|{tta_desc}]', dynamic_ncols=True):

        seq  = sample['seq']
        stem = sample['stem']
        pred_dir = os.path.join(output_dir, 'sequences', seq, 'predictions')
        out_path = os.path.join(pred_dir, stem + '.label')
        if os.path.exists(out_path):
            label_files_written += 1
            continue

        fns = tta_fns if use_tta else [_tta_identity]
        preds, confs = _infer_one_frame(models, sample, device,
                                        rv_proj, pb_proj, fns, tta_workers, R_max,
                                        mc.get('use_geometric_pt_features', False),
                                        mc.get('pb_continuous_coords', False),
                                        mc_dropout_passes=mc_dropout_passes)

        # 阶段 E：test 无 GT，不做 ignore 过滤
        if use_knn:
            rv_c = sample['rv_coords'][:, 0, :].numpy()
            pb_c = sample['pb_coords'][:, 0, :].numpy()
            preds = knn_refine(rv_c, pb_c, preds,
                               dc['rv_H'], dc['rv_W'],
                               dc['pb_H'], dc['pb_W'],
                               num_cls, knn_rad, knn_dual,
                               confs=confs,
                               knn_cfg=knn_cfg)

        # 映射回原始点云空间
        raw_N      = sample['raw_N']
        valid_mask = sample['valid_mask']
        full_train = np.full(raw_N, 255, dtype=np.uint8)
        full_train[valid_mask] = preds.astype(np.uint8)
        full_raw = _TRAIN2RAW_LUT[full_train].astype(np.uint32)

        os.makedirs(pred_dir, exist_ok=True)
        full_raw.tofile(out_path)
        label_files_written += 1

    elapsed = time.time() - t0
    print(f'\n预测完成：{label_files_written} 帧  |  耗时 {elapsed:.1f}s'
          f'  ({elapsed / max(label_files_written, 1) * 1000:.1f} ms/frame)  [{ens_desc}{mcdo_desc}|{tta_desc}]')
    print(f'输出目录：{os.path.abspath(output_dir)}')
    print('\n提交步骤：')
    print('  1. 将整个 predictions/ 目录打包为 zip')
    print('  2. 上传至 http://semantic-kitti.org/tasks.html')


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='SemanticKITTI 推理脚本（val mIoU / test 官方格式输出）')
    p.add_argument('--cfg',  required=True, help='YAML 配置文件路径')
    p.add_argument('--ckpt', default=None,
                   help='单个 checkpoint 路径（.pth）')
    p.add_argument('--ckpts', nargs='+', default=None,
                   help='多个 checkpoint 路径（空格分隔，建议 3~5 个）')
    p.add_argument('--ckpt_dir', default=None,
                   help='checkpoint 目录（*.pth），可配合 --topk 自动选优')
    p.add_argument('--topk', type=int, default=None,
                   help='按 checkpoint 内 val mIoU 选 top-k（默认仅 ckpt_dir 时取 3）')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--output_dir', default='predictions',
                   help='test 模式预测文件输出根目录（默认 predictions/）')
    p.add_argument('--no_ema', action='store_true',
                   help='强制使用 state_dict 而非 ema_state_dict')
    p.add_argument('--mc_dropout_passes', type=int, default=1,
                   help='MC Dropout 前向次数（>1 启用，推理更慢但更稳）')
    p.add_argument('--save_val_pcd_dir', default=None,
                   help='val 模式下导出预测 PCD 的目录（不填则关闭）')
    p.add_argument('--save_val_pcd_every', type=int, default=1,
                   help='val 模式每 N 帧导出 1 帧 PCD（默认1）')
    p.add_argument('--save_val_pcd_max_frames', type=int, default=0,
                   help='val 模式最多导出多少帧 PCD（0=不限制）')
    p.add_argument('--save_val_pcd_with_gt', action='store_true',
                   help='导出 PCD 时附带 gt 字段（仅 val 有效）')
    p.add_argument('--save_val_pcd_with_conf', action='store_true',
                   help='导出 PCD 时附带 conf 字段（预测置信度）')

    # KNN 覆盖开关
    p.add_argument('--use_knn',  action='store_true', default=None)
    p.add_argument('--no_knn',   action='store_true')
    p.add_argument('--knn_dual', action='store_true', default=None)
    p.add_argument('--knn_radius', type=int, default=None)

    # TTA 覆盖开关（未指定时使用 config 中的值）
    p.add_argument('--use_tta',    action='store_true', default=None,
                   help='开启 TTA（覆盖 config eval.use_tta）')
    p.add_argument('--no_tta',     action='store_true',
                   help='关闭 TTA（覆盖 config eval.use_tta）')
    p.add_argument('--tta_augs',   type=int, default=None,
                   help='TTA 变换数量 1~4（覆盖 config eval.tta_augs）')
    p.add_argument('--tta_workers', type=int, default=None,
                   help='TTA 预处理线程数（覆盖 config eval.tta_workers）')

    args = p.parse_args()
    if args.topk is not None and args.topk < 1:
        p.error('--topk 必须 >= 1')
    if args.mc_dropout_passes < 1:
        p.error('--mc_dropout_passes 必须 >= 1')

    ckpt_paths = []
    if args.ckpt:
        ckpt_paths.append(args.ckpt)
    if args.ckpts:
        ckpt_paths.extend(args.ckpts)
    if args.ckpt_dir:
        ckpt_paths.extend(sorted(glob.glob(os.path.join(args.ckpt_dir, '*.pth'))))

    ckpt_paths = _dedup_paths(ckpt_paths)
    if not ckpt_paths:
        p.error('请提供 --ckpt / --ckpts / --ckpt_dir 至少一个 checkpoint 来源')

    rank_topk = args.topk
    if args.ckpt_dir and rank_topk is None:
        rank_topk = 3
    if rank_topk is not None:
        ckpt_paths = _rank_checkpoints_by_miou(
            ckpt_paths, rank_topk
        )
        if not ckpt_paths:
            p.error('未找到可用 checkpoint，请检查 --ckpt_dir 或传入的 --ckpts')

    cfg = load_cfg(args.cfg)
    ec  = cfg.get('eval', {})

    # ── KNN 配置 ────────────────────────────────────────────
    if args.no_knn:
        use_knn = False
    elif args.use_knn:
        use_knn = True
    else:
        use_knn = ec.get('use_knn', False)

    knn_cfg = {
        'use_knn':    use_knn,
        'knn_radius': args.knn_radius if args.knn_radius is not None
                      else ec.get('knn_radius', 1),
        'knn_dual':   args.knn_dual   if args.knn_dual   is not None
                      else ec.get('knn_dual', False),
    }

    # ── TTA 配置 ────────────────────────────────────────────
    if args.no_tta:
        use_tta = False
    elif args.use_tta:
        use_tta = True
    else:
        use_tta = ec.get('use_tta', False)

    tta_augs = args.tta_augs if args.tta_augs is not None else ec.get('tta_augs', 4)
    tta_augs = max(1, min(tta_augs, len(_TTA_FNS)))  # 限制到 [1, 4]

    tta_cfg = {
        'use_tta':     use_tta,
        'tta_augs':    tta_augs,
        'tta_workers': args.tta_workers if args.tta_workers is not None
                       else ec.get('tta_workers', 2),
    }

    infer_cfg = {
        'mc_dropout_passes': args.mc_dropout_passes,
        'save_val_pcd': {
            'enable': bool(args.save_val_pcd_dir),
            'dir': args.save_val_pcd_dir or '',
            'every': args.save_val_pcd_every,
            'max_frames': args.save_val_pcd_max_frames,
            'with_gt': args.save_val_pcd_with_gt,
            'with_conf': args.save_val_pcd_with_conf,
        },
    }

    # ── CRF 配置 ────────────────────────────────────────────
    crf_cfg = {
        'use_crf': ec.get('use_crf', False),
        'crf_distance_adaptive': ec.get('crf_distance_adaptive', True),
        'crf_theta_alpha': ec.get('crf_theta_alpha', 0.3),
        'crf_theta_beta': ec.get('crf_theta_beta', 0.5),
        'crf_theta_gamma': ec.get('crf_theta_gamma', 0.15),
        'crf_compat_spatial': ec.get('crf_compat_spatial', 3.0),
        'crf_compat_bilateral': ec.get('crf_compat_bilateral', 10.0),
        'crf_num_iter': ec.get('crf_num_iter', 5),
    }

    print(f'KNN 配置: use_knn={knn_cfg["use_knn"]}  '
          f'radius={knn_cfg["knn_radius"]}  dual={knn_cfg["knn_dual"]}')
    print(f'TTA 配置: use_tta={tta_cfg["use_tta"]}  '
          f'augs={tta_cfg["tta_augs"]}  workers={tta_cfg["tta_workers"]}')
    print(f'集成配置: ckpts={len(ckpt_paths)}  '
          f'use_ema={not args.no_ema}  mc_dropout_passes={infer_cfg["mc_dropout_passes"]}')
    print(f'CRF 配置: use_crf={crf_cfg["use_crf"]}  '
          f'adaptive={crf_cfg["crf_distance_adaptive"]}  iter={crf_cfg["crf_num_iter"]}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}'
          + (f'  ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''))

    print('将加载以下 checkpoint：')
    for i, pth in enumerate(ckpt_paths, 1):
        print(f'  [{i}] {pth}')

    models = []
    for pth in ckpt_paths:
        model = build_model(cfg, device)
        load_checkpoint(model, pth, device, use_ema=not args.no_ema)
        model.eval()
        models.append(model)

    if infer_cfg['mc_dropout_passes'] > 1:
        drop_layers = _prepare_models_for_infer(
            models, mc_dropout_passes=infer_cfg['mc_dropout_passes']
        )
        print(f'MC Dropout 已启用: passes={infer_cfg["mc_dropout_passes"]}  '
              f'active_dropout_layers(total)={drop_layers}')

    if args.split == 'val':
        run_val(models, cfg, device, knn_cfg, tta_cfg, infer_cfg, crf_cfg)
    else:
        run_test(models, cfg, device, args.output_dir, knn_cfg, tta_cfg, infer_cfg)


def infer_single_bin(bin_path: str, cfg_path: str, ckpt_path: str,
                     use_ema: bool = True, device: str = 'cuda') -> np.ndarray:
    """
    单帧 .bin 文件推理

    Args:
        bin_path: 点云文件路径 (N×4 float32)
        cfg_path: 配置文件路径
        ckpt_path: 模型权重路径
        use_ema: 是否使用 EMA 权重
        device: 'cuda' 或 'cpu'

    Returns:
        pred_labels: (N,) uint32，train_id (0-18, 255=ignore)
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(device)
    model = build_model(cfg, device)
    load_checkpoint(model, ckpt_path, device, use_ema=use_ema)
    model.eval()

    # 读取点云
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

    # 投影
    dc = cfg['data']
    rv_proj = RangeImageProjector(
        H=dc['rv_H'], W=dc['rv_W'],
        fov_up=dc['fov_up'], fov_down=dc['fov_down'],
        R_max=dc['R_max']
    )
    pb_proj = PolarBEVProjector(
        H_p=dc['pb_H'], W_p=dc['pb_W'], R_max=dc['R_max']
    )

    rv_img, rv_mask = rv_proj.project(pts[:, :3], pts[:, 3])
    pb_img = pb_proj.project(pts[:, :3], pts[:, 3])
    rv_coords = rv_proj.compute_sample_coords(pts[:, :3]).astype(np.float32)
    pb_coords = pb_proj.compute_sample_coords(pts[:, :3]).astype(np.float32)

    # 点特征（与 InferenceDataset 保持一致）
    use_geo_pt = cfg.get('model', {}).get('use_geometric_pt_features', False)
    if use_geo_pt:
        pt_feat = _compute_geo_features(pts, rv_img, pb_img, rv_proj, pb_proj, dc['R_max'])  # (N, 9)
    else:
        pt_feat = pts.astype(np.float32)  # (N, 4)

    # 转 tensor
    rv_img_t = torch.from_numpy(rv_img).unsqueeze(0).to(device)
    pb_img_t = torch.from_numpy(pb_img).unsqueeze(0).to(device)
    rv_coords_t = torch.from_numpy(rv_coords).unsqueeze(0).unsqueeze(2).to(device)
    pb_coords_t = torch.from_numpy(pb_coords).unsqueeze(0).unsqueeze(2).to(device)
    pt_feat_t = torch.from_numpy(pt_feat).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(rv_img_t, pb_img_t, rv_coords_t, pb_coords_t, pt_feat_t)['logits']  # (1, N, 19)

    pred = logits[0].argmax(dim=-1).cpu().numpy().astype(np.uint8)
    return pred


if __name__ == '__main__':
    main()
