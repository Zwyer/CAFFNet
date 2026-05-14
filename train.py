"""
PRFNet 训练脚本 v2
- 所有配置从 YAML 读取，无硬编码数值
- TensorBoard + txt 双格式日志
用法：
    python train.py --cfg prfnet/configs/prfnet_semantickitti.yaml
"""

import os
import sys
import time
import argparse
import logging
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel

from prfnet.models.prfnet import PRFNet
from prfnet.datasets.semantickitti import (
    SemanticKITTIDataset, collate_fn, CLASS_NAMES,
)
from prfnet.utils.loss import PRFNetLoss


# ─────────────────────────────────────────────────────────────
# 配置加载
# ─────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
# 日志系统
# ─────────────────────────────────────────────────────────────

def setup_logger(save_dir: str) -> logging.Logger:
    """同时输出到 stdout 和 txt 文件"""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'train.log')

    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger('prfnet')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 文件 handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 控制台 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ─────────────────────────────────────────────────────────────
# mIoU 评估
# ─────────────────────────────────────────────────────────────

class IoUMetric:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.C = num_classes
        self.ig = ignore_index
        self.reset()

    def reset(self):
        self.intersect = np.zeros(self.C, dtype=np.float64)
        self.union     = np.zeros(self.C, dtype=np.float64)

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        p = preds.cpu().numpy().flatten().astype(np.int64)
        g = labels.cpu().numpy().flatten().astype(np.int64)
        valid = g != self.ig
        p, g = p[valid], g[valid]
        if len(p) == 0:
            return
        # 混淆矩阵：O(N) bincount 替代 O(C×N) Python 循环
        idx = g * self.C + p
        cm  = np.bincount(idx, minlength=self.C * self.C).reshape(self.C, self.C)
        self.intersect += np.diag(cm)
        self.union     += cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)

    def per_class_iou(self) -> np.ndarray:
        valid = self.union > 0
        iou = np.where(valid, self.intersect / (self.union + 1e-10), np.nan)
        return iou * 100.

    def miou(self) -> float:
        iou = self.per_class_iou()
        return float(np.nanmean(iou))


def _batch_class_hist(labels: torch.Tensor,
                      num_classes: int,
                      ignore_index: int) -> np.ndarray:
    """统计一个 batch 的类别直方图（忽略 ignore_index）。"""
    valid = labels != ignore_index
    if not valid.any():
        return np.zeros(num_classes, dtype=np.float64)
    hist = torch.bincount(labels[valid].view(-1), minlength=num_classes)
    return hist.detach().cpu().numpy().astype(np.float64)


def _build_low_label_subset(dataset: SemanticKITTIDataset,
                            subset_frames: int,
                            subset_seed: int,
                            balance_by_seq: bool = True):
    """
    从 train split 采样固定帧数，构造低标注子集（100/200 帧等）。
    返回：
      subset_ds:   torch.utils.data.Subset
      chosen_idx:  选中的原始帧索引（升序）
      seq_counts:  各序列命中帧数
    """
    total = len(dataset)
    if subset_frames <= 0 or subset_frames >= total:
        all_idx = np.arange(total, dtype=np.int64)
        seq_counts = {}
        for i in all_idx.tolist():
            seq = dataset.frames[i]['seq']
            seq_counts[seq] = seq_counts.get(seq, 0) + 1
        return Subset(dataset, all_idx.tolist()), all_idx.tolist(), seq_counts

    rng = np.random.default_rng(subset_seed)

    if not balance_by_seq:
        chosen = np.sort(rng.choice(total, size=subset_frames, replace=False)).astype(np.int64)
    else:
        seq_to_idx = {}
        for i, frame in enumerate(dataset.frames):
            seq_to_idx.setdefault(frame['seq'], []).append(i)
        seqs = sorted(seq_to_idx.keys())

        # 按序列平均分配，再按剩余容量补齐
        base = subset_frames // len(seqs)
        rem = subset_frames % len(seqs)
        alloc = {s: base for s in seqs}
        for s in seqs[:rem]:
            alloc[s] += 1

        chosen_list = []
        spill = 0
        for s in seqs:
            idxs = np.asarray(seq_to_idx[s], dtype=np.int64)
            want = alloc[s]
            take = min(want, len(idxs))
            if take > 0:
                pick = rng.choice(idxs, size=take, replace=False)
                chosen_list.extend(pick.tolist())
            spill += (want - take)

        if spill > 0:
            chosen_set = set(chosen_list)
            pool = np.asarray([i for i in range(total) if i not in chosen_set], dtype=np.int64)
            if len(pool) > 0:
                extra_take = min(spill, len(pool))
                extra = rng.choice(pool, size=extra_take, replace=False)
                chosen_list.extend(extra.tolist())

        chosen = np.sort(np.asarray(chosen_list, dtype=np.int64))
        # 极端情况下若不足（例如 total<subset_frames，理论上前面已拦截），做保护
        if len(chosen) < subset_frames:
            remain_pool = np.asarray([i for i in range(total) if i not in set(chosen.tolist())], dtype=np.int64)
            need = min(subset_frames - len(chosen), len(remain_pool))
            if need > 0:
                more = rng.choice(remain_pool, size=need, replace=False)
                chosen = np.sort(np.concatenate([chosen, more.astype(np.int64)], axis=0))
        if len(chosen) > subset_frames:
            chosen = np.sort(rng.choice(chosen, size=subset_frames, replace=False)).astype(np.int64)

    seq_counts = {}
    for i in chosen.tolist():
        seq = dataset.frames[int(i)]['seq']
        seq_counts[seq] = seq_counts.get(seq, 0) + 1

    return Subset(dataset, chosen.tolist()), chosen.tolist(), seq_counts


def _frame_key_from_frame(frame: dict) -> str:
    seq = str(frame["seq"])
    stem = os.path.splitext(os.path.basename(frame["velo"]))[0]
    return f"{seq}/{stem}"


def _parse_csv_seqs(value):
    if value is None:
        return None
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",") if x.strip()]
        return items if items else None
    if isinstance(value, (list, tuple)):
        items = [str(x).strip() for x in value if str(x).strip()]
        return items if items else None
    return None


def _normalize_label_mapping(raw_mapping):
    if raw_mapping is None:
        return None
    out = {}
    for k, v in raw_mapping.items():
        out[int(k)] = int(v)
    return out


def _resolve_class_names(cfg: dict):
    data_cfg = cfg.get('data', {})
    names = data_cfg.get('class_names', None)
    n_cls = int(data_cfg.get('num_classes', len(CLASS_NAMES)))
    if names is None:
        return CLASS_NAMES[:n_cls]
    names = [str(x) for x in names]
    if len(names) < n_cls:
        names = names + [f'class_{i}' for i in range(len(names), n_cls)]
    return names[:n_cls]


def _filter_state_dict_by_shape(model: nn.Module, state_dict: dict):
    """
    过滤掉与当前模型 shape 不一致的权重（常见于类别数变化导致的分类头不匹配）。
    返回：
      filtered_state: 可安全加载的子集
      skipped: [(key, ckpt_shape, model_shape), ...]
    """
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k not in model_state:
            # 不在当前模型中的 key 由 load_state_dict(strict=False) 统一处理为 unexpected
            filtered[k] = v
            continue
        if hasattr(v, 'shape') and hasattr(model_state[k], 'shape'):
            ckpt_shape = tuple(v.shape)
            model_shape = tuple(model_state[k].shape)
            if ckpt_shape != model_shape:
                skipped.append((k, ckpt_shape, model_shape))
                continue
        filtered[k] = v
    return filtered, skipped


def _load_selected_frame_keys(list_path: str):
    """
    读取选帧清单。
    允许每行格式：
      seq/stem
      seq/stem  <任意附加信息>
    """
    keys: List[str] = []
    malformed = 0
    with open(list_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            token = s.split()[0].replace("\\", "/")
            if "/" not in token:
                malformed += 1
                continue
            seq, stem = token.split("/", 1)
            seq = seq.strip()
            stem = stem.strip()
            if (not seq) or (not stem):
                malformed += 1
                continue
            keys.append(f"{seq}/{stem}")
    # 去重并保留顺序
    uniq = []
    seen = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq, malformed


def _build_low_label_subset_from_file(dataset: SemanticKITTIDataset,
                                      list_path: str,
                                      subset_frames: int,
                                      subset_seed: int,
                                      strict: bool = False,
                                      fill_random_if_insufficient: bool = True):
    """
    根据外部选帧清单构造低标注子集。
    """
    key_to_idx = {}
    for i, frame in enumerate(dataset.frames):
        key_to_idx[_frame_key_from_frame(frame)] = i

    selected_keys, malformed = _load_selected_frame_keys(list_path)
    found_idx: List[int] = []
    missing_keys: List[str] = []
    seen_idx = set()

    for k in selected_keys:
        idx = key_to_idx.get(k, None)
        if idx is None:
            missing_keys.append(k)
            continue
        if idx not in seen_idx:
            seen_idx.add(idx)
            found_idx.append(int(idx))

    if strict and (malformed > 0 or len(missing_keys) > 0):
        raise ValueError(
            f"selected frame list invalid: malformed={malformed}, missing={len(missing_keys)}"
        )

    target = subset_frames if subset_frames > 0 else len(found_idx)
    chosen = list(found_idx)

    # 如果清单不足目标帧数，可选用随机补齐（避免训练帧数低于预期）
    if len(chosen) < target and fill_random_if_insufficient:
        rng = np.random.default_rng(subset_seed)
        chosen_set = set(chosen)
        remain_pool = np.asarray([i for i in range(len(dataset)) if i not in chosen_set], dtype=np.int64)
        need = min(target - len(chosen), len(remain_pool))
        if need > 0:
            extra = rng.choice(remain_pool, size=need, replace=False).astype(np.int64).tolist()
            chosen.extend([int(x) for x in extra])

    # 清单多于目标时，按清单顺序截断，保证可复现
    if target > 0 and len(chosen) > target:
        chosen = chosen[:target]

    if len(chosen) <= 0:
        raise ValueError("selected frame list produced empty subset")

    seq_counts = {}
    for i in chosen:
        seq = dataset.frames[int(i)]["seq"]
        seq_counts[seq] = seq_counts.get(seq, 0) + 1

    stats = {
        "list_total": len(selected_keys),
        "malformed": int(malformed),
        "found": len(found_idx),
        "missing": len(missing_keys),
        "missing_examples": missing_keys[:5],
        "final_used": len(chosen),
    }
    return Subset(dataset, chosen), chosen, seq_counts, stats


def _collect_trainable_params(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def _set_stage_trainable(model: PRFNet, head_only: bool, logger: logging.Logger):
    """
    两阶段微调参数冻结控制：
    - head_only=True: 仅训练分类相关头部（aggregator + rv_aux）
    - head_only=False: 全量可训练
    """
    if not head_only:
        for p in model.parameters():
            p.requires_grad = True
        logger.info('Two-stage: set full-model trainable.')
        return

    for p in model.parameters():
        p.requires_grad = False
    for m in [model.aggregator, model.rv_aux]:
        for p in m.parameters():
            p.requires_grad = True
    logger.info('Two-stage: stage1 head-only trainable (aggregator + rv_aux).')


def _build_optimizer_with_layerwise_lr(model: PRFNet, tc: dict, logger: logging.Logger):
    """
    构建优化器：
    - 默认：单组参数（与历史行为一致）
    - layerwise_lr.enable=True：按模块分组设置不同 lr 倍率
    """
    lwc = tc.get('layerwise_lr', {})
    enable = bool(lwc.get('enable', False))
    base_lr = float(tc['lr'])
    base_wd = float(tc['weight_decay'])

    if not enable:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=base_wd,
        )
        logger.info('Layer-wise LR: disabled (single param group)')
        return opt

    lr_mult_backbone = float(lwc.get('lr_mult_backbone', 0.2))
    lr_mult_mid = float(lwc.get('lr_mult_mid', 0.5))
    lr_mult_head = float(lwc.get('lr_mult_head', 1.0))

    wd_mult_backbone = float(lwc.get('wd_mult_backbone', 1.0))
    wd_mult_mid = float(lwc.get('wd_mult_mid', 1.0))
    wd_mult_head = float(lwc.get('wd_mult_head', 1.0))

    # 按 PRFNet 结构分组
    backbone_params = []
    for m in [model.rv_stem, model.pb_stem, model.rv_enc, model.pb_enc]:
        backbone_params.extend(_collect_trainable_params(m))

    mid_params = []
    for m in [model.aaffs, model.rv_dec, model.pb_dec]:
        mid_params.extend(_collect_trainable_params(m))

    head_params = []
    for m in [model.aggregator, model.rv_aux]:
        head_params.extend(_collect_trainable_params(m))

    seen = set()
    def _uniq(params):
        out = []
        for p in params:
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                out.append(p)
        return out

    backbone_params = _uniq(backbone_params)
    mid_params = _uniq(mid_params)
    head_params = _uniq(head_params)

    # 兜底：未被显式分到上述三组的参数，归入 head 组
    other_params = []
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            other_params.append(p)
    if len(other_params) > 0:
        head_params.extend(other_params)

    param_groups = []
    if len(backbone_params) > 0:
        param_groups.append({
            'name': 'backbone',
            'params': backbone_params,
            'lr': base_lr * lr_mult_backbone,
            'weight_decay': base_wd * wd_mult_backbone,
        })
    if len(mid_params) > 0:
        param_groups.append({
            'name': 'mid',
            'params': mid_params,
            'lr': base_lr * lr_mult_mid,
            'weight_decay': base_wd * wd_mult_mid,
        })
    if len(head_params) > 0:
        param_groups.append({
            'name': 'head',
            'params': head_params,
            'lr': base_lr * lr_mult_head,
            'weight_decay': base_wd * wd_mult_head,
        })

    if len(param_groups) == 0:
        # 极端保护：回退单组
        logger.warning('Layer-wise LR enabled but no param groups found, fallback to single group.')
        param_groups = [{
            'name': 'all',
            'params': [p for p in model.parameters() if p.requires_grad],
            'lr': base_lr,
            'weight_decay': base_wd,
        }]

    opt = torch.optim.AdamW(param_groups)

    for i, g in enumerate(opt.param_groups):
        n_params = sum(p.numel() for p in g['params'])
        gname = g.get('name', f'group{i}')
        logger.info(
            f'Layer-wise LR group[{i}] {gname}: '
            f'params={n_params/1e6:.2f}M  lr={g["lr"]:.2e}  wd={g["weight_decay"]:.2e}'
        )
    return opt


def _build_scheduler(optimizer, tc: dict, total_steps: int, logger: logging.Logger):
    sched_type = tc.get('scheduler', 'onecycle')
    if sched_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps, 1),
            eta_min=tc.get('eta_min', 1e-6),
        )
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=tc['lr'],
            total_steps=max(total_steps, 1),
            pct_start=tc['pct_start'],
        )
    logger.info(
        f'Scheduler: {sched_type}  lr={tc["lr"]}  '
        f'{"eta_min=" + str(tc.get("eta_min", 1e-6)) if sched_type == "cosine" else "pct_start=" + str(tc["pct_start"])}  '
        f'total_steps={max(total_steps, 1)}'
    )
    return scheduler


# ─────────────────────────────────────────────────────────────
# 辅助：生成像素级标签（用于辅助损失）
# ─────────────────────────────────────────────────────────────

def make_pixel_labels(labels_pt: torch.Tensor,
                      rv_point_idx: torch.Tensor,
                      H: int, W: int,
                      ignore: int = 255) -> torch.Tensor:
    """
    根据 rv_point_idx（每个像素对应的点索引），
    将逐点标签映射回 Range Image 的像素标签。
    labels_pt:     (B, N)   int64
    rv_point_idx:  (B, H, W) int32（-1 表示空格）
    Returns: (B, H, W) int64
    """
    B = labels_pt.shape[0]
    out = torch.full((B, H, W), ignore, dtype=torch.long, device=labels_pt.device)
    for b in range(B):
        mask = rv_point_idx[b] >= 0
        idx  = rv_point_idx[b][mask].long()
        out[b][mask] = labels_pt[b][idx]
    return out


# ─────────────────────────────────────────────────────────────
# KNN 后处理（推理精化）
# ─────────────────────────────────────────────────────────────

def _grid_votes(coords_norm: np.ndarray,
                preds: np.ndarray,
                grid_H: int,
                grid_W: int,
                num_classes: int,
                radius: int) -> np.ndarray:
    """
    内部辅助：在单个 2D 投影网格上做邻域投票，返回 votes 矩阵。

    W 方向（方位角）循环填充：θ=-π 与 θ=+π 互为邻居。
    H 方向硬边界：俯仰角/距离带不循环。

    Args:
        coords_norm: (N, 2) grid_sample 归一化坐标 [-1,1]
                     coords_norm[:, 0] = x（W / 方位角），
                     coords_norm[:, 1] = y（H / 俯仰角 或 距离带）
        preds:       (N,) int，有效点的预测类别（已去除 padding）
        grid_H/W:    投影网格分辨率
        num_classes: 类别总数
        radius:      窗口半径；1 = 3×3（9 邻居），2 = 5×5（25 邻居）

    Returns:
        votes: (N, num_classes) int32，各类别得票数
    """
    N = len(preds)

    # 归一化坐标 → 像素坐标
    cols = np.clip(
        np.round((coords_norm[:, 0] + 1.0) * 0.5 * (grid_W - 1)).astype(np.int32),
        0, grid_W - 1,
    )
    rows = np.clip(
        np.round((coords_norm[:, 1] + 1.0) * 0.5 * (grid_H - 1)).astype(np.int32),
        0, grid_H - 1,
    )

    # 构建预测图（多点落同格取最后写入值，稠密点云中极少发生）
    pred_map = np.full((grid_H, grid_W), -1, dtype=np.int32)
    pred_map[rows, cols] = preds

    # 填充：W 方向循环（方位角 ±π 相邻），H 方向用 -1 硬边界
    padded = np.pad(pred_map, pad_width=radius, mode='wrap')
    padded[:radius, :]  = -1   # 覆盖 H 顶部 wrap 产生的伪邻居
    padded[-radius:, :] = -1   # 覆盖 H 底部 wrap 产生的伪邻居

    # 向量化邻域投票
    pr = rows + radius   # padded 坐标系行索引
    pc = cols + radius   # padded 坐标系列索引

    votes = np.zeros((N, num_classes), dtype=np.int32)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            neighbor = padded[pr + dr, pc + dc]   # (N,)
            valid = neighbor >= 0
            if valid.any():
                votes[valid, neighbor[valid]] += 1

    return votes


def knn_refine_rv(rv_coords_norm: np.ndarray,
                  preds: np.ndarray,
                  rv_H: int,
                  rv_W: int,
                  num_classes: int,
                  radius: int = 1) -> np.ndarray:
    """
    Range Image 单视图 2D 邻域投票后处理。

    在 RV（方位角 × 俯仰角，64×1024）网格上搜索邻居，
    平滑俯仰/beam 方向的边界噪声。每帧 ~0.003s，无额外依赖。

    Args:
        rv_coords_norm: (N, 2) RV 归一化坐标（grid_sample 格式）
        preds:          (N,) 有效点预测类别
        rv_H, rv_W:     RV 分辨率（默认 64×1024）
        num_classes:    类别数
        radius:         窗口半径；1 = 3×3，2 = 5×5

    Returns:
        refined: (N,) 投票后预测标签
    """
    votes = _grid_votes(rv_coords_norm, preds, rv_H, rv_W, num_classes, radius)
    has_votes = votes.sum(axis=1) > 0
    refined = preds.copy()
    refined[has_votes] = votes[has_votes].argmax(axis=1).astype(preds.dtype)
    return refined


def knn_refine_dual(rv_coords_norm: np.ndarray,
                    pb_coords_norm: np.ndarray,
                    preds: np.ndarray,
                    rv_H: int, rv_W: int,
                    pb_H: int, pb_W: int,
                    num_classes: int,
                    radius: int = 1) -> np.ndarray:
    """
    RV + Polar BEV 双视图融合邻域投票后处理。

    两视图互补性：
      RV（方位角 × 俯仰角）：平滑竖向/beam 方向边界噪声
      PB（方位角 × 距离带）：平滑径向/距离方向边界噪声

    两者的 W 轴（方位角）天然对齐（AAFF 的几何基础），
    等权合并投票可覆盖 RV 无法感知的径向边界
    （如同方位角下不同距离的两个目标之间的过渡区域）。

    实现：分别在 RV、PB 网格上调用 _grid_votes，
    两个 votes 矩阵直接相加（1:1 等权），取 argmax。
    额外耗时约为单视图的 1 倍，总计仍 < 30s/次。

    Args:
        rv_coords_norm: (N, 2) RV 归一化坐标（grid_sample 格式）
        pb_coords_norm: (N, 2) PB 归一化坐标（grid_sample 格式）
        preds:          (N,) 有效点预测类别
        rv_H/rv_W:      RV 分辨率（默认 64×1024）
        pb_H/pb_W:      PB 分辨率（默认 480×1024）
        num_classes:    类别数
        radius:         两视图共用的窗口半径

    Returns:
        refined: (N,) 双视图投票后预测标签
    """
    votes_rv = _grid_votes(rv_coords_norm, preds, rv_H, rv_W, num_classes, radius)
    votes_pb = _grid_votes(pb_coords_norm, preds, pb_H, pb_W, num_classes, radius)
    votes    = votes_rv + votes_pb   # 等权融合，两个视图各贡献其邻域信息

    has_votes = votes.sum(axis=1) > 0
    refined = preds.copy()
    refined[has_votes] = votes[has_votes].argmax(axis=1).astype(preds.dtype)
    return refined


# ─────────────────────────────────────────────────────────────
# 训练主函数
# ─────────────────────────────────────────────────────────────

def main(args):
    cfg = load_cfg(args.cfg)
    dc  = cfg['data']
    mc  = cfg['model']
    tc  = cfg['train']
    lc  = cfg['loss']
    lgc = cfg['log']

    # 实验目录（带时间戳避免覆盖）
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(lgc['save_dir'], ts)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # 备份配置
    shutil.copy(args.cfg, os.path.join(exp_dir, 'config.yaml'))

    # 日志
    logger = setup_logger(exp_dir)
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard'),
                           flush_secs=lgc['tb_flush_secs'])
    class_names = _resolve_class_names(cfg)

    logger.info(f'Experiment dir: {exp_dir}')
    logger.info(f'Config: {args.cfg}')

    # 随机种子
    seed = tc['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}  '
                f'({torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU"})')

    # ── 数据集 ─────────────────────────────────────────────
    # Copy-Paste 实例库（enable=false 时 bank=None，dataset 自动跳过）
    cp_cfg = dc.get('copy_paste', {})
    cp_bank = None
    if cp_cfg.get('enable', False):
        from prfnet.datasets.instance_bank import InstanceBank
        cp_bank = InstanceBank(
            bank_path=cp_cfg['bank_path'],
            target_cls_ids=list(cp_cfg['classes'].keys()),
        )
        logger.info(f'Copy-Paste bank 已加载: {cp_cfg["bank_path"]}')

    # ── 【创新①②】RV 特征开关 & rv_in 自动计算 ────────────
    # 由开关自动推导，避免手动维护通道数与开关不一致
    _use_normals = mc.get('use_surface_normals', False)
    _use_angle   = mc.get('use_angle_encoding',  False)
    rv_in_actual = 6 + 3 * int(_use_normals) + 3 * int(_use_angle)
    logger.info(
        f'RV channels: {rv_in_actual}  '
        f'(surface_normals={_use_normals}, angle_encoding={_use_angle})'
    )

    train_seqs = _parse_csv_seqs(dc.get('train_seqs', None))
    val_seqs = _parse_csv_seqs(dc.get('val_seqs', None))
    require_labels = bool(dc.get('require_labels', True))
    custom_label_mapping = _normalize_label_mapping(dc.get('label_mapping', None))

    train_ds = SemanticKITTIDataset(
        root=dc['root'], split='train',
        seqs=train_seqs,
        require_labels=require_labels,
        label_mapping=custom_label_mapping,
        rv_H=dc['rv_H'], rv_W=dc['rv_W'],
        pb_H=dc['pb_H'], pb_W=dc['pb_W'],
        augment=dc['augment'],
        rotate=dc.get('rotate', True),
        flip=dc.get('flip', True),
        scale_min=dc.get('scale_min', 0.95),
        scale_max=dc.get('scale_max', 1.05),
        drop_p=dc.get('drop_p', 0.05),
        R_max=dc.get('R_max', 80.0),
        use_polarmix=dc['use_polarmix'],
        use_lasermix=dc.get('use_lasermix', False),
        fov_up=dc['fov_up'], fov_down=dc['fov_down'],
        max_points=dc['max_points'],
        polarmix_p=dc['polarmix_p'],
        polarmix_sectors=dc['polarmix_sectors'],
        lasermix_p=dc.get('lasermix_p', 0.5),
        use_surface_normals=_use_normals,
        use_angle_encoding=_use_angle,
        copy_paste_bank=cp_bank,
        copy_paste_classes={int(k): v for k, v in cp_cfg.get('classes', {}).items()},
        cp_drop_p_base=cp_cfg.get('drop_p_base', 0.05),
        cp_min_keep=cp_cfg.get('min_keep_pts', 15),
        cp_max_tries=cp_cfg.get('max_tries', 50),
    )
    full_train_len = len(train_ds)
    low_label_cfg = dc.get('low_label_finetune', {})
    ll_enable = bool(low_label_cfg.get('enable', False))
    ll_frames = int(low_label_cfg.get('num_frames', 0))
    ll_seed = int(low_label_cfg.get('seed', seed))
    ll_balance_by_seq = bool(low_label_cfg.get('balance_by_seq', True))
    ll_dump_path = low_label_cfg.get('dump_indices_path', 'selected_train_indices.txt')
    ll_selected_path_raw = str(low_label_cfg.get('selected_frames_path', '')).strip()
    ll_selected_strict = bool(low_label_cfg.get('selected_frames_strict', False))
    ll_selected_fill_random = bool(low_label_cfg.get('selected_frames_fill_random', True))

    ll_selected_path = ''
    if ll_selected_path_raw:
        if os.path.isabs(ll_selected_path_raw):
            ll_selected_path = ll_selected_path_raw
        else:
            cfg_rel = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(args.cfg)), ll_selected_path_raw))
            cwd_rel = os.path.abspath(ll_selected_path_raw)
            if os.path.isfile(cwd_rel):
                ll_selected_path = cwd_rel
            elif os.path.isfile(cfg_rel):
                ll_selected_path = cfg_rel
            else:
                ll_selected_path = cwd_rel

    if ll_enable and ll_frames > 0:
        if ll_selected_path:
            if not os.path.isfile(ll_selected_path):
                raise FileNotFoundError(
                    f'Low-label selected_frames_path not found: {ll_selected_path}'
                )
            train_ds, chosen_idx, seq_counts, sel_stats = _build_low_label_subset_from_file(
                train_ds,
                list_path=ll_selected_path,
                subset_frames=ll_frames,
                subset_seed=ll_seed,
                strict=ll_selected_strict,
                fill_random_if_insufficient=ll_selected_fill_random,
            )
            logger.info(
                'Low-label finetune enabled from selected list: '
                f'used={len(chosen_idx)} target={ll_frames} seed={ll_seed} '
                f'list={ll_selected_path}'
            )
            logger.info(
                'Selected-list stats: '
                f'total={sel_stats["list_total"]} found={sel_stats["found"]} '
                f'missing={sel_stats["missing"]} malformed={sel_stats["malformed"]} '
                f'final_used={sel_stats["final_used"]}'
            )
            if sel_stats["missing"] > 0:
                logger.info(
                    'Selected-list missing examples: ' +
                    ', '.join(sel_stats["missing_examples"])
                )
        else:
            train_ds, chosen_idx, seq_counts = _build_low_label_subset(
                train_ds, subset_frames=ll_frames, subset_seed=ll_seed,
                balance_by_seq=ll_balance_by_seq,
            )
            logger.info(
                f'Low-label finetune enabled: sampled {len(chosen_idx)} train frames '
                f'(seed={ll_seed}, balance_by_seq={ll_balance_by_seq})'
            )
        logger.info(
            'Low-label seq distribution: ' +
            ', '.join([f'{k}:{v}' for k, v in sorted(seq_counts.items())])
        )
        if ll_dump_path:
            ll_dump_abs = os.path.join(exp_dir, ll_dump_path)
            with open(ll_dump_abs, 'w', encoding='utf-8') as f:
                for i in chosen_idx:
                    frame = train_ds.dataset.frames[int(i)]
                    velo_path = frame['velo']
                    stem = os.path.splitext(os.path.basename(velo_path))[0]
                    f.write(f'{frame["seq"]}/{stem}\tidx={int(i)}\n')
            logger.info(f'Low-label frame list saved: {ll_dump_abs}')
    else:
        logger.info('Low-label finetune disabled: using full train split.')

    val_ds = SemanticKITTIDataset(
        root=dc['root'], split='val',
        seqs=val_seqs,
        require_labels=True,
        label_mapping=custom_label_mapping,
        rv_H=dc['rv_H'], rv_W=dc['rv_W'],
        pb_H=dc['pb_H'], pb_W=dc['pb_W'],
        R_max=dc.get('R_max', 80.0),
        augment=False, use_polarmix=False,
        fov_up=dc['fov_up'], fov_down=dc['fov_down'],
        max_points=dc['max_points'],
        use_surface_normals=_use_normals,
        use_angle_encoding=_use_angle,
    )

    train_loader = DataLoader(
        train_ds, batch_size=dc['batch_size'],
        shuffle=True, num_workers=dc['num_workers'],
        collate_fn=collate_fn, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=dc.get('val_batch_size', 4), shuffle=False,
        num_workers=dc.get('val_num_workers', 8),
        collate_fn=collate_fn, pin_memory=True,
        persistent_workers=True,
    )

    logger.info(f'Train: {len(train_ds)} frames  |  Val: {len(val_ds)} frames')
    writer.add_scalar('data/train_frames', float(len(train_ds)), 0)
    writer.add_scalar('data/train_frames_ratio_to_full',
                      float(len(train_ds)) / max(float(full_train_len), 1.0), 0)

    # ── 模型 ───────────────────────────────────────────────
    model = PRFNet(
        rv_in=rv_in_actual, pb_in=mc['pb_in'],
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
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {n_params/1e6:.2f}M')

    # ── EMA 影子模型（验证/保存均使用 EMA 权重）────────────
    ema_decay = tc.get('ema_decay', 0.999)
    ema_model = AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
        use_buffers=True,   # 同步 BN running_mean/var
    )
    logger.info(f'EMA decay: {ema_decay}')

    # ── 损失 ───────────────────────────────────────────────
    ec = cfg.get('eval', {})   # 评估配置（use_knn 等）

    cw = torch.tensor(lc['class_weights'], dtype=torch.float32).to(device)
    criterion = PRFNetLoss(
        class_weights=cw,
        ignore_index=dc['ignore_index'],
        num_classes=dc['num_classes'],
        lambda_aux=lc['lambda_aux'],
        lambda_lovász=lc['lambda_lovasz'],
        label_smoothing=lc.get('label_smoothing', 0.0),
        use_focal=lc.get('use_focal', False),
        focal_gamma=lc.get('focal_gamma', 2.0),
    )

    # ── 两阶段微调配置（可选，默认关闭）──────────────────────
    two_stage_cfg = tc.get('two_stage', {})
    two_stage_enable = bool(two_stage_cfg.get('enable', False))
    stage1_epochs = int(two_stage_cfg.get('stage1_epochs', 0))
    if stage1_epochs < 0:
        stage1_epochs = 0
    if stage1_epochs >= int(tc['epochs']):
        stage1_epochs = max(int(tc['epochs']) - 1, 0)
    two_stage_active = two_stage_enable and stage1_epochs > 0

    stage1_tc = dict(tc)
    if two_stage_active:
        if 'stage1_lr' in two_stage_cfg:
            stage1_tc['lr'] = float(two_stage_cfg['stage1_lr'])
        if 'stage1_weight_decay' in two_stage_cfg:
            stage1_tc['weight_decay'] = float(two_stage_cfg['stage1_weight_decay'])
        if 'stage1_scheduler' in two_stage_cfg:
            stage1_tc['scheduler'] = str(two_stage_cfg['stage1_scheduler'])
        if 'stage1_eta_min' in two_stage_cfg:
            stage1_tc['eta_min'] = float(two_stage_cfg['stage1_eta_min'])
        if 'stage1_pct_start' in two_stage_cfg:
            stage1_tc['pct_start'] = float(two_stage_cfg['stage1_pct_start'])
        if not bool(two_stage_cfg.get('stage1_layerwise', False)):
            stage1_tc['layerwise_lr'] = {'enable': False}

        _set_stage_trainable(model, head_only=True, logger=logger)
        logger.info(
            f'Two-stage finetune enabled: stage1(head-only)={stage1_epochs} epochs, '
            f'stage2(full-layerwise)={int(tc["epochs"]) - stage1_epochs} epochs'
        )
    else:
        _set_stage_trainable(model, head_only=False, logger=logger)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Trainable parameters: {n_trainable/1e6:.2f}M')

    # ── 优化器 ─────────────────────────────────────────────
    phase_tc = stage1_tc if two_stage_active else tc
    phase_name = 'stage1_head_only' if two_stage_active else 'single_stage_full'
    optimizer = _build_optimizer_with_layerwise_lr(model, phase_tc, logger)
    total_steps = len(train_loader) * (stage1_epochs if two_stage_active else int(tc['epochs']))
    scheduler = _build_scheduler(optimizer, phase_tc, total_steps, logger)
    logger.info(f'Optimization phase: {phase_name}')
    scaler = torch.amp.GradScaler('cuda', enabled=tc['amp'])

    # ── Checkpoint 热启动（仅加载模型权重，optimizer/scheduler 从头开始）──
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        # --resume-ema：加载 EMA 权重（评估时使用的权重，通常比 model 权重更好）
        # 默认加载 model 权重
        if args.resume_ema and 'ema_state_dict' in ckpt:
            raw_state = ckpt['ema_state_dict']
            logger.info('Resume: 使用 ema_state_dict（EMA 权重，对应验证 mIoU 的实际状态）')
        else:
            raw_state = ckpt.get('state_dict', ckpt)

        state = raw_state
        if args.strict_false:
            state, skipped_mismatch = _filter_state_dict_by_shape(model, raw_state)
            if skipped_mismatch:
                show_n = 8
                head = '; '.join(
                    [f'{k}: ckpt{cs}!=model{ms}' for k, cs, ms in skipped_mismatch[:show_n]]
                )
                logger.info(
                    f'Resume: skip {len(skipped_mismatch)} shape-mismatched keys '
                    f'(likely classifier/head due to class-count change): {head}'
                    f'{"; ..." if len(skipped_mismatch) > show_n else ""}'
                )

        missing, unexpected = model.load_state_dict(
            state, strict=not args.strict_false
        )
        if missing:
            logger.info(f'Resume: {len(missing)} missing keys '
                        f'(new modules, will be randomly init): '
                        f'{missing[:5]}{"..." if len(missing)>5 else ""}')
        if unexpected:
            logger.info(f'Resume: {len(unexpected)} unexpected keys (ignored): '
                        f'{unexpected[:5]}{"..." if len(unexpected)>5 else ""}')
        ep_loaded = ckpt.get('epoch', '?')
        logger.info(f'Hotstart from {args.resume}  (saved at ep{ep_loaded})  '
                    f'strict={not args.strict_false}')

    # ── 训练循环 ───────────────────────────────────────────
    best_miou = 0.
    save_topk_ckpts = int(tc.get('save_topk_ckpts', 5))
    topk_ckpts = []  # list[(miou, epoch, path)]
    logger.info(f'Checkpoint policy: latest + best + top{save_topk_ckpts}')
    global_step = 0

    for epoch in range(1, tc['epochs'] + 1):
        # 两阶段切换：进入 stage2 时解冻全量并重建优化器/调度器
        if two_stage_active and epoch == (stage1_epochs + 1):
            _set_stage_trainable(model, head_only=False, logger=logger)
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'Two-stage switch -> stage2 full finetune. Trainable={n_trainable/1e6:.2f}M')

            optimizer = _build_optimizer_with_layerwise_lr(model, tc, logger)
            stage2_epochs = int(tc['epochs']) - stage1_epochs
            stage2_steps = len(train_loader) * max(stage2_epochs, 1)
            scheduler = _build_scheduler(optimizer, tc, stage2_steps, logger)
            logger.info('Optimization phase: stage2_full_layerwise')

        model.train()
        epoch_loss = 0.
        epoch_ce = epoch_lov = epoch_aux = 0.
        t0 = time.time()
        epoch_step_time = 0.0
        epoch_data_time = 0.0
        epoch_grad_norm = 0.0
        epoch_points_valid = 0.0
        epoch_points_total = 0.0
        epoch_rv_occ = 0.0
        epoch_pb_occ = 0.0
        epoch_cls_hist = np.zeros(dc['num_classes'], dtype=np.float64)
        _iter_t_prev = time.time()

        # 原型漂移诊断：保存本 epoch 开始时的原型快照
        if model.aggregator.use_proto:
            _prev_protos = model.aggregator.prototypes.detach().float().cpu().clone()

        for step, batch in enumerate(train_loader, 1):
            global_step += 1
            _iter_t_now = time.time()
            data_dt = _iter_t_now - _iter_t_prev

            rv_img    = batch['rv_img'].to(device, non_blocking=True)
            pb_img    = batch['pb_img'].to(device, non_blocking=True)
            rv_coords = batch['rv_coords'].to(device, non_blocking=True)
            pb_coords = batch['pb_coords'].to(device, non_blocking=True)
            points    = batch['points'].to(device, non_blocking=True)
            labels    = batch['labels'].to(device, non_blocking=True)
            rv_labels = batch['rv_labels'].to(device, non_blocking=True)  # (B,H,W)
            labels_cpu = batch['labels']  # cpu 统计用
            _step_t0 = time.time()

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=tc['amp']):
                outputs   = model(rv_img, pb_img, rv_coords, pb_coords, points)
                loss_dict = criterion(outputs, labels, rv_labels=rv_labels)
                loss      = loss_dict['total']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), tc['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema_model.update_parameters(model)   # EMA 权重同步
            model.aggregator.update_prototypes(labels)  # 创新⑤ 原型 EMA 更新

            epoch_loss += loss.item()
            epoch_ce   += loss_dict['ce'].item()
            epoch_lov  += loss_dict['lovász'].item()
            epoch_aux  += loss_dict['aux'].item()
            epoch_data_time += data_dt
            epoch_step_time += (time.time() - _step_t0)
            epoch_grad_norm += float(grad_norm.detach().cpu().item()
                                     if torch.is_tensor(grad_norm) else grad_norm)

            # 数据质量/分布统计
            valid_mask = labels_cpu != dc['ignore_index']
            epoch_points_valid += float(valid_mask.sum().item())
            epoch_points_total += float(labels_cpu.numel())
            epoch_rv_occ += float((batch['rv_img'][:, 3:4] > 0).float().mean().item())
            epoch_pb_occ += float(batch['pb_img'][:, 8:9].float().mean().item())
            epoch_cls_hist += _batch_class_hist(
                labels_cpu, dc['num_classes'], dc['ignore_index']
            )

            # 步级日志：终端始终打印，log_steps=false 时不写入文件
            if step % lgc['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                avg = epoch_loss / step
                step_msg = (
                    f'Ep[{epoch:03d}/{tc["epochs"]}] '
                    f'Step[{step:04d}/{len(train_loader)}] '
                    f'loss={avg:.4f}  ce={loss_dict["ce"]:.4f}  '
                    f'lov={loss_dict["lovász"]:.4f}  '
                    f'aux={loss_dict["aux"]:.4f}  '
                    f'lr={lr:.2e}  '
                    f'gnorm={float(grad_norm):.2f}'
                )
                print(step_msg, flush=True)          # 终端始终可见
                if lgc.get('log_steps', True):
                    logger.info(step_msg)            # log 文件按开关控制
                writer.add_scalar('train/loss_step',   avg,                  global_step)
                writer.add_scalar('train/ce_step',     loss_dict['ce'],      global_step)
                writer.add_scalar('train/lovász_step', loss_dict['lovász'],  global_step)
                writer.add_scalar('train/aux_step',    loss_dict['aux'],     global_step)
                writer.add_scalar('train/lr',          lr,                   global_step)
                writer.add_scalar('diag/grad_norm_step', float(grad_norm), global_step)
                writer.add_scalar('diag/rv_occ_step', float((batch['rv_img'][:, 3:4] > 0).float().mean().item()), global_step)
                writer.add_scalar('diag/pb_occ_step', float(batch['pb_img'][:, 8:9].float().mean().item()), global_step)
                writer.add_scalar('diag/points_valid_ratio_step',
                                  float(valid_mask.sum().item()) / max(float(labels_cpu.numel()), 1.0),
                                  global_step)
                if device.type == 'cuda':
                    writer.add_scalar('diag/gpu_mem_alloc_mb_step',
                                      torch.cuda.memory_allocated(device) / 1024.0 / 1024.0,
                                      global_step)
                    writer.add_scalar('diag/gpu_mem_reserved_mb_step',
                                      torch.cuda.memory_reserved(device) / 1024.0 / 1024.0,
                                      global_step)
            _iter_t_prev = time.time()

        # Epoch 级日志
        n = len(train_loader)
        dt = time.time() - t0
        logger.info(
            f'── Epoch {epoch:03d} train  '
            f'loss={epoch_loss/n:.4f}  '
            f'ce={epoch_ce/n:.4f}  '
            f'lov={epoch_lov/n:.4f}  '
            f'aux={epoch_aux/n:.4f}  '
            f'time={dt:.0f}s'
        )
        writer.add_scalar('train/loss_epoch',   epoch_loss / n, epoch)
        writer.add_scalar('train/ce_epoch',     epoch_ce   / n, epoch)
        writer.add_scalar('train/lovász_epoch', epoch_lov  / n, epoch)
        writer.add_scalar('train/aux_epoch',    epoch_aux  / n, epoch)
        writer.add_scalar('diag/step_time_s_epoch', epoch_step_time / n, epoch)
        writer.add_scalar('diag/data_time_s_epoch', epoch_data_time / n, epoch)
        writer.add_scalar('diag/grad_norm_epoch', epoch_grad_norm / n, epoch)
        writer.add_scalar('diag/rv_occ_epoch', epoch_rv_occ / n, epoch)
        writer.add_scalar('diag/pb_occ_epoch', epoch_pb_occ / n, epoch)
        writer.add_scalar(
            'diag/points_valid_ratio_epoch',
            epoch_points_valid / max(epoch_points_total, 1.0), epoch
        )
        cls_hist_sum = epoch_cls_hist.sum()
        if cls_hist_sum > 0:
            cls_freq = epoch_cls_hist / cls_hist_sum
            for i, name in enumerate(class_names):
                writer.add_scalar(f'data/class_freq_{name}', float(cls_freq[i]), epoch)
        if device.type == 'cuda':
            writer.add_scalar('diag/gpu_mem_alloc_mb_epoch',
                              torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0,
                              epoch)
            writer.add_scalar('diag/gpu_mem_reserved_mb_epoch',
                              torch.cuda.max_memory_reserved(device) / 1024.0 / 1024.0,
                              epoch)
            torch.cuda.reset_peak_memory_stats(device)

        # ── 诊断指标（VCG gate entropy + SPM proto drift）──────
        diag = model.aggregator.get_and_reset_diagnostics()
        if 'gate_entropy' in diag:
            ge = diag['gate_entropy']
            logger.info(f'   gate_entropy={ge:.4f}  '
                        f'(1.0=undecided → ↓ means gate is learning view selection)')
            writer.add_scalar('diag/gate_entropy', ge, epoch)

        if model.aggregator.use_proto:
            cur   = model.aggregator.prototypes.detach().float().cpu()
            cur   = cur / cur.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            pre   = _prev_protos / _prev_protos.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            drift = (1 - (cur * pre).sum(-1))                 # (19,)
            _PROTO_CLASS_NAMES = [
                'car','bicycle','motorcycle','truck','other-veh',
                'person','bicyclist','motorcyclist','road','parking',
                'sidewalk','other-gnd','building','fence','vegetation',
                'trunk','terrain','pole','traffic-sign',
            ]
            logger.info(
                f'   proto_drift  mean={drift.mean():.4f}  '
                f'motorcyclist={drift[7]:.4f}  '
                f'bicyclist={drift[6]:.4f}  '
                f'person={drift[5]:.4f}  '
                f'other-veh={drift[4]:.4f}'
            )
            for i, name in enumerate(_PROTO_CLASS_NAMES):
                writer.add_scalar(f'proto_drift/{name}', drift[i].item(), epoch)

        # ── 验证 ──────────────────────────────────────────
        if epoch % tc['val_every'] == 0:
            miou, per_cls, per_prec, per_rec, dist_iou = evaluate(
                ema_model, val_loader, device,
                dc['num_classes'], dc['ignore_index'],
                use_knn=ec.get('use_knn', False),
                knn_radius=ec.get('knn_radius', 1),
                knn_dual=ec.get('knn_dual', False),
                rv_H=dc['rv_H'], rv_W=dc['rv_W'],
                pb_H=dc['pb_H'], pb_W=dc['pb_W'],
            )
            logger.info(f'── Epoch {epoch:03d} val  mIoU = {miou:.2f}%')

            # 逐类 IoU
            cls_line = '  '.join(
                f'{name[:6]}:{iou:.1f}' if not np.isnan(iou) else f'{name[:6]}:--'
                for name, iou in zip(class_names, per_cls)
            )
            logger.info(f'   Per-class: {cls_line}')
            prec_line = '  '.join(
                f'{name[:6]}:{p:.1f}' if not np.isnan(p) else f'{name[:6]}:--'
                for name, p in zip(class_names, per_prec)
            )
            rec_line = '  '.join(
                f'{name[:6]}:{r:.1f}' if not np.isnan(r) else f'{name[:6]}:--'
                for name, r in zip(class_names, per_rec)
            )
            logger.info(f'   Precision: {prec_line}')
            logger.info(f'   Recall:    {rec_line}')
            logger.info(
                '   Dist-IoU: '
                f"near={dist_iou.get('near', float('nan')):.2f}%  "
                f"mid={dist_iou.get('mid', float('nan')):.2f}%  "
                f"far={dist_iou.get('far', float('nan')):.2f}%"
            )

            writer.add_scalar('val/mIoU', miou, epoch)
            for name, iou in zip(class_names, per_cls):
                if not np.isnan(iou):
                    writer.add_scalar(f'val/iou_{name}', iou, epoch)
            for name, p in zip(class_names, per_prec):
                if not np.isnan(p):
                    writer.add_scalar(f'val/precision_{name}', p, epoch)
            for name, r in zip(class_names, per_rec):
                if not np.isnan(r):
                    writer.add_scalar(f'val/recall_{name}', r, epoch)
            for k, v in dist_iou.items():
                if np.isfinite(v):
                    writer.add_scalar(f'val/dist_iou_{k}', v, epoch)

            # 保存 checkpoint
            ckpt = {
                'epoch':           epoch,
                'state_dict':      model.state_dict(),
                'ema_state_dict':  ema_model.module.state_dict(),
                'optimizer':       optimizer.state_dict(),
                'miou':            miou,
                'cfg':             cfg,
            }
            # 每个 epoch 保存 latest
            torch.save(ckpt, os.path.join(ckpt_dir, 'latest.pth'))

            if miou > best_miou:
                best_miou = miou
                torch.save(ckpt, os.path.join(ckpt_dir, 'best.pth'))
                logger.info(f'   ★ New best  mIoU={best_miou:.2f}%  saved.')

            # 维护 top-k checkpoint（按 val mIoU 排序）
            if save_topk_ckpts > 0:
                ep_ckpt_path = os.path.join(ckpt_dir, f'epoch{epoch:03d}_miou{miou:.2f}.pth')
                torch.save(ckpt, ep_ckpt_path)
                topk_ckpts.append((float(miou), int(epoch), ep_ckpt_path))
                topk_ckpts.sort(key=lambda x: (x[0], x[1]), reverse=True)

                if len(topk_ckpts) > save_topk_ckpts:
                    _, _, drop_path = topk_ckpts.pop(-1)
                    if os.path.exists(drop_path):
                        os.remove(drop_path)

                topk_line = ' | '.join(
                    [f'ep{ep:03d}:{score:.2f}' for score, ep, _ in topk_ckpts]
                )
                logger.info(f'   Top-{save_topk_ckpts} ckpts: {topk_line}')

    writer.close()
    logger.info(f'Training done.  Best val mIoU = {best_miou:.2f}%')
    logger.info(f'Logs & checkpoints: {exp_dir}')


# ─────────────────────────────────────────────────────────────
# 验证
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_index,
             use_knn: bool = False, knn_radius: int = 1,
             knn_dual: bool = False,
             rv_H: int = 64,  rv_W: int = 1024,
             pb_H: int = 480, pb_W: int = 1024):
    """
    验证循环。

    use_knn:    True → 推理后做邻域投票精化（约 +1~2% mIoU）
                False → 直接用模型原始输出
    knn_radius: 投票窗口半径；1 = 3×3，2 = 5×5
    knn_dual:   True → RV + PB 双视图融合投票（修正径向边界，耗时约 ×2，仍 < 30s）
                False → 仅 RV 单视图投票
    rv_H/rv_W:  RV 分辨率，用于像素坐标换算
    pb_H/pb_W:  PB 分辨率，knn_dual=True 时使用

    GPU 利用率优化：
      1. batch 内多样本 KNN 并行（ThreadPoolExecutor，numpy 释放 GIL）
      2. 流水线：KNN 在后台线程处理当前 batch 时，GPU 已开始下一 batch forward
    """
    model.eval()
    metric = IoUMetric(num_classes, ignore_index)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    # 距离分段统计：near/mid/far
    bins = [(0.0, 20.0), (20.0, 40.0), (40.0, np.inf)]
    bin_names = ['near', 'mid', 'far']
    bin_inter = {k: np.zeros(num_classes, dtype=np.float64) for k in bin_names}
    bin_union = {k: np.zeros(num_classes, dtype=np.float64) for k in bin_names}

    def _refine_batch(preds_np, labels_np, rv_coords_np, pb_coords_np):
        """在线程池内并行精化一个 batch 内的所有样本，返回精化后的 preds (B, N) ndarray"""
        B_cur = len(preds_np)

        def _refine_one(b_idx):
            pred  = preds_np[b_idx].copy()
            valid = labels_np[b_idx] != ignore_index
            if valid.any():
                if knn_dual:
                    pred[valid] = knn_refine_dual(
                        rv_coords_np[b_idx][valid], pb_coords_np[b_idx][valid],
                        pred[valid],
                        rv_H=rv_H, rv_W=rv_W,
                        pb_H=pb_H, pb_W=pb_W,
                        num_classes=num_classes,
                        radius=knn_radius,
                    )
                else:
                    pred[valid] = knn_refine_rv(
                        rv_coords_np[b_idx][valid], pred[valid],
                        rv_H=rv_H, rv_W=rv_W,
                        num_classes=num_classes,
                        radius=knn_radius,
                    )
            return pred

        if B_cur == 1:
            return [_refine_one(0)]
        with ThreadPoolExecutor(max_workers=B_cur) as pool:
            return list(pool.map(_refine_one, range(B_cur)))

    # 流水线状态：记录上一 batch 的后台 KNN future
    knn_executor = ThreadPoolExecutor(max_workers=1) if use_knn else None
    pending_future  = None   # Future -> list of refined preds (ndarray)
    pending_labels  = None   # (B, N) tensor，与 pending_future 对应

    for batch in loader:
        # ── 1. 提交 GPU forward（尽早入队，与前一 batch KNN 并行）──
        rv_img    = batch['rv_img'].to(device)
        pb_img    = batch['pb_img'].to(device)
        rv_coords = batch['rv_coords'].to(device)
        pb_coords = batch['pb_coords'].to(device)
        points    = batch['points'].to(device)
        labels    = batch['labels']            # (B, N) CPU

        # ── 2. 等待上一 batch 的 KNN 完成并更新 metric ─────────
        if pending_future is not None:
            refined_list = pending_future.result()
            preds_refined = torch.from_numpy(
                np.stack(refined_list, axis=0))          # (B, N)
            metric.update(preds_refined, pending_labels)
            # 混淆矩阵
            p = preds_refined.numpy().reshape(-1).astype(np.int64)
            g = pending_labels.numpy().reshape(-1).astype(np.int64)
            valid = g != ignore_index
            if valid.any():
                idx = g[valid] * num_classes + p[valid]
                cm += np.bincount(
                    idx, minlength=num_classes * num_classes
                ).reshape(num_classes, num_classes)

        # ── 3. GPU forward ─────────────────────────────────────
        outputs = model(rv_img, pb_img, rv_coords, pb_coords, points)
        preds   = outputs['logits'].argmax(dim=-1).cpu()  # (B, N)

        # ── 4. 提交 KNN 到后台线程（或直接更新 metric）──────────
        if use_knn:
            B_cur = preds.shape[0]
            preds_np      = [preds[i].numpy()                          for i in range(B_cur)]
            labels_np     = [labels[i].numpy()                         for i in range(B_cur)]
            rv_coords_np  = [batch['rv_coords'][i, :, 0, :].numpy()   for i in range(B_cur)]
            pb_coords_np  = ([batch['pb_coords'][i, :, 0, :].numpy()  for i in range(B_cur)]
                             if knn_dual else [None] * B_cur)

            pending_future = knn_executor.submit(
                _refine_batch, preds_np, labels_np, rv_coords_np, pb_coords_np)
            pending_labels = labels
        else:
            metric.update(preds, labels)
            p = preds.numpy().reshape(-1).astype(np.int64)
            g = labels.numpy().reshape(-1).astype(np.int64)
            valid = g != ignore_index
            if valid.any():
                idx = g[valid] * num_classes + p[valid]
                cm += np.bincount(
                    idx, minlength=num_classes * num_classes
                ).reshape(num_classes, num_classes)
            pending_future = None
            pending_labels = None

    # ── 5. 处理最后一个 batch 的 KNN ───────────────────────────
    if pending_future is not None:
        refined_list = pending_future.result()
        preds_refined = torch.from_numpy(np.stack(refined_list, axis=0))
        metric.update(preds_refined, pending_labels)
        p = preds_refined.numpy().reshape(-1).astype(np.int64)
        g = pending_labels.numpy().reshape(-1).astype(np.int64)
        valid = g != ignore_index
        if valid.any():
            idx = g[valid] * num_classes + p[valid]
            cm += np.bincount(
                idx, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    if knn_executor is not None:
        knn_executor.shutdown(wait=False)

    # 距离分段 IoU（需要原始 points）
    for batch in loader:
        rv_img    = batch['rv_img'].to(device)
        pb_img    = batch['pb_img'].to(device)
        rv_coords = batch['rv_coords'].to(device)
        pb_coords = batch['pb_coords'].to(device)
        points    = batch['points'].to(device)
        labels    = batch['labels']  # cpu

        outputs = model(rv_img, pb_img, rv_coords, pb_coords, points)
        preds = outputs['logits'].argmax(dim=-1).cpu().numpy().astype(np.int64)
        gts   = labels.numpy().astype(np.int64)
        dist  = torch.norm(points[..., :3], dim=-1).cpu().numpy()
        for b in range(preds.shape[0]):
            for (lo, hi), name in zip(bins, bin_names):
                m = (dist[b] >= lo) & (dist[b] < hi) & (gts[b] != ignore_index)
                if not np.any(m):
                    continue
                p = preds[b][m]
                g = gts[b][m]
                idx = g * num_classes + p
                cm_bin = np.bincount(
                    idx, minlength=num_classes * num_classes
                ).reshape(num_classes, num_classes)
                inter = np.diag(cm_bin).astype(np.float64)
                union = cm_bin.sum(1) + cm_bin.sum(0) - np.diag(cm_bin)
                bin_inter[name] += inter
                bin_union[name] += union

    precision = np.where(cm.sum(axis=0) > 0, np.diag(cm) / (cm.sum(axis=0) + 1e-10), np.nan) * 100.0
    recall    = np.where(cm.sum(axis=1) > 0, np.diag(cm) / (cm.sum(axis=1) + 1e-10), np.nan) * 100.0
    dist_iou = {}
    for name in bin_names:
        valid = bin_union[name] > 0
        iou = np.where(valid, bin_inter[name] / (bin_union[name] + 1e-10), np.nan) * 100.0
        dist_iou[name] = float(np.nanmean(iou)) if np.any(valid) else float('nan')

    model.train()
    return metric.miou(), metric.per_class_iou(), precision, recall, dist_iou


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='prfnet/configs/prfnet_semantickitti.yaml',
                   help='Path to YAML config file')
    p.add_argument('--resume', default=None,
                   help='Path to checkpoint (.pth) for weight hotstart')
    p.add_argument('--strict-false', action='store_true',
                   help='Use strict=False when loading checkpoint '
                        '(needed when adding new modules like VCG/SPM)')
    p.add_argument('--resume-ema', action='store_true',
                   help='Load ema_state_dict instead of state_dict '
                        '(recommended: EMA weights = the weights used for validation mIoU)')
    args = p.parse_args()
    main(args)
