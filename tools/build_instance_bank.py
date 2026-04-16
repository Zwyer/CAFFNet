"""
离线预处理：扫描 SemanticKITTI 训练集，提取稀有类实例，保存为 instance_bank.npz。

只需运行一次，约 3-5 分钟（多进程）。输出文件 < 40 MB。

用法：
    python tools/build_instance_bank.py \
        --root /data/semantickitti \
        --out  instance_bank.npz \
        [--num-workers 8] [--min-pts 10] [--max-dist 50] [--ground-radius 2.0]
"""

import os
import sys
import argparse
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

# ── 路径 ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prfnet.datasets.semantickitti import LABEL_MAPPING, TRAIN_SEQS

# ── 目标类别（things 类中的稀有类）──────────────────────────────
# 与 config.yaml copy_paste.classes 保持一致（含 truck）
RARE_CLASSES = {
    7: 'motorcyclist',
    6: 'bicyclist',
    5: 'person',
    3: 'truck',
    2: 'motorcycle',
    1: 'bicycle',
    4: 'other_vehicle',
}

# 地面类（mapped id），用于估算局部地面高度
GROUND_MAPPED = frozenset([8, 9, 10, 11])   # road, parking, sidewalk, other-ground

# ── 标签映射表（模块级，进程 fork 后继承，无需重复构建）──────────
_LUT = np.full(65536, 255, dtype=np.int32)
for _raw, _tid in LABEL_MAPPING.items():
    if 0 <= _raw < 65536:
        _LUT[_raw] = _tid

# truck 等大目标要求更多点才算有效实例（远距点稀疏时过滤低质量实例）
CLASS_MIN_PTS = {
    3: 30,   # truck：至少30点，过滤极远/遮挡严重的残缺实例
}

_WORKER_PARAMS: dict = {}


def _worker_init(params: dict):
    """Pool initializer：每个 worker 进程保存处理参数"""
    global _WORKER_PARAMS
    _WORKER_PARAMS = params


def _process_frame(frame_paths: tuple) -> list:
    """
    单帧处理函数（顶层函数，可被 pickle 传入子进程）。

    Args:
        frame_paths: (velo_path, label_path)

    Returns:
        list of dict: {'cls_id', 'pts', 'zoff', 'dist'}
        pts: (M, 4) float32，XY 已中心化
    """
    velo_path, label_path = frame_paths
    p = _WORKER_PARAMS
    min_pts      = p['min_pts']
    max_dist     = p['max_dist']
    ground_radius = p['ground_radius']
    min_ground_pts = p['min_ground_pts']

    # ── 加载点云与标签 ────────────────────────────────────────
    try:
        pts = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
    except Exception:
        return []

    dist_all = np.linalg.norm(pts[:, :3], axis=1)
    valid    = dist_all > 0.5
    pts      = pts[valid]
    dist_all = dist_all[valid]

    if not os.path.exists(label_path):
        return []
    try:
        raw = np.fromfile(label_path, dtype=np.uint32)[valid]
    except Exception:
        return []

    sem_raw    = (raw & 0xFFFF).astype(np.int32)
    inst_id    = (raw >> 16).astype(np.int32)
    sem_mapped = _LUT[sem_raw]

    # 快速跳过不含稀有类的帧
    rare_in_frame = set(np.unique(sem_mapped).tolist()) & set(RARE_CLASSES)
    if not rare_in_frame:
        return []

    # 地面点缓存
    ground_mask = np.isin(sem_mapped, list(GROUND_MAPPED))
    ground_pts  = pts[ground_mask]

    results = []
    for cls_id in rare_in_frame:
        cls_mask = (sem_mapped == cls_id)
        iids     = np.unique(inst_id[cls_mask & (inst_id > 0)])

        for iid in iids:
            mask     = cls_mask & (inst_id == iid)
            inst_pts = pts[mask]

            # ── 过滤 ──────────────────────────────────────────
            effective_min_pts = CLASS_MIN_PTS.get(cls_id, min_pts)
            if len(inst_pts) < effective_min_pts:
                continue
            cx, cy = float(inst_pts[:, 0].mean()), float(inst_pts[:, 1].mean())
            dist   = float(np.sqrt(cx**2 + cy**2))
            if dist > max_dist:
                continue

            # ── 地面高度估算 ──────────────────────────────────
            ground_z = None
            if len(ground_pts):
                dx = ground_pts[:, 0] - cx
                dy = ground_pts[:, 1] - cy
                g_d = np.sqrt(dx*dx + dy*dy)
                near = ground_pts[g_d < ground_radius]
                if len(near) >= min_ground_pts:
                    # 10th percentile 比 min 更稳健（排除离群低点）
                    ground_z = float(np.percentile(near[:, 2], 10))

            if ground_z is None:
                ground_z = float(inst_pts[:, 2].min()) - 0.05   # fallback

            z_offset = float(inst_pts[:, 2].min()) - ground_z
            z_offset = max(z_offset, 0.0)   # 不允许负值（物体不陷入地面）

            # ── XY 中心化 ─────────────────────────────────────
            centered = inst_pts.copy()
            centered[:, 0] -= cx
            centered[:, 1] -= cy

            results.append({
                'cls_id': cls_id,
                'pts':    centered.astype(np.float32),
                'zoff':   np.float32(z_offset),
                'dist':   np.float32(dist),
            })

    return results


def build_bank(root: str,
               num_workers: int = 8,
               min_pts: int = 10,
               max_dist: float = 50.0,
               ground_radius: float = 2.0,
               min_ground_pts: int = 5) -> dict:
    """
    并行扫描所有训练序列，返回 dict: cls_id -> list of instance dicts
    """
    # 收集所有帧路径
    all_frames = []
    for seq in TRAIN_SEQS:
        velo_dir  = os.path.join(root, 'sequences', seq, 'velodyne')
        label_dir = os.path.join(root, 'sequences', seq, 'labels')
        if not os.path.isdir(velo_dir):
            print(f'[WARN] 序列 {seq} 不存在，跳过', flush=True)
            continue
        for b in sorted(f for f in os.listdir(velo_dir) if f.endswith('.bin')):
            stem = b.replace('.bin', '')
            all_frames.append((
                os.path.join(velo_dir, b),
                os.path.join(label_dir, stem + '.label'),
            ))

    total = len(all_frames)
    print(f'共 {total} 帧，使用 {num_workers} 个进程扫描…', flush=True)

    worker_params = dict(
        min_pts=min_pts, max_dist=max_dist,
        ground_radius=ground_radius, min_ground_pts=min_ground_pts,
    )

    bank = {cls_id: [] for cls_id in RARE_CLASSES}

    # ── 多进程处理，tqdm 实时更新 ────────────────────────────────
    with Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(worker_params,),
    ) as pool:
        with tqdm(total=total, unit='frame', ncols=90,
                  desc='扫描帧') as pbar:
            for frame_results in pool.imap_unordered(
                _process_frame, all_frames, chunksize=16
            ):
                for inst in frame_results:
                    bank[inst['cls_id']].append({
                        'pts':  inst['pts'],
                        'zoff': inst['zoff'],
                        'dist': inst['dist'],
                    })
                pbar.update(1)

    # ── 统计输出 ─────────────────────────────────────────────────
    print('\n── 实例统计 ──────────────────────────────────────')
    for cls_id, name in RARE_CLASSES.items():
        insts = bank[cls_id]
        if insts:
            cnts = [len(x['pts']) for x in insts]
            print(f'  {name:15s} [{cls_id}]  实例数={len(insts):5d}  '
                  f'点数: min={min(cnts):3d}  '
                  f'med={int(np.median(cnts)):3d}  '
                  f'max={max(cnts):4d}')
        else:
            print(f'  {name:15s} [{cls_id}]  实例数=    0  (训练集无此类)')
    print()

    return bank


def save_bank(bank: dict, out_path: str):
    """以 CSR 格式保存：所有 class 的实例 pts 拼接为单一 float32 数组"""
    # 确保后缀正确
    if not out_path.endswith('.npz'):
        out_path += '.npz'

    arrays = {}
    for cls_id, name in RARE_CLASSES.items():
        insts = bank[cls_id]
        if not insts:
            arrays[f'{name}_pts']     = np.zeros((0, 4), dtype=np.float32)
            arrays[f'{name}_offsets'] = np.array([0],    dtype=np.int32)
            arrays[f'{name}_zoff']    = np.zeros((0,),   dtype=np.float32)
            arrays[f'{name}_dists']   = np.zeros((0,),   dtype=np.float32)
            continue

        pts_chunks = [x['pts'] for x in insts]
        offsets    = np.zeros(len(insts) + 1, dtype=np.int32)
        for i, c in enumerate(pts_chunks):
            offsets[i + 1] = offsets[i] + len(c)

        arrays[f'{name}_pts']     = np.concatenate(pts_chunks, axis=0)
        arrays[f'{name}_offsets'] = offsets
        arrays[f'{name}_zoff']    = np.array([x['zoff'] for x in insts], dtype=np.float32)
        arrays[f'{name}_dists']   = np.array([x['dist'] for x in insts], dtype=np.float32)

    np.savez_compressed(out_path, **arrays)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'已保存至 {out_path}  ({size_mb:.1f} MB)')


def main():
    parser = argparse.ArgumentParser(
        description='Build SemanticKITTI instance bank (multi-process)')
    parser.add_argument('--root',            required=True,
                        help='数据集根目录（含 sequences/）')
    parser.add_argument('--out',             default='instance_bank.npz',
                        help='输出文件路径')
    parser.add_argument('--num-workers',     type=int,   default=8,
                        help='并行进程数（建议 = CPU 核心数，默认 8）')
    parser.add_argument('--min-pts',         type=int,   default=10,
                        help='实例最少点数过滤阈值（默认 10）')
    parser.add_argument('--max-dist',        type=float, default=70.0,
                        help='实例最大距离过滤阈值 m（默认 70，truck 等大目标远处可见）')
    parser.add_argument('--ground-radius',   type=float, default=2.0,
                        help='局部地面高度估算半径 m（默认 2.0）')
    parser.add_argument('--min-ground-pts',  type=int,   default=5,
                        help='地面点不足此数时使用 fallback（默认 5）')
    args = parser.parse_args()

    bank = build_bank(
        root=args.root,
        num_workers=args.num_workers,
        min_pts=args.min_pts,
        max_dist=args.max_dist,
        ground_radius=args.ground_radius,
        min_ground_pts=args.min_ground_pts,
    )
    save_bank(bank, args.out)


if __name__ == '__main__':
    main()
