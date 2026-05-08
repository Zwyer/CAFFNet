"""
SemanticKITTI Dataset Loader
支持：单帧点云、数据增强、Range Image + Polar BEV 双视图生成
"""

import os
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..utils.projection import RangeImageProjector, PolarBEVProjector
from .instance_bank import InstanceBank


GROUND_MAPPED = frozenset([8, 9, 10, 11])   # road, parking, sidewalk, other-ground

# ─────────────────────────────────────────────────────────────
# SemanticKITTI 标签映射（19 个语义类）
# ─────────────────────────────────────────────────────────────

# 原始 label → 训练 label (0~18, 255=ignore)
LABEL_MAPPING = {
    0: 255,   # unlabeled
    1: 255,   # outlier
    10: 0,    # car
    11: 1,    # bicycle
    13: 4,    # bus → other-vehicle
    15: 2,    # motorcycle
    16: 4,    # on-rails → other-vehicle
    18: 3,    # truck
    20: 4,    # other-vehicle
    30: 5,    # person
    31: 6,    # bicyclist
    32: 7,    # motorcyclist
    40: 8,    # road
    44: 9,    # parking
    48: 10,   # sidewalk
    49: 11,   # other-ground
    50: 12,   # building
    51: 13,   # fence
    52: 255,  # other-structure → ignore
    60: 8,    # lane-marking → road
    70: 14,   # vegetation
    71: 15,   # trunk
    72: 16,   # terrain
    80: 17,   # pole
    81: 18,   # traffic-sign
    99: 255,  # other-object → ignore
    252: 0,   # moving-car → car
    253: 6,   # moving-bicyclist
    254: 5,   # moving-person
    255: 7,   # moving-motorcyclist
    256: 4,   # moving-on-rails
    257: 4,   # moving-bus
    258: 3,   # moving-truck
    259: 4,   # moving-other-vehicle
}

CLASS_NAMES = [
    'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
    'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
    'trunk', 'terrain', 'pole', 'traffic-sign',
]

# 类别频率倒数权重（log-inverse，手动校准）
CLASS_WEIGHTS = torch.tensor([
    1.00, 5.20, 5.20, 3.50, 4.50,
    8.10, 7.00, 9.50, 1.00, 3.20,
    1.50, 6.00, 1.20, 3.00, 1.00,
    4.00, 1.50, 6.00, 8.00,
], dtype=torch.float32)

# 训练/验证序列
TRAIN_SEQS = ['00','01','02','03','04','05','06','07','09','10']
VAL_SEQS   = ['08']
TEST_SEQS  = [f'{i:02d}' for i in range(11, 22)]


# ─────────────────────────────────────────────────────────────
# 数据增强
# ─────────────────────────────────────────────────────────────

class PointCloudAugmentor:
    """点云空间增强（在投影之前对点云操作）"""

    def __init__(self,
                 rotate: bool = True,
                 flip: bool = True,
                 scale: Tuple[float,float] = (0.95, 1.05),
                 drop_p: float = 0.05):
        self.rotate = rotate
        self.flip   = flip
        self.scale  = scale
        self.drop_p = drop_p

    def __call__(self, points: np.ndarray, labels: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        points: (N,4) x,y,z,intensity
        labels: (N,)
        """
        N = points.shape[0]

        # 全局旋转（绕 z 轴）
        if self.rotate:
            angle = np.random.uniform(-np.pi, np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
            points[:, :3] = points[:, :3] @ R.T

        # 随机翻转
        if self.flip:
            if np.random.rand() < 0.5:
                points[:, 0] *= -1   # x-flip
            if np.random.rand() < 0.5:
                points[:, 1] *= -1   # y-flip

        # 随机缩放
        if self.scale is not None:
            s = np.random.uniform(*self.scale)
            points[:, :3] *= s

        # 点随机丢弃
        if self.drop_p > 0:
            mask = np.random.rand(N) > self.drop_p
            points = points[mask]
            labels = labels[mask]

        return points, labels


class LaserMix:
    """
    LaserMix (CVPR 2023)：在点云域中，按仰角（激光束行号）随机交换两帧的连续波束段。
    与 PolarMix 形成互补——PolarMix 在方位角维度混合，LaserMix 在仰角维度混合。
    """

    def __init__(self, num_beams: int = 64,
                 fov_up: float = 2.0, fov_down: float = -24.8):
        self.num_beams = num_beams
        self.fov_up   = np.deg2rad(fov_up)
        self.fov_down = np.deg2rad(fov_down)
        self.fov = self.fov_up - self.fov_down

    def _beam_idx(self, points: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(points[:, :3], axis=1).clip(1e-5)
        phi = np.arcsin(np.clip(points[:, 2] / r, -1.0, 1.0))
        row = (self.fov_up - phi) / self.fov * (self.num_beams - 1)
        return np.clip(np.round(row).astype(np.int32), 0, self.num_beams - 1)

    def __call__(self, pts_a, lbl_a, pts_b, lbl_b):
        beams_a = self._beam_idx(pts_a)
        beams_b = self._beam_idx(pts_b)

        # 随机选取连续波束段（长度在 [H/4, H/2]，起点随机）
        n_swap = np.random.randint(self.num_beams // 4, self.num_beams // 2 + 1)
        start  = np.random.randint(0, self.num_beams)
        swap_beams = np.arange(start, start + n_swap) % self.num_beams

        mask_a = np.isin(beams_a, swap_beams)
        mask_b = np.isin(beams_b, swap_beams)

        pts_new = np.concatenate([pts_a[~mask_a], pts_b[mask_b]], axis=0)
        lbl_new = np.concatenate([lbl_a[~mask_a], lbl_b[mask_b]], axis=0)
        return pts_new, lbl_new


class PolarMix:
    """
    PolarMix (NeurIPS 2022) 的简化版本：
    随机选取一个方位角扇区，将两帧点云在该扇区内互换。
    """

    def __init__(self, num_sectors: int = 8, swap_p: float = 0.5):
        self.num_sectors = num_sectors
        self.swap_p = swap_p

    def __call__(self, pts_a, lbl_a, pts_b, lbl_b):
        # 概率控制已移至 __getitem__ 外层，此处直接执行扇区替换

        theta_a = np.arctan2(pts_a[:, 1], pts_a[:, 0])
        theta_b = np.arctan2(pts_b[:, 1], pts_b[:, 0])

        # 随机选择连续扇区
        n = np.random.randint(1, self.num_sectors // 2 + 1)
        start = np.random.uniform(-np.pi, np.pi)
        end   = start + n * (2 * np.pi / self.num_sectors)

        def in_sector(t):
            t = ((t - start) % (2*np.pi))
            e = ((end - start) % (2*np.pi))
            return t < e

        mask_a = in_sector(theta_a)
        mask_b = in_sector(theta_b)

        # 将 b 的扇区点拼接进 a
        pts_new = np.concatenate([pts_a[~mask_a], pts_b[mask_b]], axis=0)
        lbl_new = np.concatenate([lbl_a[~mask_a], lbl_b[mask_b]], axis=0)
        return pts_new, lbl_new


# ─────────────────────────────────────────────────────────────
# 数据集主类
# ─────────────────────────────────────────────────────────────

class SemanticKITTIDataset(Dataset):
    """
    SemanticKITTI 数据集加载器。

    目录结构：
        dataset/
          sequences/
            00/ velodyne/*.bin  labels/*.label
            01/ ...
            ...

    Args:
        root:      数据集根目录（含 sequences/）
        split:     'train' | 'val' | 'test'
        rv_H, rv_W: Range Image 分辨率
        pb_H, pb_W: Polar BEV 分辨率
        augment:   是否使用数据增强（训练时 True）
        use_polarmix: 是否使用 PolarMix 增强
    """

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 seqs: Optional[Union[str, Sequence[str]]] = None,
                 require_labels: bool = True,
                 rv_H: int = 64,  rv_W: int = 512,
                 pb_H: int = 240, pb_W: int = 512,
                 augment: bool = True,
                 rotate: bool = True,
                 flip: bool = True,
                 scale_min: float = 0.95,
                 scale_max: float = 1.05,
                 drop_p: float = 0.05,
                 R_max: float = 80.0,
                 use_polarmix: bool = True,
                 use_lasermix: bool = True,
                 fov_up: float = 2.0,
                 fov_down: float = -24.8,
                 max_points: int = 65536,
                 polarmix_p: float = 0.5,
                 polarmix_sectors: int = 8,
                 lasermix_p: float = 0.5,
                 # ── 【创新①②】RV 特征增强开关 ────────────────
                 use_surface_normals: bool = False,
                 use_angle_encoding: bool = False,
                 # ── Copy-Paste 实例增强 ────────────────────
                 copy_paste_bank: Optional[InstanceBank] = None,
                 copy_paste_classes: Optional[dict] = None,
                 cp_drop_p_base: float = 0.05,
                 cp_min_keep: int = 15,
                 cp_max_tries: int = 50):
        self.root = root
        self.split = split
        self.augment = augment and (split == 'train')
        self.max_points = max_points
        self.R_max = R_max
        self.rv_H = rv_H
        self.require_labels = require_labels and (split != 'test')

        if seqs is None:
            if split == 'train':
                seq_list = list(TRAIN_SEQS)
            elif split == 'val':
                seq_list = list(VAL_SEQS)
            else:
                seq_list = list(TEST_SEQS)
        else:
            if isinstance(seqs, str):
                seq_list = [s.strip() for s in seqs.split(',') if s.strip()]
            else:
                seq_list = [str(s).strip() for s in seqs if str(s).strip()]
        self.seqs = seq_list

        # 构建帧文件列表
        self.frames: List[Dict[str, str]] = []
        for seq in self.seqs:
            velo_dir  = os.path.join(root, 'sequences', seq, 'velodyne')
            label_dir = os.path.join(root, 'sequences', seq, 'labels')
            if not os.path.isdir(velo_dir):
                continue
            bins = sorted(os.listdir(velo_dir))
            for b in bins:
                stem = b.replace('.bin', '')
                entry = {
                    'velo': os.path.join(velo_dir, b),
                    'seq':  seq,
                }
                if split != 'test':
                    label_path = os.path.join(label_dir, stem + '.label')
                    if os.path.isfile(label_path):
                        entry['label'] = label_path
                    elif self.require_labels:
                        continue
                self.frames.append(entry)

        # 建立 label 映射表（uint16 → train_id）
        self._build_label_lut()

        # 投影器
        self.rv_proj = RangeImageProjector(rv_H, rv_W, fov_up, fov_down, R_max,
                                           use_surface_normals=use_surface_normals,
                                           use_angle_encoding=use_angle_encoding)
        self.pb_proj = PolarBEVProjector(pb_H, pb_W, R_max)

        # 增强
        self.augmentor = PointCloudAugmentor(
            rotate=rotate, flip=flip,
            scale=(scale_min, scale_max), drop_p=drop_p,
        ) if self.augment else None
        self.polarmix_p  = polarmix_p
        self.polarmix    = PolarMix(num_sectors=polarmix_sectors, swap_p=polarmix_p) \
                           if (use_polarmix and self.augment) else None
        self.lasermix_p  = lasermix_p
        self.lasermix    = LaserMix(num_beams=rv_H, fov_up=fov_up, fov_down=fov_down) \
                           if (use_lasermix and self.augment) else None

        # ── Copy-Paste 实例增强 ─────────────────────────────
        # cp_bank=None 时整个功能关闭（验证集/未建库时自动跳过）
        self.cp_bank     = copy_paste_bank if self.augment else None
        self.cp_classes  = copy_paste_classes or {}
        self.cp_drop_p   = cp_drop_p_base
        self.cp_min_keep = cp_min_keep
        self.cp_max_tries = cp_max_tries

    def _copy_paste_augment(self,
                             pts: np.ndarray,
                             labels: np.ndarray) -> tuple:
        """
        在点云域中粘贴稀有类实例。

        步骤：
          1. 建 BEV 占用格栅，标记已有点
          2. 缓存地面点，供局部地面高度估算
          3. 对每个目标类，按配置的概率/数量尝试粘贴
             a. 从 bank 采样 → DropPoints → yaw 旋转 → 缩放
             b. BEV 碰撞检测找合法位置（矩形重叠检查，最多 max_tries 次）
             c. 地面高度对齐（z_offset 方案）
             d. 记录最终实例位置与 3D bbox
          4. 批量移除被遮挡的原始点（3D bbox 内的背景点）
          5. 拼接并返回
        """
        # ── BEV 占用格栅 ────────────────────────────────────────
        GRID_RES   = 0.2          # m/cell
        GRID_RANGE = 52.0         # ±52 m
        G = int(2 * GRID_RANGE / GRID_RES)   # 520 × 520

        def _to_cell(xy2d):
            gx = np.floor((xy2d[:, 0] + GRID_RANGE) / GRID_RES).astype(np.int32)
            gy = np.floor((xy2d[:, 1] + GRID_RANGE) / GRID_RES).astype(np.int32)
            return np.clip(gx, 0, G - 1), np.clip(gy, 0, G - 1)

        bev = np.zeros((G, G), dtype=bool)
        gx0, gy0 = _to_cell(pts[:, :2])
        bev[gy0, gx0] = True

        # ── 地面点缓存 ────────────────────────────────────────
        gnd_mask = np.isin(labels, list(GROUND_MAPPED))
        gnd_pts  = pts[gnd_mask]   # (M, 4)

        # ── 收集待粘贴实例 ────────────────────────────────────
        pasted_pts_list   = []
        pasted_label_list = []
        paste_bboxes      = []    # [(xyz_min, xyz_max), ...]

        for cls_id, cfg in self.cp_classes.items():
            if cls_id not in self.cp_bank:
                continue
            k    = int(cfg.get('k', 1))
            prob = float(cfg.get('prob', 0.5))

            for _ in range(k):
                if np.random.rand() > prob:
                    continue

                # ── 采样 + 实例级增强 ─────────────────────────
                inst_pts, z_offset = self.cp_bank.sample(
                    cls_id, self.cp_drop_p, self.cp_min_keep)

                # yaw 旋转（绕 z 轴；XY 已中心化，旋转不引入平移误差）
                yaw = np.random.uniform(-np.pi, np.pi)
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                x_rot = cos_y * inst_pts[:, 0] - sin_y * inst_pts[:, 1]
                y_rot = sin_y * inst_pts[:, 0] + cos_y * inst_pts[:, 1]
                inst_pts = inst_pts.copy()
                inst_pts[:, 0] = x_rot
                inst_pts[:, 1] = y_rot

                # 随机缩放（xyz 同比，保持 intensity 不变）
                scale = np.random.uniform(0.95, 1.05)
                inst_pts[:, :3] *= scale
                z_offset         *= scale   # z_offset 随缩放同步调整

                # ── 寻找合法放置位置 ──────────────────────────
                placed = False
                for _ in range(self.cp_max_tries):
                    dist_t  = np.random.uniform(5.0, 40.0)
                    angle_t = np.random.uniform(-np.pi, np.pi)
                    tx = dist_t * np.cos(angle_t)
                    ty = dist_t * np.sin(angle_t)

                    trial_xy = inst_pts[:, :2] + [tx, ty]
                    # 超出格栅范围则跳过
                    if not np.all(np.abs(trial_xy) < GRID_RANGE - 1.0):
                        continue
                    # BEV 碰撞检测（矩形内任意格点被占用即跳过）
                    tgx, tgy = _to_cell(trial_xy)
                    if bev[tgy, tgx].any():
                        continue

                    # ── 地面高度估算 ──────────────────────────
                    if len(gnd_pts):
                        g_d = np.linalg.norm(gnd_pts[:, :2] - [tx, ty], axis=1)
                        near_gnd = gnd_pts[g_d < 2.0]
                        target_gz = float(np.percentile(near_gnd[:, 2], 10)) \
                                    if len(near_gnd) >= 5 else -1.73
                    else:
                        target_gz = -1.73   # HDL-64E 传感器安装高度 fallback

                    # z 平移：实例底部 = target_gz + z_offset
                    z_shift   = (target_gz + z_offset) - inst_pts[:, 2].min()
                    final_pts = inst_pts.copy()
                    final_pts[:, 0] += tx
                    final_pts[:, 1] += ty
                    final_pts[:, 2] += z_shift

                    # 标记格栅
                    bev[tgy, tgx] = True

                    pasted_pts_list.append(final_pts)
                    pasted_label_list.append(
                        np.full(len(final_pts), cls_id, dtype=np.int32))
                    paste_bboxes.append((
                        final_pts[:, :3].min(axis=0) - 0.15,
                        final_pts[:, :3].max(axis=0) + 0.15,
                    ))
                    break

        if not pasted_pts_list:
            return pts, labels

        # ── 批量剔除被遮挡的原始背景点 ──────────────────────
        # 对所有 bbox 联合计算，避免逐 bbox O(B·N) 循环
        keep = np.ones(len(pts), dtype=bool)
        for (bmin, bmax) in paste_bboxes:
            in_box = (
                (pts[:, 0] >= bmin[0]) & (pts[:, 0] <= bmax[0]) &
                (pts[:, 1] >= bmin[1]) & (pts[:, 1] <= bmax[1]) &
                (pts[:, 2] >= bmin[2]) & (pts[:, 2] <= bmax[2])
            )
            keep &= ~in_box

        pts_out    = np.concatenate([pts[keep]]    + pasted_pts_list,   axis=0)
        labels_out = np.concatenate([labels[keep]] + pasted_label_list, axis=0)
        return pts_out, labels_out

    def _build_label_lut(self):
        lut = np.full(65536, 255, dtype=np.int32)
        for raw, train in LABEL_MAPPING.items():
            if raw < 65536:
                lut[raw] = train
        self.label_lut = lut

    def __len__(self):
        return len(self.frames)

    def _load_frame(self, idx: int):
        entry = self.frames[idx]
        # 加载点云（N,4）
        pts = np.fromfile(entry['velo'], dtype=np.float32).reshape(-1, 4)
        # 过滤零点
        dist = np.linalg.norm(pts[:, :3], axis=1)
        pts = pts[dist > 0.5]

        if 'label' in entry and os.path.isfile(entry['label']):
            raw_labels = np.fromfile(entry['label'], dtype=np.uint32)
            raw_labels = raw_labels.reshape(-1) & 0xFFFF   # 取低16位语义标签
            raw_labels = raw_labels[dist > 0.5]
            labels = self.label_lut[raw_labels.astype(np.int32)]
        else:
            labels = np.zeros(len(pts), dtype=np.int32)

        return pts, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pts, labels = self._load_frame(idx)

        # PolarMix：以 polarmix_p 概率取另一帧混合
        if self.polarmix is not None and np.random.rand() < self.polarmix_p:
            idx2 = np.random.randint(0, len(self))
            pts2, lbl2 = self._load_frame(idx2)
            pts, labels = self.polarmix(pts, labels, pts2, lbl2)

        # LaserMix：以 lasermix_p 概率取另一帧混合（与 PolarMix 独立）
        if self.lasermix is not None and np.random.rand() < self.lasermix_p:
            idx2 = np.random.randint(0, len(self))
            pts2, lbl2 = self._load_frame(idx2)
            pts, labels = self.lasermix(pts, labels, pts2, lbl2)

        # Copy-Paste 实例增强（场景混合之后、整体几何增强之前）
        if self.cp_bank is not None and self.cp_classes:
            pts, labels = self._copy_paste_augment(pts, labels)

        # 数据增强
        if self.augmentor is not None:
            pts, labels = self.augmentor(pts, labels)

        # 裁剪到 R_max
        dist = np.linalg.norm(pts[:, :3], axis=1)
        valid = dist < self.R_max
        pts, labels = pts[valid], labels[valid]

        # 点数上限（随机下采样）
        N = pts.shape[0]
        if N > self.max_points:
            idx_s = np.random.choice(N, self.max_points, replace=False)
            pts    = pts[idx_s]
            labels = labels[idx_s]

        intensity = pts[:, 3].copy()

        # 生成 Range Image（rv_point_idx 记录每像素对应的点索引，用于像素级 aux 标签）
        rv_img, rv_point_idx = self.rv_proj.project(pts[:, :3], intensity)
        rv_coords = self.rv_proj.compute_sample_coords(pts[:, :3])

        # 像素级 Range Image 标签（用于 rv_aux 辅助损失）
        # rv_point_idx[h,w] >= 0 时取该点的语义标签，否则填 255（ignore）
        rv_labels_np = np.full(rv_point_idx.shape, 255, dtype=np.int64)
        valid_pix = rv_point_idx >= 0
        rv_labels_np[valid_pix] = labels[rv_point_idx[valid_pix]]

        # 生成 Polar BEV
        pb_img = self.pb_proj.project(pts[:, :3], intensity)
        pb_coords = self.pb_proj.compute_sample_coords(pts[:, :3])

        # 转 Tensor
        rv_img_t    = torch.from_numpy(rv_img)                        # (rv_channels, H, W)
        pb_img_t    = torch.from_numpy(pb_img)                        # (9, H_p, W_p)
        pts_t       = torch.from_numpy(pts.astype(np.float32))        # (N, 4)
        labels_t    = torch.from_numpy(labels.astype(np.int64))       # (N,)
        rv_labels_t = torch.from_numpy(rv_labels_np)                  # (H, W)

        # grid_sample 格式坐标: (N, 1, 2)
        rv_coords_t = torch.from_numpy(rv_coords).unsqueeze(1)        # (N,1,2)
        pb_coords_t = torch.from_numpy(pb_coords).unsqueeze(1)        # (N,1,2)

        return {
            'rv_img':    rv_img_t,
            'pb_img':    pb_img_t,
            'rv_coords': rv_coords_t,
            'pb_coords': pb_coords_t,
            'points':    pts_t,
            'labels':    labels_t,
            'rv_labels': rv_labels_t,   # (H, W) 用于 RV 辅助损失
        }


# ─────────────────────────────────────────────────────────────
# 可变长度点云的 collate_fn（pad 到批次内最大 N）
# ─────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    将一个 batch 的样本对齐到最大点数（用 255 pad 标签，0 pad 坐标）。
    Range Image、Polar BEV、rv_labels 为固定尺寸，直接 stack。
    """
    rv_imgs    = torch.stack([b['rv_img']    for b in batch])
    pb_imgs    = torch.stack([b['pb_img']    for b in batch])
    rv_labels  = torch.stack([b['rv_labels'] for b in batch])   # (B, H, W)

    max_N = max(b['points'].shape[0] for b in batch)
    B = len(batch)

    pts_pad    = torch.zeros(B, max_N, 4,  dtype=torch.float32)
    labels_pad = torch.full((B, max_N),   255, dtype=torch.int64)
    rv_coords  = torch.zeros(B, max_N, 1, 2, dtype=torch.float32)
    pb_coords  = torch.zeros(B, max_N, 1, 2, dtype=torch.float32)

    for i, b in enumerate(batch):
        N = b['points'].shape[0]
        pts_pad[i, :N]    = b['points']
        labels_pad[i, :N] = b['labels']
        rv_coords[i, :N]  = b['rv_coords']
        pb_coords[i, :N]  = b['pb_coords']

    return {
        'rv_img':    rv_imgs,
        'pb_img':    pb_imgs,
        'rv_coords': rv_coords,
        'pb_coords': pb_coords,
        'points':    pts_pad,
        'labels':    labels_pad,
        'rv_labels': rv_labels,   # (B, H, W) 像素级 GT，用于 rv_aux 辅助损失
    }
