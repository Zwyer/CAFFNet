"""
投影工具：Range Image 和 Polar BEV 的点云投影
所有操作均兼容 RKNN 静态图部署（坐标预计算，grid_sample 友好）
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────────────────────
# SemanticKITTI HDL-64E 传感器参数
# ─────────────────────────────────────────────────────────────
SENSOR = {
    "fov_up":   2.0,    # 度
    "fov_down": -24.8,  # 度
    "H": 64,            # 激光线数
    "W": 1024,          # 方位角分辨率
    "R_max": 80.0,      # 最大距离(m)
    "R_min": 0.5,       # 最小距离(m)
}

POLAR_BEV = {
    "H_p": 480,   # 径距方向格数
    "W_p": 1024,  # 方位角格数（与 RV 对齐）
    "R_max": 80.0,
}


# ─────────────────────────────────────────────────────────────
# Range Image 投影
# ─────────────────────────────────────────────────────────────

class RangeImageProjector:
    """
    将点云投影到球面 Range Image。

    基础通道（始终启用，共 6 ch）：
        [x/R, y/R, z/R, r/R, intensity, cos_phi]

    【创新②】use_angle_encoding=True 追加 3 ch：
        [sin_phi, cos_theta, sin_theta]
        角度通道对所有像素（含空像素）按网格坐标填充，形成完整几何坐标系。
        同时将 ch[5] cos_phi 也改为全像素网格填充，修正原版在空像素处
        错误显示 0（实际 cos(-10°)≈0.98）的问题。

    【创新①】use_surface_normals=True 追加 3 ch：
        [nx, ny, nz]
        通过 3D 坐标中心差分叉乘估计局部表面法向量，等价于球面投影
        雅可比 (∂p/∂θ × ∂p/∂φ) 的二阶中心差分近似。
        仅在像素本身及其上下左右 4 邻域均有效时填充，否则置 0，
        以避免空像素 xyz=(0,0,0) 污染切向量方向。

    通道顺序（均开时共 12 ch）：
        0-4 : x/R, y/R, z/R, r/R, intensity        （点级，空像素=0）
        5   : cos_phi                               （网格级，全像素）
        6-8 : sin_phi, cos_theta, sin_theta         （网格级，use_angle_encoding）
        9-11: nx, ny, nz                            （点级，use_surface_normals）

    总通道数由 rv_channels 属性查询：
        6（均关）/ 9（任一开）/ 12（均开）
    """

    def __init__(self, H=64, W=1024,
                 fov_up=2.0, fov_down=-24.8, R_max=80.0,
                 use_surface_normals: bool = False,
                 use_angle_encoding: bool = False):
        self.H = H
        self.W = W
        self.fov_up   = np.deg2rad(fov_up)
        self.fov_down = np.deg2rad(fov_down)
        self.fov = self.fov_up - self.fov_down
        self.R_max = R_max
        self.use_surface_normals = use_surface_normals
        self.use_angle_encoding  = use_angle_encoding

        # 预计算网格级角度坐标（__init__ 时一次性计算，复用于每帧）
        # phi_grid[i, j]   = 第 i 行对应的名义仰角（均匀近似）
        # theta_grid[i, j] = 第 j 列对应的方位角
        phi_rows   = (self.fov_up
                      - np.arange(H, dtype=np.float32) / max(H - 1, 1) * self.fov)
        theta_cols = np.linspace(-np.pi, np.pi, W, dtype=np.float32)
        self._phi_grid   = phi_rows[:, np.newaxis] * np.ones((1, W), dtype=np.float32)
        self._theta_grid = np.ones((H, 1), dtype=np.float32) * theta_cols[np.newaxis, :]

    @property
    def rv_channels(self) -> int:
        """返回 Range Image 的总通道数（由开关决定）"""
        return 6 + 3 * int(self.use_angle_encoding) + 3 * int(self.use_surface_normals)

    def _compute_surface_normals(self,
                                 range_img: np.ndarray,
                                 valid: np.ndarray) -> np.ndarray:
        """
        从 Range Image 的 3D 坐标估计表面法向量。

        方法：对每个像素用其上下左右 4 邻域的 3D 坐标构造中心差分切向量，
        再叉乘得法线。这等价于球面投影雅可比的二阶中心差分近似：
            tangent_j ≈ 2Δθ · (∂p/∂θ) = 2Δθ · (r·cos_φ·ê_θ + ∂r/∂θ·ê_r)
            tangent_i ≈ 2Δφ · (∂p/∂φ) = 2Δφ · (r·ê_φ      + ∂r/∂φ·ê_r)
        归一化后比例因子 2Δθ、2Δφ 消去，结果等同于解析推导。

        仅对 4 邻域均有效的像素计算法向量，其余置 0，
        以避免空像素 xyz=(0,0,0) 污染差分方向。

        Args:
            range_img: (rv_channels, H, W) — 前 3 ch 已填充 x/R, y/R, z/R
            valid:     (H, W) bool — 有效像素掩码

        Returns:
            normals: (3, H, W) float32
        """
        # 还原真实 3D 坐标（前 3 通道存的是 x/R_max）
        xyz = range_img[:3] * self.R_max   # (3, H, W)

        # ── 邻域有效性掩码 ───────────────────────────────────────
        # j 轴（方位角）：周期循环，直接 roll
        valid_j_p = np.roll(valid, -1, axis=1)   # valid[i, j+1]
        valid_j_m = np.roll(valid,  1, axis=1)   # valid[i, j-1]

        # i 轴（仰角）：非周期，用 False 填充边界外
        vpad      = np.pad(valid, 1, mode='constant', constant_values=False)
        valid_i_p = vpad[2:,  1:-1]              # valid[i+1, j]
        valid_i_m = vpad[:-2, 1:-1]              # valid[i-1, j]

        # 四邻域均有效 → 法向量可信
        valid_normal = valid & valid_j_p & valid_j_m & valid_i_p & valid_i_m

        # ── 切向量（中心差分）──────────────────────────────────
        # j 轴：方位角方向，周期循环
        tangent_j = np.roll(xyz, -1, axis=2) - np.roll(xyz, 1, axis=2)   # (3,H,W)

        # i 轴：仰角方向，边界用 edge padding（边界点用自身值重复，差分为 0）
        xyz_pad   = np.pad(xyz, ((0, 0), (1, 1), (0, 0)), mode='edge')
        tangent_i = xyz_pad[:, 2:, :] - xyz_pad[:, :-2, :]               # (3,H,W)

        # ── 叉乘 n = tangent_j × tangent_i ─────────────────────
        tj, ti = tangent_j, tangent_i
        nx = tj[1] * ti[2] - tj[2] * ti[1]
        ny = tj[2] * ti[0] - tj[0] * ti[2]
        nz = tj[0] * ti[1] - tj[1] * ti[0]

        normals  = np.stack([nx, ny, nz], axis=0)   # (3, H, W)

        # ── 归一化 ───────────────────────────────────────────────
        norm_len = np.sqrt(nx**2 + ny**2 + nz**2).clip(1e-8)
        normals /= norm_len[np.newaxis]

        # ── 清零：邻域不完整的像素（含退化叉积）────────────────
        normals[:, ~valid_normal] = 0.0

        return normals.astype(np.float32)

    def project(self, points: np.ndarray, remissions: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            points:     (N, 3) float32 — x,y,z
            remissions: (N,)   float32 — intensity [0,1]

        Returns:
            range_img:  (rv_channels, H, W) float32
            point_idx:  (H, W)    int32  — 每格存放哪个点的索引(-1=空)
        """
        N = points.shape[0]
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2).clip(1e-5, self.R_max)

        phi   = np.arcsin(np.clip(z / r, -1.0, 1.0))  # 仰角，clip 防止浮点误差越界
        theta = np.arctan2(y, x)                       # 方位角 [-pi, pi]

        # 像素坐标（浮点）
        row_f = (self.fov_up - phi) / self.fov * (self.H - 1)
        col_f = (theta + np.pi) / (2 * np.pi) * (self.W - 1)

        row = np.clip(np.round(row_f).astype(np.int32), 0, self.H - 1)
        col = np.clip(np.round(col_f).astype(np.int32), 0, self.W - 1)

        # 按距离从远到近排序，近点覆盖远点（保留最近点）
        order = np.argsort(-r)
        row_s, col_s = row[order], col[order]

        point_idx = np.full((self.H, self.W), -1, dtype=np.int32)
        point_idx[row_s, col_s] = order

        # 分配特征图（通道数由开关决定）
        range_img = np.zeros((self.rv_channels, self.H, self.W), dtype=np.float32)
        valid = point_idx >= 0
        vi    = point_idx[valid]

        # ── 通道 0-4：基础点级特征（仅 valid 像素）─────────────
        range_img[0][valid] = x[vi] / self.R_max
        range_img[1][valid] = y[vi] / self.R_max
        range_img[2][valid] = z[vi] / self.R_max
        range_img[3][valid] = r[vi] / self.R_max
        range_img[4][valid] = remissions[vi]

        # ── 通道 5：cos_phi（网格级，全像素填充）────────────────
        # 修复原版空像素处 cos_phi=0 的问题（cos(-10°)≈0.98，不是 0）
        range_img[5] = np.cos(self._phi_grid)

        # ── 【创新②】仰角+方位角位置编码（全像素，网格级）──────
        ch = 6
        if self.use_angle_encoding:
            range_img[ch]     = np.sin(self._phi_grid)    # sin_phi
            range_img[ch + 1] = np.cos(self._theta_grid)  # cos_theta
            range_img[ch + 2] = np.sin(self._theta_grid)  # sin_theta
            ch += 3

        # ── 【创新①】表面法向量（3D 叉乘，仅完整 4 邻域）───────
        if self.use_surface_normals:
            normals = self._compute_surface_normals(range_img, valid)
            range_img[ch:ch + 3] = normals

        return range_img, point_idx

    def compute_sample_coords(self, points: np.ndarray) -> np.ndarray:
        """
        计算每个点在 Range Image 上的归一化采样坐标 (u,v) ∈ [-1,1]²。
        用于 grid_sample 逐点特征提取。

        Returns:
            coords: (N, 2) float32 — [col_norm, row_norm]，顺序同 grid_sample (x→col, y→row)
        """
        N = points.shape[0]
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2).clip(1e-5)

        phi   = np.arcsin(np.clip(z / r, -1, 1))
        theta = np.arctan2(y, x)

        row_f = (self.fov_up - phi) / self.fov  # [0,1]
        col_f = (theta + np.pi) / (2 * np.pi)   # [0,1]

        # grid_sample 期望 [-1, 1]: 2*norm - 1
        u = (2 * col_f - 1).astype(np.float32)   # 方位角 → x 方向
        v = (2 * row_f - 1).astype(np.float32)   # 仰角   → y 方向

        coords = np.stack([u, v], axis=-1)  # (N, 2)
        return coords


# ─────────────────────────────────────────────────────────────
# Polar BEV 投影
# ─────────────────────────────────────────────────────────────

class PolarBEVProjector:
    """
    将点云聚合到极坐标 BEV 格。
    返回 9 通道图像：[x̄, ȳ, z̄, z_min, z_max, ī, log_count, r̄, occ]

    径距采用 sqrt-spacing，近处格子更小：
        ρ_edges = linspace(0, sqrt(R_max), H_p+1) ** 2
    """

    def __init__(self, H_p=480, W_p=1024, R_max=80.0):
        self.H_p = H_p
        self.W_p = W_p
        self.R_max = R_max

        # 径距边界（sqrt-spacing）
        self.rho_edges = np.linspace(0, np.sqrt(R_max), H_p + 1) ** 2
        # 方位角边界（均匀）
        self.theta_edges = np.linspace(-np.pi, np.pi, W_p + 1)

    def _point_to_grid(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """返回每个点对应的 (row, col) 格子索引"""
        x, y = points[:, 0], points[:, 1]
        rho   = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        row = np.searchsorted(self.rho_edges[1:], rho, side='left')
        col = np.searchsorted(self.theta_edges[1:], theta, side='left')

        row = np.clip(row, 0, self.H_p - 1)
        col = np.clip(col, 0, self.W_p - 1)
        return row, col

    def project(self, points: np.ndarray, remissions: np.ndarray
                ) -> np.ndarray:
        """
        Args:
            points:     (N, 3) float32
            remissions: (N,)   float32

        Returns:
            bev: (9, H_p, W_p) float32
        """
        N = points.shape[0]
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)

        row, col = self._point_to_grid(points)
        flat_idx = row * self.W_p + col  # (N,)

        H, W = self.H_p, self.W_p
        bev = np.zeros((9, H * W), dtype=np.float32)

        # 统计每格点数
        count = np.bincount(flat_idx, minlength=H * W).astype(np.float32)

        # 各通道 sum（后续除以 count 得均值）
        bev[0] = np.bincount(flat_idx, weights=x,            minlength=H*W)
        bev[1] = np.bincount(flat_idx, weights=y,            minlength=H*W)
        bev[2] = np.bincount(flat_idx, weights=z,            minlength=H*W)
        bev[5] = np.bincount(flat_idx, weights=remissions,   minlength=H*W)
        bev[7] = np.bincount(flat_idx, weights=r,            minlength=H*W)

        # z_min, z_max 需要逐格处理（用 np.ufunc.at）
        z_min = np.full(H * W,  np.inf, dtype=np.float32)
        z_max = np.full(H * W, -np.inf, dtype=np.float32)
        np.minimum.at(z_min, flat_idx, z)
        np.maximum.at(z_max, flat_idx, z)

        # 归一化
        valid = count > 0
        bev[0][valid] /= count[valid]
        bev[1][valid] /= count[valid]
        bev[2][valid] /= count[valid]
        bev[3] = np.where(valid, z_min, 0) / self.R_max
        bev[4] = np.where(valid, z_max, 0) / self.R_max
        bev[0][valid] /= self.R_max
        bev[1][valid] /= self.R_max
        bev[2][valid] /= self.R_max
        bev[5][valid] /= count[valid]
        bev[6] = np.log1p(count) / np.log1p(64)   # 对数归一化
        bev[7][valid] /= (count[valid] * self.R_max)
        bev[8] = valid.astype(np.float32)          # 占用标志

        return bev.reshape(9, H, W)

    def compute_sample_coords(self, points: np.ndarray) -> np.ndarray:
        """
        计算每个点在 Polar BEV 上的归一化采样坐标 (u,v) ∈ [-1,1]²。

        Returns:
            coords: (N, 2) float32 — [col_norm, row_norm]
        """
        x, y = points[:, 0], points[:, 1]
        rho   = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        # 归一化到 [0,1]
        rho_norm   = np.searchsorted(self.rho_edges[1:], rho) / self.H_p
        theta_norm = (theta + np.pi) / (2 * np.pi)

        # 转 [-1,1]
        u = (2 * theta_norm - 1).astype(np.float32)
        v = (2 * rho_norm   - 1).astype(np.float32)

        return np.stack([u, v], axis=-1)  # (N, 2)
