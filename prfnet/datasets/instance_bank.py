"""
InstanceBank：加载离线预建的实例库，提供随机采样与 DropPoints 增强。

离线库由 tools/build_instance_bank.py 生成（npz, CSR 格式）。
在 DataLoader 多进程下，Linux fork 后各 worker 共享内存（copy-on-write），
无需每个 worker 独立加载，内存开销小。
"""

import numpy as np
from typing import Optional

# cls_id (0-18) → 库中的字段名前缀
CLS_ID_TO_NAME = {
    7: 'motorcyclist',
    6: 'bicyclist',
    5: 'person',
    2: 'motorcycle',
    1: 'bicycle',
    4: 'other_vehicle',
}


class InstanceBank:
    """
    从 npz 加载实例库，支持随机采样。

    每个类存储为 CSR 格式：
      {name}_pts      float32 (total_pts, 4)   所有实例拼接，XY 已中心化
      {name}_offsets  int32   (n_inst+1,)       inst[i] = pts[off[i]:off[i+1]]
      {name}_zoff     float32 (n_inst,)         底部相对地面高度差
      {name}_dists    float32 (n_inst,)         原始距离（供 DropPoints 使用）
    """

    def __init__(self, bank_path: str, target_cls_ids: Optional[list] = None):
        """
        Args:
            bank_path:       npz 文件路径
            target_cls_ids:  只加载这些类（None = 全部）
        """
        data = np.load(bank_path, allow_pickle=False)
        self.bank: dict = {}

        load_ids = target_cls_ids if target_cls_ids is not None \
                   else list(CLS_ID_TO_NAME.keys())

        for cls_id in load_ids:
            name = CLS_ID_TO_NAME.get(cls_id)
            if name is None:
                raise ValueError(f'cls_id {cls_id} 不在支持列表中: {list(CLS_ID_TO_NAME)}')

            key_pts = f'{name}_pts'
            if key_pts not in data:
                raise KeyError(f'bank 文件缺少字段 {key_pts}，请重新运行 build_instance_bank.py')

            pts_flat  = data[f'{name}_pts']      # (total_pts, 4)
            offsets   = data[f'{name}_offsets']  # (n+1,)
            zoff      = data[f'{name}_zoff']     # (n,)
            dists     = data[f'{name}_dists']    # (n,)
            n_inst    = len(zoff)

            self.bank[cls_id] = {
                'pts_flat': pts_flat,
                'offsets':  offsets,
                'zoff':     zoff,
                'dists':    dists,
                'n':        n_inst,
            }

        # 统计信息
        info = ', '.join(
            f'{CLS_ID_TO_NAME[c]}×{self.bank[c]["n"]}'
            for c in sorted(self.bank)
        )
        print(f'[InstanceBank] 已加载: {info}  (来自 {bank_path})')

    def __contains__(self, cls_id: int) -> bool:
        return cls_id in self.bank and self.bank[cls_id]['n'] > 0

    def sample(self, cls_id: int,
               drop_p_base: float = 0.05,
               min_keep: int = 15) -> tuple:
        """
        随机采样一个实例，应用距离自适应 DropPoints。

        Returns:
            pts      (M, 4) float32，XY 已中心化，z 保留原始传感器坐标
            z_offset float，实例底部相对于被采集时地面的高度差（≥ 0）
        """
        b   = self.bank[cls_id]
        idx = np.random.randint(b['n'])
        s, e = int(b['offsets'][idx]), int(b['offsets'][idx + 1])
        pts  = b['pts_flat'][s:e].copy()         # (M, 4)
        dist = float(b['dists'][idx])

        # ── 距离自适应 DropPoints ─────────────────────────
        # 近处 (~20m) drop_p_base，远处最多 15%
        drop_p = float(np.clip(drop_p_base * (dist / 20.0), drop_p_base, 0.15))
        n_keep = max(min_keep, int(len(pts) * (1.0 - drop_p)))
        if n_keep < len(pts):
            keep = np.random.choice(len(pts), n_keep, replace=False)
            pts  = pts[keep]

        return pts, float(b['zoff'][idx])
