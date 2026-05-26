"""
RankMe: 无监督表征质量评估（Effective Rank）。

参考思想：
  给定特征矩阵 X (N, D)，对其奇异值谱做概率化并计算熵，
  最终分数为 exp(H)，范围约在 [1, min(N,D)]。
"""

from typing import Optional

import torch


@torch.no_grad()
def effective_rank_from_features(
    features: torch.Tensor,
    center: bool = True,
    eps: float = 1.0e-12,
) -> float:
    """
    Args:
        features: (N, D) float tensor
        center: 是否按列去均值
    Returns:
        effective rank (float)
    """
    if features.dim() != 2:
        raise ValueError(f"features must be 2D, got {tuple(features.shape)}")
    if features.shape[0] < 2 or features.shape[1] < 2:
        return 1.0

    x = features.float()
    if center:
        x = x - x.mean(dim=0, keepdim=True)

    # 使用奇异值平方作为能量谱
    s = torch.linalg.svdvals(x)
    p = (s * s).clamp_min(0.0)
    z = p.sum()
    if z <= eps:
        return 1.0
    p = (p / z).clamp_min(eps)
    h = -(p * torch.log(p)).sum()
    r = torch.exp(h).item()
    return float(r)

