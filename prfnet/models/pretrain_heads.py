"""
Pretraining heads for NOMAE + PCP-MAE style self-supervised learning.

Design goals:
- Training-only heads, removable at inference.
- Keep implementation lightweight and stable for SemanticKITTI RV/PB setup.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_random_mask_like(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """
    Build per-sample random spatial mask.

    Args:
        x: (B, C, H, W)
        mask_ratio: fraction in [0, 1)
    Returns:
        mask: (B, 1, H, W), 1 means masked.
    """
    B, _, H, W = x.shape
    total = H * W
    n_mask = max(1, int(total * max(0.0, min(mask_ratio, 0.95))))
    mask = torch.zeros(B, total, device=x.device, dtype=x.dtype)
    for b in range(B):
        idx = torch.randperm(total, device=x.device)[:n_mask]
        mask[b, idx] = 1.0
    return mask.reshape(B, 1, H, W)


class NOMAEPCPPretrainHead(nn.Module):
    """
    A minimal NOMAE + PCP style head:
    - NOMAE branch predicts occupancy for masked tokens.
    - PCP branch predicts masked cell center (xyz) proxy.
    """

    def __init__(self, in_c: int):
        super().__init__()
        hidden = max(in_c // 2, 32)
        self.occ_head = nn.Sequential(
            nn.Conv2d(in_c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        self.center_head = nn.Sequential(
            nn.Conv2d(in_c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, 3, 1, bias=True),
        )

    def forward(
        self,
        feat: torch.Tensor,
        mask: torch.Tensor,
        target_occ: torch.Tensor,
        target_center: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat: (B, C, H, W)
            mask: (B, 1, H, W), 1 means masked.
            target_occ: (B, 1, H, W), occupancy target in {0,1}
            target_center: (B, 3, H, W), xyz center target
        """
        pred_occ = self.occ_head(feat)
        pred_center = self.center_head(feat)

        # Only supervise masked cells.
        occ_loss_map = F.binary_cross_entropy_with_logits(
            pred_occ, target_occ, reduction="none"
        )
        center_loss_map = F.smooth_l1_loss(
            pred_center, target_center, reduction="none"
        ).mean(dim=1, keepdim=True)

        denom = mask.sum().clamp(min=1.0)
        loss_occ = (occ_loss_map * mask).sum() / denom
        loss_pcp = (center_loss_map * mask).sum() / denom

        return {
            "pred_occ": pred_occ,
            "pred_center": pred_center,
            "loss_occ": loss_occ,
            "loss_pcp": loss_pcp,
        }
