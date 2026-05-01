"""
Pretraining heads for NOMAE + PCP-MAE style self-supervised learning.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _clamp_ratio(x: float, lo: float = 0.0, hi: float = 0.95) -> float:
    return float(max(lo, min(float(x), hi)))


def build_random_mask_like(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Build per-sample random spatial mask; 1 means masked."""
    B, _, H, W = x.shape
    total = H * W
    n_mask = max(1, int(total * _clamp_ratio(mask_ratio)))
    mask = torch.zeros(B, total, device=x.device, dtype=x.dtype)
    for b in range(B):
        idx = torch.randperm(total, device=x.device)[:n_mask]
        mask[b, idx] = 1.0
    return mask.reshape(B, 1, H, W)


def build_mask_with_pos_ratio_control(
    x: torch.Tensor,
    target_occ: torch.Tensor,
    mask_ratio: float,
    enable_control: bool = False,
    min_pos_ratio: float = 0.08,
    max_pos_ratio: float = 0.50,
    max_tries: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build mask with optional masked-positive-ratio control.
    Returns:
      mask: (B,1,H,W)
      resample_count: scalar tensor
    """
    if not enable_control:
        return build_random_mask_like(x, mask_ratio), x.new_tensor(0.0)

    B = x.shape[0]
    out_mask: List[torch.Tensor] = []
    resample_count = 0.0
    min_pos_ratio = float(max(0.0, min(1.0, min_pos_ratio)))
    max_pos_ratio = float(max(min_pos_ratio, min(1.0, max_pos_ratio)))
    occ_bool = target_occ > 0.5

    for b in range(B):
        accepted = None
        one_x = x[b : b + 1]
        one_occ = occ_bool[b : b + 1]
        for t in range(max(1, int(max_tries))):
            candidate = build_random_mask_like(one_x, mask_ratio)
            masked_cnt = candidate.sum().item()
            if masked_cnt <= 0:
                continue
            pos_ratio = float((one_occ.float() * candidate).sum().item() / max(masked_cnt, 1.0))
            if min_pos_ratio <= pos_ratio <= max_pos_ratio:
                accepted = candidate
                resample_count += float(t)
                break
        if accepted is None:
            accepted = build_random_mask_like(one_x, mask_ratio)
            resample_count += float(max(0, int(max_tries) - 1))
        out_mask.append(accepted)

    return torch.cat(out_mask, dim=0), x.new_tensor(resample_count / max(float(B), 1.0))


class NOMAEPCPPretrainHead(nn.Module):
    """
    NOMAE + PCP style head with:
    - multi-scale occupancy modeling
    - PCP anti-leakage two-stage center prediction
    """

    def __init__(
        self,
        in_c: int,
        occ_scales: Sequence[int] = (1, 3, 5),
        occ_loss_type: str = "bce_pos_weight",
        occ_pos_weight: float = 5.0,
        occ_focal_gamma: float = 2.0,
        pcp_stopgrad_replace: bool = True,
    ):
        super().__init__()
        hidden = max(in_c // 2, 32)
        self.occ_scales = [int(max(1, s)) for s in occ_scales]
        self.occ_loss_type = str(occ_loss_type).lower()
        self.occ_pos_weight = float(max(1e-6, occ_pos_weight))
        self.occ_focal_gamma = float(max(0.0, occ_focal_gamma))
        self.pcp_stopgrad_replace = bool(pcp_stopgrad_replace)

        self.occ_head = nn.Sequential(
            nn.Conv2d(in_c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, len(self.occ_scales), 1, bias=True),
        )
        self.center_head_stage1 = nn.Sequential(
            nn.Conv2d(in_c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, 3, 1, bias=True),
        )
        self.center_head_stage2 = nn.Sequential(
            nn.Conv2d(in_c + 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, 3, 1, bias=True),
        )

    def _build_occ_targets(self, target_occ: torch.Tensor) -> torch.Tensor:
        """
        Build multi-scale neighborhood occupancy targets.
        target_occ: (B,1,H,W)
        returns: (B,S,H,W)
        """
        outs = []
        for k in self.occ_scales:
            if k <= 1:
                outs.append(target_occ)
            else:
                pad = k // 2
                # Neighborhood occupancy: if any occupied in kxk, target=1.
                outs.append(F.max_pool2d(target_occ, kernel_size=k, stride=1, padding=pad))
        return torch.cat(outs, dim=1)

    def _occ_loss(
        self,
        pred_occ: torch.Tensor,
        target_occ_ms: torch.Tensor,
        supervise_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        pred_occ: (B,S,H,W)
        target_occ_ms: (B,S,H,W)
        supervise_mask: (B,1,H,W)
        """
        if self.occ_loss_type == "focal":
            bce = F.binary_cross_entropy_with_logits(pred_occ, target_occ_ms, reduction="none")
            prob = torch.sigmoid(pred_occ)
            p_t = prob * target_occ_ms + (1.0 - prob) * (1.0 - target_occ_ms)
            focal = (1.0 - p_t).pow(self.occ_focal_gamma)
            loss_map = bce * focal
        elif self.occ_loss_type == "bce_pos_weight":
            pw = pred_occ.new_tensor(self.occ_pos_weight)
            loss_map = F.binary_cross_entropy_with_logits(
                pred_occ, target_occ_ms, reduction="none", pos_weight=pw
            )
        else:
            loss_map = F.binary_cross_entropy_with_logits(pred_occ, target_occ_ms, reduction="none")

        s_mask = supervise_mask.expand_as(loss_map)
        denom = s_mask.sum().clamp(min=1.0)
        return (loss_map * s_mask).sum() / denom

    def forward(
        self,
        feat: torch.Tensor,
        mask: torch.Tensor,
        target_occ: torch.Tensor,
        target_center: torch.Tensor,
        informative_only: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
          feat: (B,C,H,W)
          mask: (B,1,H,W), 1 means masked
          target_occ: (B,1,H,W), occupancy target in {0,1}
          target_center: (B,3,H,W), xyz target center
        """
        pred_occ = self.occ_head(feat)  # (B,S,H,W)
        occ_tgt_ms = self._build_occ_targets(target_occ)

        # NOMAE-style informative neighborhood supervision.
        if informative_only:
            informative = (occ_tgt_ms.max(dim=1, keepdim=True).values > 0.0).float()
            occ_sup_mask = mask * informative
        else:
            occ_sup_mask = mask
        if occ_sup_mask.sum() <= 0:
            occ_sup_mask = mask

        loss_occ = self._occ_loss(pred_occ, occ_tgt_ms, occ_sup_mask)

        # PCP anti-leakage:
        # stage1 predicts masked centers, then replaced center tokens are used by stage2.
        pred_center_stage1 = self.center_head_stage1(feat)
        pred_token = pred_center_stage1.detach() if self.pcp_stopgrad_replace else pred_center_stage1
        replaced_center = torch.where(mask.expand_as(target_center) > 0.5, pred_token, target_center)
        pred_center = self.center_head_stage2(torch.cat([feat, replaced_center], dim=1))

        center_loss_map = F.smooth_l1_loss(pred_center, target_center, reduction="none").mean(
            dim=1, keepdim=True
        )
        denom = mask.sum().clamp(min=1.0)
        loss_pcp = (center_loss_map * mask).sum() / denom

        masked_total = mask.sum().clamp(min=1.0)
        masked_pos_ratio = ((target_occ > 0.5).float() * mask).sum() / masked_total
        occ_effective_ratio = occ_sup_mask.sum() / masked_total

        return {
            "pred_occ": pred_occ,
            "pred_center": pred_center,
            "loss_occ": loss_occ,
            "loss_pcp": loss_pcp,
            "masked_pos_ratio": masked_pos_ratio.detach(),
            "occ_effective_ratio": occ_effective_ratio.detach(),
        }
