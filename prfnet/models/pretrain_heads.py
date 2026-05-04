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


def build_block_mask_like(
    x: torch.Tensor,
    mask_ratio: float,
    block_h_min: int = 4,
    block_h_max: int = 16,
    block_w_min: int = 16,
    block_w_max: int = 64,
) -> torch.Tensor:
    """Build rectangular block mask; 1 means masked."""
    B, _, H, W = x.shape
    total = H * W
    n_mask = max(1, int(total * _clamp_ratio(mask_ratio)))
    out = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
    for b in range(B):
        filled = 0
        tries = 0
        while filled < n_mask and tries < 512:
            tries += 1
            bh = int(torch.randint(max(1, block_h_min), max(block_h_min + 1, block_h_max + 1), (1,), device=x.device).item())
            bw = int(torch.randint(max(1, block_w_min), max(block_w_min + 1, block_w_max + 1), (1,), device=x.device).item())
            bh = min(bh, H)
            bw = min(bw, W)
            top = int(torch.randint(0, max(1, H - bh + 1), (1,), device=x.device).item())
            left = int(torch.randint(0, max(1, W - bw + 1), (1,), device=x.device).item())
            out[b, 0, top : top + bh, left : left + bw] = 1.0
            filled = int(out[b, 0].sum().item())
        if filled > n_mask:
            idx = torch.nonzero(out[b, 0].reshape(-1) > 0.5, as_tuple=False).squeeze(1)
            keep = idx[torch.randperm(idx.numel(), device=x.device)[:n_mask]]
            one = torch.zeros(total, device=x.device, dtype=x.dtype)
            one[keep] = 1.0
            out[b, 0] = one.reshape(H, W)
    return out


def build_band_mask_like(
    x: torch.Tensor,
    mask_ratio: float,
    axis: str = "row",
) -> torch.Tensor:
    """Build stripe/band mask along rows or cols; 1 means masked."""
    B, _, H, W = x.shape
    out = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
    axis = str(axis).lower()
    if axis in ("row", "h"):
        n = max(1, int(H * _clamp_ratio(mask_ratio)))
        for b in range(B):
            idx = torch.randperm(H, device=x.device)[:n]
            out[b, 0, idx, :] = 1.0
    else:
        n = max(1, int(W * _clamp_ratio(mask_ratio)))
        for b in range(B):
            idx = torch.randperm(W, device=x.device)[:n]
            out[b, 0, :, idx] = 1.0
    return out


def build_structured_mask_like(
    x: torch.Tensor,
    mask_ratio: float,
    strategy: str = "random",
    band_axis: str = "row",
    mix_random: float = 0.5,
    mix_block: float = 0.3,
    mix_band: float = 0.2,
    block_h_min: int = 4,
    block_h_max: int = 16,
    block_w_min: int = 16,
    block_w_max: int = 64,
) -> torch.Tensor:
    """
    Build mask with strategy:
      - random
      - block
      - band
      - mixed (sample per-batch item from random/block/band)
    """
    st = str(strategy).lower()
    if st == "random":
        return build_random_mask_like(x, mask_ratio)
    if st == "block":
        return build_block_mask_like(
            x, mask_ratio,
            block_h_min=block_h_min, block_h_max=block_h_max,
            block_w_min=block_w_min, block_w_max=block_w_max,
        )
    if st == "band":
        return build_band_mask_like(x, mask_ratio, axis=band_axis)
    if st != "mixed":
        return build_random_mask_like(x, mask_ratio)

    probs = torch.tensor(
        [max(0.0, float(mix_random)), max(0.0, float(mix_block)), max(0.0, float(mix_band))],
        device=x.device,
        dtype=torch.float32,
    )
    if probs.sum() <= 0:
        probs = torch.tensor([1.0, 0.0, 0.0], device=x.device)
    probs = probs / probs.sum()

    B = x.shape[0]
    masks: List[torch.Tensor] = []
    choices = torch.multinomial(probs, num_samples=B, replacement=True)
    for b in range(B):
        one = x[b : b + 1]
        c = int(choices[b].item())
        if c == 0:
            m = build_random_mask_like(one, mask_ratio)
        elif c == 1:
            m = build_block_mask_like(
                one, mask_ratio,
                block_h_min=block_h_min, block_h_max=block_h_max,
                block_w_min=block_w_min, block_w_max=block_w_max,
            )
        else:
            m = build_band_mask_like(one, mask_ratio, axis=band_axis)
        masks.append(m)
    return torch.cat(masks, dim=0)


def build_mask_with_pos_ratio_control(
    x: torch.Tensor,
    target_occ: torch.Tensor,
    mask_ratio: float,
    strategy: str = "random",
    band_axis: str = "row",
    mix_random: float = 0.5,
    mix_block: float = 0.3,
    mix_band: float = 0.2,
    block_h_min: int = 4,
    block_h_max: int = 16,
    block_w_min: int = 16,
    block_w_max: int = 64,
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
    def _build(one_x: torch.Tensor) -> torch.Tensor:
        return build_structured_mask_like(
            one_x, mask_ratio,
            strategy=strategy,
            band_axis=band_axis,
            mix_random=mix_random,
            mix_block=mix_block,
            mix_band=mix_band,
            block_h_min=block_h_min,
            block_h_max=block_h_max,
            block_w_min=block_w_min,
            block_w_max=block_w_max,
        )

    if not enable_control:
        return _build(x), x.new_tensor(0.0)

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
            candidate = _build(one_x)
            masked_cnt = candidate.sum().item()
            if masked_cnt <= 0:
                continue
            pos_ratio = float((one_occ.float() * candidate).sum().item() / max(masked_cnt, 1.0))
            if min_pos_ratio <= pos_ratio <= max_pos_ratio:
                accepted = candidate
                resample_count += float(t)
                break
        if accepted is None:
            accepted = _build(one_x)
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
        occ_pos_weight_adaptive: bool = False,
        occ_pos_weight_min: float = 1.0,
        occ_pos_weight_max: float = 12.0,
    ):
        super().__init__()
        hidden = max(in_c // 2, 32)
        self.occ_scales = [int(max(1, s)) for s in occ_scales]
        self.occ_loss_type = str(occ_loss_type).lower()
        self.occ_pos_weight = float(max(1e-6, occ_pos_weight))
        self.occ_focal_gamma = float(max(0.0, occ_focal_gamma))
        self.pcp_stopgrad_replace = bool(pcp_stopgrad_replace)
        self.occ_pos_weight_adaptive = bool(occ_pos_weight_adaptive)
        self.occ_pos_weight_min = float(max(1e-6, occ_pos_weight_min))
        self.occ_pos_weight_max = float(max(self.occ_pos_weight_min, occ_pos_weight_max))

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
            pw_val = self.occ_pos_weight
            if self.occ_pos_weight_adaptive:
                s_mask = supervise_mask.expand_as(target_occ_ms)
                eff = (s_mask > 0.5)
                if eff.any():
                    pos = (target_occ_ms[eff] > 0.5).float().mean().item()
                    pos = max(1e-4, min(1.0 - 1e-4, float(pos)))
                    pw_val = (1.0 - pos) / pos
            pw_val = max(self.occ_pos_weight_min, min(self.occ_pos_weight_max, float(pw_val)))
            pw = pred_occ.new_tensor(pw_val)
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
        pcp_informative_only: bool = True,
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
        if pcp_informative_only:
            pcp_sup_mask = mask * (target_occ > 0.5).float()
            if pcp_sup_mask.sum() <= 0:
                pcp_sup_mask = mask
        else:
            pcp_sup_mask = mask
        denom = pcp_sup_mask.sum().clamp(min=1.0)
        loss_pcp = (center_loss_map * pcp_sup_mask).sum() / denom

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
