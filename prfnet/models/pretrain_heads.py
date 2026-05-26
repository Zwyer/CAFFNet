"""
Pretraining heads for NOMAE + PCP-MAE style self-supervised learning.

Changes vs previous version:
  - build_random_mask_like / build_band_mask_like: fully vectorized (no Python for loops)
  - build_hmg_mask_like: new Hierarchical Mask Generation (HMG), fully vectorized
  - build_structured_mask_like: supports "hmg" strategy + mix_hmg weight
  - build_mask_with_pos_ratio_control: passes HMG params through
  - NOMAEPCPPretrainHead._occ_loss: guard adaptive pos_weight EMA update to training mode only
  - NOMAEPCPPretrainHead.forward:
      * pcp_near_range_max now uses Euclidean distance (not raw X coordinate)
      * neighbor_sup_only_visible=True: true NOMAE supervision — only masked cells
        adjacent to VISIBLE (non-masked) occupied cells receive gradient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Tuple


def _clamp_ratio(x: float, lo: float = 0.0, hi: float = 0.95) -> float:
    return float(max(lo, min(float(x), hi)))


# ─────────────────────────────────────────────────────────────
# Vectorized mask builders
# ─────────────────────────────────────────────────────────────

def build_random_mask_like(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Per-sample random spatial mask; 1 means masked. Fully vectorized (no Python for-loop)."""
    B, _, H, W = x.shape
    total = H * W
    n_mask = max(1, int(total * _clamp_ratio(mask_ratio)))
    noise = torch.rand(B, total, device=x.device)
    ids = noise.argsort(dim=-1)
    mask = torch.zeros(B, total, device=x.device, dtype=x.dtype)
    mask.scatter_(1, ids[:, :n_mask], 1.0)
    return mask.reshape(B, 1, H, W)


def build_block_mask_like(
    x: torch.Tensor,
    mask_ratio: float,
    block_h_min: int = 4,
    block_h_max: int = 16,
    block_w_min: int = 16,
    block_w_max: int = 64,
) -> torch.Tensor:
    """Rectangular block mask; 1 means masked. Per-item loop (hard to fully vectorize)."""
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
    """Stripe/band mask along rows or cols; 1 means masked. Fully vectorized."""
    B, _, H, W = x.shape
    axis = str(axis).lower()
    if axis in ("row", "h"):
        n = max(1, int(H * _clamp_ratio(mask_ratio)))
        noise = torch.rand(B, H, device=x.device)
        idx = noise.argsort(dim=-1)[:, :n]          # (B, n)
        row_mask = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        row_mask.scatter_(1, idx, 1.0)               # (B, H)
        out = row_mask.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, W).contiguous()
    else:
        n = max(1, int(W * _clamp_ratio(mask_ratio)))
        noise = torch.rand(B, W, device=x.device)
        idx = noise.argsort(dim=-1)[:, :n]           # (B, n)
        col_mask = torch.zeros(B, W, device=x.device, dtype=x.dtype)
        col_mask.scatter_(1, idx, 1.0)               # (B, W)
        out = col_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, H, -1).contiguous()
    return out


def build_hmg_mask_like(
    x: torch.Tensor,
    mask_ratio: float,
    coarse_stride: int = 8,
    fine_extra_ratio: float = 0.05,
) -> torch.Tensor:
    """
    Hierarchical Mask Generation (HMG).

    1. Apply random masking at a coarse grid (spatial stride = coarse_stride).
       Each coarse cell maps to a coarse_stride×coarse_stride patch.
    2. Upsample coarse mask to full resolution (nearest → spatially coherent blocks).
    3. Optionally scatter a small fraction of extra fine-grained masking inside
       visible (non-masked) regions to break block boundary artifacts.

    Fully vectorized across batch — no Python for-loops.
    """
    B, _, H, W = x.shape
    cH = max(1, (H + coarse_stride - 1) // coarse_stride)
    cW = max(1, (W + coarse_stride - 1) // coarse_stride)
    n_coarse = cH * cW
    n_mask = max(1, int(n_coarse * _clamp_ratio(mask_ratio)))

    # Coarse-scale random masking (vectorized)
    noise = torch.rand(B, n_coarse, device=x.device)
    ids = noise.argsort(dim=-1)
    coarse_mask = torch.zeros(B, n_coarse, device=x.device, dtype=x.dtype)
    coarse_mask.scatter_(1, ids[:, :n_mask], 1.0)
    coarse_mask = coarse_mask.reshape(B, 1, cH, cW)

    # Upsample to original resolution (nearest → block-consistent)
    mask = F.interpolate(coarse_mask, size=(H, W), mode="nearest")

    # Fine-grained extra masking within visible cells (breaks block boundary predictability)
    if fine_extra_ratio > 0.0:
        extra = (torch.rand(B, 1, H, W, device=x.device) < float(fine_extra_ratio)).to(x.dtype)
        extra = extra * (1.0 - mask)
        mask = (mask + extra).clamp(0.0, 1.0)

    return mask


def build_structured_mask_like(
    x: torch.Tensor,
    mask_ratio: float,
    strategy: str = "random",
    band_axis: str = "row",
    mix_random: float = 0.5,
    mix_block: float = 0.3,
    mix_band: float = 0.2,
    mix_hmg: float = 0.0,
    block_h_min: int = 4,
    block_h_max: int = 16,
    block_w_min: int = 16,
    block_w_max: int = 64,
    hmg_coarse_stride: int = 8,
    hmg_fine_extra_ratio: float = 0.05,
) -> torch.Tensor:
    """
    Build mask with strategy:
      - random
      - block
      - band
      - hmg  (Hierarchical Mask Generation — fully vectorized)
      - mixed (sample per-batch item from random/block/band/hmg)
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
    if st == "hmg":
        return build_hmg_mask_like(
            x, mask_ratio,
            coarse_stride=hmg_coarse_stride,
            fine_extra_ratio=hmg_fine_extra_ratio,
        )
    if st != "mixed":
        return build_random_mask_like(x, mask_ratio)

    # Mixed: sample one strategy per batch item
    probs = torch.tensor(
        [
            max(0.0, float(mix_random)),
            max(0.0, float(mix_block)),
            max(0.0, float(mix_band)),
            max(0.0, float(mix_hmg)),
        ],
        device=x.device,
        dtype=torch.float32,
    )
    if probs.sum() <= 0:
        probs = torch.tensor([1.0, 0.0, 0.0, 0.0], device=x.device)
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
        elif c == 2:
            m = build_band_mask_like(one, mask_ratio, axis=band_axis)
        else:
            m = build_hmg_mask_like(
                one, mask_ratio,
                coarse_stride=hmg_coarse_stride,
                fine_extra_ratio=hmg_fine_extra_ratio,
            )
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
    mix_hmg: float = 0.0,
    block_h_min: int = 4,
    block_h_max: int = 16,
    block_w_min: int = 16,
    block_w_max: int = 64,
    hmg_coarse_stride: int = 8,
    hmg_fine_extra_ratio: float = 0.05,
    enable_control: bool = False,
    min_pos_ratio: float = 0.08,
    max_pos_ratio: float = 0.50,
    max_tries: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build mask with optional masked-positive-ratio control.
    Returns:
      mask: (B,1,H,W)
      resample_count: scalar tensor (avg resamples per item)
    """
    def _build(one_x: torch.Tensor) -> torch.Tensor:
        return build_structured_mask_like(
            one_x, mask_ratio,
            strategy=strategy,
            band_axis=band_axis,
            mix_random=mix_random,
            mix_block=mix_block,
            mix_band=mix_band,
            mix_hmg=mix_hmg,
            block_h_min=block_h_min,
            block_h_max=block_h_max,
            block_w_min=block_w_min,
            block_w_max=block_w_max,
            hmg_coarse_stride=hmg_coarse_stride,
            hmg_fine_extra_ratio=hmg_fine_extra_ratio,
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


# ─────────────────────────────────────────────────────────────
# Pretraining head
# ─────────────────────────────────────────────────────────────

class NOMAEPCPPretrainHead(nn.Module):
    """
    NOMAE + PCP style head with:
    - multi-scale occupancy modeling
    - PCP anti-leakage two-stage center prediction
    - true NOMAE neighborhood supervision (neighbor_sup_only_visible)
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
        occ_pos_weight_ema_decay: float = 0.95,
        pcp_pos_weight: float = 1.0,
        pcp_near_range_max: float = 10.0,  # Euclidean distance (m) threshold for near-range
        pcp_near_weight: float = 1.5,
        pcp_residual_center: bool = True,  # predict delta from visible-cell mean (removes position shortcut)
        pcp_far_only: bool = False,         # only supervise cells far from visible occupied cells
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
        self.occ_pos_weight_ema_decay = float(max(0.0, min(0.9999, occ_pos_weight_ema_decay)))
        self.pcp_pos_weight = float(max(1e-6, pcp_pos_weight))
        self.pcp_near_range_max = float(max(1e-6, pcp_near_range_max))
        self.pcp_near_weight = float(max(1.0, pcp_near_weight))
        self.pcp_residual_center = bool(pcp_residual_center)
        self.pcp_far_only = bool(pcp_far_only)
        self.register_buffer("_occ_pw_ema", torch.tensor(float(self.occ_pos_weight), dtype=torch.float32))

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
        Multi-scale neighborhood occupancy targets from original (unmasked) occupancy.
        target_occ: (B,1,H,W)
        returns: (B,S,H,W)
        """
        outs = []
        for k in self.occ_scales:
            if k <= 1:
                outs.append(target_occ)
            else:
                pad = k // 2
                outs.append(F.max_pool2d(target_occ, kernel_size=k, stride=1, padding=pad))
        return torch.cat(outs, dim=1)

    def _build_visible_neighborhood(
        self, visible_occ: torch.Tensor
    ) -> torch.Tensor:
        """
        Dilate VISIBLE occupied cells to find their multi-scale neighborhood.
        visible_occ: (B,1,H,W) — occupied AND not masked
        returns: (B,S,H,W) — 1 where within k/2 radius of any visible occupied cell
        """
        outs = []
        for k in self.occ_scales:
            if k <= 1:
                outs.append(visible_occ)
            else:
                pad = k // 2
                outs.append(F.max_pool2d(visible_occ, kernel_size=k, stride=1, padding=pad))
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
                pw_new = max(self.occ_pos_weight_min, min(self.occ_pos_weight_max, float(pw_val)))
                pw_ema = (
                    self.occ_pos_weight_ema_decay * float(self._occ_pw_ema.item())
                    + (1.0 - self.occ_pos_weight_ema_decay) * pw_new
                )
                # P1 fix: only update EMA during training — do NOT pollute eval runs
                if self.training:
                    self._occ_pw_ema.fill_(float(pw_ema))
                pw_val = float(self._occ_pw_ema.item())
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
        neighbor_sup_only_visible: bool = True,
        pcp_residual_center: bool = True,
        pcp_far_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
          feat:          (B,C,H,W)
          mask:          (B,1,H,W), 1 means masked
          target_occ:    (B,1,H,W), occupancy target in {0,1}
          target_center: (B,3,H,W), xyz target center (meters)
          neighbor_sup_only_visible: only supervise OCC in VISIBLE occupied cell neighborhoods.
          pcp_residual_center: predict delta from visible-cell mean XYZ (removes grid-pos shortcut).
          pcp_far_only: only supervise PCP for cells with no visible occupied neighbor (hard cases).
        """
        pred_occ = self.occ_head(feat)          # (B,S,H,W)
        occ_tgt_ms = self._build_occ_targets(target_occ)

        # Visible occupied cells — shared by OCC supervision and PCP tasks
        vis_occ = target_occ * (1.0 - mask)    # (B,1,H,W): occupied AND not masked

        # Compute visible neighborhood once; reused for OCC + PCP far-only filter
        need_vis_nbhd = (informative_only and neighbor_sup_only_visible) or pcp_far_only
        visible_nbhd_max = None
        if need_vis_nbhd:
            visible_nbhd = self._build_visible_neighborhood(vis_occ)        # (B,S,H,W)
            visible_nbhd_max = visible_nbhd.max(dim=1, keepdim=True).values  # (B,1,H,W)

        # ── Occupancy supervision mask ──────────────────────────────────────────
        if informative_only:
            if neighbor_sup_only_visible and visible_nbhd_max is not None:
                informative = (visible_nbhd_max > 0.0).float()
            else:
                informative = (occ_tgt_ms.max(dim=1, keepdim=True).values > 0.0).float()
            occ_sup_mask = mask * informative
        else:
            occ_sup_mask = mask
        if occ_sup_mask.sum() <= 0:
            occ_sup_mask = mask

        loss_occ = self._occ_loss(pred_occ, occ_tgt_ms, occ_sup_mask)

        # ── PCP: residual target (Method 1) ────────────────────────────────────
        # Remove "predict grid position" shortcut: predict delta from visible-cell mean.
        if pcp_residual_center:
            n_vis = vis_occ.sum(dim=[2, 3], keepdim=True).clamp(min=1.0)  # (B,1,1,1)
            mean_xyz = (target_center * vis_occ).sum(dim=[2, 3], keepdim=True) / n_vis  # (B,3,1,1)
            target_pcp = target_center - mean_xyz  # delta: much smaller variance, harder to fake
        else:
            target_pcp = target_center

        # ── PCP anti-leakage two-stage center prediction ────────────────────────
        pred_center_stage1 = self.center_head_stage1(feat)
        pred_token = pred_center_stage1.detach() if self.pcp_stopgrad_replace else pred_center_stage1
        # Stage 2 context: visible cells see true delta/center; masked cells see stage1 prediction
        replaced_center = torch.where(mask.expand_as(target_pcp) > 0.5, pred_token, target_pcp)
        pred_center = self.center_head_stage2(torch.cat([feat, replaced_center], dim=1))

        center_loss_map = F.smooth_l1_loss(pred_center, target_pcp, reduction="none").mean(
            dim=1, keepdim=True
        )

        # ── PCP supervision mask ───────────────────────────────────────────────
        if pcp_informative_only:
            pcp_sup_mask = mask * (target_occ > 0.5).float()
            if pcp_sup_mask.sum() <= 0:
                pcp_sup_mask = mask
        else:
            pcp_sup_mask = mask

        # Far-only filter (Method 3): remove cells adjacent to visible occupied cells.
        # Forces long-range reconstruction; adjacent cells can be trivially interpolated.
        if pcp_far_only and visible_nbhd_max is not None:
            far_from_visible = (visible_nbhd_max == 0.0).float()
            pcp_far_mask = pcp_sup_mask * far_from_visible
            if pcp_far_mask.sum() > 0:
                pcp_sup_mask = pcp_far_mask
            # else: fallback to standard mask (don't starve the loss)

        # P1 fix: near-range weight still uses absolute distance (not delta)
        range_dist = target_center.norm(dim=1, keepdim=True)   # sqrt(x²+y²+z²) in meters
        near = (range_dist <= self.pcp_near_range_max).float()

        pcp_weight = torch.ones_like(pcp_sup_mask)
        pcp_weight = pcp_weight + (target_occ > 0.5).float() * (self.pcp_pos_weight - 1.0)
        pcp_weight = pcp_weight + near * (self.pcp_near_weight - 1.0)
        eff_weight = pcp_sup_mask * pcp_weight

        denom = eff_weight.sum().clamp(min=1.0)
        loss_pcp = (center_loss_map * eff_weight).sum() / denom

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
