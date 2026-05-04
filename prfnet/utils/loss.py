"""
损失函数：加权交叉熵 + Lovász-Softmax + 辅助损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Lovász-Softmax Loss (多类别 mIoU 代理)
# ─────────────────────────────────────────────────────────────

def lovász_grad(gt_sorted):
    """计算 Lovász 扩展的梯度"""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    """
    Lovász-Softmax Loss，直接优化 mIoU。
    忽略 ignore_index 类别。
    """

    def __init__(self, ignore_index: int = 255, num_classes: int = 19):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, N, C) or (B, C, H, W)
        labels: (B, N)    or (B, H, W)  — int64
        """
        # 统一变形为 (M, C) 和 (M,)
        if logits.dim() == 4:
            B, C, H, W = logits.shape
            logits = logits.permute(0,2,3,1).reshape(-1, C)
            labels = labels.reshape(-1)
        else:
            B, N, C = logits.shape
            logits = logits.reshape(-1, C)
            labels = labels.reshape(-1)

        # 过滤 ignore
        valid = labels != self.ignore_index
        logits = logits[valid]
        labels = labels[valid]

        if logits.numel() == 0:
            return logits.sum() * 0.

        probs = F.softmax(logits, dim=-1)  # (M, C)
        losses = []

        for c in range(self.num_classes):
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            errors = (fg - probs[:, c]).abs()
            errors_sorted, perm = errors.sort(descending=True)
            fg_sorted = fg[perm]
            losses.append((errors_sorted * lovász_grad(fg_sorted)).sum())

        if len(losses) == 0:
            return logits.sum() * 0.
        return torch.stack(losses).mean()


# ─────────────────────────────────────────────────────────────
# Focal Loss（稀有类增强）
# ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss — 对难分样本加权，自动抑制易分类样本的梯度贡献。

    公式：FL(pt) = (1 - pt)^γ · CE(x, y)
      其中 pt = exp(-CE)，是模型预测正确类的概率估计。
      γ 越大，对难样本的聚焦越强（推荐 γ=2）。

    类别频率不均衡由外部 class_weights（α）负责，
    难易程度不均衡由 γ 负责，两者正交互补。

    注意：Focal Loss 不支持 label_smoothing，
          启用时请将 label_smoothing 设为 0。

    Args:
        gamma:        聚焦系数，通常取 1.5~3.0，推荐 2.0
        weight:       (C,) 类别权重（同 nn.CrossEntropyLoss 的 weight）
        ignore_index: 忽略的标签值
    """

    def __init__(self,
                 gamma: float = 2.0,
                 weight: Optional[torch.Tensor] = None,
                 ignore_index: int = 255):
        super().__init__()
        self.gamma        = gamma
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, C, N) 逐点  或  (B, C, H, W) 像素级
        targets: (B, N)          或  (B, H, W)        int64
        """
        # reduction='none' → 保留每个位置的 loss 值
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none',
        )
        # ignore_index 位置：F.cross_entropy 返回 0，exp(0)=1 → focal 权重=(1-1)^γ=0，天然屏蔽
        pt   = torch.exp(-ce)                    # 预测正确概率的代理估计
        loss = (1.0 - pt) ** self.gamma * ce     # 难样本权重高，易样本权重低

        # 只对非 ignore 位置求均值（避免 ignore 位置的 0 稀释梯度）
        valid = targets != self.ignore_index
        return loss[valid].mean() if valid.any() else loss.sum() * 0.


# ─────────────────────────────────────────────────────────────
# 综合损失
# ─────────────────────────────────────────────────────────────

class PRFNetLoss(nn.Module):
    """
    L_total = L_ce + λ_lovász · L_lovász + λ_aux · L_aux_rv

    其中 L_ce 可选 CrossEntropyLoss（含 label_smoothing）或 FocalLoss，
    由 use_focal 开关控制。

    Args:
        class_weights:   (C,) float 各类别损失权重
        ignore_index:    忽略标签（SemanticKITTI 用 255）
        lambda_aux:      辅助损失权重
        lambda_lovász:   Lovász 损失权重
        label_smoothing: 标签平滑系数，仅 use_focal=False 时生效
        use_focal:       True → 使用 FocalLoss；False → 使用 CrossEntropyLoss
        focal_gamma:     Focal Loss 的聚焦系数 γ，仅 use_focal=True 时生效
    """

    def __init__(self,
                 class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = 255,
                 num_classes: int = 19,
                 lambda_aux: float = 0.3,
                 lambda_lovász: float = 1.0,
                 label_smoothing: float = 0.0,
                 use_focal: bool = False,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.lambda_aux    = lambda_aux
        self.lambda_lovász = lambda_lovász

        if use_focal:
            # Focal Loss：自动聚焦难样本，对 motorcyclist/other-ground 等稀有类更友好
            # label_smoothing 与 focal 机制冲突（平滑会削弱 pt 信号），此处不使用
            self.ce = FocalLoss(
                gamma=focal_gamma,
                weight=class_weights,
                ignore_index=ignore_index,
            )
        else:
            # 标准加权交叉熵，label_smoothing 缓解标注噪声
            self.ce = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=ignore_index,
                reduction='mean',
                label_smoothing=label_smoothing,
            )
        self.lovász = LovaszSoftmax(ignore_index, num_classes)

    def forward(self,
                outputs: dict,
                labels: torch.Tensor,                          # (B, N) 逐点真实标签
                rv_labels: Optional[torch.Tensor] = None,      # (B, H, W) RV 像素级标签
                ) -> dict:
        """
        Returns dict: {'total': Tensor, 'ce': Tensor, 'lovász': Tensor, 'aux': Tensor}
        """
        logits = outputs['logits']   # (B, N, C)
        B, N, C = logits.shape

        # 主损失：CE (期望输入 (B,C,N))
        l_ce  = self.ce(logits.permute(0, 2, 1), labels)
        l_lov = self.lovász(logits, labels)

        l_aux = torch.tensor(0., device=logits.device)

        # RV 辅助损失（训练时才有 rv_aux，且需要像素级标签）
        if 'rv_aux' in outputs and rv_labels is not None:
            rv_pred = outputs['rv_aux']           # (B, C, H, W)
            l_aux   = self.ce(rv_pred, rv_labels) # rv_labels: (B, H, W) int64

        total = l_ce + self.lambda_lovász * l_lov + self.lambda_aux * l_aux

        return {
            'total':  total,
            'ce':     l_ce.detach(),
            'lovász': l_lov.detach(),
            'aux':    l_aux.detach(),
        }


class NOMAEPCPLoss(nn.Module):
    """
    Aggregate loss for NOMAE + PCP pretraining.

    L = lambda_occ_rv * L_occ_rv + lambda_occ_pb * L_occ_pb +
        lambda_pcp_rv * L_pcp_rv + lambda_pcp_pb * L_pcp_pb
    """

    def __init__(
        self,
        lambda_occ: float = 1.0,
        lambda_pcp: float = 0.5,
        lambda_occ_rv: float | None = None,
        lambda_occ_pb: float | None = None,
        lambda_pcp_rv: float | None = None,
        lambda_pcp_pb: float | None = None,
    ):
        super().__init__()
        occ = float(lambda_occ)
        pcp = float(lambda_pcp)
        self.lambda_occ_rv = float(occ if lambda_occ_rv is None else lambda_occ_rv)
        self.lambda_occ_pb = float(occ if lambda_occ_pb is None else lambda_occ_pb)
        self.lambda_pcp_rv = float(pcp if lambda_pcp_rv is None else lambda_pcp_rv)
        self.lambda_pcp_pb = float(pcp if lambda_pcp_pb is None else lambda_pcp_pb)

    def forward(self, outputs: dict) -> dict:
        l_occ_rv = outputs["loss_occ_rv"]
        l_occ_pb = outputs["loss_occ_pb"]
        l_pcp_rv = outputs["loss_pcp_rv"]
        l_pcp_pb = outputs["loss_pcp_pb"]
        l_occ = l_occ_rv + l_occ_pb
        l_pcp = l_pcp_rv + l_pcp_pb
        total = (
            self.lambda_occ_rv * l_occ_rv
            + self.lambda_occ_pb * l_occ_pb
            + self.lambda_pcp_rv * l_pcp_rv
            + self.lambda_pcp_pb * l_pcp_pb
        )
        return {
            "total": total,
            "occ": l_occ.detach(),
            "pcp": l_pcp.detach(),
            "occ_rv": l_occ_rv.detach(),
            "occ_pb": l_occ_pb.detach(),
            "pcp_rv": l_pcp_rv.detach(),
            "pcp_pb": l_pcp_pb.detach(),
        }
