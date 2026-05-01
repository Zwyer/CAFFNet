"""
PRFNet: Polar-Range Fusion Network
主模型，整合双分支编码器 + AAFF × 4 + 双分支解码器 + 逐点分类头。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

from .modules import (
    LightEncoder, LightDecoder, AAFF, DepthStratifiedAAFF,
    PointSampleAggregator, AuxHead, conv_bn_relu6,
)
from .pretrain_heads import (
    NOMAEPCPPretrainHead,
    build_mask_with_pos_ratio_control,
)


class PRFNet(nn.Module):
    """
    Polar-Range Fusion Network for LiDAR Point Cloud Semantic Segmentation.

    两个视图（Range Image + Polar BEV）共享方位角维度 W，
    通过 AAFF（方位角对齐特征融合）在解码器的 4 个尺度上双向增强。

    Args:
        rv_in:         Range Image 输入通道数 (default: 6)
        pb_in:         Polar BEV  输入通道数 (default: 9)
        num_classes:   语义类别数 (SemanticKITTI: 19)
        enc_channels:  各阶段特征通道 [S1,S2,S3,S4]
        dec_out_c:     解码器输出通道（送入分类头）
        rv_strides:    RV 编码器各阶段步长
        pb_strides:    PB 编码器各阶段步长
        expand_ratios: 各阶段 expand ratio，早期大分辨率用小值
    """

    def __init__(
        self,
        rv_in: int = 6,
        pb_in: int = 9,
        num_classes: int = 19,
        enc_channels: List[int] = (64, 128, 256, 256),
        dec_out_c: int = 64,
        rv_H: int = 64,
        pb_H: int = 480,
        rv_strides: List[Tuple] = ((1,2),(2,2),(2,2),(2,2)),
        pb_strides: List[Tuple] = ((2,2),(2,2),(2,2),(2,2)),
        expand_ratios: List[int] = (2, 2, 4, 4),
        aspp_rates: List[int] = (1, 3, 6, 9),
        use_ds_aaff: bool = True,
        ds_aaff_K: int = 4,
        head_dropout: float = 0.1,
        use_vcg: bool = True,    # 创新④ 视图置信度门控
        use_proto: bool = True,  # 创新⑤ 原型记忆辅助分类
        proto_dim: int = 64,     # 创新⑤ 原型向量维度
    ):
        super().__init__()
        self.num_classes = num_classes
        self.rv_in = rv_in   # 保存供 export_onnx 使用，避免硬编码通道数
        self.pb_in = pb_in

        # ── 双分支 stem（输出同时作为 decoder 最终跳连接）─────
        # LightEncoder 无内部 stem，避免双重 stem
        self.rv_stem = conv_bn_relu6(rv_in,  enc_channels[0], 3, 1, 1)
        self.pb_stem = conv_bn_relu6(pb_in,  enc_channels[0], 3, 1, 1)

        # stem 输出通道 = enc_channels[0]，作为编码器第一阶段的输入
        self.rv_enc = LightEncoder(enc_channels[0], enc_channels, rv_strides, expand_ratios, aspp_rates)
        self.pb_enc = LightEncoder(enc_channels[0], enc_channels, pb_strides, expand_ratios, aspp_rates)

        # ── 【创新点】DS-AAFF 或原始 AAFF（由 use_ds_aaff 控制）──
        # 推导各编码器阶段输出的 H 尺寸（用于 DS-AAFF 距离带预计算）
        if use_ds_aaff:
            enc_rv_Hs: List[int] = []
            enc_pb_Hs: List[int] = []
            h_rv, h_pb = rv_H, pb_H
            for s_rv, s_pb in zip(rv_strides, pb_strides):
                h_rv = h_rv // s_rv[0]
                h_pb = h_pb // s_pb[0]
                enc_rv_Hs.append(h_rv)
                enc_pb_Hs.append(h_pb)
            self.aaffs = nn.ModuleList([
                DepthStratifiedAAFF(enc_channels[i], enc_rv_Hs[i], enc_pb_Hs[i],
                                    K=ds_aaff_K)
                for i in range(4)
            ])
        else:
            self.aaffs = nn.ModuleList([
                AAFF(enc_channels[i]) for i in range(4)
            ])

        # ── 双分支解码器 ──────────────────────────────────────
        self.rv_dec = LightDecoder(list(enc_channels), dec_out_c)
        self.pb_dec = LightDecoder(list(enc_channels), dec_out_c)

        # ── 逐点聚合 + 分类头 ─────────────────────────────────
        self.aggregator = PointSampleAggregator(
            rv_c=dec_out_c, pb_c=dec_out_c,
            num_classes=num_classes, pt_dim=4,
            head_dropout=head_dropout,
            use_vcg=use_vcg, use_proto=use_proto, proto_dim=proto_dim,
        )

        # ── 辅助预测头（训练时使用，仅 RV 分支——可直接得到像素级 GT）
        self.rv_aux = AuxHead(dec_out_c, num_classes)
        # pb_aux 已移除：Polar BEV 无直接像素→点映射，难以生成像素级 GT

        # ── 预训练头（NOMAE + PCP，训练期可选启用）──────────────
        self.pretrain_head_rv = NOMAEPCPPretrainHead(dec_out_c)
        self.pretrain_head_pb = NOMAEPCPPretrainHead(dec_out_c)
        self.rv_mask_token = nn.Parameter(torch.zeros(1, rv_in, 1, 1))
        self.pb_mask_token = nn.Parameter(torch.zeros(1, pb_in, 1, 1))

    # ─────────────────────────────────────────────────────────

    def _encode(self, rv_img: torch.Tensor, pb_img: torch.Tensor):
        """
        stem → 编码器 → AAFF 融合。
        stem 输出同时保留，供解码器最终阶段跳连接使用。
        """
        # Stem（保留全分辨率特征作为解码器最终跳连接）
        rv_stem = self.rv_stem(rv_img)   # (B, C0, H_rv, W)
        pb_stem = self.pb_stem(pb_img)   # (B, C0, H_pb, W)

        # 4 阶段编码（LightEncoder 无内部 stem，直接从 stem 输出开始）
        rv_feats = self.rv_enc(rv_stem)  # [S1,S2,S3,S4]
        pb_feats = self.pb_enc(pb_stem)  # [S1,S2,S3,S4]

        # 在每个尺度执行 AAFF
        fused_rv, fused_pb = [], []
        for i in range(4):
            # 两个分支在同一尺度的方位角维度 W 需对齐
            # rv 和 pb 步长不同，W 维度已对齐（同为 W→W/2→W/4→W/8）
            # H 维度不同（rv:64→8, pb:480→30），AAFF 会压缩掉 H 维度
            f_rv, f_pb = self.aaffs[i](rv_feats[i], pb_feats[i])
            fused_rv.append(f_rv)
            fused_pb.append(f_pb)

        return fused_rv, fused_pb, rv_stem, pb_stem

    def forward(
        self,
        rv_img:    torch.Tensor,      # (B, 6, 64,  W)
        pb_img:    torch.Tensor,      # (B, 9, 480, W)
        rv_coords: torch.Tensor,      # (B, N, 1, 2) grid_sample coords
        pb_coords: torch.Tensor,      # (B, N, 1, 2)
        points:    torch.Tensor,      # (B, N, 4)  x,y,z,intensity
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
              'logits':  (B, N, C) — 逐点语义 logits（主输出）
              'rv_aux':  (B, C, H_rv, W) — RV 辅助预测（仅训练时）
        """
        # 编码
        fused_rv, fused_pb, rv_stem, pb_stem = self._encode(rv_img, pb_img)

        # 解码
        rv_out = self.rv_dec(fused_rv, rv_stem)   # (B, dec_out_c, H_rv, W)
        pb_out = self.pb_dec(fused_pb, pb_stem)   # (B, dec_out_c, H_pb, W)

        # 逐点分类（主预测）
        logits = self.aggregator(rv_out, pb_out, rv_coords, pb_coords, points)

        result = {'logits': logits}

        # 辅助预测（只在训练时计算，避免推理开销）
        if self.training:
            result['rv_aux'] = self.rv_aux(rv_out)  # (B, num_classes, H_rv, W)

        return result

    def forward_pretrain(
        self,
        rv_img: torch.Tensor,
        pb_img: torch.Tensor,
        rv_mask_ratio: float = 0.7,
        pb_mask_ratio: float = 0.7,
        input_masking_enable: bool = True,
        input_masking_mode: str = "zero",
        mask_pos_ratio_control_enable: bool = False,
        mask_pos_ratio_min: float = 0.08,
        mask_pos_ratio_max: float = 0.50,
        mask_resample_max_tries: int = 5,
        occ_scales: Optional[List[int]] = None,
        occ_loss_type: str = "bce_pos_weight",
        occ_pos_weight: float = 5.0,
        occ_focal_gamma: float = 2.0,
        pcp_stopgrad_replace: bool = True,
        informative_occ_only: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Self-supervised pretraining forward.
        - NOMAE: occupancy reconstruction on masked cells.
        - PCP: masked center regression.

        Returns dict with losses and diagnostics.
        """
        # Targets from original inputs.
        rv_occ_tgt = (rv_img[:, 3:4] > 0).float()
        pb_occ_tgt = pb_img[:, 8:9].float()
        rv_center_tgt = rv_img[:, 0:3]
        pb_center_tgt = pb_img[:, 0:3]

        # Build masks (optionally occupancy-aware).
        rv_mask, rv_resample = build_mask_with_pos_ratio_control(
            rv_img,
            rv_occ_tgt,
            rv_mask_ratio,
            enable_control=mask_pos_ratio_control_enable,
            min_pos_ratio=mask_pos_ratio_min,
            max_pos_ratio=mask_pos_ratio_max,
            max_tries=mask_resample_max_tries,
        )
        pb_mask, pb_resample = build_mask_with_pos_ratio_control(
            pb_img,
            pb_occ_tgt,
            pb_mask_ratio,
            enable_control=mask_pos_ratio_control_enable,
            min_pos_ratio=mask_pos_ratio_min,
            max_pos_ratio=mask_pos_ratio_max,
            max_tries=mask_resample_max_tries,
        )

        # Input-level masking to block leakage shortcuts.
        if input_masking_enable:
            mode = str(input_masking_mode).lower()
            if mode == "zero":
                rv_img_in = rv_img * (1.0 - rv_mask)
                pb_img_in = pb_img * (1.0 - pb_mask)
            elif mode == "token":
                rv_tok = self.rv_mask_token.expand(rv_img.shape[0], -1, rv_img.shape[2], rv_img.shape[3])
                pb_tok = self.pb_mask_token.expand(pb_img.shape[0], -1, pb_img.shape[2], pb_img.shape[3])
                rv_img_in = rv_img * (1.0 - rv_mask) + rv_tok * rv_mask
                pb_img_in = pb_img * (1.0 - pb_mask) + pb_tok * pb_mask
            else:
                # Fallback to zero-mask for unsupported modes.
                rv_img_in = rv_img * (1.0 - rv_mask)
                pb_img_in = pb_img * (1.0 - pb_mask)
        else:
            rv_img_in = rv_img
            pb_img_in = pb_img

        fused_rv, fused_pb, rv_stem, pb_stem = self._encode(rv_img_in, pb_img_in)
        rv_out = self.rv_dec(fused_rv, rv_stem)   # (B, dec_out_c, H_rv, W)
        pb_out = self.pb_dec(fused_pb, pb_stem)   # (B, dec_out_c, H_pb, W)

        # Sync pretrain head options dynamically from config.
        if occ_scales is not None and len(occ_scales) == len(self.pretrain_head_rv.occ_scales):
            self.pretrain_head_rv.occ_scales = [int(max(1, s)) for s in occ_scales]
            self.pretrain_head_pb.occ_scales = [int(max(1, s)) for s in occ_scales]
        self.pretrain_head_rv.occ_loss_type = str(occ_loss_type).lower()
        self.pretrain_head_pb.occ_loss_type = str(occ_loss_type).lower()
        self.pretrain_head_rv.occ_pos_weight = float(max(1e-6, occ_pos_weight))
        self.pretrain_head_pb.occ_pos_weight = float(max(1e-6, occ_pos_weight))
        self.pretrain_head_rv.occ_focal_gamma = float(max(0.0, occ_focal_gamma))
        self.pretrain_head_pb.occ_focal_gamma = float(max(0.0, occ_focal_gamma))
        self.pretrain_head_rv.pcp_stopgrad_replace = bool(pcp_stopgrad_replace)
        self.pretrain_head_pb.pcp_stopgrad_replace = bool(pcp_stopgrad_replace)

        rv_ret = self.pretrain_head_rv(
            rv_out, rv_mask, rv_occ_tgt, rv_center_tgt,
            informative_only=informative_occ_only,
        )
        pb_ret = self.pretrain_head_pb(
            pb_out, pb_mask, pb_occ_tgt, pb_center_tgt,
            informative_only=informative_occ_only,
        )

        return {
            "loss_occ_rv": rv_ret["loss_occ"],
            "loss_occ_pb": pb_ret["loss_occ"],
            "loss_pcp_rv": rv_ret["loss_pcp"],
            "loss_pcp_pb": pb_ret["loss_pcp"],
            "rv_mask_ratio": rv_mask.mean().detach(),
            "pb_mask_ratio": pb_mask.mean().detach(),
            "rv_masked_pos_ratio": rv_ret["masked_pos_ratio"],
            "pb_masked_pos_ratio": pb_ret["masked_pos_ratio"],
            "rv_occ_effective_ratio": rv_ret["occ_effective_ratio"],
            "pb_occ_effective_ratio": pb_ret["occ_effective_ratio"],
            "rv_mask_resample": rv_resample.detach(),
            "pb_mask_resample": pb_resample.detach(),
        }

    # ─────────────────────────────────────────────────────────
    # RKNN 导出辅助
    # ─────────────────────────────────────────────────────────

    def export_forward(
        self,
        rv_img:    torch.Tensor,
        pb_img:    torch.Tensor,
        rv_coords: torch.Tensor,
        pb_coords: torch.Tensor,
        points:    torch.Tensor,
    ) -> torch.Tensor:
        """
        纯推理前向（去除辅助头），用于 ONNX 导出。
        所有输入形状须为静态。
        """
        self.eval()
        with torch.no_grad():
            fused_rv, fused_pb, rv_stem, pb_stem = self._encode(rv_img, pb_img)
            rv_out = self.rv_dec(fused_rv, rv_stem)
            pb_out = self.pb_dec(fused_pb, pb_stem)
            logits = self.aggregator(rv_out, pb_out, rv_coords, pb_coords, points)
        return logits  # (B, N, num_classes)


# ─────────────────────────────────────────────────────────────
# ONNX 导出工具
# ─────────────────────────────────────────────────────────────

def export_onnx(model: PRFNet,
                save_path: str,
                N_points: int = 131072,
                rv_H=64, rv_W=1024,
                pb_H=480, pb_W=1024,
                opset: int = 13):
    """
    将 PRFNet 导出为 ONNX 静态图（RK3588 RKNN 部署用）。
    N_points: 固定点数（不足则 pad，超出则截断）。
    """
    import torch

    model.eval()
    # 替换 forward 为 export_forward（去除辅助头）
    orig_forward = model.forward
    model.forward = model.export_forward

    dummy_rv  = torch.zeros(1, model.rv_in, rv_H, rv_W)
    dummy_pb  = torch.zeros(1, 9,  pb_H, pb_W)
    dummy_rv_c = torch.zeros(1, N_points, 1, 2)
    dummy_pb_c = torch.zeros(1, N_points, 1, 2)
    dummy_pts  = torch.zeros(1, N_points, 4)

    torch.onnx.export(
        model,
        (dummy_rv, dummy_pb, dummy_rv_c, dummy_pb_c, dummy_pts),
        save_path,
        opset_version=opset,
        input_names=['rv_img', 'pb_img', 'rv_coords', 'pb_coords', 'points'],
        output_names=['logits'],
        dynamic_axes=None,   # 静态形状，RKNN 要求
        do_constant_folding=True,
    )
    model.forward = orig_forward
    print(f"ONNX saved to {save_path}  (N={N_points}, RV={rv_H}×{rv_W}, PB={pb_H}×{pb_W})")
