"""
PRFNet 模型核心模块：
  - InvertedResidual: MobileNetV2 式可分离卷积块
  - ASPP: 轻量空洞卷积池化金字塔
  - LightEncoder: 双视图轻量编码器
  - AAFF: 方位角对齐特征融合（RK3588 友好）
  - LightDecoder: 上采样解码器
  - PointHead: 逐点分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────
# 基础积木
# ─────────────────────────────────────────────────────────────

def conv_bn_relu6(in_c, out_c, k=3, s=1, p=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    """
    MobileNetV2 Inverted Residual Block。
    expand_ratio=1 时退化为 DWConv+PW，适合浅层。
    RK3588 NPU 原生支持 DWConv。
    """

    def __init__(self, in_c, out_c, stride=(1, 1), expand_ratio=4):
        super().__init__()
        hidden = in_c * expand_ratio
        # 统一转 tuple：兼容 int（来自代码直接传参）和 list（来自 YAML 读取）
        stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.use_res = (stride == (1, 1)) and (in_c == out_c)

        layers = []
        if expand_ratio != 1:
            layers.append(conv_bn_relu6(in_c, hidden, 1, 1, 0))
        layers += [
            conv_bn_relu6(hidden, hidden, 3, stride, 1, groups=hidden),  # DWConv
            nn.Conv2d(hidden, out_c, 1, 1, 0, bias=False),               # PW (线性)
            nn.BatchNorm2d(out_c),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return out + x if self.use_res else out


class ASPP(nn.Module):
    """
    轻量 ASPP：4 个不同空洞率的 DWConv + 1×1 融合。
    用 DWConv 代替普通空洞卷积，减少约 9× 参数量。
    不含 GlobalAvgPool（RKNN 静态图不友好）。
    """

    def __init__(self, in_c, out_c, rates=(1, 3, 6, 9)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, 1, r, dilation=r, groups=in_c, bias=False),
                nn.Conv2d(in_c, out_c // len(rates), 1, bias=False),
                nn.BatchNorm2d(out_c // len(rates)),
                nn.ReLU6(inplace=True),
            )
            for r in rates
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        return self.fuse(torch.cat(feats, dim=1))


# ─────────────────────────────────────────────────────────────
# 轻量编码器
# ─────────────────────────────────────────────────────────────

class LightEncoder(nn.Module):
    """
    4 阶段 MobileNetV2 风格编码器，输出 4 个尺度的特征图。
    不含 stem——由外部（PRFNet）统一处理 stem，避免双重 stem。

    expand_ratios: 每个阶段的 InvertedResidual expand 比例。
      大空间分辨率的早期阶段用小 ratio（降显存），
      小空间分辨率的后期阶段用大 ratio（增容量）。
      默认 [1, 1, 4, 4]。
    """

    def __init__(self, in_channels: int,
                 channels: List[int] = (64, 128, 256, 256),
                 strides: List[Tuple] = ((1,2),(2,2),(2,2),(2,2)),
                 expand_ratios: List[int] = (1, 1, 4, 4),
                 aspp_rates: Tuple = (1, 3, 6, 9)):
        super().__init__()
        # 4 个编码阶段（无 stem，输入已由 PRFNet.stem 处理）
        self.stages = nn.ModuleList()
        in_c = in_channels
        for out_c, s, er in zip(channels, strides, expand_ratios):
            self.stages.append(nn.Sequential(
                InvertedResidual(in_c, out_c, stride=s,      expand_ratio=er),
                InvertedResidual(out_c, out_c, stride=(1,1), expand_ratio=er),
                InvertedResidual(out_c, out_c, stride=(1,1), expand_ratio=er),
            ))
            in_c = out_c

        self.aspp = ASPP(channels[-1], channels[-1], rates=aspp_rates)

    def forward(self, x):
        """x: stem 输出特征（已由外部 stem 处理）。Returns [S1, S2, S3, S4(+ASPP)]"""
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        feats[-1] = self.aspp(feats[-1])
        return feats  # [S1,S2,S3,S4]


# ─────────────────────────────────────────────────────────────
# AAFF: 方位角对齐特征融合
# ─────────────────────────────────────────────────────────────

class AAFF(nn.Module):
    """
    Azimuth-Aligned Feature Fusion (AAFF)

    利用 Range Image 与 Polar BEV 共享的方位角维度 (W 轴)，
    通过压缩非共享维度 → Conv1d 门控，实现跨视图特征增强。

    所有算子均为 RK3588 RKNN NPU 原生支持：
        AvgPool2d、MaxPool2d、Concat、Conv2d(kernel=(1,k))、Sigmoid、Mul、Add

    Args:
        channels: 两个分支的公共通道数 C
    """

    def __init__(self, channels: int, reduced: int = None):
        super().__init__()
        reduced = reduced or channels // 2
        # 聚合两视图的方位角描述（concat 后为 4C）
        self.aggregate = nn.Sequential(
            # 用 kernel=(1,3) 的 Conv2d 实现 Conv1d（W 方向）
            nn.Conv2d(channels * 4, reduced, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU6(inplace=True),
        )
        # 分别生成 RV 和 PB 的方位角门控
        self.gate_rv = nn.Sequential(
            nn.Conv2d(reduced, channels, kernel_size=(1, 1), bias=True),
            nn.Sigmoid(),
        )
        self.gate_pb = nn.Sequential(
            nn.Conv2d(reduced, channels, kernel_size=(1, 1), bias=True),
            nn.Sigmoid(),
        )

    def forward(self,
                f_rv: torch.Tensor,
                f_pb: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_rv: (B, C, H_r, W)  — Range Image 特征
            f_pb: (B, C, H_p, W)  — Polar BEV 特征
        Returns:
            f_rv_out: (B, C, H_r, W)
            f_pb_out: (B, C, H_p, W)
        """
        # Step 1: 沿各自高度维度做 AvgPool + MaxPool，压缩到 (B, C, 1, W)
        rv_avg = f_rv.mean(dim=2, keepdim=True)   # (B,C,1,W)
        rv_max = f_rv.max(dim=2, keepdim=True)[0]
        pb_avg = f_pb.mean(dim=2, keepdim=True)
        pb_max = f_pb.max(dim=2, keepdim=True)[0]

        # Step 2: 拼接 → (B, 4C, 1, W)
        ctx = torch.cat([rv_avg, rv_max, pb_avg, pb_max], dim=1)

        # Step 3: Conv1d(沿 W) 提取方位角上下文 → (B, reduced, 1, W)
        ctx = self.aggregate(ctx)

        # Step 4: 分别生成门控 (B, C, 1, W)
        gate_rv = self.gate_rv(ctx)
        gate_pb = self.gate_pb(ctx)

        # Step 5: 广播乘以原特征 + 残差
        f_rv_out = f_rv * gate_rv + f_rv
        f_pb_out = f_pb * gate_pb + f_pb

        return f_rv_out, f_pb_out


# ─────────────────────────────────────────────────────────────
# 【创新点】DS-AAFF：深度分层方位角对齐融合
# ─────────────────────────────────────────────────────────────

class DepthStratifiedAAFF(nn.Module):
    """
    【创新点】DS-AAFF（Depth-Stratified Azimuth-Aligned Feature Fusion）
    深度分层方位角对齐融合

    在原始 AAFF 的两处根本缺陷上做了改进：

    缺陷 1 — 全局 H 压缩导致远近混淆：
      原 AAFF 把整个 H 维度 AvgPool 为一个全局向量，
      使 3m 近处行人与 50m 远处路面的特征被等权混合参与门控。
      → 本模块按近→远顺序将 H 分成 K 个距离带分别池化，
        band[k]_rv 与 band[k]_pb 对应同一 3D 距离区间，做带内对齐。

    缺陷 2 — 无跨视图偏差感知（本次增强核心）：
      原 DS-AAFF 仅用每带 AvgPool 做门控上下文，
      网络看不到两视图在同一距离带的特征是否一致。
      → 新增差值信号 diff[k] = rv_avg[k] − pb_avg[k]，
        直接量化两视图的带级特征偏差；差值大的方位角列
        （通常是小目标边界、遮挡区域）门控响应自动增强，
        引导网络在特征不一致处加大对齐力度。

    同时将每带的统计量从单一 AvgPool 扩展为 Avg+Max，
    与原始 AAFF 的全局 avg+max 对应，捕捉带内特征的均值与峰值。

    聚合输入：cat([rv_avg(CK), rv_max(CK), pb_avg(CK), pb_max(CK), diff(CK)])
    = 5CK 通道（原始 DS-AAFF 的 2.5 倍），携带更完整的几何对齐信息。

    几何约定（K=4，HDL-64E，等行数分割）：
      RV 近→远：行号从高到低（图像底部→顶部）
        band[0]=近 rows[H_r-bs:H_r], band[K-1]=远 rows[0:bs]
      PB 近→远：行号从低到高（图像顶部→底部）
        band[0]=近 rows[0:bs],        band[K-1]=远 rows[H_p-bs:H_p]

    RKNN 兼容：所有切片索引在 __init__ 时计算为 Python int，
    ONNX 导出时由 constant_folding 消除，无动态 shape。
    """

    def __init__(self, channels: int, H_rv: int, H_pb: int, K: int = 4):
        super().__init__()
        self.K = K
        self.C = channels

        # ── 预计算距离带边界（Python int，RKNN 常量折叠）──────────
        # RV：近 = 高行号（图底），排列为 [近, 中近, 中远, 远]
        rv_bs = H_rv // K
        self.rv_bands: List[Tuple[int, int]] = []
        for k in range(K):
            end   = H_rv - k * rv_bs
            start = H_rv - (k + 1) * rv_bs if k < K - 1 else 0
            self.rv_bands.append((start, end))

        # PB：近 = 低行号（图顶），排列为 [近, 中近, 中远, 远]
        pb_bs = H_pb // K
        self.pb_bands: List[Tuple[int, int]] = []
        for k in range(K):
            start = k * pb_bs
            end   = (k + 1) * pb_bs if k < K - 1 else H_pb
            self.pb_bands.append((start, end))

        # 重建顺序：将 near→far 列表按原始行号升序排回，用于 cat 恢复 H
        self._rv_recon: List[int] = sorted(range(K), key=lambda i: self.rv_bands[i][0])
        self._pb_recon: List[int] = sorted(range(K), key=lambda i: self.pb_bands[i][0])

        # ── 网络层 ───────────────────────────────────────────────
        reduced = max(channels // 4, 32)

        # 聚合输入：rv_avg(CK) + rv_max(CK) + pb_avg(CK) + pb_max(CK) + diff(CK) = 5CK
        # kernel=(1,3) 同时感知 W 方向上下文与 K 带间的距离关系
        self.aggregate = nn.Sequential(
            nn.Conv2d(5 * K * channels, reduced,
                      kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU6(inplace=True),
        )
        # 每个视图生成 K 个距离带各自的门控，每带 C 通道
        self.gate_rv = nn.Sequential(
            nn.Conv2d(reduced, channels * K, kernel_size=(1, 1), bias=True),
            nn.Sigmoid(),
        )
        self.gate_pb = nn.Sequential(
            nn.Conv2d(reduced, channels * K, kernel_size=(1, 1), bias=True),
            nn.Sigmoid(),
        )

    @staticmethod
    def _pool_avg(feat: torch.Tensor,
                  bands: List[Tuple[int, int]]) -> torch.Tensor:
        """每带 AvgPool → (B, C*K, 1, W)，近→远顺序"""
        return torch.cat(
            [feat[:, :, s:e, :].mean(dim=2, keepdim=True) for s, e in bands],
            dim=1,
        )

    @staticmethod
    def _pool_max(feat: torch.Tensor,
                  bands: List[Tuple[int, int]]) -> torch.Tensor:
        """每带 MaxPool → (B, C*K, 1, W)，近→远顺序"""
        return torch.cat(
            [feat[:, :, s:e, :].max(dim=2, keepdim=True)[0] for s, e in bands],
            dim=1,
        )

    def _apply_gates_and_reassemble(self,
                                    feat: torch.Tensor,
                                    gates: torch.Tensor,
                                    bands: List[Tuple[int, int]],
                                    recon_order: List[int]) -> torch.Tensor:
        """
        对每个距离带用对应门控做残差调制，
        再按原始行号升序 cat，恢复 (B, C, H, W)。
        gates: (B, C*K, 1, W)
        """
        K = self.K
        gate_list = gates.chunk(K, dim=1)          # K × (B, C, 1, W)
        modulated = [
            feat[:, :, s:e, :] * gate_list[k] + feat[:, :, s:e, :]
            for k, (s, e) in enumerate(bands)
        ]
        # recon_order 将 near→far 顺序映射回 top→bottom（行号升序）
        return torch.cat([modulated[recon_order[i]] for i in range(K)], dim=2)

    def forward(self,
                f_rv: torch.Tensor,    # (B, C, H_r, W)
                f_pb: torch.Tensor,    # (B, C, H_p, W)
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Step 1：每带 Avg+Max 双统计量（近→远排列）
        rv_avg = self._pool_avg(f_rv, self.rv_bands)   # (B, CK, 1, W)
        rv_max = self._pool_max(f_rv, self.rv_bands)   # (B, CK, 1, W)
        pb_avg = self._pool_avg(f_pb, self.pb_bands)   # (B, CK, 1, W)
        pb_max = self._pool_max(f_pb, self.pb_bands)   # (B, CK, 1, W)

        # Step 2：【核心增强】跨视图差值信号——量化同距离带的特征偏差
        # diff[k] 在小目标/遮挡边界处幅值大，引导门控在此处更强响应
        diff = rv_avg - pb_avg                          # (B, CK, 1, W)

        # Step 3：5CK 联合上下文 → 方位角 × 距离带 × 跨视图偏差
        ctx = torch.cat([rv_avg, rv_max, pb_avg, pb_max, diff], dim=1)
        ctx = self.aggregate(ctx)                       # (B, reduced, 1, W)

        # Step 4：为每个距离带生成独立门控
        gate_rv = self.gate_rv(ctx)                     # (B, CK, 1, W)
        gate_pb = self.gate_pb(ctx)                     # (B, CK, 1, W)

        # Step 5：分带残差调制 + 恢复原始 H 顺序
        f_rv_out = self._apply_gates_and_reassemble(
            f_rv, gate_rv, self.rv_bands, self._rv_recon)
        f_pb_out = self._apply_gates_and_reassemble(
            f_pb, gate_pb, self.pb_bands, self._pb_recon)

        return f_rv_out, f_pb_out


# ─────────────────────────────────────────────────────────────
# 解码器
# ─────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """单个解码器上采样块：Bilinear Up + Concat Skip + DWConv + PW"""

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            conv_bn_relu6(in_c + skip_c, out_c, 3, 1, 1, groups=1),
            InvertedResidual(out_c, out_c, stride=(1,1), expand_ratio=4),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LightDecoder(nn.Module):
    """
    4 阶段解码器，在第 4,3,2,1 阶段各执行一次 AAFF 融合（外部传入）。
    输出最终特征图（1× 原始分辨率）。
    """

    def __init__(self,
                 enc_channels: List[int] = (64, 128, 256, 256),
                 out_channels: int = 64):
        super().__init__()
        ch = list(reversed(enc_channels))  # [256,256,128,64]

        self.blocks = nn.ModuleList([
            DecoderBlock(ch[0], ch[1], ch[1]),   # S4→S3
            DecoderBlock(ch[1], ch[2], ch[2]),   # S3→S2
            DecoderBlock(ch[2], ch[3], ch[3]),   # S2→S1
            DecoderBlock(ch[3], ch[3], out_channels),  # S1→full res
        ])
        # 最后一层上采样到 stem 分辨率，跳连接用 stem 特征
        self.out_c = out_channels

    def forward(self, enc_feats: List[torch.Tensor],
                stem_feat: torch.Tensor) -> torch.Tensor:
        """
        enc_feats: [S1,S2,S3,S4] (浅→深)
        stem_feat:  stem 输出（用于最后一阶段的跳连接）
        """
        s1, s2, s3, s4 = enc_feats
        x = self.blocks[0](s4, s3)
        x = self.blocks[1](x,  s2)
        x = self.blocks[2](x,  s1)
        x = self.blocks[3](x,  stem_feat)
        return x


# ─────────────────────────────────────────────────────────────
# 逐点特征聚合头
# ─────────────────────────────────────────────────────────────

class PointSampleAggregator(nn.Module):
    """
    利用预计算的归一化坐标，从两个特征图中双线性采样每个点的特征，
    拼接后送入分类 MLP。

    静态形状：点数 N 需在导出 ONNX 时固定（pad/sample 到 N_fixed）。

    创新④ View Confidence Gating (VCG)：
        基于 rv+pb 联合特征预测 per-channel 置信度权重（gate ∈ [0,2]），
        门控后再拼接，允许模型对不同视图的不同通道给予不同置信度。
        零初始化确保训练初期与原始行为完全等价，兼容现有 checkpoint 热启动。

    创新⑤ Semantic Prototype Memory (SPM)：
        维护 num_classes 个 EMA 原型向量，将点特征与原型的余弦相似度
        作为额外分类信号叠加到主 logits，为稀有类（motorcyclist 等）提供
        跨 batch 的稳定类别参考。EMA 更新在 train.py 中显式调用，
        ONNX 导出时原型冻结为静态常量，RKNN 完全兼容。
    """

    def __init__(self, rv_c: int, pb_c: int, num_classes: int,
                 pt_dim: int = 4,
                 use_vcg: bool = True,
                 use_proto: bool = True,
                 proto_dim: int = 64,
                 head_dropout: float = 0.1):
        super().__init__()
        self.rv_c      = rv_c
        self.use_vcg   = use_vcg
        self.use_proto = use_proto
        head_dropout = max(0.0, min(float(head_dropout), 0.9))

        # 点自身特征编码（x,y,z,intensity → 32d）
        self.pt_enc = nn.Sequential(
            nn.Linear(pt_dim, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU6(inplace=True),
        )
        in_dim = rv_c + pb_c + 32
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU6(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(128, num_classes),
        )

        # ── 创新④ VCG（Bayesian 归一化）────────────────────────
        # w_rv = sigmoid(Linear(rv||pb)) ∈ (0,1)，w_pb = 1 - w_rv
        # w_rv + w_pb ≡ 1（Bayesian 归一化，信号幅度守恒）
        # 乘以 2 使初始 gate = sigmoid(0)×2 = 1（与无门控等价）
        # 零初始化确保训练起点恒等，兼容已有 checkpoint
        if use_vcg:
            assert rv_c == pb_c, \
                f"VCG Bayesian 归一化要求 rv_c==pb_c，实际 rv_c={rv_c} pb_c={pb_c}"
            self.view_gate = nn.Linear(rv_c + pb_c, rv_c, bias=True)  # 输出 rv_c 个标量
            nn.init.zeros_(self.view_gate.weight)
            nn.init.zeros_(self.view_gate.bias)
            self._gate_entropy_sum   = 0.0
            self._gate_entropy_count = 0

        # ── 创新⑤ SPM ────────────────────────────────────────
        # proto_proj：将 in_dim 特征投影到 proto_dim 的 L2-归一化空间
        # prototypes：EMA 更新的类中心，register_buffer → 保存到 state_dict，
        #             ONNX 导出时嵌入为编译期常量
        # proto_weight：可学习标量，初始 0.1（训练初期 sim≈0 → 自动不干扰）
        if use_proto:
            self.proto_proj   = nn.Linear(in_dim, proto_dim, bias=False)
            self.register_buffer(
                'prototypes',
                F.normalize(torch.randn(num_classes, proto_dim), dim=-1)
            )
            self.proto_weight          = nn.Parameter(torch.tensor(0.1))
            self._num_classes_for_proto = num_classes

    # ─────────────────────────────────────────────────────────

    def forward(self,
                f_rv: torch.Tensor,       # (B, C_rv, H_rv, W_rv)
                f_pb: torch.Tensor,       # (B, C_pb, H_pb, W_pb)
                rv_coords: torch.Tensor,  # (B, N, 1, 2) — grid_sample 格式
                pb_coords: torch.Tensor,  # (B, N, 1, 2)
                points: torch.Tensor,     # (B, N, 4) — x,y,z,intensity
                ) -> torch.Tensor:        # (B, N, num_classes)

        B, N, _ = points.shape

        # 双线性采样
        rv_feat = F.grid_sample(f_rv, rv_coords, mode='bilinear',
                                align_corners=False, padding_mode='border')
        rv_feat = rv_feat.squeeze(-1).permute(0, 2, 1)   # (B, N, C_rv)

        pb_feat = F.grid_sample(f_pb, pb_coords, mode='bilinear',
                                align_corners=False, padding_mode='border')
        pb_feat = pb_feat.squeeze(-1).permute(0, 2, 1)   # (B, N, C_pb)

        # 点自身特征：球坐标 (r, θ, φ, intensity) 与 RV/PB 投影方式自然对齐
        # r=径向距离，θ=仰角，φ=方位角；在 float32 下计算，避免 AMP 精度问题
        _pt_dtype  = points.dtype
        _xyz       = points[..., :3].float()                       # (B,N,3) float32
        _intensity = points[..., 3:4]                              # (B,N,1) 原始 dtype
        _r         = _xyz.norm(dim=-1, keepdim=True).clamp(min=1e-6)   # (B,N,1)
        _theta     = torch.asin((_xyz[..., 2:3] / _r)
                                .clamp(-1.0 + 1e-6, 1.0 - 1e-6))      # 仰角 ∈(-π/2,π/2)
        _phi       = torch.atan2(_xyz[..., 1:2], _xyz[..., 0:1])       # 方位角 ∈(-π,π)
        _pt_input  = torch.cat([_r, _theta, _phi,
                                _intensity.float()], dim=-1).to(_pt_dtype)  # (B,N,4)
        pt_feat = self.pt_enc(_pt_input.reshape(B * N, -1)).reshape(B, N, 32)

        # ── 创新④ VCG：Bayesian per-channel 视图置信度门控 ──────
        # w_rv = sigmoid(logit) ∈ (0,1)，w_pb = 1 - w_rv
        # 归一化约束：w_rv + w_pb = 1（信号幅度守恒），×2 使初始 gate=1
        if self.use_vcg:
            gate_logit = self.view_gate(
                torch.cat([rv_feat, pb_feat], dim=-1).reshape(B * N, -1)
            ).reshape(B, N, -1)                               # (B,N,rv_c)
            w_rv = torch.sigmoid(gate_logit)                  # ∈(0,1)，初始≈0.5
            w_pb = 1.0 - w_rv                                 # 归一化互补
            rv_feat = rv_feat * (w_rv * 2.0)                  # 初始 gate=1（等价无门控）
            pb_feat = pb_feat * (w_pb * 2.0)
            # 诊断：累积 gate entropy（仅训练，不进入梯度图）
            if self.training:
                with torch.no_grad():
                    p = w_rv.float()                          # ∈(0,1)，float32 避免下溢
                    h = -(p * torch.log2(p + 1e-7)
                          + (1 - p) * torch.log2(1 - p + 1e-7))
                    self._gate_entropy_sum   += h.mean().item()
                    self._gate_entropy_count += 1

        feat     = torch.cat([rv_feat, pb_feat, pt_feat], dim=-1)  # (B,N,in_dim)
        feat_flat = feat.reshape(B * N, -1)

        # 主分类头
        main_logits = self.head(feat_flat).reshape(B, N, -1)       # (B,N,C)

        # ── 创新⑤ SPM：原型余弦相似度辅助分类 ──────────────────
        if self.use_proto:
            feat_proj  = self.proto_proj(feat_flat)                  # (B*N, proto_dim)
            feat_norm  = F.normalize(feat_proj, dim=-1)
            proto_norm = F.normalize(self.prototypes.to(feat_norm.dtype), dim=-1)  # (C_cls, proto_dim)
            sim        = (feat_norm @ proto_norm.T).reshape(B, N, -1)  # (B,N,C_cls)
            logits     = main_logits + self.proto_weight.clamp(min=0.0) * sim
            if self.training:
                # 缓存 float32 特征供外部 EMA 更新（不参与梯度图）
                self._cached_proto_feat = feat_norm.detach().float().reshape(B, N, -1)
        else:
            logits = main_logits

        return logits

    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_prototypes(self, labels: torch.Tensor,
                          ema_decay: float = 0.99,
                          ignore_index: int = 255):
        """
        自适应 EMA 更新类别原型。在每个 train step 的 optimizer step 之后调用。

        自适应策略：稀有类（本 batch 点数少于均匀期望值）的 EMA decay 降低（更新更快），
        以补偿其出现频次低导致的梯度信号稀疏。最高加速 5×（decay 不低于 0.9）。
        常见类（点数≥期望值）保持 base_decay=0.99，不受影响。
        """
        if not (self.use_proto and hasattr(self, '_cached_proto_feat')):
            return
        feat        = self._cached_proto_feat            # (B, N, proto_dim), float32
        B, N, D     = feat.shape
        total       = B * N                              # 本 batch 总点数
        feat_flat   = feat.reshape(total, D)
        labels_flat = labels.reshape(total)
        valid       = (labels_flat != ignore_index)

        # 每类均匀期望点数（用于判断"稀有"与"常见"）
        expected_n  = total / self._num_classes_for_proto

        for c in range(self._num_classes_for_proto):
            mask = valid & (labels_flat == c)
            n_c  = mask.sum().item()
            if n_c > 0:
                mean_feat = feat_flat[mask].mean(0)      # (proto_dim,) float32

                # 自适应 decay：稀有类加速（最多 5×），decay ∈ [0.9, base_decay]
                speedup        = max(1.0, min(expected_n / (n_c + 1.0), 5.0))
                adaptive_decay = max(0.9, 1.0 - (1.0 - ema_decay) * speedup)

                updated = (adaptive_decay * self.prototypes[c].float()
                           + (1.0 - adaptive_decay) * mean_feat)
                self.prototypes[c] = F.normalize(updated, dim=0)  # 保持单位范数

    def get_and_reset_diagnostics(self) -> dict:
        """返回本 epoch 的诊断指标并重置累积器。在 epoch 结束时调用。"""
        result = {}
        if self.use_vcg and self._gate_entropy_count > 0:
            result['gate_entropy'] = (
                self._gate_entropy_sum / self._gate_entropy_count
            )
        self._gate_entropy_sum   = 0.0
        self._gate_entropy_count = 0
        return result


# ─────────────────────────────────────────────────────────────
# 辅助预测头（训练时使用）
# ─────────────────────────────────────────────────────────────

class AuxHead(nn.Module):
    """在特征图上输出像素级辅助预测（用于 RV/PB 分支的辅助损失）"""

    def __init__(self, in_c: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_c, in_c // 2, 1, bias=False),
            nn.BatchNorm2d(in_c // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_c // 2, num_classes, 1),
        )

    def forward(self, x):
        return self.head(x)
