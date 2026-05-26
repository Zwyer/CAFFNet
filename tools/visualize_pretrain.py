#!/usr/bin/env python3
"""
预训练效果可视化（无需任何标注）

- PCA → RGB 点云着色：从 encoder+decoder 拿逐点融合特征，PCA 投影到 3 维，
  分通道归一化后作为 RGB，导出 .ply（3D）与 BEV .png（俯视图）。
  视觉上同一物体颜色相近、不同物体颜色明显分块 → 说明预训练学到了语义结构。
- MAE 重建三联图：用与预训练相同的掩码策略生成 mask，前向 NOMAE 头部，
  导出 occ_target / masked_input / pred_occ 的对比图（RV 与 PB 各一张）。

Usage:
    python tools/visualize_pretrain.py \\
        --cfg prfnet/configs/prfnet_16lidar_unified.yaml \\
        --ckpt runs/prfnet_16lidar_pretrain_rankme/<ts>/checkpoints/best_rankme.pth \\
        --list selected_frames_select.txt \\
        --out_dir runs/viz_pretrain \\
        --max_frames 20

Outputs (per frame `<seq>/<stem>`):
    <out_dir>/<seq>_<stem>_pca.ply           - 3D 点云 PCA 着色
    <out_dir>/<seq>_<stem>_pca_bev.png       - BEV 预览
    <out_dir>/<seq>_<stem>_mae_rv.png        - RV 重建三联图
    <out_dir>/<seq>_<stem>_mae_pb.png        - PB 重建三联图
"""

import argparse
import os
import sys
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from prfnet.datasets.semantickitti import SemanticKITTIDataset
from prfnet.models.prfnet import PRFNet
from prfnet.models.pretrain_heads import build_mask_with_pos_ratio_control


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_list(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seq, stem = line.split("/", 1)
            out.append((seq, stem))
    return out


def write_ply_ascii(path, xyz, rgb_u8):
    n = xyz.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(
                f"{xyz[i,0]:.4f} {xyz[i,1]:.4f} {xyz[i,2]:.4f} "
                f"{int(rgb_u8[i,0])} {int(rgb_u8[i,1])} {int(rgb_u8[i,2])}\n"
            )


def pca_to_rgb(feat: torch.Tensor) -> np.ndarray:
    """feat: (N, D) → rgb uint8 (N, 3) via PCA→分位归一化."""
    f = feat - feat.mean(dim=0, keepdim=True)
    _, _, V = torch.pca_lowrank(f, q=3, niter=2)
    proj = f @ V[:, :3]
    lo = torch.quantile(proj.float(), 0.02, dim=0)
    hi = torch.quantile(proj.float(), 0.98, dim=0)
    rgb = ((proj - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
    return (rgb * 255.0).to(torch.uint8).cpu().numpy()


def save_bev_png(path, xyz, rgb_u8, r_max=80.0, dpi=120):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.scatter(xyz[:, 0], xyz[:, 1], c=rgb_u8 / 255.0, s=0.5, marker=".")
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(os.path.basename(path).replace("_pca_bev.png", ""))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_mae_triplet(path, view, occ_tgt, mask, pred_occ, dpi=120):
    """occ_tgt/mask/pred_occ: (H, W) numpy, value ∈ [0,1]."""
    H, W = occ_tgt.shape
    # 宽-高比按图像比例缩放，避免 16×1024 这种细长 RV 被压扁
    aspect = max(W / max(H, 1), 1.0)
    fig, axes = plt.subplots(
        4, 1, figsize=(min(16, 0.012 * W + 4), 0.06 * H * 4 + 1), dpi=dpi
    )
    axes[0].imshow(occ_tgt, aspect=aspect, vmin=0, vmax=1, cmap="gray")
    axes[0].set_title(f"{view}: occupancy target")
    axes[0].axis("off")

    axes[1].imshow(mask, aspect=aspect, vmin=0, vmax=1, cmap="magma")
    axes[1].set_title(f"{view}: mask (1 = masked)")
    axes[1].axis("off")

    masked_input = occ_tgt * (1.0 - mask)
    axes[2].imshow(masked_input, aspect=aspect, vmin=0, vmax=1, cmap="gray")
    axes[2].set_title(f"{view}: masked input (visible occupancy)")
    axes[2].axis("off")

    axes[3].imshow(pred_occ, aspect=aspect, vmin=0, vmax=1, cmap="viridis")
    axes[3].set_title(f"{view}: predicted occupancy (sigmoid, scale 1)")
    axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_model_from_cfg(cfg, device):
    dc = dict(cfg["data"])
    mc = dict(cfg["model"])
    if "pretrain_data" in cfg:
        dc.update(cfg["pretrain_data"])
    if "pretrain_model" in cfg:
        mc.update(cfg["pretrain_model"])
    rv_in = 6 + 3 * int(dc.get("use_surface_normals", False)) + 3 * int(
        dc.get("use_angle_encoding", False)
    )
    model = PRFNet(
        rv_in=rv_in,
        pb_in=mc["pb_in"],
        num_classes=19,
        enc_channels=mc["enc_channels"],
        dec_out_c=mc["dec_out_c"],
        expand_ratios=mc["expand_ratios"],
        rv_strides=mc["rv_strides"],
        pb_strides=mc["pb_strides"],
        aspp_rates=mc["aspp_rates"],
        rv_H=dc["rv_H"],
        pb_H=dc["pb_H"],
        use_ds_aaff=mc.get("use_ds_aaff", True),
        ds_aaff_K=mc.get("ds_aaff_K", 2),
        head_dropout=mc.get("head_dropout", 0.1),
        use_vcg=mc.get("use_vcg", False),
        use_proto=mc.get("use_proto", False),
        proto_dim=mc.get("proto_dim", 64),
    ).to(device)
    return model, dc, mc


def build_dataset_for_seqs(dc, seqs):
    return SemanticKITTIDataset(
        root=dc["root"],
        split="train",
        seqs=seqs,
        require_labels=False,
        rv_H=dc["rv_H"],
        rv_W=dc["rv_W"],
        pb_H=dc["pb_H"],
        pb_W=dc["pb_W"],
        augment=False,
        rotate=False,
        flip=False,
        scale_min=1.0,
        scale_max=1.0,
        drop_p=0.0,
        R_max=dc.get("R_max", 80.0),
        use_polarmix=False,
        use_lasermix=False,
        fov_up=dc["fov_up"],
        fov_down=dc["fov_down"],
        max_points=dc.get("max_points", 65536),
        polarmix_p=0.0,
        polarmix_sectors=1,
        lasermix_p=0.0,
        use_surface_normals=dc.get("use_surface_normals", False),
        use_angle_encoding=dc.get("use_angle_encoding", False),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="预训练用的 yaml 配置")
    ap.add_argument("--ckpt", required=True, help="预训练 .pth (含 state_dict)")
    ap.add_argument("--list", required=True, help="selected_frames_select.txt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_frames", type=int, default=0,
                    help="0 = 全部；否则只处理列表前 N 帧")
    ap.add_argument("--rv_mask_ratio", type=float, default=-1.0,
                    help="<0 时从 cfg.pretrain.rv_mask_ratio_end 读取")
    ap.add_argument("--pb_mask_ratio", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=0, help="掩码随机种子，固定后可复现")
    ap.add_argument("--occ_scale_idx", type=int, default=0,
                    help="占据多尺度预测取第几个 scale 画图（0=最细）")
    ap.add_argument("--skip_ply", action="store_true",
                    help="跳过 .ply 输出（仅画 png）")
    ap.add_argument(
        "--feat_mode", default="fused",
        choices=["fused", "full", "pt"],
        help=(
            "fused: 只用 rv_feat+pb_feat（encoder+decoder 输出，去掉 pt_enc 的 intensity 干扰）；"
            "full: rv+pb+pt 全部（原始行为，PCA 易被 intensity 主导）；"
            "pt: 只看 pt_enc（32 维，用于确认 intensity 是否主导颜色）"
        ),
    )
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    pc = cfg.get("pretrain", {})
    rv_mr = args.rv_mask_ratio if args.rv_mask_ratio >= 0 else float(
        pc.get("rv_mask_ratio_end", pc.get("rv_mask_ratio", 0.7))
    )
    pb_mr = args.pb_mask_ratio if args.pb_mask_ratio >= 0 else float(
        pc.get("pb_mask_ratio_end", pc.get("pb_mask_ratio", 0.7))
    )

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, dc, mc = build_model_from_cfg(cfg, device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[viz] loaded ckpt: {args.ckpt}")
    print(f"[viz]   missing keys: {len(missing)}  unexpected: {len(unexpected)}")
    model.eval()

    frames = parse_list(args.list)
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    seqs = sorted({s for s, _ in frames})
    print(f"[viz] frames={len(frames)} seqs={seqs}  rv_mask={rv_mr}  pb_mask={pb_mr}")

    ds = build_dataset_for_seqs(dc, seqs)
    idx_map = {}
    for i, entry in enumerate(ds.frames):
        stem = os.path.splitext(os.path.basename(entry["velo"]))[0]
        idx_map[(entry["seq"], stem)] = i
    r_max = float(dc.get("R_max", 80.0))

    n_done = n_miss = 0
    for seq, stem in frames:
        key = (seq, stem)
        if key not in idx_map:
            print(f"[MISS] {seq}/{stem} 不在数据集中（检查 bin 是否存在）")
            n_miss += 1
            continue
        sample = ds[idx_map[key]]
        rv_img = sample["rv_img"].unsqueeze(0).to(device)
        pb_img = sample["pb_img"].unsqueeze(0).to(device)
        rv_coords = sample["rv_coords"].unsqueeze(0).to(device)
        pb_coords = sample["pb_coords"].unsqueeze(0).to(device)
        points = sample["points"].unsqueeze(0).to(device)

        with torch.no_grad():
            # ── PCA → RGB ─────────────────────────────────────────
            feat_full = model.extract_pretrain_point_features(
                rv_img, pb_img, rv_coords, pb_coords, points
            )[0]  # (N, rv_c + pb_c + 32)
            dec_c = mc["dec_out_c"]  # 每个视图的 decoder 输出维度
            if args.feat_mode == "fused":
                feat = feat_full[:, : dec_c * 2]   # rv_feat + pb_feat，去掉 pt_enc
            elif args.feat_mode == "pt":
                feat = feat_full[:, dec_c * 2 :]   # 只看 pt_enc（intensity 主导的 32 维）
            else:
                feat = feat_full                    # 全部（full，原始行为）
            rgb = pca_to_rgb(feat.float())
            xyz = points[0, :, :3].cpu().numpy()

            base = f"{seq}_{stem}"
            if not args.skip_ply:
                write_ply_ascii(
                    os.path.join(args.out_dir, base + "_pca.ply"), xyz, rgb
                )
            save_bev_png(
                os.path.join(args.out_dir, base + "_pca_bev.png"),
                xyz, rgb, r_max=r_max,
            )

            # ── MAE 重建（直接调用 head，拿 pred_occ）─────────────
            rv_occ_tgt = (rv_img[:, 3:4] > 0).float()
            pb_occ_tgt = pb_img[:, 8:9].float()
            rv_center_tgt = rv_img[:, 0:3]
            pb_center_tgt = pb_img[:, 0:3]

            g = torch.Generator(device=device).manual_seed(int(args.seed))
            # 固定种子：把 torch 全局 RNG 也同步，因为 mask builder 用全局 RNG
            torch.manual_seed(int(args.seed))
            rv_mask, _ = build_mask_with_pos_ratio_control(
                rv_img, rv_occ_tgt, rv_mr,
                strategy=str(pc.get("mask_strategy", "mixed")),
                band_axis=str(pc.get("rv_band_axis", "row")),
                mix_random=float(pc.get("mask_mix_random", 0.5)),
                mix_block=float(pc.get("mask_mix_block", 0.3)),
                mix_band=float(pc.get("mask_mix_band", 0.2)),
                mix_hmg=float(pc.get("mask_mix_hmg", 0.0)),
                block_h_min=int(pc.get("block_h_min", 4)),
                block_h_max=int(pc.get("block_h_max", 16)),
                block_w_min=int(pc.get("block_w_min", 16)),
                block_w_max=int(pc.get("block_w_max", 64)),
                hmg_coarse_stride=int(pc.get("hmg_coarse_stride", 8)),
                hmg_fine_extra_ratio=float(pc.get("hmg_fine_extra_ratio", 0.05)),
                enable_control=bool(pc.get("mask_pos_ratio_control_enable", False)),
                min_pos_ratio=float(pc.get("mask_pos_ratio_min", 0.08)),
                max_pos_ratio=float(pc.get("mask_pos_ratio_max", 0.50)),
                max_tries=int(pc.get("mask_resample_max_tries", 5)),
            )
            pb_mask, _ = build_mask_with_pos_ratio_control(
                pb_img, pb_occ_tgt, pb_mr,
                strategy=str(pc.get("mask_strategy", "mixed")),
                band_axis=str(pc.get("pb_band_axis", "col")),
                mix_random=float(pc.get("mask_mix_random", 0.5)),
                mix_block=float(pc.get("mask_mix_block", 0.3)),
                mix_band=float(pc.get("mask_mix_band", 0.2)),
                mix_hmg=float(pc.get("mask_mix_hmg", 0.0)),
                block_h_min=int(pc.get("block_h_min", 4)),
                block_h_max=int(pc.get("block_h_max", 16)),
                block_w_min=int(pc.get("block_w_min", 16)),
                block_w_max=int(pc.get("block_w_max", 64)),
                hmg_coarse_stride=int(pc.get("hmg_coarse_stride", 8)),
                hmg_fine_extra_ratio=float(pc.get("hmg_fine_extra_ratio", 0.05)),
                enable_control=bool(pc.get("mask_pos_ratio_control_enable", False)),
                min_pos_ratio=float(pc.get("mask_pos_ratio_min", 0.08)),
                max_pos_ratio=float(pc.get("mask_pos_ratio_max", 0.50)),
                max_tries=int(pc.get("mask_resample_max_tries", 5)),
            )

            rv_img_in = rv_img * (1.0 - rv_mask)
            pb_img_in = pb_img * (1.0 - pb_mask)
            fused_rv, fused_pb, rv_stem_f, pb_stem_f = model._encode(rv_img_in, pb_img_in)
            rv_out = model.rv_dec(fused_rv, rv_stem_f)
            pb_out = model.pb_dec(fused_pb, pb_stem_f)

            rv_ret = model.pretrain_head_rv(
                rv_out, rv_mask, rv_occ_tgt, rv_center_tgt,
                informative_only=bool(pc.get("informative_occ_only", True)),
                pcp_informative_only=bool(pc.get("pcp_informative_only", True)),
                neighbor_sup_only_visible=bool(pc.get("neighbor_sup_only_visible", True)),
                pcp_residual_center=bool(pc.get("pcp_residual_center", True)),
                pcp_far_only=bool(pc.get("pcp_far_only", False)),
            )
            pb_ret = model.pretrain_head_pb(
                pb_out, pb_mask, pb_occ_tgt, pb_center_tgt,
                informative_only=bool(pc.get("informative_occ_only", True)),
                pcp_informative_only=bool(pc.get("pcp_informative_only", True)),
                neighbor_sup_only_visible=bool(pc.get("neighbor_sup_only_visible", True)),
                pcp_residual_center=bool(pc.get("pcp_residual_center", True)),
                pcp_far_only=bool(pc.get("pcp_far_only", False)),
            )

            s_idx = min(
                max(0, int(args.occ_scale_idx)),
                rv_ret["pred_occ"].shape[1] - 1,
            )
            rv_pred = torch.sigmoid(rv_ret["pred_occ"][0, s_idx]).cpu().numpy()
            pb_pred = torch.sigmoid(pb_ret["pred_occ"][0, s_idx]).cpu().numpy()
            rv_occ = rv_occ_tgt[0, 0].cpu().numpy()
            pb_occ = pb_occ_tgt[0, 0].cpu().numpy()
            rv_m = rv_mask[0, 0].cpu().numpy()
            pb_m = pb_mask[0, 0].cpu().numpy()

            save_mae_triplet(
                os.path.join(args.out_dir, base + "_mae_rv.png"),
                "RV", rv_occ, rv_m, rv_pred,
            )
            save_mae_triplet(
                os.path.join(args.out_dir, base + "_mae_pb.png"),
                "PB", pb_occ, pb_m, pb_pred,
            )

        n_done += 1
        if n_done % 5 == 0:
            print(f"  ... {n_done}/{len(frames)}")

    print(f"[viz] done={n_done} miss={n_miss} -> {args.out_dir}")


if __name__ == "__main__":
    main()
