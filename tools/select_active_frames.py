#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
低偏差 SELECT 风格主动选帧（LiDAR 语义分割）

改进点：
1) 多 checkpoint 集成（降低单模型偏置）
2) MC Dropout + Ensemble 不确定性（MI/Entropy/Variation Ratio）
3) 软概率类别均衡（避免硬伪标签偏差）
4) 候选池并联：信息性 top-k 与代表性 farthest-point top-k 的并集
5) 序列时间去重约束（同序列最小帧间隔）
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from prfnet.models.prfnet import PRFNet
from prfnet.utils.projection import RangeImageProjector, PolarBEVProjector


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device) -> PRFNet:
    dc = cfg["data"]
    mc = cfg["model"]
    rv_in = 6 + 3 * int(dc.get("use_surface_normals", False)) + 3 * int(dc.get("use_angle_encoding", False))
    model = PRFNet(
        rv_in=rv_in,
        pb_in=mc["pb_in"],
        num_classes=dc.get("num_classes", 19),
        enc_channels=mc["enc_channels"],
        dec_out_c=mc["dec_out_c"],
        expand_ratios=mc["expand_ratios"],
        rv_strides=mc["rv_strides"],
        pb_strides=mc["pb_strides"],
        aspp_rates=mc["aspp_rates"],
        rv_H=dc["rv_H"],
        pb_H=dc["pb_H"],
        use_ds_aaff=mc.get("use_ds_aaff", True),
        ds_aaff_K=mc.get("ds_aaff_K", 4),
        head_dropout=mc.get("head_dropout", 0.1),
        use_vcg=mc.get("use_vcg", False),
        use_proto=mc.get("use_proto", False),
        proto_dim=mc.get("proto_dim", 64),
    ).to(device)
    return model


def load_ckpt_state(model: PRFNet, ckpt_path: str, device: torch.device, resume_ema: bool = True):
    ckpt = torch.load(ckpt_path, map_location=device)
    if resume_ema and isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
        state = ckpt["ema_state_dict"]
    else:
        state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()


def parse_csv(arg: Optional[str]) -> List[str]:
    if arg is None:
        return []
    out = []
    for x in arg.split(","):
        x = x.strip()
        if len(x) > 0:
            out.append(x)
    return out


def resolve_ckpt_list(primary_ckpt: str,
                      ckpt_list_csv: Optional[str],
                      ckpt_glob: Optional[str],
                      ensemble_max_models: int) -> List[str]:
    paths = []

    # primary 允许逗号分隔
    for p in parse_csv(primary_ckpt):
        paths.append(p)

    for p in parse_csv(ckpt_list_csv):
        paths.append(p)

    if ckpt_glob is not None and len(ckpt_glob.strip()) > 0:
        paths.extend(sorted(glob.glob(ckpt_glob.strip())))

    uniq = []
    seen = set()
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    out = [p for p in uniq if os.path.isfile(p)]
    if len(out) == 0:
        raise RuntimeError("no valid checkpoints resolved")

    m = max(1, int(ensemble_max_models))
    return out[:m]


def scan_frames(root: str, seqs: List[str]) -> List[Dict[str, str]]:
    frames: List[Dict[str, str]] = []
    for seq in seqs:
        velo_dir = os.path.join(root, "sequences", seq, "velodyne")
        if not os.path.isdir(velo_dir):
            continue
        bins = sorted(glob.glob(os.path.join(velo_dir, "*.bin")))
        for bp in bins:
            stem = os.path.splitext(os.path.basename(bp))[0]
            stem_int = None
            try:
                stem_int = int(stem)
            except Exception:
                stem_int = None
            frames.append({"seq": seq, "stem": stem, "stem_int": stem_int, "velo": bp})
    return frames


def enable_mc_dropout_only(model: torch.nn.Module):
    model.eval()
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()


def preprocess_points(
    bin_path: str,
    r_min: float,
    r_max: float,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if pts.shape[0] == 0:
        return pts

    dist = np.linalg.norm(pts[:, :3], axis=1)
    keep = (dist > r_min) & (dist < r_max)
    pts = pts[keep]

    if pts.shape[0] > max_points:
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts


def _build_inputs(
    pts: np.ndarray,
    rv_proj: RangeImageProjector,
    pb_proj: PolarBEVProjector,
    device: torch.device,
):
    xyz = pts[:, :3]
    inten = pts[:, 3].astype(np.float32, copy=False)

    rv_img, _ = rv_proj.project(xyz, inten)
    pb_img = pb_proj.project(xyz, inten)
    rv_coords = rv_proj.compute_sample_coords(xyz)
    pb_coords = pb_proj.compute_sample_coords(xyz)

    rv_img_t = torch.from_numpy(rv_img).unsqueeze(0).to(device)
    pb_img_t = torch.from_numpy(pb_img).unsqueeze(0).to(device)
    rv_coords_t = torch.from_numpy(rv_coords).unsqueeze(0).unsqueeze(2).to(device)
    pb_coords_t = torch.from_numpy(pb_coords).unsqueeze(0).unsqueeze(2).to(device)
    pts_t = torch.from_numpy(pts).unsqueeze(0).to(device)
    return rv_img_t, pb_img_t, rv_coords_t, pb_coords_t, pts_t


@torch.no_grad()
def frame_forward_ensemble(
    models: List[PRFNet],
    pts: np.ndarray,
    rv_proj: RangeImageProjector,
    pb_proj: PolarBEVProjector,
    device: torch.device,
    mc_passes: int,
    uncertainty_use_entropy: bool,
    uncertainty_use_mi: bool,
    uncertainty_use_vr: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      unc_vec: [mi, entropy, vr]
      frame_embedding: (D,)
      soft_hist: (C,) from mean probability
    """
    if pts.shape[0] == 0:
        return (
            np.zeros((3,), dtype=np.float32),
            np.zeros((64,), dtype=np.float32),
            np.zeros((19,), dtype=np.float32),
        )

    rv_img_t, pb_img_t, rv_coords_t, pb_coords_t, pts_t = _build_inputs(pts, rv_proj, pb_proj, device)

    feat_embs = []
    probs_all = []
    for model in models:
        # embedding
        feat = model.extract_pretrain_point_features(rv_img_t, pb_img_t, rv_coords_t, pb_coords_t, pts_t)
        feat = feat.squeeze(0)  # (N,D)
        feat_n = F.normalize(feat, dim=-1)
        feat_embs.append(feat_n.mean(dim=0, keepdim=False).float())

        # uncertainty samples
        t_pass = max(1, int(mc_passes))
        if t_pass > 1:
            enable_mc_dropout_only(model)
        else:
            model.eval()

        for _ in range(t_pass):
            out = model(rv_img_t, pb_img_t, rv_coords_t, pb_coords_t, pts_t)
            logits = out["logits"].squeeze(0)  # (N, C)
            probs = F.softmax(logits, dim=-1)
            probs_all.append(probs.unsqueeze(0))

        model.eval()

    # (S, N, C), S = num_models * mc_passes
    probs_stack = torch.cat(probs_all, dim=0)
    p_mean = probs_stack.mean(dim=0)  # (N, C)

    # soft histogram: per-frame class prior estimate
    soft_hist = p_mean.mean(dim=0)

    # uncertainty components
    # entropy on predictive mean
    ent_pred = -(p_mean * p_mean.clamp_min(1e-8).log()).sum(dim=-1).mean()
    # expected entropy
    ent_exp = -(probs_stack * probs_stack.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=0).mean()
    mi = (ent_pred - ent_exp)
    # variation ratio: 1 - max prob
    vr = (1.0 - p_mean.max(dim=-1).values).mean()

    # masked by switches (component-level)
    mi_v = float(mi.item()) if uncertainty_use_mi else 0.0
    ent_v = float(ent_pred.item()) if uncertainty_use_entropy else 0.0
    vr_v = float(vr.item()) if uncertainty_use_vr else 0.0

    emb = torch.stack(feat_embs, dim=0).mean(dim=0).cpu().numpy().astype(np.float32)
    hist = soft_hist.cpu().numpy().astype(np.float32)
    unc = np.asarray([mi_v, ent_v, vr_v], dtype=np.float32)
    return unc, emb, hist


def normalize01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def cosine_sim_matrix(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm = np.clip(nrm, 1e-8, None)
    xn = x / nrm
    sim = xn @ xn.T
    sim = (sim + 1.0) * 0.5
    return np.clip(sim, 0.0, 1.0).astype(np.float32)


def farthest_point_indices(emb: np.ndarray, k: int, seed: int) -> np.ndarray:
    """在 embedding 上做 farthest-point sampling（近似 k-center 代表性）。"""
    n = emb.shape[0]
    k = int(min(max(1, k), n))
    rng = np.random.default_rng(seed)

    # 归一化后用 cosine distance
    x = emb.astype(np.float32)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(nrm, 1e-8, None)

    first = int(rng.integers(0, n))
    selected = [first]

    # d(i) = 1 - max_j cos(x_i, x_sel_j)
    sim_to_first = x @ x[first]
    best_sim = sim_to_first.copy()

    for _ in range(1, k):
        d = 1.0 - best_sim
        nxt = int(np.argmax(d))
        selected.append(nxt)
        sim_new = x @ x[nxt]
        best_sim = np.maximum(best_sim, sim_new)

    return np.asarray(selected, dtype=np.int64)


def violates_temporal_gap(frames: List[Dict[str, str]],
                          cand_idx: int,
                          selected_idx: List[int],
                          min_gap: int) -> bool:
    if min_gap <= 0:
        return False
    c = frames[cand_idx]
    c_seq = c["seq"]
    c_stem = c.get("stem_int", None)
    if c_stem is None:
        return False

    for sidx in selected_idx:
        s = frames[sidx]
        if s["seq"] != c_seq:
            continue
        s_stem = s.get("stem_int", None)
        if s_stem is None:
            continue
        if abs(int(c_stem) - int(s_stem)) < min_gap:
            return True
    return False


def greedy_select_submodular(
    sim: np.ndarray,
    info_score: np.ndarray,
    bal_score: np.ndarray,
    budget: int,
    w_info: float,
    w_rep: float,
    w_bal: float,
    cand_global_idx: np.ndarray,
    all_frames: List[Dict[str, str]],
    temporal_min_stem_gap: int,
) -> List[int]:
    """
    facility-location + informativeness + class-balance 贪心。
    返回局部索引（相对于 candidate pool）。
    """
    n = sim.shape[0]
    budget = min(budget, n)
    selected_local: List[int] = []
    covered = np.zeros((n,), dtype=np.float32)
    chosen = np.zeros((n,), dtype=bool)

    selected_global: List[int] = []
    for _ in range(budget):
        best_i = -1
        best_gain = -1e18
        for j in range(n):
            if chosen[j]:
                continue

            gidx = int(cand_global_idx[j])
            if violates_temporal_gap(all_frames, gidx, selected_global, temporal_min_stem_gap):
                continue

            new_cov = np.maximum(covered, sim[:, j])
            rep_gain = float(np.sum(new_cov - covered))
            gain = (
                w_rep * rep_gain
                + w_info * float(info_score[j])
                + w_bal * float(bal_score[j])
            )
            if gain > best_gain:
                best_gain = gain
                best_i = j

        if best_i < 0:
            break

        selected_local.append(best_i)
        selected_global.append(int(cand_global_idx[best_i]))
        chosen[best_i] = True
        covered = np.maximum(covered, sim[:, best_i])

    return selected_local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="prfnet/configs/prfnet_semantickitti_unified.yaml")
    ap.add_argument("--ckpt", required=True, help="checkpoint path or comma-separated paths")
    ap.add_argument("--ckpt_list", default=None, help="additional comma-separated checkpoints")
    ap.add_argument("--ckpt_glob", default=None, help="glob for additional checkpoints")
    ap.add_argument("--resume_ema", action="store_true", help="load ema_state_dict if present")

    ap.add_argument("--seqs", default=None, help="comma-separated sequence list, e.g. 00,01,02")
    ap.add_argument("--budget", type=int, default=None)

    ap.add_argument("--candidate_pool_ratio", type=float, default=None)
    ap.add_argument("--candidate_pool_min", type=int, default=None)
    ap.add_argument("--candidate_rep_ratio", type=float, default=None)
    ap.add_argument("--candidate_rep_min", type=int, default=None)

    ap.add_argument("--mc_passes", type=int, default=None)
    ap.add_argument("--ensemble_max_models", type=int, default=None)

    ap.add_argument("--uncertainty_use_entropy", type=int, default=None)
    ap.add_argument("--uncertainty_use_mi", type=int, default=None)
    ap.add_argument("--uncertainty_use_vr", type=int, default=None)
    ap.add_argument("--w_unc_mi", type=float, default=None)
    ap.add_argument("--w_unc_entropy", type=float, default=None)
    ap.add_argument("--w_unc_vr", type=float, default=None)

    ap.add_argument("--r_min", type=float, default=None)
    ap.add_argument("--r_max", type=float, default=None)
    ap.add_argument("--max_points", type=int, default=None)

    ap.add_argument("--w_unc", type=float, default=None)
    ap.add_argument("--w_rep", type=float, default=None)
    ap.add_argument("--w_bal", type=float, default=None)

    ap.add_argument("--temporal_min_stem_gap", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_txt", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    dc = cfg["data"]
    sc = cfg.get("select", {})
    select_enable = bool(sc.get("enable", True))
    if not select_enable:
        print("[SELECT] disabled by cfg.select.enable=false, exit.")
        return

    # CLI > yaml.select > hard defaults
    args.seqs = args.seqs if args.seqs is not None else str(sc.get("seqs", "00"))
    args.budget = int(args.budget if args.budget is not None else sc.get("budget", 300))

    args.candidate_pool_ratio = float(
        args.candidate_pool_ratio if args.candidate_pool_ratio is not None else sc.get("candidate_pool_ratio", 0.5)
    )
    args.candidate_pool_min = int(
        args.candidate_pool_min if args.candidate_pool_min is not None else sc.get("candidate_pool_min", 1200)
    )
    args.candidate_rep_ratio = float(
        args.candidate_rep_ratio if args.candidate_rep_ratio is not None else sc.get("candidate_rep_ratio", 0.2)
    )
    args.candidate_rep_min = int(
        args.candidate_rep_min if args.candidate_rep_min is not None else sc.get("candidate_rep_min", 400)
    )

    args.mc_passes = int(args.mc_passes if args.mc_passes is not None else sc.get("mc_passes", 8))
    args.ensemble_max_models = int(
        args.ensemble_max_models if args.ensemble_max_models is not None else sc.get("ensemble_max_models", 3)
    )

    args.uncertainty_use_entropy = bool(
        int(args.uncertainty_use_entropy) if args.uncertainty_use_entropy is not None
        else int(sc.get("uncertainty_use_entropy", 1))
    )
    args.uncertainty_use_mi = bool(
        int(args.uncertainty_use_mi) if args.uncertainty_use_mi is not None
        else int(sc.get("uncertainty_use_mi", 1))
    )
    args.uncertainty_use_vr = bool(
        int(args.uncertainty_use_vr) if args.uncertainty_use_vr is not None
        else int(sc.get("uncertainty_use_vr", 1))
    )
    args.w_unc_mi = float(args.w_unc_mi if args.w_unc_mi is not None else sc.get("w_unc_mi", 1.0))
    args.w_unc_entropy = float(args.w_unc_entropy if args.w_unc_entropy is not None else sc.get("w_unc_entropy", 0.6))
    args.w_unc_vr = float(args.w_unc_vr if args.w_unc_vr is not None else sc.get("w_unc_vr", 0.6))

    args.r_min = float(args.r_min if args.r_min is not None else sc.get("r_min", 0.5))
    args.r_max = float(args.r_max if args.r_max is not None else sc.get("r_max", 85.0))
    args.max_points = int(args.max_points if args.max_points is not None else sc.get("max_points", 131072))

    args.w_unc = float(args.w_unc if args.w_unc is not None else sc.get("w_unc", 1.0))
    args.w_rep = float(args.w_rep if args.w_rep is not None else sc.get("w_rep", 1.0))
    args.w_bal = float(args.w_bal if args.w_bal is not None else sc.get("w_bal", 0.6))

    args.temporal_min_stem_gap = int(
        args.temporal_min_stem_gap if args.temporal_min_stem_gap is not None else sc.get("temporal_min_stem_gap", 5)
    )
    args.seed = int(args.seed if args.seed is not None else sc.get("seed", 42))
    args.out_txt = str(args.out_txt if args.out_txt is not None else sc.get("out_txt", "selected_frames_select.txt"))

    ckpt_list_yaml = sc.get("ckpt_list", None)
    ckpt_glob_yaml = sc.get("ckpt_glob", None)
    ckpt_list_use = args.ckpt_list if args.ckpt_list is not None else ckpt_list_yaml
    ckpt_glob_use = args.ckpt_glob if args.ckpt_glob is not None else ckpt_glob_yaml

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_paths = resolve_ckpt_list(
        primary_ckpt=args.ckpt,
        ckpt_list_csv=ckpt_list_use,
        ckpt_glob=ckpt_glob_use,
        ensemble_max_models=args.ensemble_max_models,
    )
    print(f"[SELECT] ensemble ckpts ({len(ckpt_paths)}):")
    for p in ckpt_paths:
        print(f"  - {p}")

    models: List[PRFNet] = []
    for p in ckpt_paths:
        m = build_model(cfg, device)
        load_ckpt_state(m, p, device=device, resume_ema=args.resume_ema)
        models.append(m)

    rv_proj = RangeImageProjector(
        H=dc["rv_H"],
        W=dc["rv_W"],
        fov_up=dc["fov_up"],
        fov_down=dc["fov_down"],
        R_max=dc.get("R_max", 85.0),
        use_surface_normals=dc.get("use_surface_normals", False),
        use_angle_encoding=dc.get("use_angle_encoding", False),
    )
    pb_proj = PolarBEVProjector(
        H_p=dc["pb_H"],
        W_p=dc["pb_W"],
        R_max=dc.get("R_max", 85.0),
    )

    seqs = parse_csv(args.seqs)
    frames = scan_frames(dc["root"], seqs)
    if len(frames) == 0:
        raise RuntimeError(f"no frames found under root={dc['root']} seqs={seqs}")

    print(f"[SELECT] frames={len(frames)}  device={device}")
    unc_vecs = []
    embs = []
    soft_hists = []

    for i, fr in enumerate(frames, 1):
        pts = preprocess_points(
            fr["velo"],
            r_min=args.r_min,
            r_max=args.r_max,
            max_points=args.max_points,
            rng=rng,
        )
        unc_vec, emb, soft_hist = frame_forward_ensemble(
            models=models,
            pts=pts,
            rv_proj=rv_proj,
            pb_proj=pb_proj,
            device=device,
            mc_passes=args.mc_passes,
            uncertainty_use_entropy=args.uncertainty_use_entropy,
            uncertainty_use_mi=args.uncertainty_use_mi,
            uncertainty_use_vr=args.uncertainty_use_vr,
        )
        unc_vecs.append(unc_vec)
        embs.append(emb)
        soft_hists.append(soft_hist)
        if i % 100 == 0:
            print(f"[SELECT] processed {i}/{len(frames)}")

    unc_arr = np.stack(unc_vecs, axis=0).astype(np.float32)  # (N,3)
    emb = np.stack(embs, axis=0).astype(np.float32)          # (N,D)
    soft_hist = np.stack(soft_hists, axis=0).astype(np.float32)  # (N,C)

    # uncertainty score
    mi_n = normalize01(unc_arr[:, 0])
    ent_n = normalize01(unc_arr[:, 1])
    vr_n = normalize01(unc_arr[:, 2])
    unc_mix = args.w_unc_mi * mi_n + args.w_unc_entropy * ent_n + args.w_unc_vr * vr_n
    unc_mix = normalize01(unc_mix)

    # class-balance score (soft)
    cls_freq = soft_hist.mean(axis=0)
    cls_w = 1.0 / np.sqrt(np.clip(cls_freq, 1e-6, None))
    cls_w = cls_w / np.mean(cls_w)
    bal_raw = (soft_hist * cls_w[None, :]).sum(axis=1)
    bal_n = normalize01(bal_raw)

    info = args.w_unc * unc_mix + args.w_bal * bal_n

    # Candidate-A: informativeness top-k
    n_total = len(frames)
    cand_k_info = int(max(args.candidate_pool_min, round(n_total * args.candidate_pool_ratio)))
    cand_k_info = min(cand_k_info, n_total)
    cand_info = np.argsort(-info)[:cand_k_info]

    # Candidate-B: representativeness (farthest-point)
    cand_k_rep = int(max(args.candidate_rep_min, round(n_total * args.candidate_rep_ratio)))
    cand_k_rep = min(cand_k_rep, n_total)
    cand_rep = farthest_point_indices(emb, k=cand_k_rep, seed=args.seed)

    # union
    cand_idx = np.unique(np.concatenate([cand_info, cand_rep], axis=0)).astype(np.int64)

    emb_c = emb[cand_idx]
    info_c = info[cand_idx]
    bal_c = bal_n[cand_idx]

    sim = cosine_sim_matrix(emb_c)
    picked_local = greedy_select_submodular(
        sim=sim,
        info_score=info_c,
        bal_score=bal_c,
        budget=args.budget,
        w_info=1.0,
        w_rep=args.w_rep,
        # bal 已在 info_score 中体现，贪心阶段不重复加权
        w_bal=0.0,
        cand_global_idx=cand_idx,
        all_frames=frames,
        temporal_min_stem_gap=args.temporal_min_stem_gap,
    )
    picked_global = cand_idx[np.asarray(picked_local, dtype=np.int64)]

    with open(args.out_txt, "w", encoding="utf-8") as f:
        for idx in picked_global.tolist():
            fr = frames[int(idx)]
            f.write(f"{fr['seq']}/{fr['stem']}\n")

    print(f"[SELECT] candidate_info={len(cand_info)} candidate_rep={len(cand_rep)} union={len(cand_idx)}")
    print(f"[SELECT] selected={len(picked_global)} -> {args.out_txt}")


if __name__ == "__main__":
    main()
