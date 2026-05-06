"""
NOMAE + PCP-MAE pretraining script for SemanticKITTI on PRFNet.

Usage:
    python tools/pretrain_nomae_pcp.py \
        --cfg prfnet/configs/pretrain_nomae_pcp_semantickitti.yaml
"""

import argparse
import logging
import os
import shutil
import time
from datetime import datetime
import sys
from collections import deque
from typing import Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from prfnet.datasets.semantickitti import SemanticKITTIDataset, collate_fn
from prfnet.models.prfnet import PRFNet
from prfnet.utils.loss import NOMAEPCPLoss


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "pretrain.log")

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("prfnet_pretrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def _linear_mask_ratio(
    global_step_1b: int,
    total_steps: int,
    warmup_ratio: float,
    ratio_start: float,
    ratio_end: float,
) -> float:
    if total_steps <= 0:
        return float(ratio_end)
    warmup_steps = int(max(0, min(total_steps, round(total_steps * float(warmup_ratio)))))
    if global_step_1b <= warmup_steps:
        return float(ratio_start)
    den = max(total_steps - warmup_steps, 1)
    alpha = min(max((global_step_1b - warmup_steps) / den, 0.0), 1.0)
    return float(ratio_start + alpha * (ratio_end - ratio_start))


def _ma(values: deque) -> float:
    if len(values) == 0:
        return 0.0
    return float(sum(values) / float(len(values)))


def run_linear_probe_eval(
    model: PRFNet,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    num_classes: int = 19,
    ignore_index: int = 255,
    train_steps: int = 80,
    eval_steps: int = 20,
    lr: float = 1.0e-2,
    weight_decay: float = 0.0,
) -> float:
    """
    Lightweight proxy evaluation for pretraining quality:
    freeze backbone features, train a linear head shortly, report mIoU.
    """
    was_training = model.training
    model.eval()

    probe_head = None
    opt = None
    train_iters = 0
    eval_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in train_loader:
        rv_img = batch["rv_img"].to(device, non_blocking=True)
        pb_img = batch["pb_img"].to(device, non_blocking=True)
        rv_coords = batch["rv_coords"].to(device, non_blocking=True)
        pb_coords = batch["pb_coords"].to(device, non_blocking=True)
        points = batch["points"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        feat = model.extract_pretrain_point_features(rv_img, pb_img, rv_coords, pb_coords, points)
        B, N, D = feat.shape
        feat_flat = feat.reshape(B * N, D)
        labels_flat = labels.reshape(B * N)
        valid = labels_flat != ignore_index
        if valid.sum().item() <= 0:
            continue

        if probe_head is None:
            probe_head = nn.Linear(D, num_classes).to(device)
            opt = torch.optim.AdamW(probe_head.parameters(), lr=lr, weight_decay=weight_decay)

        if train_iters >= train_steps:
            break
        probe_head.train()
        logits = probe_head(feat_flat[valid])
        loss = nn.functional.cross_entropy(logits, labels_flat[valid])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_iters += 1

    if probe_head is None:
        if was_training:
            model.train()
        return 0.0

    for batch in eval_loader:
        if eval_steps <= 0:
            break
        rv_img = batch["rv_img"].to(device, non_blocking=True)
        pb_img = batch["pb_img"].to(device, non_blocking=True)
        rv_coords = batch["rv_coords"].to(device, non_blocking=True)
        pb_coords = batch["pb_coords"].to(device, non_blocking=True)
        points = batch["points"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        feat = model.extract_pretrain_point_features(rv_img, pb_img, rv_coords, pb_coords, points)
        B, N, _ = feat.shape
        feat_flat = feat.reshape(B * N, -1)
        labels_flat = labels.reshape(B * N)
        valid = labels_flat != ignore_index
        if valid.sum().item() <= 0:
            continue

        probe_head.eval()
        logits = probe_head(feat_flat)
        preds = logits.argmax(dim=-1)
        p = preds[valid].detach().cpu().numpy().astype(np.int64)
        g = labels_flat[valid].detach().cpu().numpy().astype(np.int64)
        idx = g * num_classes + p
        eval_cm += np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        eval_steps -= 1

    if was_training:
        model.train()
    inter = np.diag(eval_cm).astype(np.float64)
    union = eval_cm.sum(axis=1) + eval_cm.sum(axis=0) - np.diag(eval_cm)
    valid_cls = union > 0
    iou = np.where(valid_cls, inter / (union + 1e-10), np.nan)
    return float(np.nanmean(iou) * 100.0) if np.any(valid_cls) else 0.0


def build_probe_loader(
    cfg_data: dict,
    split: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    ds = SemanticKITTIDataset(
        root=cfg_data["root"],
        split=split,
        rv_H=cfg_data["rv_H"],
        rv_W=cfg_data["rv_W"],
        pb_H=cfg_data["pb_H"],
        pb_W=cfg_data["pb_W"],
        augment=(split == "train"),
        rotate=cfg_data.get("rotate", True),
        flip=cfg_data.get("flip", True),
        scale_min=cfg_data.get("scale_min", 0.95),
        scale_max=cfg_data.get("scale_max", 1.05),
        drop_p=cfg_data.get("drop_p", 0.05),
        R_max=cfg_data.get("R_max", 80.0),
        use_polarmix=(cfg_data.get("use_polarmix", False) if split == "train" else False),
        use_lasermix=(cfg_data.get("use_lasermix", False) if split == "train" else False),
        fov_up=cfg_data["fov_up"],
        fov_down=cfg_data["fov_down"],
        max_points=cfg_data.get("max_points", 131072),
        polarmix_p=cfg_data.get("polarmix_p", 0.0),
        polarmix_sectors=cfg_data.get("polarmix_sectors", 8),
        lasermix_p=cfg_data.get("lasermix_p", 0.0),
        use_surface_normals=cfg_data.get("use_surface_normals", False),
        use_angle_encoding=cfg_data.get("use_angle_encoding", False),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )


def build_model(cfg: dict, device: torch.device) -> PRFNet:
    dc = cfg["data"]
    mc = cfg["model"]
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
        ds_aaff_K=mc.get("ds_aaff_K", 4),
        head_dropout=mc.get("head_dropout", 0.1),
        use_vcg=mc.get("use_vcg", False),
        use_proto=mc.get("use_proto", False),
        proto_dim=mc.get("proto_dim", 64),
    ).to(device)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg",
        default="prfnet/configs/prfnet_semantickitti_unified.yaml",
        help="pretraining config path",
    )
    ap.add_argument("--dry-run", action="store_true", help="run one step and exit")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    dc = dict(cfg["data"])
    mc = dict(cfg["model"])
    pc = cfg["pretrain"]
    lg = cfg.get("pretrain_log", cfg.get("log", {"save_dir": "runs/prfnet_pretrain_nomae_pcp"}))

    # Optional overrides for pretraining-only experiment knobs.
    if "pretrain_data" in cfg:
        dc.update(cfg["pretrain_data"])
    if "pretrain_model" in cfg:
        mc.update(cfg["pretrain_model"])

    cfg_for_model = {"data": dc, "model": mc}

    seed = int(pc.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dry_run:
        # Reduce tensor sizes to keep dry-run fast and within GPU memory.
        dc["rv_H"] = min(int(dc["rv_H"]), 64)
        dc["rv_W"] = min(int(dc["rv_W"]), 256)
        dc["pb_H"] = min(int(dc["pb_H"]), 128)
        dc["pb_W"] = min(int(dc["pb_W"]), 256)
        dc["batch_size"] = min(int(dc.get("batch_size", 2)), 1)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(lg["save_dir"], ts)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = setup_logger(exp_dir)
    logger.info(f"Experiment dir: {exp_dir}")
    logger.info(f"Config: {args.cfg}")
    shutil.copy(args.cfg, os.path.join(exp_dir, "config.yaml"))
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, "tensorboard"), flush_secs=30)
    logger.info(f"Device: {device}")

    loader = None
    try:
        ds = SemanticKITTIDataset(
            root=dc["root"],
            split="train",
            rv_H=dc["rv_H"],
            rv_W=dc["rv_W"],
            pb_H=dc["pb_H"],
            pb_W=dc["pb_W"],
            augment=dc.get("augment", True),
            rotate=dc.get("rotate", True),
            flip=dc.get("flip", True),
            scale_min=dc.get("scale_min", 0.95),
            scale_max=dc.get("scale_max", 1.05),
            drop_p=dc.get("drop_p", 0.05),
            R_max=dc.get("R_max", 80.0),
            use_polarmix=dc.get("use_polarmix", False),
            use_lasermix=dc.get("use_lasermix", False),
            fov_up=dc["fov_up"],
            fov_down=dc["fov_down"],
            max_points=dc.get("max_points", 131072),
            polarmix_p=dc.get("polarmix_p", 0.0),
            polarmix_sectors=dc.get("polarmix_sectors", 8),
            lasermix_p=dc.get("lasermix_p", 0.0),
            use_surface_normals=dc.get("use_surface_normals", False),
            use_angle_encoding=dc.get("use_angle_encoding", False),
        )
        loader = DataLoader(
            ds,
            batch_size=dc.get("batch_size", 2),
            shuffle=True,
            num_workers=dc.get("num_workers", 8),
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
    except (FileNotFoundError, PermissionError) as e:
        if not args.dry_run:
            raise
        logger.warning(f"dataset unavailable for dry-run, fallback to synthetic: {e}")

    model = build_model(cfg_for_model, device)
    criterion = NOMAEPCPLoss(
        lambda_occ=pc.get("lambda_occ", 1.0),
        lambda_pcp=pc.get("lambda_pcp", 0.5),
        lambda_occ_rv=pc.get("lambda_occ_rv", None),
        lambda_occ_pb=pc.get("lambda_occ_pb", None),
        lambda_pcp_rv=pc.get("lambda_pcp_rv", None),
        lambda_pcp_pb=pc.get("lambda_pcp_pb", None),
        lambda_cv=float(pc.get("lambda_cv", 0.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pc.get("lr", 8e-4),
        weight_decay=pc.get("weight_decay", 1e-4),
    )
    total_steps = (len(loader) if loader is not None else 1) * int(pc.get("epochs", 10))
    sched_type = str(pc.get("scheduler", "cosine")).lower()
    amp_enabled_global = bool(pc.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled_global)

    # ── LR warmup (P2) ────────────────────────────────────────────────────────
    lr_warmup_ratio = float(pc.get("lr_warmup_ratio", 0.0))
    warmup_steps = int(total_steps * max(0.0, min(1.0, lr_warmup_ratio))) if sched_type != "onecycle" else 0
    if sched_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(pc.get("lr", 8e-4)),
            total_steps=max(total_steps, 1),
            pct_start=float(pc.get("pct_start", 0.1)),
        )
    elif warmup_steps > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0e-3,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=float(pc.get("eta_min", 1e-6)),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps],
        )
        logger.info(f"Scheduler: cosine+warmup({lr_warmup_ratio:.0%} = {warmup_steps} steps)")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps, 1), eta_min=pc.get("eta_min", 1e-6)
        )

    rv_mask_ratio = float(pc.get("rv_mask_ratio", 0.7))
    pb_mask_ratio = float(pc.get("pb_mask_ratio", 0.7))
    mask_curriculum_enable = bool(pc.get("mask_curriculum_enable", False))
    mask_curriculum_warmup_ratio = float(pc.get("mask_curriculum_warmup_ratio", 0.20))
    rv_mask_ratio_start = float(pc.get("rv_mask_ratio_start", rv_mask_ratio))
    rv_mask_ratio_end = float(pc.get("rv_mask_ratio_end", rv_mask_ratio))
    pb_mask_ratio_start = float(pc.get("pb_mask_ratio_start", pb_mask_ratio))
    pb_mask_ratio_end = float(pc.get("pb_mask_ratio_end", pb_mask_ratio))
    input_masking_enable = bool(pc.get("input_masking_enable", True))
    input_masking_mode = str(pc.get("input_masking_mode", "zero"))
    occ_scales = [int(x) for x in pc.get("occ_scales", [1, 3, 5])]
    occ_loss_type = str(pc.get("occ_loss_type", "bce_pos_weight"))
    occ_pos_weight = float(pc.get("occ_pos_weight", 5.0))
    occ_focal_gamma = float(pc.get("occ_focal_gamma", 2.0))
    pcp_stopgrad_replace = bool(pc.get("pcp_stopgrad_replace", True))
    informative_occ_only = bool(pc.get("informative_occ_only", True))
    mask_pos_ratio_control_enable = bool(pc.get("mask_pos_ratio_control_enable", False))
    mask_pos_ratio_min = float(pc.get("mask_pos_ratio_min", 0.08))
    mask_pos_ratio_max = float(pc.get("mask_pos_ratio_max", 0.50))
    mask_resample_max_tries = int(pc.get("mask_resample_max_tries", 5))
    mask_strategy = str(pc.get("mask_strategy", "mixed"))
    rv_band_axis = str(pc.get("rv_band_axis", "row"))
    pb_band_axis = str(pc.get("pb_band_axis", "col"))
    mask_mix_random = float(pc.get("mask_mix_random", 0.5))
    mask_mix_block = float(pc.get("mask_mix_block", 0.3))
    mask_mix_band = float(pc.get("mask_mix_band", 0.2))
    mask_mix_hmg = float(pc.get("mask_mix_hmg", 0.0))
    block_h_min = int(pc.get("block_h_min", 4))
    block_h_max = int(pc.get("block_h_max", 16))
    block_w_min = int(pc.get("block_w_min", 16))
    block_w_max = int(pc.get("block_w_max", 64))
    hmg_coarse_stride = int(pc.get("hmg_coarse_stride", 8))
    hmg_fine_extra_ratio = float(pc.get("hmg_fine_extra_ratio", 0.05))
    pcp_informative_only = bool(pc.get("pcp_informative_only", True))
    occ_pos_weight_adaptive = bool(pc.get("occ_pos_weight_adaptive", False))
    occ_pos_weight_min = float(pc.get("occ_pos_weight_min", 1.0))
    occ_pos_weight_max = float(pc.get("occ_pos_weight_max", 12.0))
    neighbor_sup_only_visible = bool(pc.get("neighbor_sup_only_visible", True))
    pcp_residual_center = bool(pc.get("pcp_residual_center", True))
    pcp_far_only = bool(pc.get("pcp_far_only", False))
    cross_view_consistency_enable = bool(pc.get("cross_view_consistency_enable", False))
    cv_stop_grad = bool(pc.get("cv_stop_grad", False))
    cv_only_visible = bool(pc.get("cv_only_visible", True))

    early_stop_enable = bool(pc.get("early_stop_enable", False))
    early_stop_mode = str(pc.get("early_stop_mode", "relative")).lower()
    early_stop_window = int(pc.get("early_stop_window", 200))
    early_stop_occ_thr = float(pc.get("early_stop_occ_thr", 1.0e-3))
    early_stop_pcp_thr = float(pc.get("early_stop_pcp_thr", 1.0e-3))
    early_stop_gnorm_thr = float(pc.get("early_stop_gnorm_thr", 0.05))
    early_stop_patience_steps = int(pc.get("early_stop_patience_steps", 600))
    early_stop_min_steps = int(pc.get("early_stop_min_steps", 2000))
    early_stop_rel_tol = float(pc.get("early_stop_rel_tol", 1.0e-3))
    early_stop_rel_window = int(pc.get("early_stop_rel_window", 200))
    early_stop_counter = 0
    ma_occ = deque(maxlen=max(1, early_stop_window))
    ma_pcp = deque(maxlen=max(1, early_stop_window))
    ma_gn = deque(maxlen=max(1, early_stop_window))
    occ_rel_hist = deque(maxlen=max(2, early_stop_rel_window + 1))
    pcp_rel_hist = deque(maxlen=max(2, early_stop_rel_window + 1))  # P2: track PCP relative improvement

    logger.info(
        f"Pretrain knobs: input_masking={input_masking_enable}/{input_masking_mode}, "
        f"occ_loss={occ_loss_type}, occ_scales={occ_scales}, "
        f"mask_strategy={mask_strategy}, "
        f"neighbor_sup_only_visible={neighbor_sup_only_visible}, "
        f"pcp_residual_center={pcp_residual_center}, pcp_far_only={pcp_far_only}, "
        f"pos_ratio_control={mask_pos_ratio_control_enable}, "
        f"cross_view_consistency={cross_view_consistency_enable}"
    )
    if mask_curriculum_enable:
        logger.info(
            f"Mask curriculum enabled: warmup={mask_curriculum_warmup_ratio}, "
            f"rv {rv_mask_ratio_start:.2f}->{rv_mask_ratio_end:.2f}, "
            f"pb {pb_mask_ratio_start:.2f}->{pb_mask_ratio_end:.2f}"
        )
    if early_stop_enable:
        logger.info(
            f"Early-stop enabled: mode={early_stop_mode}, "
            f"window={early_stop_window}, patience={early_stop_patience_steps} steps"
        )

    log_interval = int(pc.get("log_interval", 20))
    grad_clip = float(pc.get("grad_clip", 10.0))
    epochs = int(pc.get("epochs", 10))
    probe_eval_enable = bool(pc.get("probe_eval_enable", False))
    probe_eval_every = int(pc.get("probe_eval_every", 1))
    probe_train_steps = int(pc.get("probe_train_steps", 80))
    probe_eval_steps = int(pc.get("probe_eval_steps", 20))
    probe_lr = float(pc.get("probe_lr", 1.0e-2))
    probe_weight_decay = float(pc.get("probe_weight_decay", 0.0))
    probe_batch_size = int(pc.get("probe_batch_size", dc.get("batch_size", 2)))
    probe_num_workers = int(pc.get("probe_num_workers", dc.get("num_workers", 8)))
    probe_patience_epochs = int(pc.get("probe_patience_epochs", 0))   # 0 = disabled
    best_probe_miou = -1.0
    probe_no_improve_count = 0

    probe_train_loader = None
    probe_eval_loader = None
    if probe_eval_enable and loader is not None and not args.dry_run:
        probe_train_loader = build_probe_loader(
            dc, split="train", batch_size=probe_batch_size, num_workers=probe_num_workers
        )
        probe_eval_loader = build_probe_loader(
            dc, split="val", batch_size=probe_batch_size, num_workers=probe_num_workers
        )

    global_step = 0
    overflow_steps = 0
    skipped_sched_steps = 0
    model.train()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        ep_total = ep_occ = ep_pcp = 0.0
        ep_step_time = 0.0
        ep_grad_norm = 0.0
        ep_rv_occ = ep_pb_occ = 0.0
        ep_mask_rv = ep_mask_pb = 0.0
        _iter_t_prev = time.time()
        if loader is None:
            rv_in = 6 + 3 * int(dc.get("use_surface_normals", False)) + 3 * int(
                dc.get("use_angle_encoding", False)
            )
            B = int(dc.get("batch_size", 2))
            rv_img = torch.randn(B, rv_in, dc["rv_H"], dc["rv_W"], device=device)
            pb_img = torch.randn(B, 9, dc["pb_H"], dc["pb_W"], device=device)
            # keep occupancy targets meaningful
            rv_img[:, 3:4] = torch.sigmoid(rv_img[:, 3:4])
            pb_img[:, 8:9] = torch.sigmoid(pb_img[:, 8:9])
            iters = [(1, {"rv_img": rv_img, "pb_img": pb_img})]
        else:
            iters = enumerate(loader, 1)

        for step, batch in iters:
            global_step += 1
            _iter_t_now = time.time()
            data_dt = _iter_t_now - _iter_t_prev
            rv_img = batch["rv_img"].to(device, non_blocking=True)
            pb_img = batch["pb_img"].to(device, non_blocking=True)
            labels_cpu = batch.get("labels", None)
            _step_t0 = time.time()

            optimizer.zero_grad(set_to_none=True)
            if mask_curriculum_enable:
                cur_rv_mask_ratio = _linear_mask_ratio(
                    global_step, max(total_steps, 1),
                    mask_curriculum_warmup_ratio,
                    rv_mask_ratio_start, rv_mask_ratio_end,
                )
                cur_pb_mask_ratio = _linear_mask_ratio(
                    global_step, max(total_steps, 1),
                    mask_curriculum_warmup_ratio,
                    pb_mask_ratio_start, pb_mask_ratio_end,
                )
            else:
                cur_rv_mask_ratio = rv_mask_ratio
                cur_pb_mask_ratio = pb_mask_ratio

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled_global):
                outputs = model.forward_pretrain(
                    rv_img=rv_img,
                    pb_img=pb_img,
                    rv_mask_ratio=cur_rv_mask_ratio,
                    pb_mask_ratio=cur_pb_mask_ratio,
                    input_masking_enable=input_masking_enable,
                    input_masking_mode=input_masking_mode,
                    mask_strategy=mask_strategy,
                    rv_band_axis=rv_band_axis,
                    pb_band_axis=pb_band_axis,
                    mask_mix_random=mask_mix_random,
                    mask_mix_block=mask_mix_block,
                    mask_mix_band=mask_mix_band,
                    block_h_min=block_h_min,
                    block_h_max=block_h_max,
                    block_w_min=block_w_min,
                    block_w_max=block_w_max,
                    mask_pos_ratio_control_enable=mask_pos_ratio_control_enable,
                    mask_pos_ratio_min=mask_pos_ratio_min,
                    mask_pos_ratio_max=mask_pos_ratio_max,
                    mask_resample_max_tries=mask_resample_max_tries,
                    occ_scales=occ_scales,
                    occ_loss_type=occ_loss_type,
                    occ_pos_weight=occ_pos_weight,
                    occ_pos_weight_adaptive=occ_pos_weight_adaptive,
                    occ_pos_weight_min=occ_pos_weight_min,
                    occ_pos_weight_max=occ_pos_weight_max,
                    occ_pos_weight_ema_decay=float(pc.get("occ_pos_weight_ema_decay", 0.95)),
                    occ_focal_gamma=occ_focal_gamma,
                    pcp_stopgrad_replace=pcp_stopgrad_replace,
                    informative_occ_only=informative_occ_only,
                    pcp_informative_only=pcp_informative_only,
                    pcp_pos_weight=float(pc.get("pcp_pos_weight", 1.0)),
                    pcp_near_range_max=float(pc.get("pcp_near_range_max", 10.0)),
                    pcp_near_weight=float(pc.get("pcp_near_weight", 1.5)),
                    mask_mix_hmg=mask_mix_hmg,
                    hmg_coarse_stride=hmg_coarse_stride,
                    hmg_fine_extra_ratio=hmg_fine_extra_ratio,
                    neighbor_sup_only_visible=neighbor_sup_only_visible,
                    pcp_residual_center=pcp_residual_center,
                    pcp_far_only=pcp_far_only,
                    cross_view_consistency_enable=cross_view_consistency_enable,
                    cv_stop_grad=cv_stop_grad,
                    cv_only_visible=cv_only_visible,
                )
                loss_dict = criterion(outputs)
                loss = loss_dict["total"]

            if amp_enabled_global:
                scale_before = float(scaler.get_scale())
            else:
                scale_before = 1.0

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            grad_norm_val = float(grad_norm.detach().cpu().item()
                                  if torch.is_tensor(grad_norm) else grad_norm)
            grad_is_finite = math.isfinite(grad_norm_val)
            if not grad_is_finite:
                overflow_steps += 1

            scaler.step(optimizer)
            scaler.update()

            if amp_enabled_global:
                # AMP overflow 时 optimizer.step 会被跳过，scale 会下降。
                scale_after = float(scaler.get_scale())
                optimizer_stepped = scale_after >= scale_before
            else:
                optimizer_stepped = True

            if optimizer_stepped:
                scheduler.step()
            else:
                skipped_sched_steps += 1

            ep_total += float(loss.item())
            ep_occ += float(loss_dict["occ"].item())
            ep_pcp += float(loss_dict["pcp"].item())
            ep_step_time += (time.time() - _step_t0)
            if grad_is_finite:
                ep_grad_norm += grad_norm_val
            ep_rv_occ += float((rv_img[:, 3:4] > 0).float().mean().item())
            ep_pb_occ += float(pb_img[:, 8:9].float().mean().item())
            ep_mask_rv += float(outputs["rv_mask_ratio"].item())
            ep_mask_pb += float(outputs["pb_mask_ratio"].item())
            rv_masked_pos_ratio = float(outputs["rv_masked_pos_ratio"].item())
            pb_masked_pos_ratio = float(outputs["pb_masked_pos_ratio"].item())
            rv_occ_effective_ratio = float(outputs["rv_occ_effective_ratio"].item())
            pb_occ_effective_ratio = float(outputs["pb_occ_effective_ratio"].item())
            rv_mask_resample = float(outputs["rv_mask_resample"].item())
            pb_mask_resample = float(outputs["pb_mask_resample"].item())

            ma_occ.append(float(loss_dict["occ"].item()))
            ma_pcp.append(float(loss_dict["pcp"].item()))
            if grad_is_finite:
                ma_gn.append(grad_norm_val)
            occ_ma = _ma(ma_occ)
            pcp_ma = _ma(ma_pcp)
            gn_ma = _ma(ma_gn) if len(ma_gn) > 0 else float("inf")
            occ_rel_hist.append(occ_ma)
            pcp_rel_hist.append(pcp_ma)

            if early_stop_enable and len(ma_occ) >= max(1, early_stop_window):
                if global_step < max(0, early_stop_min_steps):
                    early_stop_counter = 0
                else:
                    if early_stop_mode == "absolute":
                        cond = (
                            occ_ma < early_stop_occ_thr
                            and pcp_ma < early_stop_pcp_thr
                            and gn_ma < early_stop_gnorm_thr
                        )
                    else:
                        cond = False
                        if len(occ_rel_hist) >= occ_rel_hist.maxlen and len(pcp_rel_hist) >= pcp_rel_hist.maxlen:
                            occ_old = occ_rel_hist[0]; occ_new = occ_rel_hist[-1]
                            pcp_old = pcp_rel_hist[0]; pcp_new = pcp_rel_hist[-1]
                            occ_rel_improve = (occ_old - occ_new) / max(abs(occ_old), 1e-8)
                            pcp_rel_improve = (pcp_old - pcp_new) / max(abs(pcp_old), 1e-8)
                            cond = (
                                occ_rel_improve < early_stop_rel_tol
                                and pcp_rel_improve < early_stop_rel_tol
                                and gn_ma < early_stop_gnorm_thr
                            )
                    if cond:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
            else:
                early_stop_counter = 0

            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("pretrain/loss_step", ep_total / step, global_step)
            writer.add_scalar("pretrain/occ_step", loss_dict["occ"], global_step)
            writer.add_scalar("pretrain/pcp_step", loss_dict["pcp"], global_step)
            if "occ_rv" in loss_dict:
                writer.add_scalar("pretrain/occ_rv_step", loss_dict["occ_rv"], global_step)
            if "occ_pb" in loss_dict:
                writer.add_scalar("pretrain/occ_pb_step", loss_dict["occ_pb"], global_step)
            if "pcp_rv" in loss_dict:
                writer.add_scalar("pretrain/pcp_rv_step", loss_dict["pcp_rv"], global_step)
            if "pcp_pb" in loss_dict:
                writer.add_scalar("pretrain/pcp_pb_step", loss_dict["pcp_pb"], global_step)
            if "cv" in loss_dict:
                writer.add_scalar("pretrain/cv_step", loss_dict["cv"], global_step)
            writer.add_scalar("pretrain/lr", lr, global_step)
            writer.add_scalar("pretrain/rv_mask_ratio_step", outputs["rv_mask_ratio"], global_step)
            writer.add_scalar("pretrain/pb_mask_ratio_step", outputs["pb_mask_ratio"], global_step)
            writer.add_scalar("pretrain/occ_ma", occ_ma, global_step)
            writer.add_scalar("pretrain/pcp_ma", pcp_ma, global_step)
            writer.add_scalar("diag/gnorm_ma", gn_ma, global_step)
            writer.add_scalar("diag/grad_norm_step", grad_norm_val, global_step)
            writer.add_scalar("diag/grad_norm_finite_step", 1.0 if grad_is_finite else 0.0, global_step)
            writer.add_scalar("diag/data_time_s_step", data_dt, global_step)
            writer.add_scalar("diag/step_time_s_step", time.time() - _step_t0, global_step)
            writer.add_scalar("diag/rv_occ_step", float((rv_img[:, 3:4] > 0).float().mean().item()), global_step)
            writer.add_scalar("diag/pb_occ_step", float(pb_img[:, 8:9].float().mean().item()), global_step)
            writer.add_scalar("diag/masked_positive_ratio_rv_step", rv_masked_pos_ratio, global_step)
            writer.add_scalar("diag/masked_positive_ratio_pb_step", pb_masked_pos_ratio, global_step)
            writer.add_scalar("diag/occ_effective_ratio_rv_step", rv_occ_effective_ratio, global_step)
            writer.add_scalar("diag/occ_effective_ratio_pb_step", pb_occ_effective_ratio, global_step)
            writer.add_scalar("diag/mask_resample_rv_step", rv_mask_resample, global_step)
            writer.add_scalar("diag/mask_resample_pb_step", pb_mask_resample, global_step)
            writer.add_scalar("diag/early_stop_counter_step", early_stop_counter, global_step)
            writer.add_scalar("diag/overflow_steps_step", overflow_steps, global_step)
            writer.add_scalar("diag/skipped_sched_steps_step", skipped_sched_steps, global_step)
            if labels_cpu is not None:
                valid = labels_cpu != 255
                writer.add_scalar(
                    "diag/points_valid_ratio_step",
                    float(valid.sum().item()) / max(float(labels_cpu.numel()), 1.0),
                    global_step,
                )
            if device.type == "cuda":
                writer.add_scalar("diag/gpu_mem_alloc_mb_step",
                                  torch.cuda.memory_allocated(device) / 1024.0 / 1024.0,
                                  global_step)
                writer.add_scalar("diag/gpu_mem_reserved_mb_step",
                                  torch.cuda.memory_reserved(device) / 1024.0 / 1024.0,
                                  global_step)

            if step % log_interval == 0:
                logger.info(
                    f"[pretrain] ep={epoch:03d}/{epochs} step={step:04d}/{(len(loader) if loader is not None else 1)} "
                    f"loss={ep_total/step:.4f} occ={loss_dict['occ']:.4f} "
                    f"pcp={loss_dict['pcp']:.4f} lr={lr:.2e} "
                    f"gnorm={grad_norm_val:.2f} "
                    f"occ_ma={occ_ma:.4e} pcp_ma={pcp_ma:.4e} gnorm_ma={gn_ma:.4f} "
                    f"overflow={overflow_steps} sched_skip={skipped_sched_steps}",
                )

            if early_stop_enable and early_stop_counter >= max(1, early_stop_patience_steps):
                logger.info(
                    f"[pretrain] early-stop triggered at global_step={global_step}: "
                    f"occ_ma={occ_ma:.4e}, pcp_ma={pcp_ma:.4e}, gnorm_ma={gn_ma:.4f}"
                )
                ckpt = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                    "early_stop": True,
                    "global_step": global_step,
                }
                torch.save(ckpt, os.path.join(ckpt_dir, "early_stop.pth"))
                torch.save(ckpt, os.path.join(ckpt_dir, "latest.pth"))
                writer.close()
                logger.info(f"done. outputs: {exp_dir}")
                return

            if args.dry_run:
                break
            _iter_t_prev = time.time()

        n = max(step, 1)
        dt = time.time() - t0
        logger.info(
            f"[pretrain] epoch={epoch:03d} loss={ep_total/n:.4f} "
            f"occ={ep_occ/n:.4f} pcp={ep_pcp/n:.4f} time={dt:.0f}s"
        )
        writer.add_scalar("pretrain/loss_epoch", ep_total / n, epoch)
        writer.add_scalar("pretrain/occ_epoch", ep_occ / n, epoch)
        writer.add_scalar("pretrain/pcp_epoch", ep_pcp / n, epoch)
        writer.add_scalar("pretrain/rv_mask_ratio_epoch", ep_mask_rv / n, epoch)
        writer.add_scalar("pretrain/pb_mask_ratio_epoch", ep_mask_pb / n, epoch)
        writer.add_scalar("diag/grad_norm_epoch", ep_grad_norm / max(1, len(ma_gn)), epoch)
        writer.add_scalar("diag/overflow_steps_epoch", overflow_steps, epoch)
        writer.add_scalar("diag/skipped_sched_steps_epoch", skipped_sched_steps, epoch)
        writer.add_scalar("diag/rv_occ_epoch", ep_rv_occ / n, epoch)
        writer.add_scalar("diag/pb_occ_epoch", ep_pb_occ / n, epoch)
        writer.add_scalar("diag/step_time_s_epoch", ep_step_time / n, epoch)
        if device.type == "cuda":
            writer.add_scalar("diag/gpu_mem_alloc_mb_epoch",
                              torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0,
                              epoch)
            writer.add_scalar("diag/gpu_mem_reserved_mb_epoch",
                              torch.cuda.max_memory_reserved(device) / 1024.0 / 1024.0,
                              epoch)
            torch.cuda.reset_peak_memory_stats(device)

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, "latest.pth"))
        if not args.dry_run:
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch{epoch:03d}.pth"))

        if (
            probe_eval_enable
            and probe_train_loader is not None
            and probe_eval_loader is not None
            and (epoch % max(1, probe_eval_every) == 0)
            and not args.dry_run
        ):
            probe_miou = run_linear_probe_eval(
                model=model,
                train_loader=probe_train_loader,
                eval_loader=probe_eval_loader,
                device=device,
                num_classes=19,
                ignore_index=255,
                train_steps=probe_train_steps,
                eval_steps=probe_eval_steps,
                lr=probe_lr,
                weight_decay=probe_weight_decay,
            )
            writer.add_scalar("probe/miou", probe_miou, epoch)
            logger.info(f"[probe] epoch={epoch:03d} proxy_mIoU={probe_miou:.2f}%")
            if probe_miou > best_probe_miou:
                best_probe_miou = probe_miou
                probe_no_improve_count = 0
                torch.save(ckpt, os.path.join(ckpt_dir, "best_pretrain.pth"))
                logger.info(f"[probe] new best_pretrain saved: proxy_mIoU={probe_miou:.2f}%")
            else:
                probe_no_improve_count += 1
                logger.info(f"[probe] no improvement ({probe_no_improve_count}/{probe_patience_epochs})")
            if probe_patience_epochs > 0 and probe_no_improve_count >= probe_patience_epochs:
                logger.info(
                    f"[probe] patience exhausted after {probe_no_improve_count} epochs, "
                    f"stopping pretrain. best_mIoU={best_probe_miou:.2f}%"
                )
                writer.close()
                logger.info(f"done. outputs: {exp_dir}")
                return

        if args.dry_run:
            break

    writer.close()
    logger.info(f"done. outputs: {exp_dir}")


if __name__ == "__main__":
    main()
