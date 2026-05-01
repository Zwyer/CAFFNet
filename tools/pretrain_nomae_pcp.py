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
    if args.dry_run and device.type != "cuda":
        # Keep memory safe for CPU-only CI/sandbox dry tests.
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
        )
    except (FileNotFoundError, PermissionError) as e:
        if not args.dry_run:
            raise
        logger.warning(f"dataset unavailable for dry-run, fallback to synthetic: {e}")

    model = build_model(cfg_for_model, device)
    criterion = NOMAEPCPLoss(
        lambda_occ=pc.get("lambda_occ", 1.0), lambda_pcp=pc.get("lambda_pcp", 0.5)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pc.get("lr", 8e-4),
        weight_decay=pc.get("weight_decay", 1e-4),
    )
    total_steps = (len(loader) if loader is not None else 1) * int(pc.get("epochs", 10))
    sched_type = str(pc.get("scheduler", "cosine")).lower()
    if sched_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(pc.get("lr", 8e-4)),
            total_steps=max(total_steps, 1),
            pct_start=float(pc.get("pct_start", 0.1)),
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps, 1), eta_min=pc.get("eta_min", 1e-6)
        )
    logger.info(f"Scheduler: {sched_type}")
    amp_enabled_global = bool(pc.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled_global)

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

    early_stop_enable = bool(pc.get("early_stop_enable", False))
    early_stop_window = int(pc.get("early_stop_window", 200))
    early_stop_occ_thr = float(pc.get("early_stop_occ_thr", 1.0e-3))
    early_stop_pcp_thr = float(pc.get("early_stop_pcp_thr", 1.0e-3))
    early_stop_gnorm_thr = float(pc.get("early_stop_gnorm_thr", 0.05))
    early_stop_patience_steps = int(pc.get("early_stop_patience_steps", 600))
    early_stop_counter = 0
    ma_occ = deque(maxlen=max(1, early_stop_window))
    ma_pcp = deque(maxlen=max(1, early_stop_window))
    ma_gn = deque(maxlen=max(1, early_stop_window))

    logger.info(
        f"Pretrain knobs: input_masking={input_masking_enable}/{input_masking_mode}, "
        f"occ_loss={occ_loss_type}, occ_scales={occ_scales}, "
        f"pos_ratio_control={mask_pos_ratio_control_enable}"
    )
    if mask_curriculum_enable:
        logger.info(
            f"Mask curriculum enabled: warmup={mask_curriculum_warmup_ratio}, "
            f"rv {rv_mask_ratio_start:.2f}->{rv_mask_ratio_end:.2f}, "
            f"pb {pb_mask_ratio_start:.2f}->{pb_mask_ratio_end:.2f}"
        )
    if early_stop_enable:
        logger.info(
            f"Early-stop enabled: window={early_stop_window}, occ<{early_stop_occ_thr}, "
            f"pcp<{early_stop_pcp_thr}, gnorm<{early_stop_gnorm_thr}, "
            f"patience={early_stop_patience_steps} steps"
        )

    log_interval = int(pc.get("log_interval", 20))
    grad_clip = float(pc.get("grad_clip", 10.0))
    epochs = int(pc.get("epochs", 10))

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
                    mask_pos_ratio_control_enable=mask_pos_ratio_control_enable,
                    mask_pos_ratio_min=mask_pos_ratio_min,
                    mask_pos_ratio_max=mask_pos_ratio_max,
                    mask_resample_max_tries=mask_resample_max_tries,
                    occ_scales=occ_scales,
                    occ_loss_type=occ_loss_type,
                    occ_pos_weight=occ_pos_weight,
                    occ_focal_gamma=occ_focal_gamma,
                    pcp_stopgrad_replace=pcp_stopgrad_replace,
                    informative_occ_only=informative_occ_only,
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

            if early_stop_enable and len(ma_occ) >= max(1, early_stop_window):
                if occ_ma < early_stop_occ_thr and pcp_ma < early_stop_pcp_thr and gn_ma < early_stop_gnorm_thr:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
            else:
                early_stop_counter = 0

            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("pretrain/loss_step", ep_total / step, global_step)
            writer.add_scalar("pretrain/occ_step", loss_dict["occ"], global_step)
            writer.add_scalar("pretrain/pcp_step", loss_dict["pcp"], global_step)
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

        if args.dry_run:
            break

    writer.close()
    logger.info(f"done. outputs: {exp_dir}")


if __name__ == "__main__":
    main()
