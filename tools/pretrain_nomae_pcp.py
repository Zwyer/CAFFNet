"""
NOMAE + PCP-MAE pretraining script for SemanticKITTI on PRFNet.

Usage:
    python tools/pretrain_nomae_pcp.py \
        --cfg prfnet/configs/pretrain_nomae_pcp_semantickitti.yaml
"""

import argparse
import os
import shutil
import time
from datetime import datetime
import sys

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
        default="prfnet/configs/pretrain_nomae_pcp_semantickitti.yaml",
        help="pretraining config path",
    )
    ap.add_argument("--dry-run", action="store_true", help="run one step and exit")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    dc = cfg["data"]
    pc = cfg["pretrain"]
    lg = cfg["log"]

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
    shutil.copy(args.cfg, os.path.join(exp_dir, "config.yaml"))
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, "tensorboard"), flush_secs=30)

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
        print(f"[warn] dataset unavailable for dry-run, fallback to synthetic: {e}", flush=True)

    model = build_model(cfg, device)
    criterion = NOMAEPCPLoss(
        lambda_occ=pc.get("lambda_occ", 1.0), lambda_pcp=pc.get("lambda_pcp", 0.5)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pc.get("lr", 8e-4),
        weight_decay=pc.get("weight_decay", 1e-4),
    )
    total_steps = (len(loader) if loader is not None else 1) * int(pc.get("epochs", 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=pc.get("eta_min", 1e-6)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=pc.get("amp", True))

    rv_mask_ratio = float(pc.get("rv_mask_ratio", 0.7))
    pb_mask_ratio = float(pc.get("pb_mask_ratio", 0.7))
    log_interval = int(pc.get("log_interval", 20))
    grad_clip = float(pc.get("grad_clip", 10.0))
    epochs = int(pc.get("epochs", 10))

    global_step = 0
    model.train()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        ep_total = ep_occ = ep_pcp = 0.0
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
            rv_img = batch["rv_img"].to(device, non_blocking=True)
            pb_img = batch["pb_img"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            amp_enabled = bool(pc.get("amp", True)) and device.type == "cuda"
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model.forward_pretrain(
                    rv_img=rv_img,
                    pb_img=pb_img,
                    rv_mask_ratio=rv_mask_ratio,
                    pb_mask_ratio=pb_mask_ratio,
                )
                loss_dict = criterion(outputs)
                loss = loss_dict["total"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ep_total += float(loss.item())
            ep_occ += float(loss_dict["occ"].item())
            ep_pcp += float(loss_dict["pcp"].item())

            if step % log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[pretrain] ep={epoch:03d}/{epochs} step={step:04d}/{(len(loader) if loader is not None else 1)} "
                    f"loss={ep_total/step:.4f} occ={loss_dict['occ']:.4f} "
                    f"pcp={loss_dict['pcp']:.4f} lr={lr:.2e}",
                    flush=True,
                )
                writer.add_scalar("pretrain/loss_step", ep_total / step, global_step)
                writer.add_scalar("pretrain/occ_step", loss_dict["occ"], global_step)
                writer.add_scalar("pretrain/pcp_step", loss_dict["pcp"], global_step)
                writer.add_scalar("pretrain/lr", lr, global_step)

            if args.dry_run:
                break

        n = max(step, 1)
        dt = time.time() - t0
        print(
            f"[pretrain] epoch={epoch:03d} loss={ep_total/n:.4f} "
            f"occ={ep_occ/n:.4f} pcp={ep_pcp/n:.4f} time={dt:.0f}s",
            flush=True,
        )
        writer.add_scalar("pretrain/loss_epoch", ep_total / n, epoch)
        writer.add_scalar("pretrain/occ_epoch", ep_occ / n, epoch)
        writer.add_scalar("pretrain/pcp_epoch", ep_pcp / n, epoch)

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
    print(f"done. outputs: {exp_dir}")


if __name__ == "__main__":
    main()
