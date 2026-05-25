#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/deploy_rknn.py

Convert PRFNet checkpoint (.pth) + config (.yaml) to ONNX and RKNN artifacts.

Modes:
  1) full      : export full point-wise logits model (likely contains GridSample)
  2) backbone  : export NPU-friendly backbone/decoder subgraph (rv_out, pb_out)

Why two modes:
- RKNN Toolkit2 official OP list marks ONNX GridSample as unsupported.
- PRFNet full forward uses grid_sample inside PointSampleAggregator.
- For RK3588 deployment, recommended runtime is hybrid:
  NPU(RKNN backbone) + CPU(postprocess head).
"""

import argparse
import os
import sys
import platform
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

# Ensure project root is importable when running from tools/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prfnet.models.prfnet import PRFNet  # noqa: E402


def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict, device: torch.device) -> PRFNet:
    mc = cfg["model"]
    dc = cfg["data"]

    use_normals = bool(mc.get("use_surface_normals", False))
    use_angle = bool(mc.get("use_angle_encoding", False))
    use_pb_z_hist = bool(mc.get("use_pb_z_hist", False))

    rv_in = int(mc.get("rv_in", 6 + 3 * int(use_normals) + 3 * int(use_angle)))
    pb_in = int(mc.get("pb_in", 9 + 3 * int(use_pb_z_hist)))

    model = PRFNet(
        rv_in=rv_in,
        pb_in=pb_in,
        num_classes=int(dc["num_classes"]),
        enc_channels=mc["enc_channels"],
        dec_out_c=int(mc["dec_out_c"]),
        expand_ratios=mc["expand_ratios"],
        rv_strides=mc.get("rv_strides", [[1, 2], [2, 2], [2, 2], [2, 2]]),
        pb_strides=mc.get("pb_strides", [[2, 2], [2, 2], [2, 2], [2, 2]]),
        aspp_rates=mc.get("aspp_rates", [1, 3, 6, 9]),
        rv_H=int(dc["rv_H"]),
        pb_H=int(dc["pb_H"]),
        use_ds_aaff=bool(mc.get("use_ds_aaff", True)),
        ds_aaff_K=int(mc.get("ds_aaff_K", 4)),
        head_dropout=float(mc.get("head_dropout", 0.1)),
        use_vcg=bool(mc.get("use_vcg", True)),
        use_proto=bool(mc.get("use_proto", True)),
        proto_dim=int(mc.get("proto_dim", 64)),
    ).to(device)
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device, use_ema: bool = True) -> str:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if use_ema and isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
        state = ckpt["ema_state_dict"]
        key = "ema_state_dict"
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        key = "state_dict"
    else:
        state = ckpt
        key = "raw_state_dict"

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")
    return key


class FullExportWrapper(torch.nn.Module):
    """Full PRFNet export wrapper: output point-wise logits."""

    def __init__(self, model: PRFNet):
        super().__init__()
        self.model = model

    def forward(
        self,
        rv_img: torch.Tensor,
        pb_img: torch.Tensor,
        rv_coords: torch.Tensor,
        pb_coords: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.export_forward(rv_img, pb_img, rv_coords, pb_coords, points)


class BackboneExportWrapper(torch.nn.Module):
    """NPU-friendly subgraph: encode+decode only, no grid_sample head."""

    def __init__(self, model: PRFNet):
        super().__init__()
        self.model = model

    def forward(self, rv_img: torch.Tensor, pb_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fused_rv, fused_pb, rv_stem, pb_stem = self.model._encode(rv_img, pb_img)
        rv_out = self.model.rv_dec(fused_rv, rv_stem)
        pb_out = self.model.pb_dec(fused_pb, pb_stem)
        return rv_out, pb_out


def export_onnx_full(model: PRFNet, cfg: Dict, out_path: str, max_points: int, opset: int) -> None:
    dc = cfg["data"]
    rv_h, rv_w = int(dc["rv_H"]), int(dc["rv_W"])
    pb_h, pb_w = int(dc["pb_H"]), int(dc["pb_W"])
    rv_in = int(model.rv_in)
    pb_in = int(model.pb_in)
    pt_dim = int(model.aggregator.pt_enc[0].in_features)

    wrapper = FullExportWrapper(model).eval()

    dummy_rv = torch.zeros(1, rv_in, rv_h, rv_w, dtype=torch.float32)
    dummy_pb = torch.zeros(1, pb_in, pb_h, pb_w, dtype=torch.float32)
    dummy_rv_c = torch.zeros(1, max_points, 1, 2, dtype=torch.float32)
    dummy_pb_c = torch.zeros(1, max_points, 1, 2, dtype=torch.float32)
    dummy_pts = torch.zeros(1, max_points, pt_dim, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (dummy_rv, dummy_pb, dummy_rv_c, dummy_pb_c, dummy_pts),
        out_path,
        opset_version=opset,
        input_names=["rv_img", "pb_img", "rv_coords", "pb_coords", "points"],
        output_names=["logits"],
        dynamic_axes=None,
        do_constant_folding=True,
    )


def export_onnx_backbone(model: PRFNet, cfg: Dict, out_path: str, opset: int) -> None:
    dc = cfg["data"]
    rv_h, rv_w = int(dc["rv_H"]), int(dc["rv_W"])
    pb_h, pb_w = int(dc["pb_H"]), int(dc["pb_W"])
    rv_in = int(model.rv_in)
    pb_in = int(model.pb_in)

    wrapper = BackboneExportWrapper(model).eval()

    dummy_rv = torch.zeros(1, rv_in, rv_h, rv_w, dtype=torch.float32)
    dummy_pb = torch.zeros(1, pb_in, pb_h, pb_w, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (dummy_rv, dummy_pb),
        out_path,
        opset_version=opset,
        input_names=["rv_img", "pb_img"],
        output_names=["rv_out", "pb_out"],
        dynamic_axes=None,
        do_constant_folding=True,
    )


def inspect_onnx_ops(onnx_path: str) -> List[str]:
    try:
        import onnx
    except Exception:
        print("[WARN] onnx package not installed, skip op inspection")
        return []

    model = onnx.load(onnx_path)
    ops = sorted({node.op_type for node in model.graph.node})
    return ops


def build_rknn_from_onnx(
    onnx_path: str,
    rknn_path: str,
    target_platform: str,
    quantize: bool,
    dataset: str,
) -> None:
    try:
        from rknn.api import RKNN
    except Exception as e:
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        machine = platform.machine()
        os_name = platform.system()
        msg = (
            "rknn-toolkit2 import failed.\n"
            f"Current environment: os={os_name}, arch={machine}, python={py_ver}\n"
            "Hints:\n"
            "1) RKNN-Toolkit2 supports Python 3.6~3.12 (3.13 is not supported).\n"
            "2) Official toolkit wheels in repo are manylinux builds; on Windows host, use WSL2 Ubuntu for conversion.\n"
            "3) If you only need ONNX now, rerun without --build_rknn."
        )
        raise RuntimeError(msg) from e

    rknn = RKNN(verbose=True)
    print("--> RKNN config")
    ret = rknn.config(target_platform=target_platform)
    if ret != 0:
        raise RuntimeError(f"rknn.config failed: {ret}")

    print("--> RKNN load_onnx")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx failed: {ret}")

    print("--> RKNN build")
    if quantize:
        if not dataset:
            raise ValueError("Quantization enabled but --dataset is empty")
        ret = rknn.build(do_quantization=True, dataset=dataset)
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f"rknn.build failed: {ret}")

    print("--> RKNN export")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed: {ret}")

    rknn.release()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export PRFNet pth to ONNX/RKNN for RK3588")
    p.add_argument("--cfg", required=True, help="Path to config yaml")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint pth")
    p.add_argument("--out_dir", default="test_rknn/export", help="Output directory")

    p.add_argument("--mode", choices=["full", "backbone"], default="backbone",
                   help="Export full model or NPU-friendly backbone")
    p.add_argument("--max_points", type=int, default=32768,
                   help="Static point count for full ONNX export")
    p.add_argument("--opset", type=int, default=13)
    p.add_argument("--no_ema", action="store_true", help="Use state_dict instead of ema_state_dict")

    p.add_argument("--build_rknn", action="store_true", help="Build .rknn from exported ONNX")
    p.add_argument("--target_platform", default="rk3588", help="RKNN target platform")
    p.add_argument("--quantize", action="store_true", help="Enable INT8 quantization in RKNN build")
    p.add_argument("--dataset", default="", help="Calibration dataset txt when --quantize")
    p.add_argument("--force_rknn", action="store_true",
                   help="Force RKNN build even if unsupported ops are detected")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.cfg)
    device = torch.device("cpu")
    model = build_model(cfg, device)
    state_key = load_checkpoint(model, args.ckpt, device, use_ema=(not args.no_ema))
    model.eval()

    stem = Path(args.ckpt).stem
    onnx_name = f"{stem}_{args.mode}.onnx"
    onnx_path = str(out_dir / onnx_name)

    print(f"[INFO] cfg={args.cfg}")
    print(f"[INFO] ckpt={args.ckpt}")
    print(f"[INFO] checkpoint_key={state_key}")
    print(f"[INFO] mode={args.mode}")

    if args.mode == "full":
        export_onnx_full(model, cfg, onnx_path, max_points=args.max_points, opset=args.opset)
    else:
        export_onnx_backbone(model, cfg, onnx_path, opset=args.opset)

    print(f"[OK] ONNX exported: {onnx_path}")

    ops = inspect_onnx_ops(onnx_path)
    if ops:
        print(f"[INFO] ONNX ops ({len(ops)}): {', '.join(ops)}")

    unsupported_hard = {"GridSample"}
    hit = sorted(unsupported_hard.intersection(set(ops)))
    if hit:
        print(f"[WARN] Detected ops likely unsupported by RKNN: {hit}")
        if args.build_rknn and not args.force_rknn:
            print("[ABORT] Skip RKNN build. Use --force_rknn to try anyway.")
            return

    if args.build_rknn:
        rknn_name = f"{stem}_{args.mode}_{args.target_platform}.rknn"
        rknn_path = str(out_dir / rknn_name)
        build_rknn_from_onnx(
            onnx_path=onnx_path,
            rknn_path=rknn_path,
            target_platform=args.target_platform,
            quantize=args.quantize,
            dataset=args.dataset,
        )
        print(f"[OK] RKNN exported: {rknn_path}")


if __name__ == "__main__":
    main()
