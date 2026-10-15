"""Accuracy of the torch refraction removal (core.torch_remove) against the numba path.

Usage:
    python benchmarks/accuracy_gpu.py configs/dataset.yaml [--rgb-dir DIR] [--devices cpu,mps]
                                      [--tilt 5] [--n-images 5]

For the config's camera (zoom and canvas as remove_refraction), each depth (z0 as a
constant map, and the synthetic kinds of make_synthetic_depth: smooth, plane, noisy, step,
halfres -- the latter resized like load_depth) and each device / dtype:
  * maps: torch float32 remap maps vs the numba kernel's (radtan applied), over pixels valid
    in both: max / p99 / mean |difference| (px), pixels > 0.01 px, valid-mask disagreement;
  * ground truth: torch maps without distortion are forward-traced with the analytic model
    (depth sampled bilinearly in float64): max |F(map(q), w) - q| in output px;
  * images: a real image corrected with Corrector.correct (fused remap on GPU devices)
    vs cv2.remap with the numba maps: PSNR and max grey-level difference over all pixels.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import remove_refraction as rr  # noqa: E402
from accuracy import gt_residual  # noqa: E402
from bench_gpu import scaled_cfg  # noqa: E402
from core.config import load_config, camera_params, distortion_params, housing_params  # noqa: E402
from core.flatport import FlatPort  # noqa: E402
from core.torch_remove import Corrector, TorchInverse  # noqa: E402
from core.undistort import InverseMapper  # noqa: E402
from make_synthetic_depth import synthetic_depth  # noqa: E402

KINDS = ["const", "smooth", "plane", "noisy", "step", "halfres"]


def depth_of(kind, W, H, z0, seed):
    if kind == "const":
        return np.full((H, W), z0, np.float32)
    d = synthetic_depth(kind, W, H, seed)
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(d, 1e-3, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--rgb-dir", default="/Users/fischert/robot/refrax-data/gopro10_linear")
    ap.add_argument("--devices", default="cpu,mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--float64", action="store_true", help="also torch-cpu float64")
    args = ap.parse_args()
    cfg = scaled_cfg(load_config(args.config), 1.0, args.tilt)
    W, H, fx, fy, cx, cy = camera_params(cfg)
    dist, housing = distortion_params(cfg), housing_params(cfg)
    z0 = cfg["correction"]["z0_fixed"]
    paths = sorted(glob.glob(os.path.join(args.rgb_dir, "*.png")))[::50][:len(KINDS)]
    imgs = [cv2.imread(p) for p in paths]
    model = FlatPort(**housing)
    runs = [(d, torch.float32) for d in args.devices.split(",")]
    if args.float64:
        runs.append(("cpu", torch.float64))
    print(f"tilt {args.tilt} deg, {W}x{H}, images: {[os.path.basename(p) for p in paths]}")
    for device, dtype in runs:
        corr = Corrector(cfg, device, dtype)
        corr.warmup()
        zoom, canvas = corr.zoom, corr.canvas
        nodist = TorchInverse(W, H, fx, fy, cx, cy, housing, zoom, canvas, None, device, dtype,
                              z_ref=z0)
        mapper = InverseMapper(W, H, fx, fy, cx, cy, housing, zoom, canvas)
        print(f"\n{device} {str(dtype).split('.')[-1]}  canvas {canvas[0]}x{canvas[1]}")
        print(f"{'depth':>8} | {'max':>8} {'p99':>8} {'mean':>8} {'>0.01px':>8} {'mask xor':>8} |"
              f" {'GT max':>8} | {'PSNR dB':>8} {'max dl':>6}")
        for i, kind in enumerate(KINDS):
            depth = depth_of(kind, W, H, z0, i)
            rx, ry = mapper.remap_maps(depth, dist)
            mx, my = (a[0].cpu().numpy().astype(np.float64) for a in corr.remap_maps(depth))
            vr = rr.compute_valid_mask(rx, ry, W, H) > 0
            vt = rr.compute_valid_mask(mx, my, W, H) > 0
            e = np.hypot(mx - rx, my - ry)[vr & vt]
            ux, uy = (a[0].cpu().numpy().astype(np.float64) for a in
                      nodist.remap_maps(torch.as_tensor(depth, device=device)[None]))
            ok = ux >= 0
            ux[~ok], uy[~ok] = np.nan, np.nan
            gt = gt_residual(model, ux, uy, depth.astype(np.float64), (fx, fy, cx, cy), zoom,
                             canvas)
            img = imgs[i % len(imgs)]
            ref = cv2.remap(img, rx, ry, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            out, _ = corr.correct(img, depth)
            out = out.cpu().numpy()
            d = out.astype(np.float64) - ref
            psnr = 10 * np.log10(255 ** 2 / max(np.mean(d ** 2), 1e-12))
            print(f"{kind:>8} | {e.max():8.1e} {np.percentile(e, 99):8.1e} {e.mean():8.1e} "
                  f"{int((e > 0.01).sum()):8d} {int((vr ^ vt).sum()):8d} | {gt.max():8.1e} | "
                  f"{psnr:8.1f} {int(np.abs(d).max()):6d}", flush=True)


if __name__ == "__main__":
    main()
