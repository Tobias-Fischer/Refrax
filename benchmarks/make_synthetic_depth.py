"""Synthetic per-image depth maps (<name>.npy, metres) for benchmarking depth-map mode.

Usage:
    python benchmarks/make_synthetic_depth.py <rgb_dir> <depth_dir> [--W 1920 --H 1080]

The images are assigned, in sorted order, one of the following depth kinds:
    smooth   smooth undulating surface, 0.5-3 m
    plane    tilted plane (sea floor seen obliquely), 0.6 m (bottom) - 3 m (top)
    noisy    smooth surface + 1 % multiplicative Gaussian noise per pixel
    step     2 m background with a 0.6 m disc and a 1.2 m box (depth discontinuities)
    halfres  smooth surface stored at half resolution (exercises the resize path)
"""
import argparse
import glob
import os

import numpy as np

KINDS = ["smooth", "plane", "noisy", "step", "halfres"]


def synthetic_depth(kind, W, H, seed=0):
    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    if kind in ("smooth", "noisy", "halfres"):
        ph = rng.uniform(0, 2 * np.pi, 3)
        d = (1.75 + 0.6 * np.sin(2 * np.pi * 1.3 * u + ph[0]) * np.cos(2 * np.pi * 0.9 * v + ph[1])
             + 0.35 * np.sin(2 * np.pi * (0.7 * u + 1.1 * v) + ph[2]))
        d = np.clip(d, 0.5, 3.0)
        if kind == "noisy":
            d = d * (1 + 0.01 * rng.standard_normal(d.shape))
        if kind == "halfres":
            d = d[::2, ::2]
    elif kind == "plane":
        d = 3.0 - 2.4 * v + 0.3 * (u - 0.5)
    elif kind == "step":
        d = np.full((H, W), 2.0)
        cx, cy = rng.uniform(0.3, 0.7, 2)
        d[(u - cx) ** 2 * (W / H) ** 2 + (v - cy) ** 2 < 0.15 ** 2] = 0.6
        d[(u > 0.05) & (u < 0.3) & (v > 0.6) & (v < 0.95)] = 1.2
    else:
        raise ValueError(kind)
    return d.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rgb_dir")
    ap.add_argument("depth_dir")
    ap.add_argument("--W", type=int, default=1920)
    ap.add_argument("--H", type=int, default=1080)
    args = ap.parse_args()
    os.makedirs(args.depth_dir, exist_ok=True)
    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(f"{args.rgb_dir}/*.png"))
    for i, name in enumerate(names):
        kind = KINDS[i % len(KINDS)]
        np.save(f"{args.depth_dir}/{name}.npy", synthetic_depth(kind, args.W, args.H, seed=i))
        print(name, kind)


if __name__ == "__main__":
    main()
