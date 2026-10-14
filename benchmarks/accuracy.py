"""Accuracy of the new undistortion maps against the reference and against ground truth.

Usage:
    python benchmarks/accuracy.py <config.yaml> [--depth-dir DIR] [--image IMG] [--tilt DEG]

For a fixed zoom and canvas (from the reference fit_canvas, so both implementations solve
the same problem) and each depth (z0_fixed and every <name>.npy in --depth-dir):
  * map difference new vs reference over pixels valid in both (max / mean / p99, px), and
    valid-mask agreement;
  * ground truth: each map is forward-traced (analytic model, depth sampled bilinearly) and
    compared with the output pixel it was computed for: |F(map(q)) - q| in output px;
  * remap of --image with the final float32 maps (radtan included): PSNR new vs reference
    over pixels valid in both.
--tilt rotates n_port about the camera y axis (general, non-radial solver).
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
import reference_impl as ref  # noqa: E402
from core.config import load_config, camera_params, distortion_params, housing_params  # noqa: E402
from core.flatport import FlatPort, InvDepth  # noqa: E402
from core.optics import matrix_K  # noqa: E402
from core.undistort import InverseMapper  # noqa: E402
from remove_refraction import load_depth, make_map_builder  # noqa: E402


def gt_residual(model, map_x, map_y, depth, cam, zoom, canvas):
    """|F(p, w(p)) - q| in output pixels for every finite map entry."""
    fx, fy, cx, cy = cam
    out_W, out_H, (ox, oy) = canvas
    ok = np.isfinite(map_x) & np.isfinite(map_y)
    x = (map_x[ok].astype(np.float64) - cx) / fx
    y = (map_y[ok].astype(np.float64) - cy) / fy
    w = InvDepth(depth, fx, fy, cx, cy).sample(x, y)
    Fx, Fy = model.forward(x, y, w)
    qx = fx * zoom * Fx + cx - ox
    qy = fy * zoom * Fy + cy - oy
    jj, ii = np.meshgrid(np.arange(out_W), np.arange(out_H))
    return np.hypot(qx - jj[ok], qy - ii[ok])


def stats(v):
    return f"max {v.max():.2e}  mean {v.mean():.2e}  p99 {np.percentile(v, 99):.2e}"


def psnr(a, b, valid):
    d = (a.astype(np.float64) - b.astype(np.float64))[valid]
    mse = np.mean(d ** 2)
    return np.inf if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--depth-dir")
    ap.add_argument("--image")
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--max-depth-maps", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    W, H, fx, fy, cx, cy = camera_params(cfg)
    distortion = distortion_params(cfg)
    housing = housing_params(cfg)
    if args.tilt:
        a = np.deg2rad(args.tilt)
        housing["n_port"] = np.array([np.sin(a), 0.0, np.cos(a)])
    z0 = cfg["correction"]["z0_fixed"]
    zoom, _ = ref.select_zoom(z0, W, H, fx, fy, cx, cy, housing, in_bounds=False)
    canvas = ref.fit_canvas(W, H, fx, fy, cx, cy, housing, distortion, zoom, z0)
    print(f"n_port={housing['n_port']}, zoom={zoom:.6f}, canvas={canvas}")
    out_W, out_H, origin = canvas
    _, K_inv = matrix_K(fx, fy, cx, cy)
    P2, ray_water = ref.compute_housing_geometry(H, W, K_inv, **housing)
    mapper = InverseMapper(W, H, fx, fy, cx, cy, housing, zoom, canvas)
    model = FlatPort(**housing)
    build_ref = ref.make_map_builder(W, H, fx, fy, cx, cy, housing, distortion, zoom, canvas)
    build_new = make_map_builder(W, H, fx, fy, cx, cy, housing, distortion, zoom, canvas)
    img = cv2.imread(args.image) if args.image else None

    depths = [("z0=%g" % z0, z0)]
    if args.depth_dir:
        files = sorted(glob.glob(f"{args.depth_dir}/*.npy"))[:args.max_depth_maps]
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            depths.append((name, load_depth(name, args.depth_dir, W, H)))

    for label, depth in depths:
        print(f"--- {label}")
        rx, ry = ref.build_undistort_map_closed_form(P2, ray_water, depth, fx, fy, cx, cy,
                                                     out_H, out_W, zoom=zoom, origin=origin)
        nx, ny = mapper.build(depth)
        vr, vn = np.isfinite(rx), np.isfinite(nx)
        both = vr & vn
        d = np.hypot(nx - rx, ny - ry)[both]
        print(f"valid ref {vr.sum()}  new {vn.sum()}  disagree {(vr ^ vn).sum()} "
              f"({100 * (vr ^ vn).mean():.4f} %)")
        print(f"map |new - ref|            {stats(d)}")
        d_ref = np.asarray(depth, dtype=np.float64)
        print(f"GT residual reference      {stats(gt_residual(model, rx, ry, d_ref, (fx, fy, cx, cy), zoom, canvas))}")
        print(f"GT residual new            {stats(gt_residual(model, nx, ny, d_ref, (fx, fy, cx, cy), zoom, canvas))}")
        # Final float32 remap maps (radtan, -1 invalid) and images.
        Rx, Ry = build_ref(depth)
        Nx, Ny = build_new(depth)
        mr = (Rx >= 0) & (Rx <= W - 1) & (Ry >= 0) & (Ry <= H - 1)
        mn = (Nx >= 0) & (Nx <= W - 1) & (Ny >= 0) & (Ny <= H - 1)
        dd = np.hypot(Nx - Rx, Ny - Ry)[mr & mn]
        print(f"final maps |new - ref|     {stats(dd)}   mask disagree {(mr ^ mn).sum()}")
        if img is not None:
            a = cv2.remap(img, Rx, Ry, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            b = cv2.remap(img, Nx, Ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            v = mr & mn
            print(f"image PSNR new vs ref      {psnr(a, b, v):.1f} dB   "
                  f"max |diff| {np.abs(a.astype(int) - b.astype(int))[v].max()}")


if __name__ == "__main__":
    main()
