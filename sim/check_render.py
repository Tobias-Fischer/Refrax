"""Sanity check for simulator output (any sim): removal round trip on rendered frames.

Usage:
    python -m sim.check_render SIM_DIR [--config configs/dataset.yaml]
    python -m sim.check_render SIM_DIR --make-demo     # stand-in simulator: analytic scenes

SIM_DIR holds render_camera.yaml, pinhole/, pinhole_depth/ (z-depth, m), underwater/ and
underwater_depth/ (camera-frame Z per underwater pixel) -- as written by
sim/isaacsim/render_flatport_isaacsim.py or `render_refraction.py ... apply`.

For each frame, Refrax removes the refraction from underwater/<name>.png using the per-pixel
depth; every corrected pixel is back-projected with its depth, projected into the pinhole
render camera and compared photometrically with the pinhole render (away from depth edges).
"""
import argparse
import glob
import os

import cv2
import numpy as np

from core.config import load_config, camera_params, distortion_params, housing_params
from core.render import depth_edges
from core.scale import select_zoom
from remove_refraction import make_map_builder, compute_valid_mask
from render_refraction import read_spec, write_spec


def make_demo(sim_dir, cfg):
    from core.render import FlatPortRenderer
    from sim.scenes import standard_scenes
    r = FlatPortRenderer.from_config(cfg, oversample=1.5)
    for sub in ("pinhole", "pinhole_depth"):
        os.makedirs(os.path.join(sim_dir, sub), exist_ok=True)
    write_spec(os.path.join(sim_dir, "render_camera.yaml"), r.spec, r.n_steps)
    for nm in ("spheres", "steps", "tilted_plane"):
        rgb, z, _ = standard_scenes()[nm].render_pinhole(r.spec)
        cv2.imwrite(os.path.join(sim_dir, "pinhole", nm + ".png"),
                    (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1])
        np.save(os.path.join(sim_dir, "pinhole_depth", nm + ".npy"), z.astype(np.float32))
    print(f"demo frames written to {sim_dir}")


def check(sim_dir, cfg):
    W, H, fx, fy, cx, cy = camera_params(cfg)
    housing, dist = housing_params(cfg), distortion_params(cfg)
    spec = read_spec(os.path.join(sim_dir, "render_camera.yaml"))
    zoom, _ = select_zoom(cfg["correction"]["z0_fixed"], W, H, fx, fy, cx, cy, housing)
    build_map = make_map_builder(W, H, fx, fy, cx, cy, housing, dist, zoom)
    ug, vg = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    rows = []
    for p in sorted(glob.glob(os.path.join(sim_dir, "underwater", "*.png"))):
        name = os.path.splitext(os.path.basename(p))[0]
        uw = cv2.imread(p).astype(np.float32)
        Zu = np.load(os.path.join(sim_dir, "underwater_depth", name + ".npy")).astype(np.float64)
        pin = cv2.imread(os.path.join(sim_dir, "pinhole", name + ".png")).astype(np.float32)
        fill = np.nanmedian(Zu[np.isfinite(Zu)])
        Zu_f = np.where(np.isfinite(Zu), Zu, fill).astype(np.float32)
        map_x, map_y = build_map(Zu_f)
        valid = compute_valid_mask(map_x, map_y, W, H) > 0
        corr = cv2.remap(uw, map_x, map_y, cv2.INTER_LINEAR)
        Zc = cv2.remap(np.where(np.isfinite(Zu), Zu, np.nan).astype(np.float32), map_x, map_y,
                       cv2.INTER_NEAREST)
        # back-project corrected pixels (pinhole at the camera centre, focal zoom * f)
        X = np.stack([(ug - cx) / (zoom * fx) * Zc, (vg - cy) / (zoom * fy) * Zc, Zc], -1)
        Xr = X - spec.centre
        ur = spec.fx * Xr[..., 0] / Xr[..., 2] + spec.cx
        vr = spec.fy * Xr[..., 1] / Xr[..., 2] + spec.cy
        ref = cv2.remap(pin, ur.astype(np.float32), vr.astype(np.float32), cv2.INTER_LINEAR)
        edge = depth_edges(np.where(np.isfinite(Zc) & (Zc > 0), 1 / np.where(Zc > 0, Zc, 1), 0))
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        m = valid & np.isfinite(Zc) & ~edge & (ur >= 0) & (ur <= spec.W - 1) & (vr >= 0) & (vr <= spec.H - 1)
        diff = np.abs(corr - ref)[m]
        psnr = 10 * np.log10(255.0 ** 2 / np.mean(diff ** 2))
        rows.append((name, float(m.mean()), float(diff.mean()), psnr))
        print(f"{name}: compared {m.mean():.3f} of pixels, mean |diff| {diff.mean():.2f}/255, PSNR {psnr:.1f} dB")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--make-demo", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.make_demo:
        make_demo(args.sim_dir, cfg)
        import subprocess, sys
        subprocess.run([sys.executable, "render_refraction.py", args.config, "apply", args.sim_dir],
                       check=True)
    check(args.sim_dir, cfg)


if __name__ == "__main__":
    main()
