"""How much does per-pixel depth change the correction on the real reef?

    python depth_aware/lizard_depth_effect.py --config <yaml> --z0 Z --depth-dir <run>/depth [--every 10]

For every --every-th image with a dense depth map (lizard.py --mode dense), each ideal raw
pixel (16 px grid) is corrected with the constant z0 and with its own depth, from the camera
centre and from the virtual centre. Reported per image (output px, median over images):
  total     RMS / max displacement between the two corrections
  np        the non-pinhole part: residual after the best zoom and shift (what a pinhole
            camera cannot absorb, i.e. what can bend SfM)
  np_med    the same for one depth per image (the image's median depth) vs dense
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Corrector, load_cfg  # noqa: E402
from core.config import camera_params, housing_params  # noqa: E402
from core.virtual_center import optimal_center_z  # noqa: E402


def out_positions(C, depth, g):
    gx, gy = C.ideal_to_out_grid(depth)
    return np.stack([gx[g].ravel(), gy[g].ravel()], 1)


def nonpinhole(a, b):
    """Residual of b -> a after the best zoom + shift (px)."""
    A = np.column_stack([b.ravel(), np.tile([1, 0], len(b)), np.tile([0, 1], len(b))])
    x = np.linalg.lstsq(A, a.ravel(), rcond=None)[0]
    e = np.hypot(*(a.ravel() - A @ x).reshape(-1, 2).T)
    return float(np.sqrt(np.mean(e ** 2))), float(e.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--z0", type=float, required=True)
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    W, H, fx, fy, cx, cy = camera_params(cfg)
    c_opt = optimal_center_z(W, H, fx, fy, cx, cy, housing_params(cfg), z_range=(1.0, 8.0))[0]
    files = sorted(glob.glob(f"{args.depth_dir}/*.npy"))[::args.every]
    out = {}
    for c in (0.0, c_opt):
        C = Corrector(cfg, z0=args.z0, center_z=c)
        g = (slice(8, None, 16), slice(8, None, 16))
        rows = []
        for f in files:
            D = np.load(f).astype(np.float64)
            a = out_positions(C, D, g)
            b = out_positions(C, args.z0, g)
            m = out_positions(C, float(np.median(D)), g)
            ok = np.all(np.isfinite(a) & np.isfinite(b) & np.isfinite(m), 1)
            a, b, m = a[ok], b[ok], m[ok]
            d = np.hypot(*(a - b).T)
            rows.append(dict(total_rms=float(np.sqrt(np.mean(d ** 2))), total_max=float(d.max()),
                             np=nonpinhole(a, b), np_med=nonpinhole(a, m),
                             depth_p5_p95=[float(x) for x in np.percentile(D, [5, 95])]))
        key = f"c={c * 1000:.2f}mm"
        out[key] = dict(
            n=len(rows),
            total_rms=float(np.median([r["total_rms"] for r in rows])),
            total_max=float(np.median([r["total_max"] for r in rows])),
            np_rms=float(np.median([r["np"][0] for r in rows])),
            np_max=float(np.median([r["np"][1] for r in rows])),
            np_rms_p90=float(np.percentile([r["np"][0] for r in rows], 90)),
            np_med_rms=float(np.median([r["np_med"][0] for r in rows])),
            np_med_max=float(np.median([r["np_med"][1] for r in rows])),
            depth_ratio_p95_p5=float(np.median([r["depth_p5_p95"][1] / r["depth_p5_p95"][0]
                                                for r in rows])))
        print(key, json.dumps({k: round(v, 3) if isinstance(v, float) else v
                               for k, v in out[key].items()}), flush=True)
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
