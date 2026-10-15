"""COLMAP statistics and board planarity for pipeline runs.

    python depth_aware/eval_sfm.py RUN_DIR[@CENTER_MM] ... [--labels ...]

RUN_DIR is a pipeline stage directory (it0_fixed, it1_depth, ...) containing
colmap/sparse/*_txt and, for depth runs, depth/<name>.npy. Reports registered
images, points, mean track length, mean reprojection error, and the planarity of the
3D points on the AprilGrid: points whose observations fall inside the corrected
board outline (tag corners corrected with the run's own depth) in the majority of
their images. Planarity uses evaluation/planarity.py (relative RMSE, curvature
kappa = c3 * D as in the paper).
"""
import argparse
import json
import os
import sys

import numpy as np
import cv2

from board import detect_dir
from common import Corrector, load_cfg, REPO, WORK
import colmap_utils as cu
from pipeline import bilinear

sys.path.insert(0, os.path.join(REPO, "evaluation"))
import planarity as pl  # noqa: E402

DET = f"{WORK}/board/gopro10_det.npz"


def model_dir(run):
    sp = f"{run}/colmap/sparse"
    return cu.largest_model_txt(sp, f"{run}/colmap/colmap.log")


def board_points(run, C, z0, det):
    txt = model_dir(run)
    imgs = cu.read_images(f"{txt}/images.txt")
    pts = cu.read_points3d(f"{txt}/points3D.txt")
    inside, total = {}, {}
    for name, d in imgs.items():
        n = os.path.splitext(name)[0]
        sel = d["p3d"] >= 0
        for pid in d["p3d"][sel]:
            total[pid] = total.get(pid, 0) + 1
        if n not in det or len(det[n][0]) < 8:
            continue
        raw = det[n][1].reshape(-1, 2)
        dp = f"{run}/depth/{n}.npy"
        Z = bilinear(np.load(dp).astype(np.float32), C.raw_to_ideal(raw)) if os.path.exists(dp) else z0
        poly = cv2.convexHull(C.correct_points(raw, Z).astype(np.float32))
        xy = d["xy"][sel]
        for pid, (x, y) in zip(d["p3d"][sel], xy):
            if cv2.pointPolygonTest(poly, (float(x), float(y)), True) > 5:
                inside[pid] = inside.get(pid, 0) + 1
    ids = [p for p, k in inside.items() if k >= 2 and k >= 0.5 * total[p]]
    return np.array([pts[p][0] for p in ids]), imgs, pts


def planarity(points):
    pts = pl.percentile_filter(points, 1)
    for _ in range(3):  # drop off-board outliers (> 5 MAD from the plane)
        r = pl.analyze(pts)
        mad = 1.4826 * np.median(np.abs(r["dist"] - np.median(r["dist"])))
        pts = pts[np.abs(r["dist"] - np.median(r["dist"])) < 5 * mad]
    r = pl.fit_quadric(pl.analyze(pts))
    rmse = np.sqrt(np.mean(r["dist"] ** 2))
    return dict(n=len(pts), rel_rmse_pct=float(rmse / r["D"] * 100), kappa=float(r["c3"] * r["D"]),
                rel_median_pct=float(np.median(np.abs(r["dist"])) / r["D"] * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--z0", type=float, default=1.0)
    ap.add_argument("--n-images", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    det = detect_dir(None, DET)
    cfg = load_cfg()
    rows = {}
    for i, spec in enumerate(args.runs):
        run, center = (spec.split("@") + ["0"])[:2]
        C = Corrector(cfg, z0=args.z0, center_z=float(center) / 1000)
        label = args.labels[i] if args.labels else os.path.relpath(run, WORK)
        st = cu.model_stats(model_dir(run), args.n_images)
        bp, _, _ = board_points(run, C, args.z0, det)
        st["board"] = planarity(bp) if len(bp) > 50 else None
        rows[label] = st
        b = st["board"] or {}
        print(f"{label:40s} reg {st['registered']:3d} pts {st['points']:6d} track {st['track_len']:.2f} "
              f"reproj {st['reproj']:.3f} | board n={b.get('n')} relRMSE {b.get('rel_rmse_pct', np.nan):.3f}% "
              f"kappa {b.get('kappa', np.nan):+.4f}", flush=True)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
