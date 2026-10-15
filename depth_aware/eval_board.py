"""Board-based evaluation: how pinhole-like are the corrected images? (no SfM needed)

    python depth_aware/eval_board.py --runs NAME=DEPTH_DIR_OR_Z0[@CENTER_MM] ...

For every image with a reliable AprilGrid detection, the detected tag corners are
corrected point-wise with the depth a method assigns them (a constant z0, or the
method's ideal-grid depth map sampled at the corner) and the RMS homography transfer
residual (px, corrected image) is computed on a fixed set of inlier tags. The board
layout (tag spacing, and a per-corner print/flatness offset) is self-calibrated once
from all images, so the residual floor reflects detection noise and board bending.
Prints mean/median per method and paired differences against the first method.
"""
import argparse
import json
import os

import numpy as np

from board import board_coords, detect_dir, homography_residual, inlier_tags, rms
from common import Corrector, load_cfg, WORK, REPO
from pipeline import bilinear

DET = f"{WORK}/board/gopro10_det.npz"
RHO, ORIENT = 0.5042, 3


def refine_layout(C, det, z0, iters=3):
    """Per-corner board offsets (board units) averaged over all images."""
    import cv2
    offs = {}
    for _ in range(iters):
        acc = {}
        for n, (ids, c) in det.items():
            bxy = board_coords(ids, RHO, ORIENT) + np.array([offs.get(k, (0, 0)) for k in keys(ids)])
            out = C.correct_points(c.reshape(-1, 2), z0)
            H, _ = cv2.findHomography(bxy, out, 0)
            pb = cv2.perspectiveTransform(out.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
            for k, e in zip(keys(ids), pb - bxy):
                acc.setdefault(k, []).append(e)
        for k, v in acc.items():
            if len(v) >= 5:
                offs[k] = np.array(offs.get(k, (0, 0))) + np.mean(v, 0)
    return offs


def keys(ids):
    return [int(i) * 4 + j for i in ids for j in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "figures", "board_eval.json"))
    ap.add_argument("--names", default=None, help="file with image names to restrict to")
    args = ap.parse_args()
    cfg = load_cfg()
    det_all = detect_dir(None, DET)

    C_ref = Corrector(cfg, z0=0.5)
    det = {}
    for n, (ids, c) in det_all.items():
        if len(ids) < 12:
            continue
        ok = inlier_tags(ids, board_coords(ids, RHO, ORIENT), C_ref.correct_points(c.reshape(-1, 2), 0.5))
        if ok is not None:
            det[n] = (ids[ok], c[ok])
    offs = refine_layout(C_ref, det, 0.5)

    methods = []
    for spec in args.runs:
        label, rest = spec.split("=", 1)
        center = 0.0
        if "@" in rest:
            rest, cmm = rest.split("@")
            center = float(cmm) / 1000
        methods.append((label, rest, center))

    results = {}
    common_names = None
    for label, src, center in methods:
        C = Corrector(cfg, z0=1.0, center_z=center)
        per = {}
        for n, (ids, c) in det.items():
            raw = c.reshape(-1, 2)
            if os.path.isdir(src):
                p = f"{src}/{n}.npy"
                if not os.path.exists(p):
                    continue
                D = np.load(p).astype(np.float32)
                Z = bilinear(D, C.raw_to_ideal(raw))
            else:
                Z = float(src)
            out = C.correct_points(raw, Z)
            bxy = board_coords(ids, RHO, ORIENT) + np.array([offs.get(k, (0, 0)) for k in keys(ids)])
            per[n] = rms(homography_residual(bxy, out))
        results[label] = per
        common_names = set(per) if common_names is None else common_names & set(per)
    if args.names:
        common_names &= set(open(args.names).read().split())
    common_names = sorted(common_names)
    ref = methods[0][0]
    summary = {}
    print(f"{len(common_names)} images with board detections and depth for every method")
    print(f"{'method':38s} {'mean':>7s} {'median':>7s} {'d_mean vs ' + ref:>22s} {'frac better':>11s}")
    for label, _, _ in methods:
        v = np.array([results[label][n] for n in common_names])
        r = np.array([results[ref][n] for n in common_names])
        d = v - r
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0
        summary[label] = dict(mean=float(v.mean()), median=float(np.median(v)),
                              diff_mean=float(d.mean()), diff_se=float(se),
                              frac_better=float((d < 0).mean()), n=len(d))
        print(f"{label:38s} {v.mean():7.3f} {np.median(v):7.3f} {d.mean():+12.3f} ± {se:.3f}"
              f" {(d < 0).mean():11.2f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(summary=summary, per_image=results), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
