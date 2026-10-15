"""Evaluate the BA variants of uncertainty/real_ba.py: reprojection error (unweighted, px),
Mahalanobis error with the non-absorbable covariance, AprilGrid planarity (relative RMSE
and curvature kappa = c3 D of evaluation/planarity.py) and the refined intrinsics.

    python uncertainty/real_eval.py PREP_DIR [PREP_DIR ...]   (PREP_DIR/ba/<variant>/)
Writes PREP_DIR/eval.json and prints a table.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "evaluation"))
import planarity as pl  # noqa: E402
sys.path.insert(0, REPO)
from uncertainty.real_prep import board_points  # noqa: E402

SIG_KP = 0.5


def quat2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load(d):
    cam = [l.split() for l in open(f"{d}/cameras.txt") if not l.startswith("#")][0]
    fx, fy, cx, cy = map(float, cam[4:8])
    pts = {}
    for l in open(f"{d}/points3D.txt"):
        if l.startswith("#"):
            continue
        s = l.split()
        pts[int(s[0])] = np.array(s[1:4], float)
    lines = [l for l in open(f"{d}/images.txt") if not l.startswith("#")]
    imgs = {}
    for a, b in zip(lines[::2], lines[1::2]):
        s = a.split()
        R, t = quat2R(np.array(s[1:5], float)), np.array(s[5:8], float)
        v = np.array(b.split(), float).reshape(-1, 3)
        imgs[s[9]] = (R, t, v)
    return (fx, fy, cx, cy), pts, imgs


def evaluate(d, cov, board):
    (fx, fy, cx, cy), pts, imgs = load(d)
    lut = {(n, int(i)): k for k, (n, i) in enumerate(zip(cov["name"], cov["idx"]))}
    E, M = [], []
    for name, (R, t, v) in imgs.items():
        sel = np.flatnonzero(v[:, 2] >= 0)
        X = np.array([pts[int(p)] for p in v[sel, 2]])
        Xc = X @ R.T + t
        pred = np.c_[fx * Xc[:, 0] / Xc[:, 2] + cx, fy * Xc[:, 1] / Xc[:, 2] + cy]
        e = v[sel, :2] - pred
        E.append(e)
        k = np.array([lut[(name, int(i))] for i in sel])
        C = cov["np_"][k]
        a, b, c = C[:, 0] + SIG_KP ** 2, C[:, 1], C[:, 2] + SIG_KP ** 2
        M.append((c * e[:, 0] ** 2 - 2 * b * e[:, 0] * e[:, 1] + a * e[:, 1] ** 2) /
                 (a * c - b * b))
    E, M = np.concatenate(E), np.concatenate(M)
    n = np.hypot(E[:, 0], E[:, 1])
    Xb = np.array([pts[p] for p in board if p in pts])
    # percentile filter + 3 rounds of 5-MAD plane-outlier rejection (explore/depth eval_sfm)
    Xb = pl.percentile_filter(Xb, 1)
    for _ in range(3):
        rr = pl.analyze(Xb)
        mad = 1.4826 * np.median(np.abs(rr["dist"] - np.median(rr["dist"])))
        Xb = Xb[np.abs(rr["dist"] - np.median(rr["dist"])) < 5 * mad]
    r = pl.fit_quadric(pl.analyze(Xb))
    out = dict(reproj_mean=float(n.mean()), reproj_rms=float(np.sqrt((n ** 2).mean())),
               maha_mean=float(M.mean()), n_obs=int(len(n)), board_pts=int(len(Xb)),
               plan_rel_rmse_pct=float(np.sqrt(np.mean(r["dist"] ** 2)) / r["D"] * 100),
               kappa=float(r["c3"] * r["D"]), intr=[fx, fy, cx, cy])
    return out


def main():
    for prep in sys.argv[1:]:
        cov = np.load(os.path.join(prep, "obs_cov.npz"))
        hulls = np.load(os.path.join(prep, "board_hulls.npz"), allow_pickle=True)["hulls"].item()
        res = {}
        base = os.path.join(prep, "ba")
        for v in sorted(os.listdir(base)):
            d = os.path.join(base, v)
            if not os.path.exists(os.path.join(d, "images.txt")):
                continue
            res[v] = evaluate(d, cov, board_points(d, hulls))
        json.dump(res, open(os.path.join(prep, "eval.json"), "w"), indent=1)
        print(prep)
        print(f"{'variant':16s} {'reproj mean':>11s} {'rms':>7s} {'maha':>7s} "
              f"{'board relRMSE%':>14s} {'kappa':>8s}  fx, fy, cx, cy")
        for v, r in res.items():
            print(f"{v:16s} {r['reproj_mean']:11.4f} {r['reproj_rms']:7.4f} "
                  f"{r['maha_mean']:7.3f} {r['plan_rel_rmse_pct']:14.4f} {r['kappa']:8.4f}  "
                  + ", ".join(f"{x:.2f}" for x in r["intr"]))


if __name__ == "__main__":
    main()
