"""Evaluation of COLMAP models of the Lizard Island subset (docs/explore_lizard.md).

    python depth_aware/lizard_eval.py name=<model_txt> [name=<model_txt> ...] [--json out.json]

Per model:
  reg, points, track, reproj   COLMAP statistics (reproj: mean of the per-point errors, as
                               in the paper's Table II)
  kappa_traj                   the paper's trajectory curvature (Eq. 15-18): plane fit to the
                               camera centres, kappa = c3 * D of the radially symmetric
                               quadric of the residuals
  kappa_scene                  the same for the sparse points (track >= 3, 1-99 % box
                               filter, 5 cm voxels after scaling to metres)
  gps_h_rms, gps_v_rms         similarity (Umeyama) alignment of the camera centres to the
                               cleaned GPS track (east, north, 0): horizontal RMS and RMS of
                               the vertical residual (m). The ASV keeps the camera at constant
                               height, so the vertical residual is bending (plus GPS noise
                               leaking through the alignment)
  sag_along_mm                 bending along the legs: a parabola h = a + b s + c s^2 fitted
                               per leg to the aligned height over along-track position s;
                               sagitta of the quadratic term over the leg length, mean of
                               the three legs (mm, + = leg ends higher than the middle)
  scale                        metres per model unit (from the GPS alignment)
  depth_med, depth_p5_p95      camera-to-point depth of the sparse points (m)
Sign convention: the plane normal points from the scene towards the cameras ("up"), so
kappa > 0 means the edges are bent up (a bowl seen from above) and kappa < 0 a dome. (The
paper calls negative kappa a bowl; its normal orientation is not fully specified.)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colmap_utils as cu  # noqa: E402

WORK = os.environ.get("REFRAX_LIZARD_WORK", "/Users/fischert/robot/refrax-work/lizard")


def load_gps(path=f"{WORK}/gps.csv"):
    out = {}
    for line in open(path):
        n, e, no = line.strip().split(",")
        out[n] = np.array([float(e), float(no), 0.0])
    return out


def umeyama(src, dst):
    """Similarity (s, R, t) with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / a.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def curvature(P, up):
    """kappa = c3 * D of a point set, plane normal oriented along `up`."""
    mu = P.mean(0)
    _, _, Vt = np.linalg.svd(P - mu, full_matrices=False)
    n = Vt[2] if np.dot(Vt[2], up) > 0 else -Vt[2]
    u_ax, v_ax = Vt[0], Vt[1]
    r = (P - mu) @ n
    u, v = (P - mu) @ u_ax, (P - mu) @ v_ax
    D = np.hypot(np.ptp(u), np.ptp(v))
    A = np.stack([np.ones_like(u), u, v, u * u + v * v], 1)
    c = np.linalg.lstsq(A, r, rcond=None)[0]
    return float(c[3] * D), r, u, v, D


def leg_segments(names):
    """Split the (name-ordered) images into the three survey legs at the turns, from GPS."""
    gps = load_gps()
    allnames = sorted(gps)
    xy = np.array([gps[n][:2] for n in allnames])
    h = np.radians(120.0)  # leg direction of this survey (see docs)
    s = xy @ np.array([np.sin(h), np.cos(h)])
    # turns: local extrema of the along-track coordinate (smoothed)
    k = np.convolve(s, np.ones(9) / 9, mode="same")
    turns = [i for i in range(10, len(k) - 10) if (k[i] - k[i - 10]) * (k[i + 10] - k[i]) < 0
             and abs(k[i]) == max(abs(k[i - 10:i + 11]))]
    # keep one index per turn
    cut = []
    for t in turns:
        if not cut or t - cut[-1] > 30:
            cut.append(t)
    bounds = [0] + cut + [len(allnames)]
    leg = {}
    for j in range(len(bounds) - 1):
        for n in allnames[bounds[j]:bounds[j + 1]]:
            leg[n] = j
    return leg, {n: float(v) for n, v in zip(allnames, s)}


def evaluate(model_txt, n_total=None):
    gps = load_gps()
    n_total = n_total or len(gps)
    imgs = cu.read_images(f"{model_txt}/images.txt")
    pts = cu.read_points3d(f"{model_txt}/points3D.txt")
    st = cu.model_stats(model_txt, n_total)
    names = sorted(n for n in imgs if n in gps)
    C = np.array([-imgs[n]["R"].T @ imgs[n]["t"] for n in names])
    G = np.array([gps[n] for n in names])
    s, R, t = umeyama(C, G)
    Ca = s * C @ R.T + t
    res = Ca - G
    X = np.array([p[0] for p in pts.values()])
    tl = np.array([p[2] for p in pts.values()])
    Xa = s * X[tl >= 3] @ R.T + t
    lo, hi = np.percentile(Xa, [1, 99], axis=0)
    Xa = Xa[np.all((Xa >= lo) & (Xa <= hi), 1)]
    vox = np.unique(np.floor(Xa / 0.05).astype(np.int64), axis=0, return_index=True)[1]
    Xv = Xa[vox]
    up = Ca.mean(0) - Xv.mean(0)
    up /= np.linalg.norm(up)
    k_traj, *_ = curvature(Ca, up)
    k_scene, *_ = curvature(Xv, up)
    # depths of the sparse points in metres
    dep = []
    for n in names:
        d = imgs[n]
        sel = d["p3d"] >= 0
        ids = d["p3d"][sel]
        P = np.array([pts[i][0] for i in ids if i in pts])
        if len(P):
            dep.append(s * (P @ d["R"].T + d["t"])[:, 2])
    dep = np.concatenate(dep)
    # sagitta along the legs
    leg, along = leg_segments(names)
    sags = []
    for j in sorted(set(leg[n] for n in names)):
        idx = [i for i, n in enumerate(names) if leg[n] == j]
        if len(idx) < 20:
            continue
        sa = np.array([along[names[i]] for i in idx])
        z = Ca[idx, 2]
        c2 = np.polyfit(sa, z, 2)[0]
        L = np.ptp(sa)
        sags.append(c2 * (L / 2) ** 2)
    return dict(**st, kappa_traj=k_traj, kappa_scene=k_scene,
                gps_h_rms=float(np.sqrt(np.mean(np.sum(res[:, :2] ** 2, 1)))),
                gps_v_rms=float(np.sqrt(np.mean(res[:, 2] ** 2))),
                sag_along_mm=float(1000 * np.mean(sags)) if sags else float("nan"),
                sag_legs_mm=[round(1000 * x, 1) for x in sags],
                scale=float(s), depth_med=float(np.median(dep)),
                depth_p5_p95=[float(x) for x in np.percentile(dep, [5, 95])])


def aligned(model_txt):
    """Aligned camera centres, names, aligned points (for figures)."""
    gps = load_gps()
    imgs = cu.read_images(f"{model_txt}/images.txt")
    pts = cu.read_points3d(f"{model_txt}/points3D.txt")
    names = sorted(n for n in imgs if n in gps)
    C = np.array([-imgs[n]["R"].T @ imgs[n]["t"] for n in names])
    G = np.array([gps[n] for n in names])
    s, R, t = umeyama(C, G)
    X = np.array([p[0] for p in pts.values()])
    tl = np.array([p[2] for p in pts.values()])
    return names, s * C @ R.T + t, s * X[tl >= 3] @ R.T + t, G


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="+", help="name=<model_txt>")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = {}
    for m in args.models:
        name, path = m.split("=", 1)
        out[name] = evaluate(path)
        r = out[name]
        print(f"{name:28s} reg {r['registered']:4d} pts {r['points']:6d} track {r['track_len']:.2f} "
              f"reproj {r['reproj']:.3f} k_traj {r['kappa_traj']:+.4f} k_scene "
              f"{r['kappa_scene']:+.4f} gps_h {r['gps_h_rms']:.2f} gps_v {r['gps_v_rms']:.3f} "
              f"sag {r['sag_along_mm']:+.0f}mm {r['sag_legs_mm']} depth {r['depth_med']:.2f} "
              f"{r['depth_p5_p95']}", flush=True)
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
