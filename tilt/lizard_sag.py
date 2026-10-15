"""Trajectory bending of Lizard Island models (GPS-aligned camera centres), using the
evaluation of explore/lizard (Refrax-lizard/depth_aware/lizard_eval.py, imported read-only).

    python tilt/lizard_sag.py name=MODEL_TXT|RESULT.npz [...]

For .npz results of tilt/real.py --save-model (R, t, names) only the camera-centre metrics
are computed (kappa_traj, gps_v_rms, sag_along_mm).
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/fischert/robot/Refrax-lizard/depth_aware")
import lizard_eval as le  # noqa: E402


def centres_from_txt(d):
    import colmap_utils as cu
    imgs = cu.read_images(f"{d}/images.txt")
    return {n: -v["R"].T @ v["t"] for n, v in imgs.items()}


def centres_from_npz(p):
    z = np.load(p, allow_pickle=True)
    return {str(n): -R.T @ t for n, R, t in zip(z["names"], z["R"], z["t"])}


def metrics(cent):
    gps = le.load_gps()
    names = sorted(n for n in cent if n in gps)
    C = np.array([cent[n] for n in names])
    G = np.array([gps[n] for n in names])
    s, R, t = le.umeyama(C, G)
    Ca = s * C @ R.T + t
    res = Ca - G
    up = np.array([0, 0, 1.0])  # GPS frame: cameras above the scene, z up (as lizard_eval's up)
    k_traj, *_ = le.curvature(Ca, up)
    leg, along = le.leg_segments(names)
    sags = []
    for j in sorted(set(leg[n] for n in names)):
        idx = [i for i, n in enumerate(names) if leg[n] == j]
        if len(idx) < 20:
            continue
        sa = np.array([along[names[i]] for i in idx])
        c2 = np.polyfit(sa, Ca[idx, 2], 2)[0]
        sags.append(c2 * (np.ptp(sa) / 2) ** 2)
    return dict(n=len(names), kappa_traj=k_traj,
                gps_v_rms=float(np.sqrt(np.mean(res[:, 2] ** 2))),
                sag_along_mm=float(1000 * np.mean(sags)), sag_legs_mm=[round(1000 * x) for x in sags])


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name, p = arg.split("=", 1)
        cent = centres_from_npz(p) if p.endswith(".npz") else centres_from_txt(p)
        print(f"{name:24s}", metrics(cent), flush=True)
