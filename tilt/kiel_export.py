"""Export a Refrax-corrected COLMAP model as a refractive model for Kiel's
"Refractive COLMAP" (She et al., IROS 2024; cau-git.rz.uni-kiel.de/inf-ag-koeser/colmap_underwater)
so that its bundle adjuster (FLATPORT, refine_refrac_params) can estimate the port normal
on the same observations as tilt/real.py (independent implementation cross-check).

    python tilt/kiel_export.py MODEL_TXT CALIB_YAML CONFIG Z0 SCALE OUT_DIR
    python tilt/kiel_export.py --read KIEL_MODEL_TXT     # print the estimated tilt

Keypoints are mapped back to the raw image (tilt.model.raw_from_corrected) and written in
COLMAP's pixel convention (+0.5); the camera is OPENCV with the in-air intrinsics
(principal point +0.5) followed by "FLATPORT nx ny nz d0 tglass mu_a mu_g mu_w"; poses
and points are scaled to metres (SCALE metres per model unit).
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def export(args):
    from core.config import load_config
    from tilt.real import prepare
    cfg = load_config(args.config)
    m, decl, geo, u, ok = prepare(args.model, args.calib, cfg, args.z0)
    th = geo.theta0
    hs = cfg["housing"]
    os.makedirs(args.out, exist_ok=True)
    W, H = cfg["camera"]["W"], cfg["camera"]["H"]
    fx, fy, cx, cy, k1, k2, p1, p2 = th[6:14]
    with open(os.path.join(args.out, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write(f"1 OPENCV {W} {H} {fx!r} {fy!r} {cx + 0.5!r} {cy + 0.5!r} {k1!r} {k2!r} "
                f"{p1!r} {p2!r} FLATPORT 0 0 1 {hs['d0']!r} {hs['tglass']!r} "
                f"{hs['mu_a']!r} {hs['mu_g']!r} {hs['mu_w']!r}\n")
    # images: rewrite keypoints of observations (and all other keypoints) to raw
    from tilt.model import raw_from_corrected
    src = open(os.path.join(args.model, "images.txt")).read().splitlines()
    body = [l for l in src if not l.startswith("#")]
    out = ["# Image list with two lines of data per image:"]
    for a, b in zip(body[::2], body[1::2]):
        s = a.split()
        t = np.array(s[5:8], float) * args.scale
        s[5:8] = [repr(float(v)) for v in t]
        v = np.array(b.split(), float).reshape(-1, 3) if b.strip() else np.zeros((0, 3))
        if len(v):
            raw = raw_from_corrected(v[:, :2] - 0.5, th, decl, args.z0) + 0.5
            bad = ~np.isfinite(raw).all(1)
            v[:, :2] = np.where(bad[:, None], -1e6, raw)
            v[bad, 2] = -1
        out.append(" ".join(s))
        out.append(" ".join(f"{x:.6f} {y:.6f} {int(p)}" for x, y, p in v))
    with open(os.path.join(args.out, "images.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(args.model, "points3D.txt")) as a, \
            open(os.path.join(args.out, "points3D.txt"), "w") as b:
        for l in a:
            if l.startswith("#"):
                b.write(l)
                continue
            s = l.split()
            s[1:4] = [repr(float(x) * args.scale) for x in s[1:4]]
            b.write(" ".join(s) + "\n")
    print("wrote", args.out)


def read(path):
    from tilt.model import tilts_of_n
    cam = [l.split() for l in open(os.path.join(path, "cameras.txt"))
           if not l.startswith("#")][0]
    i = cam.index("FLATPORT")
    n = np.array(cam[i + 1:i + 4], float)
    ax, ay = tilts_of_n(n)
    print(f"camera params {cam[4:i]}")
    print(f"normal {n}, d0 {cam[i + 4]}, tilt_x {np.degrees(ax):.4f} deg, "
          f"tilt_y {np.degrees(ay):.4f} deg")
    return np.degrees(ax), np.degrees(ay), float(cam[i + 4]), [float(v) for v in cam[4:i]]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--read":
        read(sys.argv[2])
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("calib")
    ap.add_argument("config")
    ap.add_argument("z0", type=float)
    ap.add_argument("scale", type=float)
    ap.add_argument("out")
    export(ap.parse_args())
