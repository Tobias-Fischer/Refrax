"""Port-tilt estimation on a real COLMAP model built on Refrax-corrected images
(docs/explore_tilt.md, §4b/c).

    python tilt/real.py MODEL_TXT CALIB_YAML CONFIG Z0 OUT_DIR [--hulls board_hulls.npz]
                        [--variants K K+tilt ...] [--max-obs N]

MODEL_TXT: COLMAP text model of the *corrected* images (fixed-depth Refrax correction at
depth Z0 from the camera centre with the config's housing; calibration.yaml gives the
corrected-image pinhole). Its keypoints are mapped back exactly to the raw images
(tilt.model.raw_from_corrected), then the refractive BA (tilt/ba.py) is run from the
model's poses / points with different sets of free globals. Prints / writes
OUT_DIR/real_tilt.json with the estimates, posterior std (scaled by the a-posteriori
residual variance), correlations, reprojection RMS (raw px) and, with --hulls, the
planarity of the AprilGrid board points.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evaluation"))
from core.config import load_config  # noqa: E402
from core import uncertainty as U  # noqa: E402
from tilt.model import Decl, raw_from_corrected  # noqa: E402
from tilt.ba import RefracBA  # noqa: E402

VARIANTS = {
    "K": ("fx", "fy", "cx", "cy"),
    "K+tilt": ("tilt_x", "tilt_y", "fx", "fy", "cx", "cy"),
    "K+tilt+d0": ("tilt_x", "tilt_y", "d0", "fx", "fy", "cx", "cy"),
    "tilt (K fixed)": ("tilt_x", "tilt_y"),
    "K+dist": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "K+dist+tilt": ("tilt_x", "tilt_y", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
}


def quat2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_model(d):
    cam = [l.split() for l in open(f"{d}/cameras.txt") if not l.startswith("#")][0]
    assert cam[1] == "PINHOLE", cam[1]
    fx, fy, cx, cy = map(float, cam[4:8])
    pid, P = [], []
    for l in open(f"{d}/points3D.txt"):
        if l.startswith("#"):
            continue
        s = l.split()
        pid.append(int(s[0]))
        P.append([float(v) for v in s[1:4]])
    pid = np.array(pid)
    lut = {p: i for i, p in enumerate(pid)}
    lines = [l for l in open(f"{d}/images.txt") if not l.startswith("#")]
    names, R, t, ci, pj, q, kp = [], [], [], [], [], [], []
    for a, b in zip(lines[::2], lines[1::2]):
        s = a.split()
        names.append(s[9])
        R.append(quat2R(np.array(s[1:5], float)))
        t.append(np.array(s[5:8], float))
        v = np.array(b.split(), float).reshape(-1, 3)
        sel = np.flatnonzero(v[:, 2] >= 0)
        ci.append(np.full(len(sel), len(names) - 1))
        pj.append(np.array([lut[int(p)] for p in v[sel, 2]], int))
        q.append(v[sel, :2])
        kp.append(sel)
    return dict(K=(fx, fy, cx, cy), names=names, R=np.array(R), t=np.array(t),
                X=np.array(P), pid=pid, ci=np.concatenate(ci), pj=np.concatenate(pj),
                q=np.concatenate(q), kp=np.concatenate(kp))


def board_ids(m, hulls):
    """Indices (into m['X']) of points inside the board outline in >= 2 and >= half of
    their observations (uncertainty/real_prep.board_points)."""
    import cv2
    inside = np.zeros(len(m["X"]), int)
    total = np.bincount(m["pj"], minlength=len(m["X"]))
    for i, name in enumerate(m["names"]):
        hull = hulls.get(os.path.splitext(name)[0])
        if hull is None:
            continue
        sel = np.flatnonzero(m["ci"] == i)
        for k in sel:
            x, y = m["q"][k]
            if cv2.pointPolygonTest(hull, (float(x), float(y)), True) > 5:
                inside[m["pj"][k]] += 1
    return np.flatnonzero((inside >= 2) & (inside >= 0.5 * total))


def planarity(X):
    import planarity as pl
    Xb = pl.percentile_filter(X, 1)
    for _ in range(3):
        rr = pl.analyze(Xb)
        mad = 1.4826 * np.median(np.abs(rr["dist"] - np.median(rr["dist"])))
        Xb = Xb[np.abs(rr["dist"] - np.median(rr["dist"])) < 5 * mad]
    r = pl.fit_quadric(pl.analyze(Xb))
    return dict(plan_rel_rmse_pct=float(np.sqrt(np.mean(r["dist"] ** 2)) / r["D"] * 100),
                kappa=float(r["c3"] * r["D"]), n=int(len(Xb)))


def prepare(model, calib, cfg, z0):
    m = load_model(model)
    cal = load_config(calib)
    f_out, c_out = cal["focal_length"], cal["principal_point"]
    assert np.allclose(m["K"], [*f_out, *c_out], rtol=1e-6), (m["K"], f_out, c_out)
    geo = U.Geometry(cfg, 1.0)          # only theta0, R0, mu_a are used
    decl = Decl(f_out, c_out, geo.R0, geo.mu_a)
    q_ref = m["q"] - 0.5                # COLMAP (0.5, 0.5) pixel centre -> Refrax (0, 0)
    u = raw_from_corrected(q_ref, geo.theta0, decl, z0)
    ok = np.isfinite(u).all(1)
    return m, decl, geo, u, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("calib")
    ap.add_argument("config")
    ap.add_argument("z0", type=float)
    ap.add_argument("out")
    ap.add_argument("--hulls")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--images", default="all", help="all | first | second (sequence half)")
    ap.add_argument("--huber", type=float, default=2.0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="metres per model unit (the refraction needs metric depth)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--tag", default="", help="suffix of the output json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.config)
    m, decl, geo, u, ok = prepare(args.model, args.calib, cfg, args.z0)
    print(f"{len(m['names'])} images, {len(m['X'])} points, {len(u)} observations, "
          f"{(~ok).sum()} unmappable", flush=True)
    sel = ok.copy()
    if args.images != "all":
        order = np.argsort(m["names"])
        half = set(order[: len(order) // 2] if args.images == "first"
                   else order[len(order) // 2:])
        sel &= np.isin(m["ci"], list(half))
    # keep points with >= 2 observations
    cnt = np.bincount(m["pj"][sel], minlength=len(m["X"]))
    sel &= cnt[m["pj"]] >= 2
    used_c = np.unique(m["ci"][sel])
    used_p = np.unique(m["pj"][sel])
    cmap = -np.ones(len(m["names"]), int)
    cmap[used_c] = np.arange(len(used_c))
    pmap = -np.ones(len(m["X"]), int)
    pmap[used_p] = np.arange(len(used_p))
    ci, pj = cmap[m["ci"][sel]], pmap[m["pj"][sel]]
    R0, t0, X0 = m["R"][used_c], args.scale * m["t"][used_c], args.scale * m["X"][used_p]
    bids = None
    out = dict(model=args.model, z0=args.z0, scale=args.scale, images=args.images, n_obs=int(sel.sum()),
               n_images=int(len(used_c)), n_points=int(len(used_p)), variants={})
    if args.hulls:
        hulls = np.load(args.hulls, allow_pickle=True)["hulls"].item()
        b = board_ids(m, hulls)
        bids = pmap[b][pmap[b] >= 0]
        out["input_planarity"] = planarity(X0[bids])
        print("input board planarity", out["input_planarity"], flush=True)
    for name in args.variants:
        free = VARIANTS[name]
        t_start = time.time()
        print(f"== {name}: free {free}", flush=True)
        ba = RefracBA(R0, t0, X0, ci, pj, u[sel], geo.theta0.copy(), decl, free=free,
                      huber=args.huber, sigma_px=None, threads=args.threads)
        ba.solve(max_iter=40)
        sm = ba.summary()
        res = dict(params=sm["params"], names=sm["names"], corr=sm["corr"],
                   rms_px=sm["rms_px"], sigma_px=float(np.sqrt(sm["sigma2"])),
                   seconds=time.time() - t_start, n_iter=len(ba.history) - 1)
        if bids is not None:
            res["planarity"] = planarity(ba.X[bids])
        out["variants"][name] = res
        print(json.dumps({k: v for k, v in res.items() if k != "corr"}), flush=True)
        if args.save_model:
            np.savez_compressed(os.path.join(args.out, f"model_{name.replace(' ', '_')}.npz"),
                                R=ba.R, t=ba.t, X=ba.X, theta=ba.theta,
                                names=np.array(m["names"])[used_c], pid=m["pid"][used_p])
        with open(os.path.join(args.out, f"real_tilt_{args.images}{args.tag}.json"), "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
