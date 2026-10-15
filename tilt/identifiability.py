"""Identifiability of the port tilt in BA (docs/explore_tilt.md, §3).

    python tilt/identifiability.py

Fisher information of the refractive BA (tilt/ba.py, raw-pixel residuals, sigma = 0.5 px)
at the true state of the synthetic scenes of uncertainty/synthetic_ba.py, with the points
marginalised (Schur complement) and the gauge fixed. For each configuration it reports the
posterior std of tilt_x / tilt_y and their largest correlations with the other globals.

Configurations: scene (tank: planar board 0.7-1.3 m; reef: 3-D terrain 0.6-2 m), free
globals (tilt only / + in-air K / + d0 / + distortion), poses free vs known (known poses
remove the "constant camera rotation" confound), field of view (all observations vs the
central part of the raw image), distance (scene scaled x3) and number of views.
Writes docs/figures/tilt_identifiability.json.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "uncertainty"))
from core.config import load_config  # noqa: E402
from core import uncertainty as U  # noqa: E402
from tilt.model import Decl  # noqa: E402
from tilt.ba import RefracBA  # noqa: E402
import synthetic_ba as S  # noqa: E402

SIG = 0.5
FREE = {
    "tilt": ("tilt_x", "tilt_y"),
    "tilt+K": ("tilt_x", "tilt_y", "fx", "fy", "cx", "cy"),
    "tilt+K+d0": ("tilt_x", "tilt_y", "d0", "fx", "fy", "cx", "cy"),
    "tilt+K+dist": ("tilt_x", "tilt_y", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "tilt+K+d0+mu_w": ("tilt_x", "tilt_y", "d0", "mu_w", "fx", "fy", "cx", "cy"),
}


def observations(X, cams, th, geo, fov=1.0):
    """Noise-free raw observations (ci, pj, u) of the scene; fov < 1 keeps only raw
    keypoints within that fraction of the half-diagonal around the image centre."""
    ci, pj, uu = [], [], []
    for c, (R, t) in enumerate(cams):
        Xc = X @ R.T + t
        idx = np.flatnonzero(Xc[:, 2] > 0.2)
        Xn = Xc[idx, :2] / Xc[idx, 2:]
        m = np.hypot(*Xn.T) < 1.0
        idx, Xn = idx[m], Xn[m]
        u = S.raw_of(Xn, 1.0 / Xc[idx, 2], th, geo)
        ok = np.isfinite(u).all(1) & (u[:, 0] > 2) & (u[:, 0] < geo.W - 3) & \
            (u[:, 1] > 2) & (u[:, 1] < geo.H - 3)
        rr = np.hypot((u[:, 0] - geo.W / 2) / (geo.W / 2), (u[:, 1] - geo.H / 2) / (geo.H / 2))
        ok &= rr / np.sqrt(2) <= fov
        ci.append(np.full(ok.sum(), c))
        pj.append(idx[ok])
        uu.append(u[ok])
    ci, pj, uu = np.concatenate(ci), np.concatenate(pj), np.concatenate(uu)
    cnt = np.bincount(pj, minlength=len(X))
    k = cnt[pj] >= 2
    ci, pj, uu = ci[k], pj[k], uu[k]
    used = np.unique(pj)
    remap = -np.ones(len(X), int)
    remap[used] = np.arange(len(used))
    return ci, remap[pj], uu, used


def info(X, cams, th, geo, free, poses_known=False, fov=1.0):
    ci, pj, u, used = observations(X, cams, th, geo, fov)
    # drop views without observations (narrow field of view)
    keep = np.unique(ci)
    cmap = -np.ones(len(cams), int)
    cmap[keep] = np.arange(len(keep))
    ci = cmap[ci]
    cams = [cams[k] for k in keep]
    R = np.stack([c[0] for c in cams])
    t = np.stack([c[1] for c in cams])
    decl = Decl(geo.f_out, geo.c_out, geo.R0, geo.mu_a)
    fixed = tuple(range(len(cams))) if poses_known else (0,)
    ba = RefracBA(R, t, X[used], ci, pj, u, th, decl, free=free, huber=None, sigma_px=SIG,
                  verbose=False)
    ba.fixed = set(fixed)
    ba.slot = -np.ones(ba.nc, int)
    fc = [i for i in range(ba.nc) if i not in ba.fixed]
    ba.slot[fc] = np.arange(len(fc))
    ba.nf = len(fc)
    if poses_known:
        ba.linearize()
        Sm, _, _, _, _ = ba.normal_equations(0.0)
        d = np.sqrt(np.diag(Sm))
        Cg = np.linalg.inv(Sm / np.outer(d, d)) / np.outer(d, d) * SIG ** 2
    else:
        Cg = ba.covariance()
    sd = np.sqrt(np.diag(Cg))
    corr = Cg / np.outer(sd, sd)
    out = dict(n_obs=int(len(u)), sd_tilt_deg=[float(np.degrees(sd[0])), float(np.degrees(sd[1]))])
    names = list(free)
    for i, n in enumerate(names[:2]):
        c = [(names[j], round(float(corr[i, j]), 3)) for j in range(2, len(names))]
        c.sort(key=lambda x: -abs(x[1]))
        out[f"corr_{n}"] = c[:3]
    if len(names) >= 2:
        out["corr_tx_ty"] = float(corr[0, 1])
    return out


def main():
    cfg = load_config(os.path.join(REPO, "configs/dataset.yaml"))
    res = {}
    for kind in ("tank", "reef"):
        rng = np.random.default_rng(10)
        X, cams = S.make_scene(kind, rng)
        Zs = np.concatenate([(X @ R.T + t)[:, 2] for R, t in cams])
        z0 = float(np.median(Zs[Zs > 0.2]))
        geo, _ = U.setup(dict(cfg, correction=dict(cfg["correction"], z0_fixed=z0)))
        th = geo.theta0.copy()
        th[4] = th[5] = np.radians(0.5)
        confs = {}
        for fname, free in FREE.items():
            confs[f"{fname}, poses free"] = info(X, cams, th, geo, free)
        confs["tilt+K, poses known"] = info(X, cams, th, geo, FREE["tilt+K"], poses_known=True)
        confs["tilt, poses known"] = info(X, cams, th, geo, FREE["tilt"], poses_known=True)
        for fov in (0.6, 0.4):
            confs[f"tilt+K, poses free, central {int(fov * 100)}% of the field"] = \
                info(X, cams, th, geo, FREE["tilt+K"], fov=fov)
        # 3x farther: scale the scene (camera centres too)
        X3 = 3 * X
        cams3 = [(R, 3 * t) for R, t in cams]
        confs["tilt+K, poses free, distance x3"] = info(X3, cams3, th, geo, FREE["tilt+K"])
        confs["tilt+K, poses free, 1/4 of the views"] = info(X, cams[::4], th, geo,
                                                             FREE["tilt+K"])
        res[kind] = dict(z0=z0, n_cams=len(cams), confs=confs)
        print(f"\n{kind} (z0 {z0:.2f} m, {len(cams)} views), sigma {SIG} px")
        for k, v in confs.items():
            print(f"  {k:52s} sd tilt x/y {v['sd_tilt_deg'][0]:.4f} / {v['sd_tilt_deg'][1]:.4f} deg"
                  f"  ({v['n_obs']} obs)  corr x: {v.get('corr_tilt_x')}  y: {v.get('corr_tilt_y')}",
                  flush=True)
    with open(os.path.join(REPO, "docs/figures/tilt_identifiability.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
