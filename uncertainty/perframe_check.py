"""Per-frame (depth-map) covariance: accuracy of the fast path and Monte Carlo check.

    python uncertainty/perframe_check.py DEPTH.npy [configs/dataset.yaml] [--device mps]

1. FastCov evaluates the parameter terms at the *reference* sources (depth z0); the exact
   per-frame value uses the frame's own sources. Both are compared on the full canvas.
2. Monte Carlo over the parameters, a per-frame depth scale (log-normal) and independent
   per-pixel depth noise (log-normal) on a stride-16 grid, against the linear full
   covariance with the frame's depth map.
3. Timing of the per-frame GPU step.
Writes docs/figures/perframe_check.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from core.config import load_config  # noqa: E402
from core.flatport import InvDepth  # noqa: E402
from core import uncertainty as U  # noqa: E402
from refraction_uncertainty import CovarianceMaps  # noqa: E402
from uncertainty.mc_validate import compare, nees  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("depth")
    ap.add_argument("config", nargs="?", default=os.path.join(REPO, "configs/dataset.yaml"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    prior = U.default_prior(None, {"depth_global": 0.10, "depth_local": 0.03})
    depth = np.load(args.depth).astype(np.float32)
    cm = CovarianceMaps(cfg, prior, device=args.device)
    geo = cm.geo
    W, H, fx, fy, cx, cy = cm.cam
    res = {}

    # 1. reference-source approximation
    full, npc, ok = cm.maps(depth)
    xs, ys = cm.source(depth)
    ok2, ur, vr = cm._valid(xs, ys)
    assert (ok == ok2).all()
    qx, qy = geo.out_grid(1)
    w = InvDepth(depth, fx, fy, cx, cy).sample((xs[ok] - cx) / fx, (ys[ok] - cy) / fy)
    ex_terms = cm.fc.terms(ur, vr, qx[ok], qy[ok])
    ef, en = cm.fc(w, terms=ex_terms, qxy=(cm.fc._t(qx[ok]), cm.fc._t(qy[ok])))
    ef, en = ef.cpu().numpy(), en.cpu().numpy()
    for name, A, E in (("full", full[ok], ef), ("np", npc[ok], en)):
        sa, se = np.sqrt(A[:, 0] + A[:, 2]), np.sqrt(E[:, 0] + E[:, 2])
        rel = np.abs(sa / se - 1)
        res[f"fast_vs_exact_{name}"] = dict(rel_med=float(np.median(rel)),
                                            rel_p99=float(np.percentile(rel, 99)),
                                            rel_max=float(rel.max()),
                                            sigma_med=float(np.median(se)))
    src_shift = np.hypot(*(np.stack(cm.source(geo.z_ref)) - np.stack([xs, ys]))[:, ok])
    res["source_shift_px"] = dict(med=float(np.nanmedian(src_shift)),
                                  max=float(np.nanmax(src_shift)))

    # 2. Monte Carlo on a grid
    st = 16
    sub = np.zeros_like(ok)
    sub[::st, ::st] = True
    m = sub[ok]
    u, v, ww = ur[m], vr[m], w[m]
    L = ef[m]
    th, rng = geo.theta0, np.random.default_rng(3)
    sig = prior.vector()
    S = np.zeros((m.sum(), 3))
    keep = []
    T = torch.as_tensor
    q0 = torch.stack(U.corrected_xy(T(th), T(u), T(v), T(ww), geo), -1)
    for s in range(0, args.n, 250):
        b = min(250, args.n - s)
        tt = T(th + sig * rng.standard_normal((b, len(th)))).T[:, :, None]
        g = prior.depth_global * rng.standard_normal((b, 1))
        loc = prior.depth_local * rng.standard_normal((b, len(u)))
        wt = T(ww[None] * np.exp(-g - loc))
        qx_, qy_ = U.corrected_xy(tt, T(u)[None], T(v)[None], wt, geo)
        d = (torch.stack([qx_, qy_], -1) - q0[None]).numpy()
        S[:, 0] += (d[..., 0] ** 2).sum(0)
        S[:, 1] += (d[..., 0] * d[..., 1]).sum(0)
        S[:, 2] += (d[..., 1] ** 2).sum(0)
        keep.append(d)
    D = np.concatenate(keep)
    res["mc_full"] = compare(S / args.n, L)
    res["mc_nees"] = nees(D, L)
    res["depth_terms_note"] = "prior depth_global 0.10, depth_local 0.03 (log-normal)"

    # 3. timing of the per-frame GPU step (w -> covariances) and the full maps() call
    wt = torch.as_tensor(w, dtype=cm.fc.dtype, device=cm.fc.device)
    kw = dict(terms=ex_terms, qxy=(cm.fc._t(qx[ok]), cm.fc._t(qy[ok])))
    cm.fc(wt, **kw)
    ts = []
    for _ in range(10):
        U_sync(args.device)
        t0 = time.perf_counter()
        f_, n_ = cm.fc(wt, **kw)
        U_sync(args.device)
        ts.append(time.perf_counter() - t0)
    res["ms_cov_step"] = 1e3 * float(np.median(ts))
    t0 = time.perf_counter()
    cm.maps(depth)
    res["ms_maps_total"] = 1e3 * (time.perf_counter() - t0)
    res["n_pixels"] = int(ok.sum())
    print(json.dumps(res, indent=1))
    with open(os.path.join(REPO, "docs", "figures", "perframe_check.json"), "w") as f:
        json.dump(res, f, indent=1)


def U_sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
