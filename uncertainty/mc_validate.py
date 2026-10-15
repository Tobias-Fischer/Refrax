"""Monte Carlo validation of the first-order per-pixel covariance (section 3 of
docs/explore_uncertainty.md).

    python uncertainty/mc_validate.py [configs/dataset.yaml] [--n 4000] [--stride 16]

For fixed-depth mode (z0 = 1 m, camera centre and virtual centre) the true parameters are
sampled, every raw pixel is pushed exactly through the true model (no linearisation:
closed-form forward model + Newton undistortion), and the per-pixel sample covariance of
dq = q_true - q is compared with the linear propagation J Sigma J^T. Cases:

  gauss       Gaussian priors (what the linear model assumes)
  realistic   mu_w uniform over S 0-40 PSU / T 0-30 C / 450-650 nm, mu_g uniform over glass
              types, depth log-normal (global 25 %, local 10 %), others Gaussian
  gauss x3    Gaussian priors x 3 (where does the linearisation start to break?)
  depth heavy realistic, but depth log-normal with 70 % global / 30 % local spread

"non-absorbable": every MC field is reduced by its own least-squares P4 fit (fx, fy, cx,
cy) over the image, and compared with the linear residual covariance.
Writes docs/figures/mc_validation.json and mc_*.png.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from core.config import load_config  # noqa: E402
from core import uncertainty as U  # noqa: E402

FIG = os.path.join(REPO, "docs", "figures")
GLASS = np.array([1.458, 1.47, 1.49, 1.517])   # fused silica, borosilicate, PMMA, BK7


def linear_cov(geo, ur, vr, qx, qy, prior, scale=1.0, lognormal=False, nodes=1):
    """Linear full and non-absorbable covariances (N, 3). lognormal: depth spreads are
    log-normal in Z (exact RMS of w); nodes: Gauss-Hermite nodes over the global depth."""
    pr = copy.deepcopy(prior)
    pr.sigma = {k: v * scale for k, v in prior.sigma.items()}
    pr.depth_global, pr.depth_local = prior.depth_global * scale, prior.depth_local * scale
    if not lognormal:  # Gaussian in w (additive): undo the log-normal conversions
        pr.depth_global = _inv_w_rms(pr.depth_global)
        pr.depth_local = _inv_w_rms(pr.depth_local) / np.exp(pr.depth_global ** 2)
    return U.propagate(geo, ur, vr, 1.0 / geo.z_ref, pr, qx, qy, nodes=nodes)


def _inv_w_rms(s):
    from scipy.optimize import brentq
    return brentq(lambda g: U.w_rms(g) - s, 0.0, 5.0) if s > 0 else 0.0


def sample_theta(rng, n, geo, prior, case):
    th0 = geo.theta0
    sig = prior.vector()
    k = 3.0 if case == "gauss x3" else 1.0
    th = th0 + k * sig * rng.standard_normal((n, len(th0)))
    g = k * prior.depth_global * rng.standard_normal(n)
    if case in ("realistic", "depth heavy"):
        S, T, L = rng.uniform(0, 40, n), rng.uniform(0, 30, n), rng.uniform(450, 650, n)
        mw = U.water_index(S, T, L)
        th[:, 3] = th0[3] + (mw - mw.mean())       # same spread, centred on the nominal
        mg = rng.choice(GLASS, n)
        th[:, 2] = th0[2] + (mg - GLASS.mean())
    return th, g


def mc(geo, ur, vr, prior, case, n, rng, basis_pinv, Bs, chunk=250):
    """MC sums of dq and its P4 residual: returns (cov_full, cov_np) (N, 3) and samples."""
    th, g = sample_theta(rng, n, geo, prior, case)
    k = 3.0 if case == "gauss x3" else 1.0
    u = torch.as_tensor(ur)[None]
    v = torch.as_tensor(vr)[None]
    q0 = torch.stack(U.corrected_xy(torch.as_tensor(geo.theta0), u[0], v[0],
                                    torch.full_like(u[0], 1.0 / geo.z_ref), geo), -1)
    S_f = np.zeros((len(ur), 3))
    S_n = np.zeros((len(ur), 3))
    keep = []
    for s in range(0, n, chunk):
        t = torch.as_tensor(th[s:s + chunk]).T[:, :, None]          # (P, b, 1)
        b = t.shape[1]
        loc = k * prior.depth_local * torch.randn(b, len(ur), dtype=torch.float64,
                                                  generator=torch.Generator().manual_seed(s))
        if case in ("realistic", "depth heavy"):
            w = (1.0 / geo.z_ref) * torch.exp(-torch.as_tensor(g[s:s + chunk])[:, None] - loc)
        else:  # linear in w: Gaussian in inverse depth
            w = (1.0 / geo.z_ref) * (1 - torch.as_tensor(g[s:s + chunk])[:, None] - loc)
        qx, qy = U.corrected_xy(t, u, v, w, geo)
        d = torch.stack([qx, qy], -1) - q0[None]                    # (b, N, 2)
        d = d.numpy()
        # Per-sample P4 fit of the systematic part only (the local term is per pixel).
        beta = np.einsum("kj,bj->bk", basis_pinv, d.reshape(b, -1))
        r = d - np.einsum("njk,bk->bnj", Bs, beta)
        for D, S in ((d, S_f), (r, S_n)):
            S[:, 0] += (D[..., 0] ** 2).sum(0)
            S[:, 1] += (D[..., 0] * D[..., 1]).sum(0)
            S[:, 2] += (D[..., 1] ** 2).sum(0)
        keep.append(d)
    return S_f / n, S_n / n, np.concatenate(keep)


def compare(Sm, Sl):
    tr_m, tr_l = Sm[:, 0] + Sm[:, 2], Sl[:, 0] + Sl[:, 2]
    ratio = np.sqrt(tr_m / tr_l)
    fro = np.sqrt(((Sm - Sl) ** 2 * [1, 2, 1]).sum(-1) / ((Sl ** 2) * [1, 2, 1]).sum(-1))
    return dict(ratio_med=float(np.median(ratio)), ratio_p5=float(np.percentile(ratio, 5)),
                ratio_p95=float(np.percentile(ratio, 95)), ratio_min=float(ratio.min()),
                ratio_max=float(ratio.max()), fro_med=float(np.median(fro)),
                fro_p95=float(np.percentile(fro, 95)),
                sigma_rms=float(np.sqrt(np.mean(tr_l))))


def nees(d, Sl):
    """Mean NEES and the fraction inside the 95 % chi^2(2) ellipse."""
    a, b, c = Sl[:, 0], Sl[:, 1], Sl[:, 2]
    det = a * c - b * b
    x, y = d[..., 0], d[..., 1]
    m = (c * x * x - 2 * b * x * y + a * y * y) / det
    return float(m.mean()), float((m < 5.991).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=os.path.join(REPO, "configs/dataset.yaml"))
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--stride", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(args.config)
    prior = U.default_prior(cfg)
    out = {}
    maps = {}
    for centre, cz in (("cc", 0.0), ("vc", U.optimal_center_z(cfg))):
        geo, canvas = U.setup(cfg, center_z=cz)
        xs, ys = U.source_pixels(geo, canvas, geo.z_ref)
        xs, ys = xs[::args.stride, ::args.stride], ys[::args.stride, ::args.stride]
        qx, qy = geo.out_grid(args.stride)
        ok = np.isfinite(xs)
        ur, vr = U.raw_from_ideal(xs[ok], ys[ok], geo)
        t0 = time.time()
        Bs = U.basis(qx[ok], qy[ok], geo, ("P4",))
        pinv = np.linalg.pinv(Bs.reshape(-1, Bs.shape[-1]))
        for case in ("gauss", "realistic", "gauss x3", "depth heavy naive", "depth heavy",
                     "depth heavy quad"):
            sc = 3.0 if case == "gauss x3" else 1.0
            pr = prior
            if case.startswith("depth heavy"):  # z0 off by ~2x, 30 % relief, log-normal in Z
                pr = copy.deepcopy(prior)
                pr.depth_global, pr.depth_local = 0.7, 0.3
            Lf, Ln = linear_cov(geo, ur, vr, qx[ok], qy[ok], pr, sc,
                                lognormal=case in ("realistic", "depth heavy", "depth heavy quad"),
                                nodes=5 if case == "depth heavy quad" else 1)
            rng = np.random.default_rng(1)
            t1 = time.time()
            Mf, Mn, D = mc(geo, ur, vr, pr, case.replace(" naive", "").replace(" quad", ""), args.n, rng, pinv, Bs)
            res = dict(full=compare(Mf, Lf), np=compare(Mn, Ln),
                       nees_full=nees(D, Lf), mc_seconds=time.time() - t1, n=args.n)
            out[f"{centre} {case}"] = res
            print(centre, case, json.dumps(res), flush=True)
            if centre == "cc":
                maps[case] = dict(Lf=Lf, Ln=Ln, Mf=Mf, Mn=Mn)
        out[f"{centre} linear_seconds"] = time.time() - t0
    with open(os.path.join(FIG, "mc_validation.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(REPO, "uncertainty", ".mc_maps.npz"), ok=ok,
                        **{f"{c}_{k}": v for c, d in maps.items() for k, v in d.items()})


if __name__ == "__main__":
    main()
