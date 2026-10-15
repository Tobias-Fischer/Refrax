"""Synthetic experiment: depth-aware vs fixed-depth correction with known geometry.

    python depth_aware/synthetic.py [--config configs/dataset.yaml]

Scenes are height fields in the camera frame (a tilted seabed with coral-bommie-like
bumps). Every ideal raw pixel is traced through the flat port (core/optics model)
onto the scene, which gives its true 3D point and the true depth map on the raw
grid. A correction with assumed depth Z' maps the pixel to map(Z'); the truth is the
pinhole projection of the 3D point from the same virtual centre. The residual is the
error left in the corrected image. "zoom-compensated" additionally removes the best
single focal-length change over all scenes (what a calibration refinement could
absorb), leaving the non-pinhole part that bends SfM reconstructions.
Depth perturbations: global scale error (metric-scale uncertainty), smooth relative
error (mono-depth-like, low frequency) and per-pixel noise.
Writes docs/figures/synthetic_*.png and docs/figures/synthetic.json.
"""
import argparse
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter

from common import REPO, load_cfg
from core.config import camera_params, housing_params
from core.optics import matrix_K
from core.scale import select_zoom
from core.undistort import compute_housing_geometry, forward_map
from core.virtual_center import optimal_center_z
from plotstyle import plt, SERIES

FIG = os.path.join(REPO, "docs", "figures")
SUB = 4


def scene_surface(kind):
    """Z = f(X, Y) in the camera frame (metres)."""
    if kind == "near_reef":   # seabed 0.9 m, tilted 25 deg, two bommies up to 0.45 m tall
        def f(X, Y):
            Z = 0.9 + np.tan(np.radians(25)) * Y
            Z -= 0.45 * np.exp(-((X - 0.15) ** 2 + (Y + 0.05) ** 2) / (2 * 0.12 ** 2))
            Z -= 0.25 * np.exp(-((X + 0.35) ** 2 + (Y - 0.15) ** 2) / (2 * 0.10 ** 2))
            return Z
    elif kind == "far_reef":  # same shape, 4x further (seabed 3.6 m, bommies 1.8 m)
        g = scene_surface("near_reef")
        def f(X, Y):
            return 4 * g(X / 4, Y / 4)
    elif kind == "flat_1m":
        def f(X, Y):
            return np.full_like(X, 1.0)
    else:
        raise ValueError(kind)
    return f


def trace_scene(P2, rw, f, z_max=12.0, n_march=600):
    """First intersection of each water ray with the height field Z = f(X, Y)
    (ray marching in depth, then bisection); occlusions are handled correctly."""
    g = lambda t: (P2[..., 2] + t * rw[..., 2]) - f(P2[..., 0] + t * rw[..., 0],
                                                  P2[..., 1] + t * rw[..., 1])
    ts = np.linspace(0.0, z_max, n_march)
    lo = np.zeros(P2.shape[:2])
    hi = np.full(P2.shape[:2], np.nan)
    prev = g(lo)
    for a, b in zip(ts[:-1], ts[1:]):
        cur = g(np.full_like(lo, b / 1.0))
        new = np.isnan(hi) & (prev < 0) & (cur >= 0)
        lo = np.where(new, a, lo)
        hi = np.where(new, b, hi)
        prev = np.where(np.isnan(hi), cur, prev)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        neg = g(mid) < 0
        lo, hi = np.where(neg, mid, lo), np.where(neg, hi, mid)
    t = 0.5 * (lo + hi)
    return P2 + t[..., None] * rw


def project(P, fx, fy, cx, cy, c):
    Zc = P[..., 2] - c
    return fx * P[..., 0] / Zc + cx, fy * P[..., 1] / Zc + cy


def residual(P2, rw, P, Zassumed, fx, fy, cx, cy, c, zoom):
    mx, my = forward_map(P2, rw, Zassumed, fx, fy, cx, cy, c)
    tx, ty = project(P, fx, fy, cx, cy, c)
    return zoom * (np.asarray(mx, np.float64) - tx), zoom * (np.asarray(my, np.float64) - ty), tx, ty


def summarise(ex, ey, tx, ty, cx, cy, k=None):
    """RMS / max error; with k, after scaling the corrected positions by k about (cx, cy)."""
    if k is not None:
        mx, my = tx + ex, ty + ey
        ex, ey = cx + k * (mx - cx) - tx, cy + k * (my - cy) - ty
    e = np.hypot(ex, ey)
    return float(np.sqrt(np.nanmean(e ** 2))), float(np.nanmax(e))


def best_k(items, cx, cy):
    num = den = 0.0
    for ex, ey, tx, ty in items:
        mx, my = tx + ex - cx, ty + ey - cy
        num += np.nansum(mx * (tx - cx) + my * (ty - cy))
        den += np.nansum(mx * mx + my * my)
    return num / den


def nonpinhole(P2, rw, P, Zassumed, fx, fy, cx, cy, c, zoom, cands=np.linspace(-0.03, 0.005, 36)):
    """Distance of the corrected image from the closest pinhole camera.

    The corrected positions are compared with the projections of the true points
    from every candidate centre c' on the optical axis, each with its best focal
    scale k; returns (rms, max, c') of the best candidate. Any pinhole is fine for
    SfM (a centre offset of a few mm is a rigid shift of the camera), so this is the
    error that can actually bend a reconstruction.
    """
    mx, my = forward_map(P2, rw, Zassumed, fx, fy, cx, cy, c)
    mx, my = np.asarray(mx, np.float64), np.asarray(my, np.float64)
    best = None
    for cp in cands:
        tx, ty = project(P, fx, fy, cx, cy, cp)
        ex, ey = mx - tx, my - ty
        k = best_k([(ex, ey, tx, ty)], cx, cy)
        r, m = summarise(zoom * ex, zoom * ey, zoom * tx, zoom * ty, zoom * cx, zoom * cy, k)
        if best is None or r < best[0]:
            mx2, my2 = cx + k * (mx - cx), cy + k * (my - cy)
            best = (r, m, float(cp), zoom * np.hypot(mx2 - tx, my2 - ty))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    W, H, fx, fy, cx, cy = camera_params(cfg)
    hs = housing_params(cfg)
    _, K_inv = matrix_K(fx, fy, cx, cy)
    P2, rw = compute_housing_geometry(H, W, K_inv, **hs)
    P2, rw = P2[::SUB, ::SUB].astype(np.float64), rw[::SUB, ::SUB].astype(np.float64)
    zoom, _ = select_zoom(1.0, W, H, fx, fy, cx, cy, hs, in_bounds=False)
    c_opt = optimal_center_z(W, H, fx, fy, cx, cy, hs, z_range=(0.3, 8.0))[0]
    rng = np.random.default_rng(0)

    out = dict(zoom=zoom, c_opt=c_opt, scenes={})
    lines = []
    for scene in ("near_reef", "far_reef", "flat_1m"):
        P = trace_scene(P2, rw, scene_surface(scene))
        Zt = P[..., 2]
        smooth = gaussian_filter(rng.standard_normal(Zt.shape), 25)
        smooth /= smooth.std()
        white = rng.standard_normal(Zt.shape)
        med = float(np.median(Zt))
        methods = {
            "fixed z0 = 1 m (config)": lambda: 1.0,
            "fixed z0 = median depth": lambda: med,
            "depth-aware, exact": lambda: Zt,
            "depth-aware, scale x0.8": lambda: 0.8 * Zt,
            "depth-aware, scale x1.25": lambda: 1.25 * Zt,
            "depth-aware, scale x2": lambda: 2.0 * Zt,
            "depth-aware, smooth 10% err": lambda: Zt * (1 + 0.10 * smooth),
            "depth-aware, pixel 10% noise": lambda: Zt * (1 + 0.10 * white),
            "depth-aware, scale x1.25 + smooth 10%": lambda: 1.25 * Zt * (1 + 0.10 * smooth),
        }
        res = {}
        for c, cname in ((0.0, "camera centre"), (c_opt, "optimal centre")):
            for mname, fn in methods.items():
                ex, ey, tx, ty = residual(P2, rw, P, fn(), fx, fy, cx, cy, c, zoom)
                r, m = summarise(ex, ey, tx, ty, cx, cy)
                k = best_k([(ex, ey, tx, ty)], cx, cy)
                rk, mk = summarise(ex, ey, tx, ty, cx, cy, k)
                nr, nm, nc, nmap = nonpinhole(P2, rw, P, fn(), fx, fy, cx, cy, c, zoom)
                res[f"{cname} | {mname}"] = dict(rms=r, max=m, rms_zoomcomp=rk, max_zoomcomp=mk,
                                                 rms_nonpinhole=nr, max_nonpinhole=nm,
                                                 best_pinhole_center=nc)
                lines.append(f"{scene:9s} {cname:15s} {mname:40s} {r:6.2f} {m:6.2f} {rk:6.2f} {mk:6.2f}"
                             f" {nr:6.2f} {nm:6.2f} {nc * 1000:6.1f}")
                if scene == "near_reef" and (
                        (c == 0.0 and mname in ("fixed z0 = median depth", "depth-aware, smooth 10% err"))
                        or (c != 0.0 and mname == "depth-aware, smooth 10% err")):
                    res.setdefault("_maps", {})[f"{mname} ({cname})"] = nmap
        maps = res.pop("_maps", None)
        out["scenes"][scene] = dict(depth_range=[float(np.nanpercentile(Zt, 1)),
                                                 float(np.nanpercentile(Zt, 99))],
                                    median=med, results=res)
        if maps:
            fig, axs = plt.subplots(1, 4, figsize=(15, 2.5))
            im0 = axs[0].imshow(Zt, cmap="viridis_r", extent=(0, W, H, 0))
            fig.colorbar(im0, ax=axs[0], label="true depth [m]")
            vmax = max(np.nanmax(m) for m in maps.values())
            for ax, (k, m) in zip(axs[1:], maps.items()):
                im = ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax, extent=(0, W, H, 0))
                ax.set_title(f"{k}\nnon-pinhole RMS {np.sqrt(np.nanmean(m ** 2)):.2f} px", fontsize=7)
            fig.colorbar(im, ax=axs[1:], label="non-pinhole residual [px]")
            for ax in axs:
                ax.set(xticks=[], yticks=[])
                ax.grid(False)
            axs[0].set_title("near_reef scene (depth on raw grid)", fontsize=8)
            fig.savefig(f"{FIG}/synthetic_near_reef.png", dpi=150)

    print(f"{'scene':9s} {'centre':15s} {'method':40s} {'rms':>6s} {'max':>6s} {'rms_zc':>6s} {'max_zc':>6s} {'rms_np':>6s} {'max_np':>6s} {'c_mm':>6s}")
    print("\n".join(lines))

    # scale-error sweep (near scene): metric-scale sensitivity of the depth-aware correction
    P = trace_scene(P2, rw, scene_surface("near_reef"))
    Zt = P[..., 2]
    ss = np.geomspace(0.25, 4, 33)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for c, lab, col in ((0.0, "camera centre", SERIES[0]), (c_opt, f"virtual centre {c_opt*1000:.1f} mm", SERIES[1])):
        rr = []
        rk = []
        for s in ss:
            ex, ey, tx, ty = residual(P2, rw, P, s * Zt, fx, fy, cx, cy, c, zoom)
            rr.append(summarise(ex, ey, tx, ty, cx, cy)[0])
            rk.append(summarise(ex, ey, tx, ty, cx, cy, best_k([(ex, ey, tx, ty)], cx, cy))[0])
        ax.loglog(ss, rr, color=col, label=f"{lab}")
        ax.loglog(ss, rk, color=col, ls="--", label=f"{lab}, zoom-compensated")
        out[f"scale_sweep_{'cam' if c == 0 else 'opt'}"] = dict(scale=ss.tolist(), rms=rr, rms_zoomcomp=rk)
    for s0, lab in ((1.0, None),):
        ax.axvline(s0, color="#52514e", lw=0.8, ls=":")
    ax.set(xlabel="depth scale factor (1 = correct metric scale)", ylabel="RMS residual [px]",
           title="near_reef: depth-aware with scale error")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/synthetic_scale_sweep.png", dpi=150)
    json.dump(out, open(f"{FIG}/synthetic.json", "w"), indent=1)


if __name__ == "__main__":
    main()
