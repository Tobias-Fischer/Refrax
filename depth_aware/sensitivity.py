"""How accurate does depth need to be? Forward-model sensitivity of the Refrax correction.

    python depth_aware/sensitivity.py [--config configs/dataset.yaml]

For a pixel that sees a point at true depth Z, correcting with an assumed depth Z'
puts it at map(Z') instead of map(Z); the difference (in corrected-image pixels,
zoom included) is the residual refraction error. Reported for relative depth errors
and for one fixed z0 over a depth range, at the camera centre (Refrax) and at the
optimal virtual centre (core/virtual_center.py). "non-pinhole" removes the best
isotropic zoom about the principal point per depth, i.e. the part a pinhole camera
cannot absorb even with a free focal length.
Writes docs/figures/sens_*.png and docs/figures/sensitivity.json.
"""
import argparse
import json
import os

import numpy as np

from common import REPO, load_cfg
from core.config import camera_params, housing_params
from core.optics import matrix_K
from core.scale import select_zoom
from core.undistort import compute_housing_geometry, forward_map
from core.virtual_center import optimal_center_z
from plotstyle import plt, SERIES, INK2

FIG = os.path.join(REPO, "docs", "figures")


def setup(cfg, sub=4):
    W, H, fx, fy, cx, cy = camera_params(cfg)
    hs = housing_params(cfg)
    _, K_inv = matrix_K(fx, fy, cx, cy)
    P2, rw = compute_housing_geometry(H, W, K_inv, **hs)
    return (W, H, fx, fy, cx, cy, hs, P2[::sub, ::sub].astype(np.float64),
            rw[::sub, ::sub].astype(np.float64))


def disp(P2, rw, fx, fy, cx, cy, Z, Zp, c, zoom, remove_scale=False):
    ax, ay = forward_map(P2, rw, Z, fx, fy, cx, cy, c)
    bx, by = forward_map(P2, rw, Zp, fx, fy, cx, cy, c)
    ax, ay, bx, by = (np.asarray(a, np.float64) for a in (ax, ay, bx, by))
    if remove_scale:  # best isotropic scale of b about (cx, cy) onto a
        da, db = np.stack([ax - cx, ay - cy]), np.stack([bx - cx, by - cy])
        k = np.nansum(da * db) / np.nansum(db * db)
        bx, by = cx + k * (bx - cx), cy + k * (by - cy)
    return zoom * np.hypot(ax - bx, ay - by)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    W, H, fx, fy, cx, cy, hs, P2, rw = setup(cfg)
    zoom, _ = select_zoom(1.0, W, H, fx, fy, cx, cy, hs, in_bounds=False)
    c_opt, rms_c, rms_0 = optimal_center_z(W, H, fx, fy, cx, cy, hs, z_range=(0.3, 8.0))
    res = dict(zoom=zoom, center_opt_m=c_opt, d0=hs["d0"])
    print(f"zoom {zoom:.4f}, optimal virtual centre {c_opt * 1000:.2f} mm")

    depths = [0.3, 0.5, 1.0, 2.0, 4.0, 8.0]
    errs = [0.05, 0.10, 0.25, 0.50]
    table = []
    for c, cname in ((0.0, "camera centre"), (c_opt, "optimal centre")):
        for Z in depths:
            for e in errs:
                d = disp(P2, rw, fx, fy, cx, cy, Z, Z * (1 + e), c, zoom)
                dn = disp(P2, rw, fx, fy, cx, cy, Z, Z * (1 + e), c, zoom, remove_scale=True)
                table.append(dict(center=cname, Z=Z, err=e, rms=float(np.sqrt(np.nanmean(d ** 2))),
                                  max=float(np.nanmax(d)), rms_nonpinhole=float(np.sqrt(np.nanmean(dn ** 2))),
                                  max_nonpinhole=float(np.nanmax(dn))))
    res["relative_error_table"] = table
    print(f"{'centre':15s} {'Z':>5s} {'err':>5s} {'rms':>7s} {'max':>7s} {'rms_np':>7s} {'max_np':>7s}")
    for r in table:
        print(f"{r['center']:15s} {r['Z']:5.1f} {r['err']:5.2f} {r['rms']:7.2f} {r['max']:7.2f} "
              f"{r['rms_nonpinhole']:7.2f} {r['max_nonpinhole']:7.2f}")

    # Figure 1: error vs true depth for 10 % and 25 % depth error, both centres
    zs = np.geomspace(0.2, 10, 40)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, e in zip(axs, (0.10, 0.25)):
        for c, lab, col in ((0.0, "camera centre (Refrax)", SERIES[0]),
                            (c_opt, f"virtual centre {c_opt * 1000:.1f} mm", SERIES[1])):
            mx = [np.nanmax(disp(P2, rw, fx, fy, cx, cy, z, z * (1 + e), c, zoom)) for z in zs]
            rm = [np.sqrt(np.nanmean(disp(P2, rw, fx, fy, cx, cy, z, z * (1 + e), c, zoom) ** 2))
                  for z in zs]
            ax.loglog(zs, mx, color=col, label=f"{lab}, max")
            ax.loglog(zs, rm, color=col, ls="--", label=f"{lab}, RMS")
        ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
        ax.text(0.21, 0.55, "0.5 px (typical SfM reprojection error)", color=INK2, fontsize=7)
        ax.set(title=f"depth wrong by {int(e * 100)} %", xlabel="true depth Z [m]")
    axs[0].set_ylabel("residual displacement [px, corrected image]")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/sens_relative_error.png", dpi=150)

    # Figure 2: fixed z0 = 1 m, error vs true depth
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for c, lab, col in ((0.0, "camera centre (Refrax)", SERIES[0]),
                        (c_opt, f"virtual centre {c_opt * 1000:.1f} mm", SERIES[1])):
        mx = [np.nanmax(disp(P2, rw, fx, fy, cx, cy, z, 1.0, c, zoom)) for z in zs]
        np_ = [np.nanmax(disp(P2, rw, fx, fy, cx, cy, z, 1.0, c, zoom, True)) for z in zs]
        ax.loglog(zs, mx, color=col, label=f"{lab}, max")
        ax.loglog(zs, np_, color=col, ls="--", label=f"{lab}, max non-pinhole")
    ax.set(xlabel="true depth Z [m]", ylabel="residual [px]", title="fixed z0 = 1 m")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/sens_fixed_z0.png", dpi=150)
    res["fixed_z0_1m"] = {f"{z:g}": dict(
        cam=float(np.nanmax(disp(P2, rw, fx, fy, cx, cy, z, 1.0, 0.0, zoom))),
        opt=float(np.nanmax(disp(P2, rw, fx, fy, cx, cy, z, 1.0, c_opt, zoom))))
        for z in (0.3, 0.5, 2.0, 4.0, 8.0)}

    # Figure 3: spatial pattern, Z = 0.5 m assumed 1.0 m
    d = disp(P2, rw, fx, fy, cx, cy, 0.5, 1.0, 0.0, zoom)
    dc = disp(P2, rw, fx, fy, cx, cy, 0.5, 1.0, c_opt, zoom)
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.8))
    vmax = np.nanmax(d)
    for ax, m, t in ((axs[0], d, "camera centre"), (axs[1], dc, f"virtual centre {c_opt * 1000:.1f} mm")):
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax, extent=(0, W, H, 0))
        ax.set(title=f"true 0.5 m, assumed 1.0 m: {t}", xticks=[], yticks=[])
        ax.grid(False)
    fig.colorbar(im, ax=axs, label="px")
    fig.savefig(f"{FIG}/sens_spatial.png", dpi=150)

    json.dump(res, open(f"{FIG}/sensitivity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
