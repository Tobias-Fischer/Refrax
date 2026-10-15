"""Figures for docs/explore_sim.md (called from sim.validate --figures)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from core.render import apply_water, _sample, project_underwater

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED = "#222222", "#777777"
BLUES = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])

plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def _err_map(r, data, step=4):
    """Per-pixel output-image error on a subsampled grid (refractive reprojection)."""
    vv, uu = np.mgrid[0:r.H:step, 0:r.W:step]
    S = data["S"][vv, uu]
    e = np.full(vv.shape, np.nan)
    ok = np.all(np.isfinite(S), -1) & data["info"]["valid"][vv, uu] & ~data["info"]["background"][vv, uu]
    u, v = project_underwater(S[ok], r.K, r.housing, uv0=(uu[ok].astype(float), vv[ok].astype(float)))
    e[ok] = np.hypot(u - uu[ok], v - vv[ok])
    return e


def overview(path, r, keep):
    names = [n for n in ("steps", "spheres") if n in keep]
    fig, axes = plt.subplots(len(names), 4, figsize=(17, 2.6 * len(names) + 0.4))
    axes = np.atleast_2d(axes)
    for row, nm in zip(axes, names):
        d = keep[nm]
        e = _err_map(r, d)
        row[0].imshow(np.clip(d["rgb_r"], 0, 1)); row[0].set_title(f"{nm}: sim pinhole render\n{r.spec.W}x{r.spec.H}, f={r.spec.fx:.0f}px")
        row[1].imshow(np.clip(d["img"], 0, 1)); row[1].set_title(f"refracted (RGB-D post-process)\n{r.W}x{r.H}")
        row[2].imshow(np.clip(d["rgb_gt"], 0, 1)); row[2].set_title("exact ray trace through housing")
        im = row[3].imshow(e, cmap=BLUES, vmin=0, vmax=1.0)
        row[3].set_title(f"px error (subsampled x4)\nmedian {np.nanmedian(e):.4f}, p99 {np.nanpercentile(e, 99):.3f}")
        fig.colorbar(im, ax=row[3], fraction=0.03, label="px")
        for a in row:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def parallax(path, r, r0):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    vmax = 1e3 * np.linalg.norm(r0.b, axis=-1).max()
    for a, rr, title in ((axes[0], r0, "render camera at camera centre"),
                         (axes[1], r, f"render camera at virtual centre (z={1e3 * r.spec.centre[2]:.1f} mm)")):
        bm = 1e3 * np.linalg.norm(rr.b, axis=-1)
        im = a.imshow(bm, cmap=BLUES, vmin=0, vmax=vmax)
        a.set_title(f"{title}\nray offset |b|: max {bm.max():.2f} mm, "
                    f"{rr.spec.fx * bm.max() * 1e-3 / 0.3:.1f} render px at 0.3 m", color=INK)
        a.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.025, label="lateral offset of water ray |b| (mm)")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def depth_sweep(path, results):
    an = results["analytic"]
    Zs = sorted({float(k.split("/")[0].split("_")[1].rstrip("m")) for k in an if k.startswith("plane_")})
    if len(Zs) < 2:
        return
    def get(tag, key):
        return [max(an[f"plane_{Z:g}m/{tag}"][key]["max"], 1e-5) for Z in Zs]
    series = [("ours, virtual centre", get("virtual", "px_err_no_edge")),
              ("ours, camera centre", get("origin", "px_err_no_edge")),
              ("central warp exact at infinity", get("virtual", "baseline_central_inf_px_err_no_edge")),
              ("central warp exact at 1 m", get("virtual", "baseline_central_1m_px_err_no_edge"))]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (lab, y), c in zip(series, SERIES):
        ax.plot(Zs, y, "-o", color=c, lw=2, ms=6, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(Zs); ax.set_xticklabels([f"{z:g}" for z in Zs])
    ax.set_xlabel("plane depth Z (m)"); ax.set_ylabel("max pixel error in underwater image (px)")
    ax.set_title("Fronto-parallel planes: per-pixel depth marching vs. single-viewpoint warps\n(errors below 1e-5 px are drawn at 1e-5; the two ours lines coincide)", color=INK)
    ax.grid(True, which="major", color="#e6e6e6", lw=0.8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def visualise_cmp(path, vis_imgs):
    img_vis, img_ren = vis_imgs
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
    axes[0].imshow(img_vis, cmap="gray"); axes[0].set_title("visualise_refraction.py (analytic forward map)")
    axes[1].imshow(img_ren, cmap="gray"); axes[1].set_title("pinhole checkerboard render -> FlatPortRenderer")
    im = axes[2].imshow(np.abs(img_vis.astype(int) - img_ren.astype(int)), cmap=BLUES, vmin=0, vmax=255)
    axes[2].set_title("|difference| (only antialiasing at square borders)")
    for a in axes:
        a.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def water(path, keep):
    if "spheres" not in keep:
        return
    d = keep["spheres"]
    clear = np.clip(d["img"].astype(np.float64), 0, 1)
    wet = apply_water(clear, d["info"]["water_range"], beta_D=[0.45, 0.12, 0.08],
                      beta_B=[0.40, 0.10, 0.07], B_inf=[0.05, 0.25, 0.32])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
    axes[0].imshow(clear); axes[0].set_title("refracted, clear water")
    axes[1].imshow(np.clip(wet, 0, 1)); axes[1].set_title("+ attenuation/backscatter on refracted water path")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_figures(out_dir, r, r0, keep, vis_imgs, results):
    os.makedirs(out_dir, exist_ok=True)
    overview(os.path.join(out_dir, "sim_overview.png"), r, keep)
    parallax(os.path.join(out_dir, "sim_parallax.png"), r, r0)
    depth_sweep(os.path.join(out_dir, "sim_depth_sweep.png"), results)
    visualise_cmp(os.path.join(out_dir, "sim_visualise_consistency.png"), vis_imgs)
    water(os.path.join(out_dir, "sim_water.png"), keep)
    print(f"figures written to {out_dir}")
