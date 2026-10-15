"""Figures for docs/explore_uncertainty.md from the outputs of sensitivity.py,
mc_validate.py, synthetic_ba.py and real_eval.py.

    python uncertainty/figures.py [--shared COV_DIR]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from core import uncertainty as U  # noqa: E402

FIG = os.path.join(REPO, "docs", "figures")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = dict(U.LABELS, z_global="scene depth, global (25 %)",
             z_local="scene depth, per pixel (10 %)")
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "figure.dpi": 130})


def to_img(vals, ok):
    img = np.full(ok.shape, np.nan)
    img[ok] = vals
    return img


def sens_maps():
    z = np.load(os.path.join(REPO, "uncertainty", ".sens_1m.npz"))
    ok = z["cc_ok"]
    names = list(U.PARAMS) + ["z_global", "z_local"]
    show = ["tilt_y", "tilt_x", "mu_w", "d0", "k1", "cx", "z_global", "z_local"]
    fig, ax = plt.subplots(len(show), 3, figsize=(9.6, 2.0 * len(show)))
    for r, p in enumerate(show):
        i = names.index(p)
        fields = [(z["cc_F"][:, :, i], "full, camera centre"),
                  (z["cc_r4"][:, :, i], "non-absorbable (P4), camera centre"),
                  (z["vc_r4"][:, :, i], "non-absorbable (P4), virtual centre")]
        for c, (F, title) in enumerate(fields):
            mag = to_img(np.hypot(F[:, 0], F[:, 1]), z["vc_ok"] if c == 2 else ok)
            a = ax[r, c]
            im = a.imshow(mag, cmap="Blues", vmin=0, vmax=np.nanmax(mag))
            a.set_axis_off()
            if r == 0:
                a.set_title(title, fontsize=9)
            if c == 0:
                a.set_ylabel(NAMES[p], fontsize=8)
            cb = fig.colorbar(im, ax=a, fraction=0.035, pad=0.02)
            cb.ax.tick_params(labelsize=7)
            cb.set_label("px", fontsize=7)
    fig.suptitle("|dq| for a 1-sigma error, scene at 1 m (GoPro 10 config)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sens_maps_1m.png"))
    plt.close(fig)


def sens_rank():
    d = json.load(open(os.path.join(FIG, "sensitivity.json")))
    rows = [r for r in d["rows"] if r["centre"] == "cc" and r["Z"] == 1.0]
    rows.sort(key=lambda r: r["np_rms"])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True)
    ax[0].barh(y, [r["full_rms"] for r in rows], color=C[0], height=0.6)
    ax[0].set_xlabel("full RMS displacement (px)")
    ax[0].set_title("full (what a fixed pinhole sees)")
    h = 0.38
    ax[1].barh(y + h / 2, [r["np_rms"] for r in rows], height=h, color=C[0],
               label="after fx, fy, cx, cy (P4)")
    ax[1].barh(y - h / 2, [r["np_rot_rms"] for r in rows], height=h, color=C[1],
               label="after P4 + constant rotation")
    ax[1].set_xlabel("non-absorbable RMS (px)")
    ax[1].set_title("non-absorbable")
    ax[1].legend(frameon=False, loc="lower right")
    ax[0].set_yticks(y, [NAMES[r["param"]] for r in rows])
    for a in ax:
        a.grid(axis="y", visible=False)
    fig.suptitle("1-sigma prior, scene at 1 m, camera-centre projection", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sens_ranking.png"))
    plt.close(fig)


def sens_depth():
    d = json.load(open(os.path.join(FIG, "sensitivity.json")))
    get = lambda p, c: [next(r["np_rms"] for r in d["rows"] if r["param"] == p and
                             r["centre"] == c and r["Z"] == Z) for Z in (0.3, 0.5, 1, 2, 5)]
    Zs = [0.3, 0.5, 1, 2, 5]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for k, p in enumerate(["z_local", "tilt_y", "tilt_x", "z_global", "k1", "mu_w", "d0"]):
        ax.plot(Zs, get(p, "cc"), color=C[k], lw=2, marker="o", ms=4, label=NAMES[p])
    ax.plot(Zs, get("z_local", "vc"), color=C[0], lw=2, ls="--", marker="o", ms=4,
            label=NAMES["z_local"] + ", virtual centre")
    ax.set(xscale="log", yscale="log", xlabel="scene depth Z (m)",
           ylabel="non-absorbable RMS (px)", xticks=Zs)
    ax.set_xticklabels([str(z) for z in Zs])
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_title("non-absorbable (P4) error vs depth, 1-sigma priors", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sens_depth.png"))
    plt.close(fig)


def mc_maps():
    z = np.load(os.path.join(REPO, "uncertainty", ".mc_maps.npz"))
    ok = z["ok"]
    tr = lambda S: np.sqrt(S[:, 0] + S[:, 2])
    cases = [("gauss", "Gaussian priors"), ("realistic", "realistic (non-Gaussian) priors"),
             ("depth heavy naive", "z0 off ~2x, first-order in Z"),
             ("depth heavy", "z0 off ~2x, exact w-moments")]
    fig, ax = plt.subplots(2, len(cases) + 1, figsize=(13, 4.6))
    for r, kind in enumerate(["f", "n"]):
        L0 = z[f"gauss_L{kind}"]
        im = ax[r, 0].imshow(to_img(tr(L0), ok), cmap="Blues", vmin=0)
        fig.colorbar(im, ax=ax[r, 0], fraction=0.04).set_label("px", fontsize=7)
        ax[r, 0].set_title(("full" if kind == "f" else "non-absorbable") +
                           " sqrt(tr Sigma), linear", fontsize=8)
        for c, (k, t) in enumerate(cases):
            ratio = tr(z[f"{k}_M{kind}"]) / tr(z[f"{k}_L{kind}"])
            im = ax[r, c + 1].imshow(to_img(ratio, ok), cmap="RdBu_r", vmin=0.5, vmax=1.5)
            ax[r, c + 1].set_title(f"{t}\nMC / linear (median {np.median(ratio):.2f})",
                                   fontsize=8)
            if c == len(cases) - 1:
                fig.colorbar(im, ax=ax[r, c + 1], fraction=0.04)
        for a in ax[r]:
            a.set_axis_off()
    fig.suptitle("Monte Carlo (4000 samples) vs first-order covariance, fixed depth 1 m, "
                 "camera centre", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mc_validation.png"))
    plt.close(fig)


def cov_maps(shared):
    full = np.load(os.path.join(shared + "_full", "shared.npy"))
    npc = np.load(os.path.join(shared, "shared.npy"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, M, t in ((ax[0], full, "full"), (ax[1], npc, "non-absorbable")):
        s = np.sqrt(M[..., 0] + M[..., 2])
        im = a.imshow(s, cmap="Blues", vmin=0)
        fig.colorbar(im, ax=a, fraction=0.03).set_label("sqrt(tr Sigma) px", fontsize=7)
        # 1-sigma ellipses, magnified
        H, W = s.shape
        k = 10.0 if t == "full" else 40.0
        for yy in np.linspace(40, H - 40, 6).astype(int):
            for xx in np.linspace(60, W - 60, 9).astype(int):
                S = np.array([[M[yy, xx, 0], M[yy, xx, 1]], [M[yy, xx, 1], M[yy, xx, 2]]])
                ev, evec = np.linalg.eigh(S)
                th = np.linspace(0, 2 * np.pi, 60)
                e = evec @ (np.sqrt(np.maximum(ev, 0))[:, None] * np.stack([np.cos(th),
                                                                            np.sin(th)]))
                a.plot(xx + k * e[0], yy + k * e[1], color=C[1], lw=1)
        a.set_title(f"{t} covariance, fixed depth 1 m (ellipses x{k:.0f})", fontsize=9)
        a.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cov_maps.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", default="/Users/fischert/robot/refrax-work/uncertainty/"
                    "gopro10_z1/cov")
    args = ap.parse_args()
    sens_maps()
    sens_rank()
    sens_depth()
    mc_maps()
    cov_maps(args.shared)


if __name__ == "__main__":
    main()
