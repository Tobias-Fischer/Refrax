"""Figures for the real-data part of docs/explore_depth.md.

    python depth_aware/figures_real.py --work <work>/gopro10
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from common import Corrector, load_cfg, REPO, WORK
from pipeline import sparse_depth
import colmap_utils as cu
from plotstyle import plt, SERIES, INK2

FIG = os.path.join(REPO, "docs", "figures")


def example(work, name=None):
    run0, run1 = f"{work}/z1_c0mm/it0_fixed", f"{work}/z1_c0mm/it1_depth"
    summ = json.load(open(f"{work}/z1_c0mm/summary.json"))
    s = summ["it1_scale"]
    txt = cu.largest_model_txt(f"{run0}/colmap/sparse", f"{run0}/colmap/colmap.log")
    sp = sparse_depth(txt)
    names = sorted(sp)
    name = name or names[len(names) // 3]
    img = cv2.imread(f"{run0}/images/{name}.png")[..., ::-1]
    disp = np.load(f"{run0}/disp/{name}.npy").astype(np.float32)
    D = np.load(f"{run1}/depth/{name}.npy").astype(np.float32)
    xy, Zc = sp[name]
    fig, axs = plt.subplots(1, 4, figsize=(16, 2.8))
    axs[0].imshow(img)
    axs[0].set_title("fixed-depth Refrax (z0 = 1 m)", fontsize=8)
    sc = axs[1].scatter(xy[:, 0], xy[:, 1], c=s * Zc, s=1, cmap="viridis_r")
    axs[1].set(xlim=(0, img.shape[1]), ylim=(img.shape[0], 0), aspect="equal")
    axs[1].set_title(f"COLMAP sparse depth ({len(Zc)} pts) [m]", fontsize=8)
    fig.colorbar(sc, ax=axs[1])
    im = axs[2].imshow(disp, cmap="magma")
    axs[2].set_title("Depth Anything V2-L disparity (relative)", fontsize=8)
    im = axs[3].imshow(D, cmap="viridis_r")
    axs[3].set_title("aligned metric depth on the raw grid [m]", fontsize=8)
    fig.colorbar(im, ax=axs[3])
    for a in axs:
        a.set(xticks=[], yticks=[])
        a.grid(False)
    fig.savefig(f"{FIG}/real_example_depth.png", dpi=130, bbox_inches="tight")


def scale_curves(work):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, (tag, lab) in enumerate((("z1_c0mm", "z0 = 1 m, camera centre"),
                                     ("z1_c-10.34mm", "z0 = 1 m, virtual centre"),
                                     ("z0.4_c0mm", "z0 = 0.4 m, camera centre"))):
        p = f"{work}/{tag}/summary.json"
        if not os.path.exists(p):
            continue
        summ = json.load(open(p))
        if "it1_scale_table" not in summ:
            continue
        txt = cu.largest_model_txt(f"{work}/{tag}/it0_fixed/colmap/sparse", "/dev/null")
        med = np.median(np.concatenate([z for _, z in sparse_depth(txt).values()]))
        s, e = np.array(summ["it1_scale_table"]).T
        base = summ.get("it1_ba_unmodified_rms")
        ax.semilogx(s * med + summ["center_z"], e, "o-", color=SERIES[i], ms=4, label=lab)
        if base:
            ax.axhline(base, color=SERIES[i], ls=":", lw=1)
        ax.axvline(summ["z0"], color=SERIES[i], lw=0.8, ls="--")
    ax.set_ylim(0.77, 1.0)  # zoom on the minimum; small scales rise to > 5 px
    ax.set(xlabel="median scene depth implied by the scale [m]",
           ylabel="RMS reprojection after BA [px]",
           title="Scale self-calibration (refractive-BA proxy)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/real_scale_selfcal.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=f"{WORK}/gopro10")
    a = ap.parse_args()
    example(a.work)
    scale_curves(a.work)
