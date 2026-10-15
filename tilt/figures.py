"""Figures for docs/explore_tilt.md from the JSON outputs of tilt/*.py.

    python tilt/figures.py [--work /Users/fischert/robot/refrax-work/tilt]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, "docs", "figures")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#2b2b29", "#6b6a63", "#e4e3dc"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def synthetic(path):
    d = json.load(open(path))
    meths = ["refrax+pinhole", "refrax(tilt)+pinhole", "refrac, no tilt", "refrac + tilt",
             "oracle"]
    labels = ["Refrax + pinhole BA (today)", "Refrax with est. tilt + pinhole BA",
              "refractive BA, no tilt", "refractive BA + tilt", "oracle (true params)"]
    fig, axes = plt.subplots(1, 2 * len(d), figsize=(5.2 * len(d) * 2, 3.6))
    for si, (kind, dd) in enumerate(d.items()):
        rows = dd["rows"]
        tilts = sorted({r["tilt"] for r in rows})
        ax = axes[2 * si]
        style(ax)
        for mi, (m, lab) in enumerate(zip(meths, labels)):
            med = [np.median([r[m]["pts_mm"] for r in rows if r["tilt"] == t]) for t in tilts]
            ax.plot(tilts, med, "-o", color=C[mi], lw=2, ms=6, label=lab)
        ax.set_yscale("log")
        ax.set_xlabel("true port tilt [deg]")
        ax.set_ylabel("3-D point error [mm] (median)")
        ax.set_title(f"{kind}: structure error", color=INK, loc="left")
        if si == 0:
            ax.legend(frameon=False, fontsize=8)
        ax = axes[2 * si + 1]
        style(ax)
        for t in tilts:
            rr = [r for r in rows if r["tilt"] == t]
            e = np.array([[r["refrac + tilt"]["params"]["tilt_x"][0] - r["tilt_true_deg"][0],
                           r["refrac + tilt"]["params"]["tilt_y"][0] - r["tilt_true_deg"][1]]
                          for r in rr])
            ax.scatter(np.full(len(e), t) - 0.04, e[:, 0], s=36, color=C[0],
                       label="tilt_x" if t == tilts[0] else None, edgecolor="white", lw=1)
            ax.scatter(np.full(len(e), t) + 0.04, e[:, 1], s=36, color=C[1],
                       label="tilt_y" if t == tilts[0] else None, edgecolor="white", lw=1)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel("true port tilt [deg]")
        ax.set_ylabel("estimated - true [deg]")
        ax.set_title(f"{kind}: tilt recovery", color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tilt_synthetic.png"), dpi=130)
    plt.close(fig)


def identifiability(path):
    d = json.load(open(path))
    fig, axes = plt.subplots(1, len(d), figsize=(7 * len(d), 4.2), sharex=True)
    for ax, (kind, dd) in zip(np.atleast_1d(axes), d.items()):
        style(ax)
        names = list(dd["confs"])
        sx = [dd["confs"][n]["sd_tilt_deg"][0] for n in names]
        sy = [dd["confs"][n]["sd_tilt_deg"][1] for n in names]
        y = np.arange(len(names))
        ax.barh(y - 0.2, sx, 0.38, color=C[0], label="tilt_x")
        ax.barh(y + 0.2, sy, 0.38, color=C[1], label="tilt_y")
        ax.set_yticks(y, names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("posterior std of the tilt [deg] (σ = 0.5 px)")
        ax.set_title(f"{kind} (z0 {dd['z0']:.2f} m, {dd['n_cams']} views)", color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tilt_identifiability.png"), dpi=130)
    plt.close(fig)


def real(entries, out):
    """entries: list of (label, tilt_x, sd_x, tilt_y, sd_y)."""
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(entries) + 1.2))
    style(ax)
    y = np.arange(len(entries))
    for i, (lab, tx, sx, ty, sy) in enumerate(entries):
        ax.errorbar(tx, i - 0.12, xerr=2 * sx, fmt="o", color=C[0], ms=6, capsize=3,
                    label="tilt_x" if i == 0 else None)
        ax.errorbar(ty, i + 0.12, xerr=2 * sy, fmt="s", color=C[1], ms=6, capsize=3,
                    label="tilt_y" if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, [e[0] for e in entries], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("estimated port tilt [deg] (bars: ±2 formal σ)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/Users/fischert/robot/refrax-work/tilt")
    args = ap.parse_args()
    p = os.path.join(FIG, "tilt_synthetic.json")
    if os.path.exists(p):
        synthetic(p)
    p = os.path.join(FIG, "tilt_identifiability.json")
    if os.path.exists(p):
        identifiability(p)
    p = os.path.join(FIG, "tilt_real.json")
    if os.path.exists(p):
        real([tuple(e) for e in json.load(open(p))], os.path.join(FIG, "tilt_real.png"))


if __name__ == "__main__":
    main()
