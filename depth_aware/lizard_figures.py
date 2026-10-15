"""Figures for docs/explore_lizard.md: trajectory side/top views and height profiles.

    python depth_aware/lizard_figures.py label=<model_txt> [label=<model_txt> ...] [--prefix lizard]

Every model is aligned to the GPS track with a similarity transform (lizard_eval.aligned).
The side view projects onto the survey's leg direction (120 deg) and the vertical, like the
README teaser; the height profile plots each camera's aligned height against along-track
position, one line per leg.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lizard_eval import aligned, leg_segments  # noqa: E402
from plotstyle import plt, SERIES, INK2  # noqa: E402
from common import REPO  # noqa: E402

FIG = os.path.join(REPO, "docs", "figures")
H = np.radians(120.0)
ALONG = np.array([np.sin(H), np.cos(H), 0.0])
ACROSS = np.array([np.cos(H), -np.sin(H), 0.0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="+")
    ap.add_argument("--prefix", default="lizard")
    args = ap.parse_args()
    runs = [m.rsplit("=", 1) for m in args.models]
    data = [(lab, *aligned(p)) for lab, p in runs]
    rng = np.random.default_rng(0)

    # side + top views
    n = len(data)
    fig, ax = plt.subplots(n, 2, figsize=(10, 1.9 * n + 0.4), squeeze=False,
                           gridspec_kw=dict(width_ratios=[2.2, 1]))
    for i, (lab, names, C, X, G) in enumerate(data):
        Xs = X[rng.choice(len(X), min(len(X), 15000), replace=False)]
        z0 = np.median(C[:, 2])
        a = ax[i, 0]
        a.scatter(Xs @ ALONG, Xs[:, 2] - z0, s=0.3, c="#9a9a96", lw=0, rasterized=True)
        a.scatter(C @ ALONG, C[:, 2] - z0, s=2, c=SERIES[i % len(SERIES)], lw=0)
        zc = C[:, 2] - z0
        a.set(ylabel="height [m]",
              ylim=(min(np.percentile(Xs[:, 2] - z0, 1), zc.min()) - 0.3, zc.max() + 0.5))
        a.set_title(f"{lab}: side view (along legs)", fontsize=9, loc="left")
        b = ax[i, 1]
        b.scatter(Xs @ ALONG, Xs @ ACROSS, s=0.3, c="#9a9a96", lw=0, rasterized=True)
        b.scatter(C @ ALONG, C @ ACROSS, s=2, c=SERIES[i % len(SERIES)], lw=0)
        b.set(aspect="equal")
        b.set_title("top view", fontsize=9, loc="left")
    ax[-1, 0].set_xlabel("along-track [m]")
    ax[-1, 1].set_xlabel("along-track [m]")
    fig.tight_layout()
    fig.savefig(f"{FIG}/{args.prefix}_views.png", dpi=150)
    plt.close(fig)

    # camera height profiles (exaggerated vertical scale)
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.6))
    for i, (lab, names, C, X, G) in enumerate(data):
        leg, along = leg_segments(names)
        z = C[:, 2] - np.median(C[:, 2])
        for j in sorted(set(leg[nm] for nm in names)):
            idx = [k for k, nm in enumerate(names) if leg[nm] == j]
            s = np.array([along[names[k]] for k in idx])
            o = np.argsort(s)
            ax.plot(s[o], z[idx][o], color=SERIES[i % len(SERIES)], lw=1.2,
                    label=lab if j == 0 else None)
    ax.set(xlabel="along-track position [m]", ylabel="camera height after GPS alignment [m]")
    ax.legend(fontsize=8, ncol=2)
    ax.axhline(0, color=INK2, lw=0.6)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{args.prefix}_height.png", dpi=150)
    plt.close(fig)
    print("wrote", f"{FIG}/{args.prefix}_views.png", f"{FIG}/{args.prefix}_height.png")


if __name__ == "__main__":
    main()
