"""Figure of the region-wise evaluation (docs/explore_lizard.md, section 9).

    python depth_aware/lizard_regions_fig.py <regions.json> --work W --out PNG [--runs a b ...]

Left: GPS track coloured by the per-image median depth, with the region labels (shallow /
deep / transition). Right: aligned camera height along-track for the selected runs, one
panel per run, shallow images blue, deep orange, transition grey (global alignment).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotstyle import plt, SERIES  # noqa: E402
from lizard_regions import load_gps  # noqa: E402

COL = {0: "#b5b4ae", 1: SERIES[0], 2: SERIES[1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("json")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    J = json.load(open(args.json))
    seg = J["segmentation"]
    gps = load_gps(f"{args.work}/gps.csv")
    names = seg["names"]
    xy = np.array([gps[n][:2] for n in names])
    lab = np.array(seg["label"])
    dep = np.array(seg["depth"])
    s = np.array(seg["s"])
    runs = args.runs or list(J["heights"])
    nr = len(runs)
    fig = plt.figure(figsize=(12, 1.5 + 1.35 * nr))
    gs = fig.add_gridspec(nr, 2, width_ratios=[1, 1.6])
    a = fig.add_subplot(gs[: max(1, nr // 2), 0])
    sc = a.scatter(*xy.T, c=dep, s=6, cmap="viridis_r", vmin=1, vmax=5.5)
    fig.colorbar(sc, ax=a, label="per-image median depth [m]", shrink=0.8)
    a.set(aspect="equal", xlabel="east [m]", ylabel="north [m]")
    a.set_title(f"{args.title} depth along the track", loc="left", fontsize=9)
    b = fig.add_subplot(gs[max(1, nr // 2):, 0])
    for k, txt in ((1, "shallow"), (2, "deep"), (0, "transition")):
        m = lab == k
        b.scatter(*xy[m].T, c=COL[k], s=6,
                  label=f"{txt} ({m.sum()}, {np.median(dep[m]):.1f} m)" if m.any() else txt)
    b.set(aspect="equal", xlabel="east [m]", ylabel="north [m]")
    b.legend(fontsize=7, loc="best")
    b.set_title("regions", loc="left", fontsize=9)
    for i, r in enumerate(runs):
        h = np.array([np.nan if x is None else x for x in J["heights"][r]])
        c = fig.add_subplot(gs[i, 1])
        for k in (0, 1, 2):
            m = lab == k
            c.scatter(s[m], h[m], c=COL[k], s=3, lw=0)
        q = J["results"][r]
        st = "" if q.get("step") is None else f"  step {q['step']:+.2f} m"
        c.set_title(f"{r}{st}", loc="left", fontsize=8)
        c.set_ylabel("h [m]")
        lim = max(0.3, np.nanpercentile(np.abs(h), 99) * 1.1)
        c.set_ylim(-lim, lim)
        if i == nr - 1:
            c.set_xlabel("along-track [m]")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)


if __name__ == "__main__":
    main()
