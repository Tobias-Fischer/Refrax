"""Figure of the sequence screening (docs/explore_lizard.md, section 9).

    python depth_aware/lizard_screen_fig.py [--out docs/figures/lizard_screen.png]

Top row: each sequence's GPS track coloured by the September sounder depth at the image
positions (lizard_screen.py), with the best stretch outlined and the SfM windows marked
(lizard_windows.py). Bottom: SfM window depth vs sounder depth per sequence.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotstyle import plt, SERIES  # noqa: E402
from lizard_screen import OUT, SEQS, load_sounder  # noqa: E402
from common import REPO  # noqa: E402

CHOSEN = {"feb_gp1": ("G0021364", "G0022563"), "mar_gp1": ("G0015281", "G0016231")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "figures", "lizard_screen.png"))
    args = ap.parse_args()
    win = json.load(open(f"{OUT}/windows.json"))
    P, D = load_sounder()
    fig, ax = plt.subplots(2, 6, figsize=(17, 6.4), gridspec_kw=dict(height_ratios=[1.25, 1]))
    a = ax[0, 0]
    sc = a.scatter(*P.T, c=D, s=1, cmap="viridis_r", vmin=0.5, vmax=5.5, rasterized=True)
    a.set_title("sounder, Sep'24 (Palf115)", loc="left", fontsize=9)
    fig.colorbar(sc, ax=ax[0, :], label="sounder depth [m]", shrink=0.8, pad=0.01)
    for j, seq in enumerate(SEQS):
        z = np.load(f"{OUT}/screen_{seq}.npz")
        names, xy, dep = list(z["names"]), z["xy"], z["dep"]
        a = ax[0, j + 1]
        a.scatter(*xy.T, c=dep, s=0.5, cmap="viridis_r", vmin=0.5, vmax=5.5, rasterized=True)
        if seq in CHOSEN:
            i0, i1 = names.index(CHOSEN[seq][0]), names.index(CHOSEN[seq][1])
            a.plot(*xy[i0:i1 + 1].T, c="r" if seq == "feb_gp1" else "k", lw=0.6)
        w = [v for v in win.values() if v["seq"] == seq]
        a.plot([xy[v["index"]][0] for v in w], [xy[v["index"]][1] for v in w], "x", c="k", ms=5)
        a.set_title(f"{seq} ({len(names)} img)", loc="left", fontsize=9)
        b = ax[1, j + 1]
        ok = [v for v in w if v.get("sfm") and v.get("registered", 0) >= 8]
        bad = [v for v in w if v not in ok]
        b.plot([v["sounder"] for v in ok], [v["sfm"] for v in ok], "o", c=SERIES[j], ms=5)
        b.plot([v["sounder"] for v in bad], [min(v.get("sfm") or 0, 11) for v in bad], "x",
               c="0.5", ms=5)
        b.plot([0, 7], [0, 7], c="0.6", lw=0.8)
        if len(ok) > 2:
            r = np.corrcoef([v["sounder"] for v in ok], [v["sfm"] for v in ok])[0, 1]
            b.set_title(f"r = {r:.2f} ({len(ok)}/{len(w)} windows)", loc="left", fontsize=9)
        b.set(xlim=(0, 6.5), ylim=(0, 11), xlabel="sounder depth [m]")
    for a in ax[0]:
        a.set(aspect="equal", xlim=(-50, 25), ylim=(-118, -58))
        a.tick_params(labelsize=7)
    ax[1, 0].axis("off")
    ax[1, 1].set_ylabel("SfM window depth [m]")
    fig.savefig(args.out, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
