"""Screening the Lizard Island sequences for a stretch with two clearly separated depths.

    python depth_aware/lizard_screen.py [--meta DIR] [--out DIR] [--fig PNG]

All five sequences (Feb'24 GP1, Mar'24 GP1/GP2, Sep'24 GP1/GP2; the HPC folder says
"Sep_25", the logs are dated 23-26 Sep 2024) survey the same ~60 x 30 m patch of South
Palfrey reef. The ASV log of the September survey (LIRS_Sep_25/pose.csv, NMEA-like
$ASVSM sentences at 0.5 Hz) carries an echo-sounder depth (field 11, metres; -1.0 and
-99.9 are invalid). Its valid samples during the Palf115 mission are a bathymetry map of
the reef. The map is used as the depth proxy for every sequence (tide offsets between the
surveys only shift the levels; they are measured with SfM windows, lizard_windows.py):

  1. sounder map: valid samples (> 0.2 m) of the mission rows, local east/north (m)
  2. each sequence: image positions from its pose CSV (cleaned like lizard_prep.py), the
     sounder depth per image = median of the 8 nearest samples within 2 m
  3. stretch search: windows of N consecutive images (N = 1200..1800 at the native
     ~0.17 m spacing, i.e. 600..900 images at every 2nd), images labelled shallow
     (sounder < 2.0 m) / deep (> 3.5 m) / other; a label only counts inside runs of
     >= 60 images (~10 m), so both regions are spatially large. Score = the smaller of
     the two counted fractions (balance), tie-broken by the level separation.
Output: <out>/screen.json (per-sequence tracks and the best stretches) and a figure.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lizard_prep import clean_gps  # noqa: E402

META = "/Users/fischert/robot/refrax-data/lizard_meta"
OUT = "/Users/fischert/robot/refrax-work/lizard2"
LAT0, LON0 = -14.6985, 145.4474
SEQS = {
    "feb_gp1": "LIRS_Feb_24/South_Palfrey_1/GoPro1/south_palf_1.csv",
    "mar_gp1": "LIRS_Mar_24/South_Palfrey_1/GoPro1/march_south_palf_1-GP1.csv",
    "mar_gp2": "LIRS_Mar_24/South_Palfrey_1/GoPro2/march_south_palf_1-GP2.csv",
    "sep_gp1": "LIRS_Sep_25/GP1/sept_south_palf_1_v3_1.csv",
    "sep_gp2": "LIRS_Sep_25/GP2/sept_south_palf_1_v3_2.csv",
}
SHALLOW, DEEP, MINRUN = 2.0, 3.5, 60


def en(lat, lon):
    R = 6371000.0
    return np.stack([np.radians(lon - LON0) * R * np.cos(np.radians(LAT0)),
                     np.radians(lat - LAT0) * R], 1)


def load_sounder(meta=META):
    rows = [l.strip().split(",") for l in open(f"{meta}/LIRS_Sep_25/pose.csv")]
    a = np.array([(float(r[4]), float(r[5]), float(r[10])) for r in rows
                  if len(r) == 18 and r[16].startswith("Palf") and r[2] != "0"])
    v = a[:, 2] > 0.2
    return en(a[v, 1], a[v, 0]), a[v, 2]


def load_track(seq, meta=META):
    r = sorted((x for x in csv.reader(open(f"{meta}/{SEQS[seq]}")) if x[0].endswith(".JPG")),
               key=lambda x: x[0])
    names = [x[0][:-4] for x in r]
    xy = en(np.array([float(x[1]) for x in r]), np.array([float(x[2]) for x in r]))
    return names, clean_gps(xy)


def sounder_depth(xy, P, D, k=8, rmax=2.0):
    d, i = cKDTree(P).query(xy, k=k)
    return np.where(d[:, -1] < rmax, np.median(D[i], 1), np.nan)


def runs(mask):
    """(start, end) of runs of True."""
    m = np.r_[False, mask, False].astype(int)
    st = np.flatnonzero(np.diff(m) == 1)
    en_ = np.flatnonzero(np.diff(m) == -1)
    return list(zip(st, en_))


def counted(lab, value, minrun=MINRUN):
    c = np.zeros(len(lab), bool)
    for a, b in runs(lab == value):
        if b - a >= minrun:
            c[a:b] = True
    return c


def labels(dep):
    lab = np.zeros(len(dep), int)  # 0 other / unknown
    # median filter over 9 images against single bad sounder samples
    from scipy.ndimage import median_filter
    d = median_filter(np.nan_to_num(dep, nan=-1), 9, mode="nearest")
    lab[(d > 0) & (d < SHALLOW)] = 1
    lab[d > DEEP] = 2
    return lab


def search(dep, lengths=(1200, 1500, 1800), step=50):
    lab = labels(dep)
    best = []
    for N in lengths:
        for s in range(0, len(dep) - N, step):
            w = lab[s:s + N]
            cs, cd = counted(w, 1), counted(w, 2)
            fs, fd = cs.mean(), cd.mean()
            sep = (np.nanmedian(dep[s:s + N][cd]) - np.nanmedian(dep[s:s + N][cs])
                   if cs.any() and cd.any() else 0.0)
            best.append(dict(start=s, n=N, f_shallow=float(fs), f_deep=float(fd),
                             score=float(min(fs, fd)), sep=float(sep),
                             z_shallow=float(np.nanmedian(dep[s:s + N][cs])) if cs.any() else None,
                             z_deep=float(np.nanmedian(dep[s:s + N][cd])) if cd.any() else None))
    best.sort(key=lambda r: (-round(r["score"], 2), -r["sep"]))
    return best, lab


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--meta", default=META)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    P, D = load_sounder(args.meta)
    print(f"sounder: {len(D)} valid samples, depth p5/50/95 "
          f"{np.percentile(D, [5, 50, 95]).round(2)}")
    res = {}
    for seq in SEQS:
        names, xy = load_track(seq, args.meta)
        dep = sounder_depth(xy, P, D)
        best, lab = search(dep)
        # non-overlapping top stretches
        top = []
        for b in best:
            if all(abs(b["start"] - t["start"]) > 600 for t in top):
                top.append(b)
            if len(top) == 3:
                break
        for t in top:
            t["first"], t["last"] = names[t["start"]], names[t["start"] + t["n"] - 1]
        res[seq] = dict(n=len(names), first=names[0], last=names[-1],
                        f_shallow=float(counted(lab, 1).mean()),
                        f_deep=float(counted(lab, 2).mean()),
                        f_valid=float(np.isfinite(dep).mean()), top=top)
        print(seq, len(names), f"shallow {res[seq]['f_shallow']:.2f} deep "
              f"{res[seq]['f_deep']:.2f}")
        for t in top:
            print(f"   {t['first']}..{t['last']} n {t['n']} shallow {t['f_shallow']:.2f} "
                  f"deep {t['f_deep']:.2f} levels {t['z_shallow']:.2f}/{t['z_deep']:.2f} m")
        np.savez(f"{args.out}/screen_{seq}.npz", names=np.array(names), xy=xy, dep=dep, lab=lab)
    json.dump(res, open(f"{args.out}/screen.json", "w"), indent=1)


if __name__ == "__main__":
    main()
