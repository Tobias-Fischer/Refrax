"""Prepare the Lizard Island subset (docs/explore_lizard.md): downscale images, clean GPS.

    python depth_aware/lizard_prep.py [--data DIR] [--work DIR] [--factor 6] [--step 2]

* Every --step-th image of <data>/raw/*GOPRO/*.JPG (GoPro HERO11, 5568x4872, 8:7) is
  downscaled by --factor with area averaging (928x812 for 6) to <work>/raw<W>/<name>.png.
  Pixel centres are preserved: a full-resolution pixel u maps to (u + 0.5) / f - 0.5.
* The pose CSV (name, lat, lon, height; heights are all 0) is converted to a local
  east/north frame in metres (equirectangular about the subset's mean, < 1 mm error over
  60 m). Single-epoch GPS jumps are repaired as in VSLAM-LAB's LIZARDISLAND loader
  (steps > 0.6 m between consecutive images are interpolated), and the receiver's
  repeated fixes (the GPS updates at about half the image rate, so ~55 % of consecutive
  images share a fix) are replaced by linear interpolation in time between fix changes.
  Output: <work>/gps.csv (name, east, north).
* Image pairs for matching: every pair whose GPS positions are within --radius metres
  (the same pair list is used for every COLMAP run, so the runs are comparable).
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

DATA = "/Users/fischert/robot/refrax-data/lizard_mar24_gp1"
WORK = os.environ.get("REFRAX_LIZARD_WORK", "/Users/fischert/robot/refrax-work/lizard")


def load_gps(csv_path, names):
    rows = {r[0]: (float(r[1]), float(r[2])) for r in csv.reader(open(csv_path))}
    lat = np.array([rows[n][0] for n in names])
    lon = np.array([rows[n][1] for n in names])
    R = 6371000.0
    e = np.radians(lon - lon.mean()) * R * np.cos(np.radians(lat.mean()))
    n = np.radians(lat - lat.mean()) * R
    return np.stack([e, n], 1)


def clean_gps(xy):
    """Repair jumps (> 0.6 m) and interpolate between fix changes."""
    xy = xy.copy()
    n = len(xy)
    d = np.hypot(*np.diff(xy, axis=0).T)
    bad = np.zeros(n, bool)
    jump = d > 0.6
    # a jump that is undone in the next step marks a single bad epoch
    for i in np.flatnonzero(jump[:-1] & jump[1:]):
        bad[i + 1] = True
    good = ~bad
    t = np.arange(n, dtype=float)
    for k in range(2):
        xy[bad, k] = np.interp(t[bad], t[good], xy[good, k])
    # repeated fixes: keep the first image of each run of identical positions as the fix
    # epoch and interpolate the others in time
    same = np.r_[False, np.all(np.diff(xy, axis=0) == 0, axis=1)]
    key = ~same
    for k in range(2):
        xy[same, k] = np.interp(t[same], t[key], xy[key, k])
    return xy


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--factor", type=int, default=6)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--radius", type=float, default=4.0)
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{args.data}/raw/*GOPRO/*.JPG"),
                   key=lambda p: os.path.basename(p))[::args.step]
    names = [os.path.basename(p) for p in paths]
    img0 = cv2.imread(paths[0])
    H0, W0 = img0.shape[:2]
    W, H = W0 // args.factor, H0 // args.factor
    out = f"{args.work}/raw{W}"
    os.makedirs(out, exist_ok=True)
    for p in paths:
        q = f"{out}/{os.path.splitext(os.path.basename(p))[0]}.png"
        if os.path.exists(q):
            continue
        img = cv2.imread(p)
        cv2.imwrite(q, cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA))
    print(f"{len(paths)} images {W0}x{H0} -> {W}x{H} in {out}")

    csv_path = glob.glob(f"{args.data}/*.csv")[0]
    # clean on the full contiguous subset (not just every step-th image), then subsample
    all_names = sorted(os.path.basename(p) for p in glob.glob(f"{args.data}/raw/*GOPRO/*.JPG"))
    xy_all = clean_gps(load_gps(csv_path, all_names))
    idx = {n: i for i, n in enumerate(all_names)}
    xy = xy_all[[idx[n] for n in names]]
    with open(f"{args.work}/gps.csv", "w") as f:
        for n, (e, no) in zip(names, xy):
            f.write(f"{os.path.splitext(n)[0]}.png,{e:.4f},{no:.4f}\n")

    D = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    i, j = np.nonzero(np.triu(D <= args.radius, 1))
    with open(f"{args.work}/pairs.txt", "w") as f:
        for a, b in zip(i, j):
            f.write(f"{os.path.splitext(names[a])[0]}.png {os.path.splitext(names[b])[0]}.png\n")
    print(f"{len(i)} pairs within {args.radius} m ({2 * len(i) / len(names):.0f} per image)")


if __name__ == "__main__":
    main()
