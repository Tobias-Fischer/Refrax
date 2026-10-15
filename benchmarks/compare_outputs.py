"""Compares two remove_refraction.py output trees (e.g. reference vs new).

Usage:
    python benchmarks/compare_outputs.py <out_a> <out_b>

Reports, per corrected image, whether sizes match and the PSNR over the common
(top-left aligned) area, plus mask agreement and the calibration files.
"""
import glob
import os
import sys

import cv2
import numpy as np


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return np.inf if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    a_dir, b_dir = sys.argv[1:3]
    names = sorted(os.path.basename(p) for p in glob.glob(f"{a_dir}/corrected/*.png"))
    same, psnrs, maxdiff, size_diff = 0, [], 0, []
    for n in names:
        a = cv2.imread(f"{a_dir}/corrected/{n}", cv2.IMREAD_UNCHANGED)
        b = cv2.imread(f"{b_dir}/corrected/{n}", cv2.IMREAD_UNCHANGED)
        if a.shape == b.shape:
            same += 1
        else:
            size_diff.append((n, a.shape, b.shape))
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        psnrs.append(psnr(a[:h, :w], b[:h, :w]))
        maxdiff = max(maxdiff, int(np.abs(a[:h, :w].astype(int) - b[:h, :w].astype(int)).max()))
    p = np.array(psnrs)
    print(f"{len(names)} images, identical size: {same}, PSNR min {p.min():.1f} "
          f"median {np.median(p):.1f} dB, max |diff| {maxdiff}")
    for s in size_diff[:10]:
        print("  size differs:", *s)
    masks = sorted(os.path.basename(p) for p in glob.glob(f"{a_dir}/mask/*.png"))
    dis = []
    for n in masks:
        ma = cv2.imread(f"{a_dir}/mask/{n}", 0)
        mb = cv2.imread(f"{b_dir}/mask/{n}", 0)
        dis.append(int((ma != mb).sum()) if ma.shape == mb.shape else -1)
    print(f"{len(masks)} masks, disagreeing pixels per mask: max {max(dis)}, "
          f"mean {np.mean(dis):.1f}")
    for f in ("new_calibration.yaml",):
        fa, fb = f"{a_dir}/{f}", f"{b_dir}/{f}"
        if os.path.exists(fa):
            print(f"--- {f}\n" + "".join(f"  {la.strip():45s} | {lb.strip()}\n" for la, lb in
                                       zip(open(fa), open(fb))))


if __name__ == "__main__":
    main()
