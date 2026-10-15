"""Lizard Island experiment driver (docs/explore_lizard.md).

    python depth_aware/lizard.py --config configs/lizard_gp11.yaml --mode fixed  --z0 Z [--center-z C]
    python depth_aware/lizard.py --config ... --mode median --z0 Z [--center-z C]
    python depth_aware/lizard.py --config ... --mode dense  --z0 Z [--center-z C] [--seed S]

All runs share the reference output frame of (z0, centre), so their calibrations are
identical and COLMAP gets one PINHOLE camera with fixed intrinsics; matching uses the GPS
pair list of lizard_prep.py. "median" and "dense" take their depth from the "fixed" run with
the same z0 and centre (run that first):
  median  one depth per image: median of the image's sparse SfM depths
  dense   Depth Anything V2 disparity fitted per image to the sparse SfM depths
          (1/Z = a disp + b), carried to the ideal raw grid (depth_aware/pipeline.py)
The metric scale (metres per SfM unit) comes from a similarity alignment of the fixed run's
camera centres to GPS, and depth from the virtual centre is Z = s * Zc + c.
Outputs under <work>/runs/<tag>/ (images, masks, depth, colmap, eval.json).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Corrector, load_cfg  # noqa: E402
import colmap_utils as cu  # noqa: E402
from lizard_eval import evaluate, umeyama, load_gps  # noqa: E402
from pipeline import correct_set, sparse_depth, mono_stage, fit_stage, to_ideal_grid  # noqa: E402
from core.virtual_center import optimal_center_z  # noqa: E402

WORK = os.environ.get("REFRAX_LIZARD_WORK", "/Users/fischert/robot/refrax-work/lizard")


def gps_scale(model_txt):
    gps = load_gps()
    imgs = cu.read_images(f"{model_txt}/images.txt")
    names = sorted(n for n in imgs if n in gps)
    C = np.array([-imgs[n]["R"].T @ imgs[n]["t"] for n in names])
    G = np.array([gps[n] for n in names])
    return umeyama(C, G)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["fixed", "median", "dense"], default="fixed")
    ap.add_argument("--z0", type=float, required=True)
    ap.add_argument("--center-z", default="0", help="metres, or 'auto'")
    ap.add_argument("--calib-tag", default="", help="prefix of the run name (calibration)")
    ap.add_argument("--seed", type=int, default=0, help="COLMAP mapper random seed")
    ap.add_argument("--every", type=int, default=1,
                    help="use every n-th image (z0 sweep); pairs are filtered accordingly")
    ap.add_argument("--source-z0", type=float, default=None,
                    help="median/dense: take depths from the fixed run at this z0 (default --z0)")
    ap.add_argument("--source-c", type=float, default=None,
                    help="median/dense: ... and this centre in metres (default: the run's)")
    ap.add_argument("--source-tag", default=None,
                    help="median/dense: calib tag of the source fixed run (default --calib-tag)")
    ap.add_argument("--invdepth-affine", nargs=2, type=float, default=None, metavar=("A", "B"),
                    help="median: use 1/Z' = A + B/Z instead of Z (two-term model, section 9)")
    ap.add_argument("--work", default=WORK)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.center_z == "auto":
        from core.config import camera_params, housing_params
        W, H, fx, fy, cx, cy = camera_params(cfg)
        c = optimal_center_z(W, H, fx, fy, cx, cy, housing_params(cfg),
                             z_range=cfg["correction"].get("depth_range", [0.5, 8.0]))[0]
    else:
        c = float(args.center_z)
    C = Corrector(cfg, z0=args.z0, center_z=c)
    rgb_dir = cfg["paths"]["rgb_dir"]
    names = [os.path.splitext(os.path.basename(p))[0]
             for p in sorted(glob.glob(f"{rgb_dir}/*.png"))][::args.every]
    base = f"{args.calib_tag}z{args.z0:g}_c{c * 1000:.2f}mm"
    sub = f"_every{args.every}" if args.every > 1 else ""
    run = (f"{args.work}/runs/{base}_{args.mode}{sub}" +
           (f"_seed{args.seed}" if args.seed else ""))
    os.makedirs(run, exist_ok=True)
    pairs = f"{args.work}/pairs.txt"
    if args.every > 1:
        keep = {f"{n}.png" for n in names}
        pairs = f"{run}/pairs.txt"
        with open(pairs, "w") as f:
            for line in open(f"{args.work}/pairs.txt"):
                a, b = line.split()
                if a in keep and b in keep:
                    f.write(line)
    calib = (C.out_W, C.out_H, C.out_fx, C.out_fy, C.out_cx, C.out_cy)
    info = dict(z0=args.z0, center_z=c, zoom=C.zoom, out=[int(C.out_W), int(C.out_H)],
                out_f=[C.out_fx, C.out_fy], out_c=[C.out_cx, C.out_cy], n_images=len(names))
    print(run, info, flush=True)

    if args.mode == "fixed":
        depth_of = lambda n: args.z0  # noqa: E731
    else:
        # The source is a fixed-depth run; its images/observations live in its own output
        # frame (z0_s, c_s), so the depth is carried to the ideal grid with its Corrector.
        z0_s = args.z0 if args.source_z0 is None else args.source_z0
        c_s = c if args.source_c is None else args.source_c
        Cs = C if (z0_s, c_s) == (args.z0, c) else Corrector(cfg, z0=z0_s, center_z=c_s)
        stag = args.calib_tag if args.source_tag is None else args.source_tag
        src = f"{args.work}/runs/{stag}z{z0_s:g}_c{c_s * 1000:.2f}mm_fixed{sub}"
        src_model = cu.largest_model_txt(f"{src}/colmap/sparse", f"{src}/colmap/colmap.log")
        s = gps_scale(src_model)
        sparse = sparse_depth(src_model)
        info.update(scale=s, source=src_model, n_sparse=len(sparse))
        med_all = float(np.median(np.concatenate([z for _, z in sparse.values()])) * s + c_s)
        info["median_depth_m"] = med_all
        print(f"scale {s:.4f} m/unit, median depth {med_all:.3f} m", flush=True)
        used = {}
        if args.mode == "median":
            for n in names:
                used[n] = float(s * np.median(sparse[n][1]) + c_s) if n in sparse else med_all
                if args.invdepth_affine:
                    A, B = args.invdepth_affine
                    used[n] = 1.0 / (A + B / used[n])
            json.dump(used, open(f"{run}/median_depths.json", "w"), indent=0)
        else:
            mono_stage(list(sparse), f"{src}/images", f"{src}/disp")
            dense, rows = fit_stage(Cs, sparse, list(sparse), f"{src}/disp")
            json.dump(rows, open(f"{run}/fit_stats.json", "w"), indent=1)
            os.makedirs(f"{run}/depth", exist_ok=True)
            for n in names:
                if n in dense:
                    p = f"{run}/depth/{n}.npy"
                    if not os.path.exists(p):
                        # float32: float16 would quantise depth to ~1e-3 relative
                        np.save(p, to_ideal_grid(Cs, dense[n], s, z0_s).astype(np.float32))
                    used[n] = p
                else:
                    used[n] = med_all
        depth_of = lambda n: (np.load(used[n]) if isinstance(used[n], str)  # noqa: E731
                              else used[n])

    img_dir, mask_dir = correct_set(C, names, rgb_dir, run, depth_of)
    model = cu.sfm(img_dir, f"{run}/colmap", calib, mask_dir, pairs=pairs, seed=args.seed)
    ev = evaluate(model, len(names))
    info["eval"] = ev
    json.dump(info, open(f"{run}/eval.json", "w"), indent=1)
    print(json.dumps(ev), flush=True)


if __name__ == "__main__":
    main()
