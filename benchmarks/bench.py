"""Per-stage benchmark of the reference (original) and the new pipeline.

Usage:
    python benchmarks/bench.py <config.yaml> [--impl ref|new|both] [--repeat 3]
                               [--n-images N] [--save-maps DIR]

Stages are timed in-process, in the order remove_refraction.main() runs them; the
image stages (read / remap / PNG encode) are timed serially per image so they are
comparable between implementations (the new CLI additionally overlaps them across
threads, see the end-to-end numbers in docs/explore_speed.md). Each configuration is
run --repeat times; the table reports the median per stage and the min-max of the total.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
import reference_impl as ref  # noqa: E402
import remove_refraction as new  # noqa: E402
from core.config import load_config, camera_params, distortion_params, housing_params  # noqa: E402
from core.optics import matrix_K  # noqa: E402
from core.scale import select_zoom  # noqa: E402


class Timer:
    def __init__(self):
        self.t = defaultdict(float)

    def __call__(self, key):
        timer = self

        class _C:
            def __enter__(self):
                self.s = time.perf_counter()

            def __exit__(self, *a):
                timer.t[key] += time.perf_counter() - self.s
        return _C()


def run(impl, cfg, n_images, save=None):
    W, H, fx, fy, cx, cy = camera_params(cfg)
    distortion = distortion_params(cfg)
    housing = housing_params(cfg)
    corr = cfg["correction"]
    z0, fit = corr["z0_fixed"], corr.get("fit_canvas", True)
    depth_dir = cfg["paths"].get("depth_dir")
    T = Timer()
    t0 = time.perf_counter()

    with T("zoom selection"):
        if impl == "ref":
            zoom, _ = ref.select_zoom(z0, W, H, fx, fy, cx, cy, housing, in_bounds=not fit)
        else:
            zoom, _ = select_zoom(z0, W, H, fx, fy, cx, cy, housing, in_bounds=not fit)
    canvas, fit_maps = (W, H, (0.0, 0.0)), None
    if fit:
        with T("fit_canvas"):
            if impl == "ref":
                canvas = ref.fit_canvas(W, H, fx, fy, cx, cy, housing, distortion, zoom, z0)
            else:
                canvas, fit_maps = new._fit_canvas(W, H, fx, fy, cx, cy, housing, distortion,
                                                   zoom, z0)
    with T("map builder setup"):
        if impl == "ref":
            build_map = ref.make_map_builder(W, H, fx, fy, cx, cy, housing, distortion, zoom,
                                             canvas)
        else:
            build_map = new.make_map_builder(W, H, fx, fy, cx, cy, housing, distortion, zoom,
                                             canvas)
    rect = ref.largest_valid_rectangle if impl == "ref" else new.largest_valid_rectangle

    saved = {}
    if depth_dir is None:
        with T("map build (per map)"):
            maps = fit_maps if fit_maps is not None else build_map(z0)
        with T("mask + crop rectangle"):
            mask = new.compute_valid_mask(*maps, W, H)
            box = rect(mask)
        saved["fixed"] = (maps, box)

    paths = new.find_rgb_paths(cfg["paths"]["rgb_dir"], 1)[:n_images]
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        with T("image read"):
            img = cv2.imread(p)
        if depth_dir is not None:
            with T("depth load"):
                depth = new.load_depth(name, depth_dir, W, H)
            with T("map build (per map)"):
                maps = build_map(depth)
            with T("mask + crop rectangle"):
                mask = new.compute_valid_mask(*maps, W, H)
                box = rect(mask)
            saved[name] = (maps, box)
        with T("remap"):
            out = cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        y0, y1, x0, x1 = box
        with T("PNG encode"):
            cv2.imencode(".png", out[y0:y1, x0:x1])
    total = time.perf_counter() - t0
    if save:
        os.makedirs(save, exist_ok=True)
        for k, ((mx, my), box) in saved.items():
            np.savez_compressed(f"{save}/{k}.npz", map_x=mx, map_y=my, box=np.array(box),
                                zoom=zoom, canvas=np.array([canvas[0], canvas[1], *canvas[2]]))
    return dict(T.t), total, len(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--impl", default="both", choices=["ref", "new", "both"])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--n-images", type=int, default=None)
    ap.add_argument("--save-maps", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    impls = ["ref", "new"] if args.impl == "both" else [args.impl]

    # Warm-up (numba cache load / JIT, file cache) not counted.
    new.make_map_builder(*camera_params(cfg), housing_params(cfg), distortion_params(cfg),
                         1.5)(1.0)
    res = {}
    for impl in impls:
        runs = []
        for r in range(args.repeat):
            save = f"{args.save_maps}/{impl}" if (args.save_maps and r == 0) else None
            runs.append(run(impl, cfg, args.n_images, save))
        res[impl] = runs
    n = runs[0][2]
    keys = list(dict.fromkeys(k for impl in impls for t, _, _ in res[impl] for k in t))
    print(f"config: {args.config}  images: {n}  repeats: {args.repeat}")
    print(f"{'stage (median, ms; per-map / per-image stages are totals)':58s}" +
          "".join(f"{i:>12s}" for i in impls))
    for k in keys:
        print(f"{k:58s}" + "".join(
            f"{np.median([t.get(k, 0.0) for t, _, _ in res[i]]) * 1e3:12.1f}" for i in impls))
    print(f"{'TOTAL (median)':58s}" + "".join(
        f"{np.median([tt for _, tt, _ in res[i]]) * 1e3:12.1f}" for i in impls))
    print(f"{'TOTAL (min-max)':58s}" + "".join(
        f"{min(tt for _, tt, _ in res[i]) * 1e3:6.0f}-{max(tt for _, tt, _ in res[i]) * 1e3:<5.0f}"
        for i in impls))


if __name__ == "__main__":
    main()
