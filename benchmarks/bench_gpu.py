"""Benchmark of refraction removal: numba (CPU) vs torch-CPU vs torch on a GPU device.

Usage:
    python benchmarks/bench_gpu.py configs/dataset.yaml [--devices cpu,mps] [--scales 0.5,1,2]
        [--repeat 20] [--warmup 3] [--tilt 5] [--batch 4] [--json out.json]

For every resolution (the config's camera scaled by --scales) and backend it reports, in ms:
  setup     constructor: zoom, fit_canvas, fixed-depth map, per-camera tensors
  compile   warm-up incl. torch.compile of every kernel variant (inductor caches on disk,
            so a second process is faster)
  fixed     per frame, fixed depth: cached map + one remap (cv2.remap / grid_sample)
  depth     per frame, depth-aware: solve the map from the frame's depth + remap. Frames
            cycle through the synthetic depth kinds smooth / plane / noisy / step
  h2d, d2h  host<->device copies per frame (uint8 BGR image + float32 depth up; uint8
            image + bool mask down); not included in fixed / depth
Per-frame numbers are the median over --repeat frames after --warmup frames, with the
10th-90th percentile range. Inputs are device resident for the per-frame timings, and the
device is synchronised around every timed call.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import remove_refraction as rr  # noqa: E402
from core.config import load_config, camera_params, distortion_params, housing_params  # noqa: E402
from core.torch_remove import Corrector, synchronize  # noqa: E402
from core.undistort import InverseMapper  # noqa: E402
from make_synthetic_depth import synthetic_depth  # noqa: E402

KINDS = ["smooth", "plane", "noisy", "step"]


def scaled_cfg(cfg, s, tilt=0.0):
    cfg = json.loads(json.dumps(cfg))
    cam = cfg["camera"]
    cam["W"], cam["H"] = int(round(cam["W"] * s)), int(round(cam["H"] * s))
    for k in ("fx", "fy"):
        cam[k] *= s
    for k in ("cx", "cy"):
        cam[k] = (cam[k] + 0.5) * s - 0.5
    if tilt:
        a = np.deg2rad(tilt)
        cfg["housing"]["n_port"] = [float(np.sin(a)), 0.0, float(np.cos(a))]
    return cfg


def timed(fn, device, n, warmup):
    for _ in range(warmup):
        fn(-1)
    ts = []
    for i in range(n):
        synchronize(device)
        t = time.perf_counter()
        fn(i)
        synchronize(device)
        ts.append(1e3 * (time.perf_counter() - t))
    return ts


def summary(ts):
    ts = np.asarray(ts)
    return dict(median=float(np.median(ts)), p10=float(np.percentile(ts, 10)),
                p90=float(np.percentile(ts, 90)), n=len(ts))


def fmt(s):
    if s is None:
        return "-"
    if isinstance(s, float):
        return f"{s:.0f}"
    return f"{s['median']:.1f} ({s['p10']:.1f}-{s['p90']:.1f})"


def load_images(cfg, n=4):
    rgb_dir = "/Users/fischert/robot/refrax-data/gopro10_linear"
    W, H = cfg["camera"]["W"], cfg["camera"]["H"]
    paths = sorted(os.listdir(rgb_dir))[:n] if os.path.isdir(rgb_dir) else []
    imgs = [cv2.imread(os.path.join(rgb_dir, p)) for p in paths]
    imgs = [cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) for im in imgs if im is not None]
    if not imgs:
        rng = np.random.default_rng(0)
        imgs = [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(n)]
    return imgs


def bench_numba(cfg, depths, imgs, repeat, warmup):
    W, H, fx, fy, cx, cy = camera_params(cfg)
    dist, housing = distortion_params(cfg), housing_params(cfg)
    z0 = cfg["correction"]["z0_fixed"]
    t = time.perf_counter()
    zoom, _ = rr.select_zoom(z0, W, H, fx, fy, cx, cy, housing, in_bounds=False)
    canvas, fixed = rr._fit_canvas(W, H, fx, fy, cx, cy, housing, dist, zoom, z0)
    mapper = InverseMapper(W, H, fx, fy, cx, cy, housing, zoom, canvas, backend="numba")
    mapper.remap_maps(depths[0], dist)  # numba kernel load (cached compile)
    setup = 1e3 * (time.perf_counter() - t)

    def fixed_fn(i):
        cv2.remap(imgs[i % len(imgs)], fixed[0], fixed[1], cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT)

    def depth_fn(i):
        mx, my = mapper.remap_maps(depths[i % len(depths)], dist)
        cv2.remap(imgs[i % len(imgs)], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def map_fn(i):
        mapper.remap_maps(depths[i % len(depths)], dist)

    return dict(setup=setup, compile=None,
                fixed=summary(timed(fixed_fn, "cpu", repeat, warmup)),
                depth=summary(timed(depth_fn, "cpu", repeat, warmup)),
                map=summary(timed(map_fn, "cpu", repeat, warmup)), h2d=None, d2h=None,
                canvas=canvas[:2])


def bench_torch(cfg, device, depths, imgs, repeat, warmup, batch=1, compile="auto"):
    t = time.perf_counter()
    corr = Corrector(cfg, device, compile=compile)
    synchronize(device)
    setup = 1e3 * (time.perf_counter() - t)
    t = time.perf_counter()
    corr.warmup(batch)
    comp = 1e3 * (time.perf_counter() - t)
    dev_imgs = [torch.as_tensor(im, device=device) for im in imgs]
    dev_depths = [torch.as_tensor(d, device=device) for d in depths]
    if batch > 1:
        dev_imgs = [torch.stack([dev_imgs[(k + j) % len(dev_imgs)] for j in range(batch)])
                    for k in range(len(dev_imgs))]
        dev_depths = [torch.stack([dev_depths[(k + j) % len(dev_depths)] for j in range(batch)])
                      for k in range(len(dev_depths))]

    def fixed_fn(i):
        corr.correct(dev_imgs[i % len(dev_imgs)])

    def depth_fn(i):
        corr.correct(dev_imgs[i % len(dev_imgs)], dev_depths[i % len(dev_depths)])

    def map_fn(i):
        corr.remap_maps(dev_depths[i % len(dev_depths)])

    res = dict(setup=setup, compile=comp,
               fixed=summary(timed(fixed_fn, device, repeat, warmup)),
               depth=summary(timed(depth_fn, device, repeat, warmup)),
               map=summary(timed(map_fn, device, repeat, warmup)), h2d=None, d2h=None,
               canvas=corr.canvas[:2])
    if torch.device(device).type != "cpu":
        def h2d(i):
            torch.as_tensor(imgs[i % len(imgs)]).to(device)
            torch.as_tensor(depths[i % len(depths)]).to(device)
        out, mask = corr.correct(dev_imgs[0][0] if batch > 1 else dev_imgs[0])

        def d2h(i):
            out.cpu().numpy()
            mask.cpu().numpy()
        res["h2d"] = summary(timed(h2d, device, repeat, warmup))
        res["d2h"] = summary(timed(d2h, device, repeat, warmup))
        if torch.device(device).type == "mps":
            res["mem_mb"] = torch.mps.driver_allocated_memory() / 2 ** 20
    if batch > 1:
        for k in ("fixed", "depth", "map"):
            res[k] = {kk: (v / batch if kk != "n" else v) for kk, v in res[k].items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--devices", default="cpu,mps" if torch.backends.mps.is_available() else
                    "cpu,cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--scales", default="0.5,1,2")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--no-numba", action="store_true")
    ap.add_argument("--cpu-repeat", type=int, default=5,
                    help="repeats for torch-cpu (eager, slow)")
    ap.add_argument("--json")
    args = ap.parse_args()
    base = load_config(args.config)
    print(f"load average: {os.getloadavg()}; torch {torch.__version__}")
    rows = []
    for s in (float(v) for v in args.scales.split(",")):
        cfg = scaled_cfg(base, s, args.tilt)
        W, H = cfg["camera"]["W"], cfg["camera"]["H"]
        depths = [synthetic_depth(k, W, H, seed=i) for i, k in enumerate(KINDS)]
        imgs = load_images(cfg)
        backends = ([] if args.no_numba else ["numba"]) + args.devices.split(",")
        for b in backends:
            if b == "numba":
                r = bench_numba(cfg, depths, imgs, args.repeat, args.warmup)
            else:
                n = args.cpu_repeat if b == "cpu" else args.repeat
                r = bench_torch(cfg, b, depths, imgs, n, min(args.warmup, n), args.batch)
            r.update(res=f"{W}x{H}", backend=b if b == "numba" else f"torch-{b}",
                     tilt=args.tilt, batch=args.batch)
            rows.append(r)
            print(f"{r['res']:>10} {r['backend']:>10} canvas {r['canvas'][0]}x{r['canvas'][1]} "
                  f"setup {fmt(r['setup'])} compile {fmt(r['compile'])} | fixed {fmt(r['fixed'])}"
                  f" | depth {fmt(r['depth'])} | map {fmt(r['map'])} | h2d {fmt(r['h2d'])} "
                  f"d2h {fmt(r['d2h'])}" + (f" | mem {r['mem_mb']:.0f} MB" if 'mem_mb' in r else ""),
                  flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
