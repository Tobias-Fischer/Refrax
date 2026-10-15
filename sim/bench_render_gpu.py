"""Benchmark: numpy vs torch (cpu / mps / cuda, eager and torch.compile) vs fused Metal kernel.

Usage:
    python -m sim.bench_render_gpu [configs/dataset.yaml] [--scales 0.5 1 2] [--repeats 20]
        [--json docs/figures/gpu_bench.json]

Output resolutions are the config camera scaled by --scales (GoPro config: 960x540, 1920x1080,
3840x2160); scene: "spheres" (occlusions + background) rendered with the planned pinhole
camera; RGB uint8 HWC (3 channels), depth float32 z-depth, as a simulator delivers them.

Reported per method (median and p25-p75 over repeats, after warm-up):
  setup     per camera/housing: numpy geometry (FlatPortRenderer) + upload of the line table
  h2d       numpy RGB + depth -> device tensors (B frames)
  frame     render with inputs resident on the device: march + refine + RGB gather +
            valid mask (no near_edge mask; its extra cost is listed separately)
  d2h       output uint8 image -> numpy
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch

from core.config import load_config
from core.render import FlatPortRenderer
from core.render_torch import TorchFlatPortRenderer, synchronize, _upload
from sim.scenes import standard_scenes
from sim.validate_torch import scaled_renderer


def timeit(fn, device, repeats, warmup=3):
    for _ in range(warmup):
        fn()
    synchronize(device)
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        synchronize(device)
        ts.append(time.perf_counter() - t)
    ts = 1e3 * np.asarray(ts)
    return dict(median=float(np.median(ts)), p25=float(np.percentile(ts, 25)),
                p75=float(np.percentile(ts, 75)), min=float(ts.min()), std=float(ts.std()),
                n=int(len(ts)))


def bandwidth_gbs(device, mb=256, repeats=20):
    """Effective device copy bandwidth (read + write) in GB/s."""
    x = torch.empty(mb * 2 ** 18, device=device)
    y = torch.empty_like(x)
    t = timeit(lambda: y.copy_(x), device, repeats)
    del x, y
    return 2 * mb * 2 ** 20 / (t["median"] * 1e-3) / 1e9


def fmt(t):
    return f"{t['median']:.1f} ({t['p25']:.1f}-{t['p75']:.1f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/dataset.yaml")
    ap.add_argument("--scales", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4, help="batch size for the batched rows (scale <= 1)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + \
              (["cuda"] if torch.cuda.is_available() else [])
    res = dict(machine=platform.platform(), processor=platform.processor(),
               torch=torch.__version__, threads=torch.get_num_threads(),
               loadavg_start=os.getloadavg(), bandwidth_gbs={}, rows=[])
    for dev in devices:
        res["bandwidth_gbs"][dev] = bandwidth_gbs(dev)
    if torch.cuda.is_available():
        res["cuda_device"] = torch.cuda.get_device_name()
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1, default=str))
    scene = standard_scenes()["spheres"]
    try:
        import warp  # noqa: F401
        have_warp = True
    except ImportError:
        have_warp = False

    def row(**kw):
        res["rows"].append(kw)
        print(" | ".join(f"{k}={fmt(v) if isinstance(v, dict) else v}" for k, v in kw.items()), flush=True)

    for scale in args.scales:
        cpu_reps = max(3, args.repeats // 4) if scale <= 1 else 3
        t = time.perf_counter(); r = scaled_renderer(cfg, scale); t_setup_np = 1e3 * (time.perf_counter() - t)
        res_s = f"{r.W}x{r.H}"
        rgb, z, _ = scene.render_pinhole(r.spec)
        rgb = (255 * rgb).astype(np.uint8)
        z = z.astype(np.float32)
        print(f"=== {res_s} (render {r.spec.W}x{r.spec.H}, {r.n_steps} march steps), numpy setup {t_setup_np:.0f} ms")
        row(res=res_s, method="numpy (cv2.remap)", B=1, setup_ms=round(t_setup_np),
            frame=timeit(lambda: r.render(rgb, z), "cpu", cpu_reps, warmup=1))

        variants = []
        for dev in devices:
            variants.append((f"torch-{dev}", dev, dict()))
            if dev != "cpu" and not args.no_compile:
                variants.append((f"torch-{dev} compile", dev, dict(compile=True)))
            if dev != "cpu" and scale == 1.0:
                variants.append((f"torch-{dev} fp16", dev, dict(dtype=torch.float16)))
            if dev == "mps":
                variants.append(("metal fused (mps)", dev, "metal"))
            if dev in ("cpu", "cuda") and have_warp:
                variants.append((f"warp fused ({dev})", dev, "warp"))
        for name, dev, kw in variants:
            batches = [1] + ([args.batch] if scale <= 1 and dev != "cpu" else [])
            t = time.perf_counter()
            if kw == "metal":
                from core.render_metal import MetalFlatPortRenderer
                tr = MetalFlatPortRenderer(r)
            elif kw == "warp":
                from core.render_warp import WarpFlatPortRenderer
                tr = WarpFlatPortRenderer(r, dev)
            else:
                tr = TorchFlatPortRenderer(r, dev, **kw)
            synchronize(dev)
            t_up = 1e3 * (time.perf_counter() - t)
            reps = cpu_reps if dev == "cpu" else args.repeats
            for B in batches:
                rgbB = np.ascontiguousarray(np.broadcast_to(rgb, (B,) + rgb.shape))
                zB = np.ascontiguousarray(np.broadcast_to(z, (B,) + z.shape))
                h2d = timeit(lambda: (_upload(rgbB, dev), _upload(zB, dev)), dev, reps)
                rgb_t, z_t = _upload(rgbB, dev), _upload(zB, dev)
                t = time.perf_counter()
                # fused kernels: image + flags (valid / too near / background) + w; the torch
                # path always returns all per-pixel outputs (map, Z, water range, masks)
                kw_r = dict(info=False) if kw in ("metal", "warp") else dict(edges=False)
                img, _ = tr.render(rgb_t, z_t, **kw_r)
                synchronize(dev)
                t_first = 1e3 * (time.perf_counter() - t)
                frame = timeit(lambda: tr.render(rgb_t, z_t, **kw_r), dev, reps)
                d2h = timeit(lambda: img.cpu().numpy(), dev, reps)
                extra = {}
                if kw in ("metal", "warp"):
                    x_k = rgb_t if kw == "metal" else rgb_t.float().contiguous()
                    k = timeit(lambda: tr.launch(z_t, x_k), dev, reps)
                    extra["kernel_only"] = dict(k, median=k["median"] / B, p25=k["p25"] / B, p75=k["p75"] / B)
                if kw == {} and B == 1:
                    e = timeit(lambda: tr.render(rgb_t, z_t, edges=True), dev, reps)
                    extra["edges_extra_ms"] = round(e["median"] - frame["median"], 1)
                per = {k: dict(v, median=v["median"] / B, p25=v["p25"] / B, p75=v["p75"] / B)
                       for k, v in (("frame", frame), ("h2d", h2d), ("d2h", d2h))}
                row(res=res_s, method=name, B=B, setup_ms=round(t_setup_np + t_up), upload_ms=round(t_up, 1),
                    first_call_ms=round(t_first), frame=per["frame"], h2d=per["h2d"], d2h=per["d2h"], **extra)
                del rgb_t, z_t, img
            del tr
            if dev == "mps":
                torch.mps.empty_cache()

    res["loadavg_end"] = os.getloadavg()
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1, default=str)

    print("\n| output | method | B | setup (ms) | frame ms/frame median (p25-p75) | H2D ms/frame | D2H ms/frame | fps (frame only) |")
    print("|---|---|---|---|---|---|---|---|")
    for rw in res["rows"]:
        f = rw["frame"]
        print(f"| {rw['res']} | {rw['method']} | {rw['B']} | {rw['setup_ms']} | {fmt(f)} | "
              f"{fmt(rw['h2d']) if 'h2d' in rw else '-'} | {fmt(rw['d2h']) if 'd2h' in rw else '-'} | "
              f"{1e3 / f['median']:.0f} |")


if __name__ == "__main__":
    main()
