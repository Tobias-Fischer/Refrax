"""Accuracy of the torch / fused-Metal renderers against the numpy renderer (core/render.py).

Usage:
    python -m sim.validate_torch configs/dataset.yaml [--scale 1] [--json docs/figures/gpu_validation.json]

For every validation scene (sim.scenes.standard_scenes, virtual-centre render camera) and
every variant (torch cpu / mps / cuda, float32 / float16, torch.compile, fused Metal and Warp kernels)
it reports the difference of the render-image sample coordinates (map_u, map_v) to the numpy
renderer (exact=True, float64 bilinear sampling), off depth edges and overall, mask
agreement, image PSNR against numpy's image (cv2 path) and the px error against the exact
analytic ray trace (same metric as sim/validate.py).
"""
import argparse
import json
import os

import numpy as np
import torch

from core.config import load_config, camera_params, housing_params
from core.render import FlatPortRenderer, _sample
from core.render_torch import TorchFlatPortRenderer
from sim.scenes import standard_scenes


def scaled_renderer(cfg, scale=1.0, **kw):
    """FlatPortRenderer for the config camera resampled to scale * (W, H) (pixel centres at
    integers: c' = (c + 0.5) * scale - 0.5)."""
    W, H, fx, fy, cx, cy = camera_params(cfg)
    return FlatPortRenderer(int(round(W * scale)), int(round(H * scale)), fx * scale, fy * scale,
                            (cx + 0.5) * scale - 0.5, (cy + 0.5) * scale - 0.5,
                            housing_params(cfg), z_near=0.2, **kw)


def available_devices():
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def make_variants(r, devices=None, fp16=True, compiled=True, metal=True):
    """{name: renderer} of the variants to compare."""
    out = {}
    for dev in devices or available_devices():
        out[f"torch-{dev}"] = TorchFlatPortRenderer(r, dev)
        if dev != "cpu" and fp16:
            out[f"torch-{dev}-fp16"] = TorchFlatPortRenderer(r, dev, dtype=torch.float16)
        if dev != "cpu" and compiled:
            out[f"torch-{dev}-compile"] = TorchFlatPortRenderer(r, dev, compile=True)
        if dev == "mps" and metal:
            from core.render_metal import MetalFlatPortRenderer
            out["metal-fused"] = MetalFlatPortRenderer(r)
        if dev in ("cpu", "cuda") and metal:
            try:
                from core.render_warp import WarpFlatPortRenderer
                out[f"warp-{dev}"] = WarpFlatPortRenderer(r, dev)
            except ImportError:
                pass
    return out


def to_np(info):
    return {k: v[0].float().cpu().numpy() if v.dtype != torch.bool else v[0].cpu().numpy()
            for k, v in info.items() if k != "grid" and v.ndim == 3}


def compare(ref_info, ref_img, info, img):
    """Coordinate / mask / image agreement of one variant's output with the numpy reference
    (ref: exact=True trace; ref_img: numpy image to compare the variant's image with)."""
    both = ref_info["valid"] & info["valid"]
    ne = both & ~ref_info["near_edge"]
    d = np.hypot(info["map_u"] - ref_info["map_u"], info["map_v"] - ref_info["map_v"])
    mse = np.mean((img[both].astype(np.float64) - ref_img[both].astype(np.float64)) ** 2)
    pct = lambda x, q: float(np.percentile(x, q)) if x.size else 0.0
    return dict(
        coord_max_no_edge=float(d[ne].max()), coord_p99_no_edge=pct(d[ne], 99),
        coord_max_all=float(d[both].max()), coord_p99_all=pct(d[both], 99),
        coord_p9999_all=pct(d[both], 99.99), frac_gt_0_1px=float((d[both] > 0.1).mean()),
        valid_mismatch=float((ref_info["valid"] != info["valid"]).mean()),
        background_mismatch=float((ref_info["background"] != info["background"]).mean()),
        psnr_vs_numpy_db=float(10 * np.log10(1.0 / max(mse, 1e-30))),
    )


def gt_px_error(r, scene, X_r, info, n_eval, rng):
    """Same metric as sim/validate.py: refractive reprojection error of the shown scene
    point, off edges (p99, max) over n_eval random hit pixels."""
    from sim.validate import pixel_error
    t_gt, _, _ = scene.trace(r.P2, r.d)
    hit = np.isfinite(t_gt) & info["valid"] & ~info["background"] & ~info["near_edge"]
    vv, uu = np.nonzero(hit)
    if vv.size > n_eval:
        sel = rng.choice(vv.size, n_eval, replace=False)
        vv, uu = vv[sel], uu[sel]
    S = _sample(X_r, info["map_u"].astype(np.float64), info["map_v"].astype(np.float64), exact=True)
    e = pixel_error(S, r, (vv, uu))
    e = e[np.isfinite(e)]
    return dict(p99=float(np.percentile(e, 99)), max=float(e.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/dataset.yaml")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--n-eval", type=int, default=10000)
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    r = scaled_renderer(cfg, args.scale)
    variants = make_variants(r)
    scenes = standard_scenes()
    names = args.scenes or list(scenes)
    rng = np.random.default_rng(0)
    res = dict(output=[r.W, r.H], render=[r.spec.W, r.spec.H], n_steps=r.n_steps, scenes={})
    batch = []
    for nm in names:
        rgb, z, X_r = scenes[nm].render_pinhole(r.spec)
        batch.append((rgb, z))
        img_np, _ = r.render(rgb, z)                       # numpy default (cv2) path
        img_x, info_x = r.render(rgb, z, exact=True)       # float64 reference
        row = dict(numpy_cv2=compare(info_x, img_x, r.trace(z), img_np))
        row["numpy_cv2"]["gt_px_err_no_edge"] = gt_px_error(r, scenes[nm], X_r, info_x, args.n_eval, rng)
        for vn, tr in variants.items():
            img, info = tr.render(rgb, z)
            info_n = to_np(info)
            info_n["near_edge"] = info_x["near_edge"]     # metal path has no edge mask
            row[vn] = compare(info_x, img_x, info_n, img[0].float().cpu().numpy())
            if not vn.endswith("fp16"):
                row[vn]["gt_px_err_no_edge"] = gt_px_error(r, scenes[nm], X_r, info_n | dict(near_edge=info_x["near_edge"]), args.n_eval, rng)
        res["scenes"][nm] = row
        print(f"--- {nm}")
        for vn, m in row.items():
            g = m.get("gt_px_err_no_edge", {})
            print(f"  {vn:20s} coord vs numpy: off-edge max {m['coord_max_no_edge']:.1e} p99 {m['coord_p99_no_edge']:.1e} | "
                  f"all max {m['coord_max_all']:.3f} p99.99 {m['coord_p9999_all']:.1e} | valid/bg mism "
                  f"{m['valid_mismatch']:.0e}/{m['background_mismatch']:.0e} | PSNR {m['psnr_vs_numpy_db']:.1f} dB | "
                  f"GT err p99 {g.get('p99', np.nan):.1e} max {g.get('max', np.nan):.1e}")

    # batched rendering == per-frame rendering
    rgbs = np.stack([b[0] for b in batch]); zs = np.stack([b[1] for b in batch])
    res["batch_vs_single_max_abs_px"] = {}
    for vn, tr in variants.items():
        img_b, info_b = tr.render(rgbs, zs)
        dmax = 0.0
        for i, (rgb, z) in enumerate(batch):
            _, info_s = tr.render(rgb, z)
            dmax = max(dmax, float((info_b["map_u"][i] - info_s["map_u"][0]).abs().max()))
        res["batch_vs_single_max_abs_px"][vn] = dmax
    print("batch vs single max |du| px:", res["batch_vs_single_max_abs_px"])
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
