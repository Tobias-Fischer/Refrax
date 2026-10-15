"""Validation of the flat-port refraction renderer (core/render.py).

Usage:
    python -m sim.validate configs/dataset.yaml [--quick] [--figures docs/figures]

(a) analytic scenes: exact ray trace through the housing vs. RGB-D post-process
    (+ two central-camera baselines that ignore the per-pixel ray offset)
(b) round trip: rendered underwater image -> Refrax removal at the true depth -> pinhole
(c) consistency with visualise_refraction.py (checkerboard at z0_fixed)

Pixel errors are measured in the *output* (underwater) image: the scene point whose colour
a pixel shows is projected back into the underwater camera with the exact refractive
projection (core.render.project_underwater) and compared with the pixel itself.
Exit status 1 if --quick thresholds are exceeded.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from core.config import load_config, camera_params, distortion_params, housing_params
from core.render import FlatPortRenderer, project_underwater, _sample, plan_render_camera
from sim.scenes import standard_scenes, Scene, Plane


def stats(e):
    e = e[np.isfinite(e)]
    if e.size == 0:
        return dict(n=0)
    return dict(n=int(e.size), mean=float(e.mean()), median=float(np.median(e)),
                p99=float(np.percentile(e, 99)), max=float(e.max()),
                frac_gt_0_5=float((e > 0.5).mean()), frac_gt_1=float((e > 1).mean()))


def pixel_error(S, r, sub):
    """|refractive projection of S - pixel| for the output pixels in `sub` (index arrays)."""
    vv, uu = sub
    u, v = project_underwater(S[vv, uu], r.K, r.housing, r.distortion,
                              uv0=(uu.astype(float), vv.astype(float)))
    return np.hypot(u - uu, v - vv)


def central_baseline(r, w0):
    """Render-image coordinates of a central (single-viewpoint) warp, exact for 1/Z = w0
    (w0 = 0: exact at infinity) -- what a LUT / per-pixel-direction lens model can do."""
    return r._uv(np.full((r.H, r.W), w0))


def eval_scene(r, scene, rng, n_eval):
    rgb_r, z_r, X_r = scene.render_pinhole(r.spec)
    t0 = time.time()
    img, info = r.render(rgb_r, z_r)
    t_render = time.time() - t0

    # ground truth: trace every output pixel's water ray through the scene
    t_gt, X_gt, id_gt = scene.trace(r.P2, r.d)
    rgb_gt = scene.shade(X_gt, id_gt)

    hit = np.isfinite(t_gt) & info["valid"] & ~info["background"]
    vv, uu = np.nonzero(hit)
    if vv.size > n_eval:
        sel = rng.choice(vv.size, n_eval, replace=False)
        vv, uu = vv[sel], uu[sel]
    edge = info["near_edge"][vv, uu]

    out = dict(scene=scene.name, render_s=t_render, n_steps=r.n_steps,
               valid_frac=float(info["valid"].mean()))
    S = _sample(X_r, info["map_u"], info["map_v"], exact=True)
    e = pixel_error(S, r, (vv, uu))
    out["px_err_all"], out["px_err_no_edge"] = stats(e), stats(e[~edge])
    out["edge_frac"] = float(edge.mean())
    # depth along the ray (marching result) vs. ground truth, away from edges
    zr = (info["Z"][vv, uu] - X_gt[vv, uu, 2]) / X_gt[vv, uu, 2]
    out["rel_depth_err_no_edge"] = stats(np.abs(zr[~edge]))
    # background classification agreement
    bg_gt = ~np.isfinite(t_gt)
    out["background_mismatch_frac"] = float((bg_gt != info["background"])[info["valid"]].mean())
    # image error vs. exact ground-truth image
    diff = np.abs(img.astype(np.float64) - rgb_gt)[info["valid"]]
    out["psnr_db"] = float(10 * np.log10(1.0 / np.mean(diff ** 2)))

    for name, w0 in (("central_inf", 0.0), ("central_1m", 1.0 / (1.0 - r.spec.centre[2]))):
        mu, mv = central_baseline(r, w0)
        Sb = _sample(X_r, mu, mv, exact=True)
        eb = pixel_error(Sb, r, (vv, uu))
        out[f"baseline_{name}_px_err_no_edge"] = stats(eb[~edge])
    return out, dict(rgb_r=rgb_r, z_r=z_r, img=img, rgb_gt=rgb_gt, info=info, X_r=X_r,
                     X_gt=X_gt, S=S)


def round_trip(cfg, r, Z_list, tilted=True, with_distortion=False):
    """Underwater render -> Refrax removal at the true depth -> pinhole; px error in the
    corrected image (pinhole at the camera centre with focal zoom * f)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from remove_refraction import make_map_builder, compute_valid_mask
    from core.scale import select_zoom

    W, H, fx, fy, cx, cy = camera_params(cfg)
    housing = housing_params(cfg)
    dist = r.distortion if r.distortion is not None else (0.0, 0.0, 0.0, 0.0)
    zoom, _ = select_zoom(cfg["correction"]["z0_fixed"], W, H, fx, fy, cx, cy, housing)
    build_map = make_map_builder(W, H, fx, fy, cx, cy, housing, dist, zoom)
    ug, vg = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))

    scenes = [(f"plane Z={Z:g} m", Scene("", [Plane([0, 0, Z], [0, 0, -1])]), Z) for Z in Z_list]
    if tilted:
        th = np.radians(35)
        scenes.append(("tilted plane 35 deg (per-pixel depth)",
                       Scene("", [Plane([0, 0, 1.0], [np.sin(th), 0, -np.cos(th)])]), None))
    res = []
    for name, scene, Z in scenes:
        _, z_r, X_r = scene.render_pinhole(r.spec)
        _, info = r.render(np.zeros(z_r.shape + (3,), np.float32), z_r)
        X_uw = _sample(X_r, info["map_u"], info["map_v"], exact=True)          # renderer
        _, X_gt, _ = scene.trace(r.P2, r.d)                                     # exact
        depth = Z if Z is not None else np.where(info["valid"], info["Z"], 1.0).astype(np.float32)
        map_x, map_y = build_map(depth)
        valid = compute_valid_mask(map_x, map_y, W, H) > 0
        row = dict(scene=name, valid_frac=float(valid.mean()))
        for tag, Xu in (("renderer", X_uw), ("exact_trace", X_gt)):
            Xc = _sample(Xu, map_x.astype(np.float64), map_y.astype(np.float64), exact=True)
            u = zoom * fx * Xc[..., 0] / Xc[..., 2] + cx
            v = zoom * fy * Xc[..., 1] / Xc[..., 2] + cy
            e = np.hypot(u - ug, v - vg)[valid]
            row[tag] = stats(e)
            # interior only (2 px away from the invalid border, where the linear-
            # interpolation hull of the inverse map is cut)
        res.append(row)
    return dict(zoom=zoom, distortion=with_distortion, rows=res)


def visualise_consistency(cfg, r):
    """Compare with visualise_refraction.py: same checkerboard image and world points."""
    from visualise_refraction import make_checkerboard, BOARD_COLS, BOARD_ROWS, SQUARE_SIZE
    from core.optics import matrix_K
    from core.undistort import compute_housing_geometry, forward_map

    W, H, fx, fy, cx, cy = camera_params(cfg)
    housing = housing_params(cfg)
    Z0 = cfg["correction"]["z0_fixed"]
    _, K_inv = matrix_K(fx, fy, cx, cy)
    P2, ray_water = compute_housing_geometry(H, W, K_inv, **housing)
    fwd_x, fwd_y = forward_map(P2, ray_water, Z0, fx, fy, cx, cy)
    P_vis = np.stack([(fwd_x - cx) / fx * Z0, (fwd_y - cy) / fy * Z0, np.full((H, W), Z0)], -1)
    img_vis = make_checkerboard(P_vis, BOARD_COLS, BOARD_ROWS, SQUARE_SIZE).astype(np.uint8)

    scene = Scene("", [Plane([0, 0, Z0], [0, 0, -1])])
    _, z_r, X_r = scene.render_pinhole(r.spec)
    _, info = r.render(np.zeros(z_r.shape + (3,), np.float32), z_r)
    S = _sample(X_r, info["map_u"], info["map_v"], exact=True)
    board_r = make_checkerboard(X_r, BOARD_COLS, BOARD_ROWS, SQUARE_SIZE).astype(np.uint8)
    img_ren = _sample(board_r, info["map_u"], info["map_v"], exact=False)   # what a sim would get
    dX = np.linalg.norm(S - P_vis, axis=-1)
    _, X_exact, _ = scene.trace(r.P2, r.d)
    dX_exact = np.linalg.norm(X_exact - P_vis, axis=-1)
    # pixels whose label differs (nearest-label agreement away from square borders)
    mism = (np.abs(img_ren.astype(int) - img_vis.astype(int)) > 64)
    return dict(Z0=Z0, world_pt_diff_mm=stats(1e3 * dX), exact_trace_vs_visualise_mm=stats(1e3 * dX_exact),
                checkerboard_label_mismatch_frac=float(mism.mean())), (img_vis, img_ren)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/dataset.yaml")
    ap.add_argument("--quick", action="store_true", help="subset of scenes, threshold check")
    ap.add_argument("--figures", default=None, help="directory for figures")
    ap.add_argument("--json", default=None, help="write all numbers to this JSON file")
    ap.add_argument("--n-eval", type=int, default=40000, help="pixels per scene for px error")
    args = ap.parse_args()

    cfg = load_config(args.config)
    W, H, fx, fy, cx, cy = camera_params(cfg)
    housing = housing_params(cfg)
    rng = np.random.default_rng(0)
    results = {}

    r = FlatPortRenderer(W, H, fx, fy, cx, cy, housing, z_near=0.2)
    r0 = FlatPortRenderer(W, H, fx, fy, cx, cy, housing, z_near=0.2, centre="origin")
    results["render_camera_virtual"] = r.spec.to_dict() | dict(n_steps=r.n_steps,
                                                                max_b_mm=1e3 * float(np.abs(r.b).max()))
    results["render_camera_origin"] = r0.spec.to_dict() | dict(n_steps=r0.n_steps,
                                                               max_b_mm=1e3 * float(np.abs(r0.b).max()))
    print(json.dumps(results, indent=1, default=float))

    scenes = standard_scenes()
    names = ["plane_1m", "steps", "spheres"] if args.quick else list(scenes)
    results["analytic"], keep = {}, {}
    for nm in names:
        for tag, rr in (("virtual", r),) if args.quick else (("virtual", r), ("origin", r0)):
            out, data = eval_scene(rr, scenes[nm], rng, args.n_eval)
            results["analytic"][f"{nm}/{tag}"] = out
            if tag == "virtual":
                keep[nm] = data
            e, eb = out["px_err_no_edge"], out["baseline_central_inf_px_err_no_edge"]
            print(f"[a] {nm:13s} {tag:7s} steps={out['n_steps']:3d} t={out['render_s']:.2f}s "
                  f"px err (no edge) median={e['median']:.4f} p99={e['p99']:.3f} max={e['max']:.3f} | "
                  f"all p99={out['px_err_all']['p99']:.3f} max={out['px_err_all']['max']:.2f} "
                  f"edge={out['edge_frac']:.4f} | central@inf p99={eb.get('p99', np.nan):.2f} "
                  f"max={eb.get('max', np.nan):.2f} | PSNR={out['psnr_db']:.1f}")

    rt = round_trip(cfg, r, [0.5, 1.0, 3.0] if not args.quick else [1.0], tilted=not args.quick)
    results["round_trip"] = rt
    for row in rt["rows"]:
        print(f"[b] {row['scene']:38s} renderer p99={row['renderer']['p99']:.3f} max={row['renderer']['max']:.3f} "
              f"| exact trace p99={row['exact_trace']['p99']:.3f} max={row['exact_trace']['max']:.3f}")
    if not args.quick:
        dist = distortion_params(cfg)
        rd = FlatPortRenderer(W, H, fx, fy, cx, cy, housing, distortion=dist, z_near=0.2)
        rt_d = round_trip(cfg, rd, [1.0], tilted=False, with_distortion=True)
        results["round_trip_radtan"] = rt_d
        for row in rt_d["rows"]:
            print(f"[b] radtan {row['scene']:31s} renderer p99={row['renderer']['p99']:.3f} "
                  f"max={row['renderer']['max']:.3f} | exact p99={row['exact_trace']['p99']:.3f}")

    vc, vis_imgs = visualise_consistency(cfg, r)
    results["visualise_consistency"] = vc
    print(f"[c] world point diff to visualise_refraction: max={vc['world_pt_diff_mm']['max']:.4f} mm, "
          f"exact trace max={vc['exact_trace_vs_visualise_mm']['max']:.2e} mm, "
          f"checkerboard label mismatch={vc['checkerboard_label_mismatch_frac']:.2e}")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1, default=float)
    if args.figures:
        from sim.figures import make_figures
        make_figures(args.figures, r, r0, keep, vis_imgs, results)

    if args.quick:
        bad = []
        for k, v in results["analytic"].items():
            if v["px_err_no_edge"]["p99"] > 0.1:
                bad.append(k)
        for row in rt["rows"]:
            if row["renderer"]["p99"] > 0.1:
                bad.append(row["scene"])
        if vc["world_pt_diff_mm"]["max"] > 0.5:
            bad.append("visualise")
        print("QUICK CHECK:", "FAIL " + ", ".join(bad) if bad else "PASS")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
