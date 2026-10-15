"""Depth-aware refraction removal: fixed-depth Refrax -> COLMAP -> sparse + mono depth -> re-correct.

    python depth_aware/pipeline.py [--config configs/dataset.yaml] [--rgb-dir DIR] [--step 2]
                             [--z0 1.0] [--center-z 0] [--scale ba|<metres per SfM unit>]
                             [--iters 1] [--work DIR]

Stages (each cached under --work, rerun-safe):
  0  correct every --step-th image at the fixed depth z0 (the Refrax baseline)
  1  COLMAP SfM with fixed PINHOLE intrinsics on the corrected images
  2  per-image sparse depth: COLMAP points in each camera frame (SfM units)
  3  Depth Anything V2 relative disparity on each corrected image
  4  robust per-image fit 1/Z ~ a * disp + b (Huber IRLS), dense depth in SfM units
  5  metric scale s (metres per SfM unit): either given, or self-calibrated from the
     refraction residual with a refractive-BA proxy (see scale_by_ba)
  6  dense metric depth Z = s * Zc + center_z resampled onto the ideal raw grid
     (<work>/<run>/depth/<name>.npy, usable as paths.depth_dir of remove_refraction.py)
  7  depth-aware correction in the reference output frame (one shared calibration)
     -> COLMAP again; optionally iterate 2-7
Results go to <work>/summary.json; docs/explore_depth.md explains the design.
"""
import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np

from common import Corrector, load_cfg, WORK
import colmap_utils as cu
import mono as mono_mod

DATA = "/Users/fischert/robot/refrax-data/gopro10_linear"


# ------------------------------------------------------------------ helpers
def bilinear(img, xy):
    """Bilinear lookup of a (H, W[, C]) array at float positions xy (N, 2); NaN outside."""
    m = cv2.remap(img.astype(np.float32), xy[:, 0:1].astype(np.float32),
                  xy[:, 1:2].astype(np.float32), cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    return m.reshape(len(xy), -1).squeeze()


def correct_set(C, names, rgb_dir, out_dir, depth_for):
    """Write corrected images + masks; depth_for(name) gives a scalar or ideal-grid map."""
    img_dir, mask_dir = f"{out_dir}/images", f"{out_dir}/masks"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    for n in names:
        if os.path.exists(f"{img_dir}/{n}.png"):
            continue
        img = cv2.imread(f"{rgb_dir}/{n}.png")
        out, mask = C.correct_image(img, depth_for(n))
        cv2.imwrite(f"{img_dir}/{n}.png", out)
        # COLMAP mask: erode so features never sit on the black border
        cv2.imwrite(f"{mask_dir}/{n}.png.png", cv2.erode(mask, np.ones((7, 7), np.uint8)))
    C.write_calibration(f"{out_dir}/calibration.yaml")
    return img_dir, mask_dir


def sparse_depth(model_txt, min_track=3, max_err=2.0):
    """{name: (xy out-frame (N,2), Zc (N,) SfM units)} from a COLMAP text model."""
    imgs = cu.read_images(f"{model_txt}/images.txt")
    pts = cu.read_points3d(f"{model_txt}/points3D.txt")
    out = {}
    for name, d in imgs.items():
        sel = d["p3d"] >= 0
        xy, ids = d["xy"][sel], d["p3d"][sel]
        good = np.array([pts[i][2] >= min_track and pts[i][1] <= max_err for i in ids], bool)
        if not good.any():
            continue
        X = np.array([pts[i][0] for i in ids[good]])
        Zc = (X @ d["R"].T + d["t"])[:, 2]
        ok = Zc > 0
        out[os.path.splitext(name)[0]] = (xy[good][ok], Zc[ok])
    return out


def ideal_obs(C, xy, depth):
    """Ideal raw pixel of out-frame observations of an image corrected with `depth`."""
    ix, iy = C.out_to_ideal_map(depth)
    return np.stack([bilinear(ix, xy), bilinear(iy, xy)], 1)


# ------------------------------------------------------------------ scale
def write_model_with_xy(src_txt, dst_txt, new_xy):
    """Copy a TXT model, replacing the 2D observations of each image."""
    os.makedirs(dst_txt, exist_ok=True)
    for f in ("cameras.txt", "points3D.txt"):
        shutil.copy(f"{src_txt}/{f}", f"{dst_txt}/{f}")
    with open(f"{src_txt}/images.txt") as f:
        lines = [l for l in f if not l.startswith("#")]
    with open(f"{dst_txt}/images.txt", "w") as f:
        for i in range(0, len(lines) - 1, 2):
            head = lines[i]
            name = os.path.splitext(head.split()[9])[0]
            pts = np.array(lines[i + 1].split(), dtype=float).reshape(-1, 3)
            if name in new_xy:
                pts[:, :2] = new_xy[name]
            f.write(head)
            f.write(" ".join(f"{x:.4f} {y:.4f} {int(p)}" for x, y, p in pts) + "\n")


def obs_depths(model_txt):
    """{name: (xy (N,2) all obs, Zc (N,) or NaN for untriangulated)}."""
    imgs = cu.read_images(f"{model_txt}/images.txt")
    pts = cu.read_points3d(f"{model_txt}/points3D.txt")
    out = {}
    for name, d in imgs.items():
        Zc = np.full(len(d["xy"]), np.nan)
        sel = np.flatnonzero(d["p3d"] >= 0)
        if len(sel):
            X = np.array([pts[i][0] for i in d["p3d"][sel]])
            Zc[sel] = (X @ d["R"].T + d["t"])[:, 2]
        out[os.path.splitext(name)[0]] = (d["xy"], Zc)
    return out


def scale_by_ba(C, model_txt, depth_for, work, scales, log):
    """Refractive-BA proxy for the metric scale.

    For each candidate s (metres per SfM unit) every observation is re-corrected
    point-wise with its own depth Z = s * Zc + center_z (the exact flat-port model,
    starting from its ideal raw pixel), and COLMAP's bundle adjuster is re-run with
    fixed intrinsics. A wrong s leaves a depth-dependent, non-pinhole residual, so the
    final reprojection error is minimal near the true scale. Returns (best s, table).
    """
    obs = obs_depths(model_txt)
    ideal = {n: ideal_obs(C, xy, depth_for(n)) for n, (xy, _) in obs.items()}
    table = []
    for s in scales:
        new_xy = {}
        for n, (xy, Zc) in obs.items():
            q = xy.copy()
            ok = np.isfinite(Zc) & np.all(np.isfinite(ideal[n]), 1)
            if ok.any():
                Z = s * Zc[ok] + C.center_z
                q[ok] = C.project_world(C.world_points_at_depth(ideal[n][ok], Z))
            new_xy[n] = q
        d = f"{work}/s_{s:.5f}"
        write_model_with_xy(model_txt, f"{d}/in", new_xy)
        os.makedirs(f"{d}/out", exist_ok=True)
        cu.run(["colmap", "bundle_adjuster", "--input_path", f"{d}/in", "--output_path",
                f"{d}/out", "--BundleAdjustment.refine_focal_length", "0",
                "--BundleAdjustment.refine_principal_point", "0",
                "--BundleAdjustment.refine_extra_params", "0",
                "--BundleAdjustment.max_num_iterations", "50"], log)
        os.makedirs(f"{d}/out_txt", exist_ok=True)
        cu.run(["colmap", "model_converter", "--input_path", f"{d}/out", "--output_path",
                f"{d}/out_txt", "--output_type", "TXT"], log)
        err = reprojection_rms(f"{d}/out_txt")
        table.append((float(s), err))
        print(f"  scale {s:.4f} m/unit -> BA reprojection RMS {err:.4f} px", flush=True)
        shutil.rmtree(f"{d}/in")
    ss, ee = np.array(table).T
    i = int(np.argmin(ee))
    if 0 < i < len(ss) - 1:  # parabola in log s through the minimum and its neighbours
        x, y = np.log(ss[i - 1:i + 2]), ee[i - 1:i + 2]
        a, b, _ = np.polyfit(x, y, 2)
        best = float(np.exp(-b / (2 * a))) if a > 0 else float(ss[i])
    else:
        best = float(ss[i])
    return best, table


def reprojection_rms(model_txt):
    """RMS reprojection error (px) over all observations of a TXT model."""
    cams = {}
    with open(f"{model_txt}/cameras.txt") as f:
        for l in f:
            if l.startswith("#"):
                continue
            e = l.split()
            cams[int(e[0])] = np.array(list(map(float, e[4:8])))
    imgs = cu.read_images(f"{model_txt}/images.txt")
    pts = cu.read_points3d(f"{model_txt}/points3D.txt")
    sq = []
    for d in imgs.values():
        fx, fy, cx, cy = cams[d["cam"]]
        sel = d["p3d"] >= 0
        X = np.array([pts[i][0] for i in d["p3d"][sel]])
        Xc = X @ d["R"].T + d["t"]
        uv = np.stack([fx * Xc[:, 0] / Xc[:, 2] + cx, fy * Xc[:, 1] / Xc[:, 2] + cy], 1)
        sq.append(((uv - d["xy"][sel]) ** 2).sum(1))
    return float(np.sqrt(np.mean(np.concatenate(sq))))


# ------------------------------------------------------------------ main stages
def mono_stage(names, img_dir, cache_dir, model=None):
    os.makedirs(cache_dir, exist_ok=True)
    for n in names:
        p = f"{cache_dir}/{n}.npy"
        if os.path.exists(p):
            continue
        if model is None:
            model = mono_mod.MonoDepth()
        rgb = cv2.imread(f"{img_dir}/{n}.png")[..., ::-1].copy()
        np.save(p, model(rgb).astype(np.float16))
    return model


def fit_stage(C, sparse, names, disp_dir, kind="affine"):
    """Per-image fits; returns ({name: dense Zc (out frame, SfM units)}, stats rows)."""
    size = (C.out_W, C.out_H)
    dense, rows = {}, []
    for n in names:
        if n not in sparse:
            continue
        xy, Zc = sparse[n]
        disp = np.load(f"{disp_dir}/{n}.npy").astype(np.float32)
        ds = bilinear(disp, xy)
        ok = np.isfinite(ds)
        xy, Zc, ds = xy[ok], Zc[ok], ds[ok]
        row = dict(name=n, n_sparse=int(len(Zc)))
        if len(Zc) < 30:
            row["status"] = "too_few"
            rows.append(row)
            continue
        for k in ("scale", "affine", "spatial"):
            row[f"holdout_{k}"] = mono_mod.holdout_error(ds, Zc, xy, size, k)
        # median relative spread of the sparse depths themselves (constant-depth baseline)
        row["holdout_const"] = float(np.median(np.abs(Zc - np.median(Zc)) / Zc))
        p, keep = mono_mod.fit_inverse_depth(ds, Zc, xy, size, kind)
        if p[0] <= 0:
            row["status"] = "negative_gain"
            p, keep = mono_mod.fit_inverse_depth(ds, Zc, xy, size, "scale")
            k_used = "scale"
        else:
            row["status"] = "ok"
            k_used = kind
        row["inlier_frac"] = float(keep.mean())
        dense[n] = mono_mod.dense_from_fit(p, disp, size, k_used, Zc)
        rows.append(row)
    return dense, rows


def to_ideal_grid(C, Zc_out, s, used_depth):
    """Metric depth on the ideal grid from an out-frame SfM-unit map of an image that
    was corrected with used_depth (scalar or ideal-grid map)."""
    gx, gy = C.ideal_to_out_grid(used_depth)
    Zm = s * Zc_out + C.center_z
    return cv2.remap(Zm.astype(np.float32), gx.astype(np.float32), gy.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--rgb-dir", default=DATA)
    ap.add_argument("--work", default=f"{WORK}/gopro10")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--z0", type=float, default=None, help="default: correction.z0_fixed")
    ap.add_argument("--center-z", type=float, default=0.0)
    ap.add_argument("--scale", default="ba", help="'ba' or metres per SfM unit")
    ap.add_argument("--scale-grid", default="", help="comma list of candidate scales for 'ba'")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--fit", default="affine", choices=["scale", "affine", "spatial"])
    ap.add_argument("--mode", default="dense", choices=["dense", "median"],
                    help="dense: mono+sparse per-pixel depth; median: one depth per image "
                         "(median of its sparse depth)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    z0 = args.z0 if args.z0 is not None else cfg["correction"]["z0_fixed"]
    C = Corrector(cfg, z0=z0, center_z=args.center_z)
    tag = f"z{z0:g}_c{args.center_z * 1000:g}mm"
    work = f"{args.work}/{tag}"
    os.makedirs(work, exist_ok=True)
    names = [os.path.splitext(os.path.basename(p))[0]
             for p in sorted(glob.glob(f"{args.rgb_dir}/*.png"))][::args.step]
    calib = (C.out_W, C.out_H, C.out_fx, C.out_fy, C.out_cx, C.out_cy)
    summary_path = f"{work}/summary.json"
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    summary.update(dict(z0=z0, center_z=args.center_z, n_images=len(names), zoom=C.zoom,
                        out_size=[int(C.out_W), int(C.out_H)]))

    # stage 0-1: fixed-depth baseline
    run = f"{work}/it0_fixed"
    print(f"[it0] fixed-depth correction at z0={z0} of {len(names)} images", flush=True)
    img_dir, mask_dir = correct_set(C, names, args.rgb_dir, run, lambda n: z0)
    model = cu.sfm(img_dir, f"{run}/colmap", calib, mask_dir)
    summary["it0_fixed"] = cu.model_stats(model, len(names))
    print("  ", summary["it0_fixed"], flush=True)
    used = {n: z0 for n in names}  # scalar depth or path of an ideal-grid .npy

    def depth_of(n):
        u = used[n]
        return np.load(u).astype(np.float32) if isinstance(u, str) else u

    model_prev = model
    mono_model = None

    for it in range(1, args.iters + 1):
        prev_run = run
        sparse = sparse_depth(model_prev)
        print(f"[it{it}] sparse depth for {len(sparse)} images; mono depth...", flush=True)
        mono_model = mono_stage(list(sparse), f"{prev_run}/images", f"{prev_run}/disp", mono_model)
        dense, rows = fit_stage(C, sparse, list(sparse), f"{prev_run}/disp", args.fit)
        json.dump(rows, open(f"{prev_run}/fit_stats.json", "w"), indent=1)

        if args.scale == "ba" and it == 1 and "it1_scale_table" in summary:
            s = summary["it1_scale"]  # depends only on the fixed-depth model: reuse
        elif args.scale == "ba":
            grid = ([float(x) for x in args.scale_grid.split(",")] if args.scale_grid else None)
            if grid is None:
                # centre the grid on the scale that puts the median depth at z0
                med = np.median(np.concatenate([z for _, z in sparse.values()]))
                s0 = (z0 - C.center_z) / med
                grid = list(s0 * np.geomspace(0.25, 4.0, 9))
            s, table = scale_by_ba(C, model_prev, depth_of, f"{prev_run}/scale_ba", grid,
                                   f"{prev_run}/scale_ba.log")
            summary[f"it{it}_scale_table"] = table
            summary[f"it{it}_ba_unmodified_rms"] = reprojection_rms(model_prev)
        else:
            s = float(args.scale)
        summary[f"it{it}_scale"] = s
        med_Z = float(np.median(np.concatenate([s * z for _, z in sparse.values()])) + C.center_z)
        summary[f"it{it}_median_depth_m"] = med_Z
        print(f"  metric scale {s:.5f} m/unit, median depth {med_Z:.3f} m", flush=True)

        run = f"{work}/it{it}_{'depth' if args.mode == 'dense' else 'median'}"
        os.makedirs(f"{run}/depth", exist_ok=True)
        new_used = {}
        for n in names:
            if args.mode == "median" and n in sparse:
                new_used[n] = float(s * np.median(sparse[n][1]) + C.center_z)
            elif args.mode == "dense" and n in dense:
                Zi = to_ideal_grid(C, dense[n], s, depth_of(n))
                np.save(f"{run}/depth/{n}.npy", Zi.astype(np.float16))
                new_used[n] = f"{run}/depth/{n}.npy"
            else:  # no depth: fall back to the median metric depth
                new_used[n] = med_Z
        used = new_used
        print(f"[it{it}] depth-aware correction ({len(dense)}/{len(names)} with dense depth)",
              flush=True)
        img_dir, mask_dir = correct_set(C, names, args.rgb_dir, run, depth_of)
        model_prev = cu.sfm(img_dir, f"{run}/colmap", calib, mask_dir)
        key = os.path.basename(run)
        summary[key] = cu.model_stats(model_prev, len(names))
        print("  ", summary[key], flush=True)
        json.dump(summary, open(summary_path, "w"), indent=1)

    json.dump(summary, open(summary_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
