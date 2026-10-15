"""Removes flat-port refraction from underwater images.

Usage:
    python remove_refraction.py configs/dataset.yaml [--overwrite]

Every image in paths.rgb_dir is warped to a refraction-free pinhole image with
focal length zoom * (fx, fy). The output is sized to hold the whole corrected
image (correction.fit_canvas, default), or has the source size. Outputs:
    <output_dir>/<name>.png  corrected image; cropped to the largest rectangle
                             without invalid pixels if correction.crop_valid_bbox,
                             otherwise BGRA with the validity mask as alpha
    <mask_dir>/mask.png      validity mask (<mask_dir>/<name>.png per image when
                             a depth map is given per image)
    <calib_path>             intrinsics of the corrected images (fixed-depth mode, or
                             depth mode with correction.depth_crop: reference)

Depth maps (paths.depth_dir) are indexed on the lens-undistorted ("ideal") source grid,
i.e. the grid the flat-port model is traced on, not on the recorded raw pixels when the
camera has radtan distortion (the two differ by up to tens of px in the corners).
"""
import argparse
import glob
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from core.config import load_config, camera_params, distortion_params, housing_params
from core.rectangle import largest_valid_rectangle
from core.scale import select_zoom
from core.virtual_center import optimal_center_z
from core.undistort import (InverseMapper, forward_map_analytic, zoom_forward_map,
                            forward_map_canvas)

IMAGE_EXTS = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG",
              "*.bmp", "*.BMP", "*.tiff", "*.TIFF"]


def find_rgb_paths(rgb_dir, step_size):
    paths = sorted(p for ext in IMAGE_EXTS for p in glob.glob(f"{rgb_dir}/{ext}"))
    return paths[::step_size]


def load_depth(name, depth_dir, W, H):
    """Per-image depth map <depth_dir>/<name>.npy resized to (W, H), or None if missing."""
    depth_path = f"{depth_dir}/{name}.npy"
    if not os.path.exists(depth_path):
        return None
    depth = np.load(depth_path).astype(np.float32)
    if depth.shape[:2] != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(depth, 1e-3, None)


def make_map_builder(W, H, fx, fy, cx, cy, housing, distortion, zoom, canvas=None,
                     center_z=0.0):
    """Returns build_map(depth) -> (map_x, map_y), float32 maps for cv2.remap.

    The maps cover canvas = (out_W, out_H, origin), the source frame by default.
    Pixels without a source are set to -1. If radtan distortion coefficients are
    non-zero, the map points into the distorted (raw) image. depth is a scalar or a
    per-source-pixel depth map (H x W).
    """
    mapper = InverseMapper(W, H, fx, fy, cx, cy, housing, zoom, canvas, center_z=center_z)

    def build_map(depth):
        return mapper.remap_maps(depth, distortion)

    return build_map


def _fit_canvas(W, H, fx, fy, cx, cy, housing, distortion, zoom, z0, center_z=0.0):
    """fit_canvas, also returning the remap maps at z0 cropped to the canvas."""
    map_x, map_y = zoom_forward_map(
        *forward_map_analytic(W, H, fx, fy, cx, cy, housing, z0, center_z), cx, cy, zoom)
    out_W, out_H, (x0, y0) = forward_map_canvas(map_x, map_y)
    out_W, out_H, x0, y0 = out_W + 2, out_H + 2, x0 - 1.0, y0 - 1.0
    if out_W > 8 * W or out_H > 8 * H:
        raise ValueError(f"fit_canvas: corrected image would be {out_W}x{out_H} for a "
                         f"{W}x{H} source; check the housing parameters and zoom ({zoom}).")

    build_map = make_map_builder(W, H, fx, fy, cx, cy, housing, distortion, zoom,
                                 canvas=(out_W, out_H, (x0, y0)), center_z=center_z)
    mx, my = build_map(z0)
    valid = compute_valid_mask(mx, my, W, H) > 0
    rows, cols = np.flatnonzero(valid.any(axis=1)), np.flatnonzero(valid.any(axis=0))
    if len(rows) == 0:
        raise ValueError("fit_canvas: the corrected image has no valid pixel.")
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    canvas = (int(c1 - c0), int(r1 - r0), (x0 + float(c0), y0 + float(r0)))
    # The maps of the cropped canvas are a crop of these (same grid, integer offset).
    maps = (np.ascontiguousarray(mx[r0:r1, c0:c1]), np.ascontiguousarray(my[r0:r1, c0:c1]))
    return canvas, maps


def fit_canvas(W, H, fx, fy, cx, cy, housing, distortion, zoom, z0, center_z=0.0):
    """Canvas (out_W, out_H, origin) holding the whole image corrected at depth z0.

    Starts from the bounding box of the zoomed forward map, with one pixel of slack
    for the interpolation hull, then shrinks to the valid pixels of the map built on
    it. origin is the source-frame coordinate of the canvas' pixel (0, 0).
    """
    return _fit_canvas(W, H, fx, fy, cx, cy, housing, distortion, zoom, z0, center_z)[0]


def compute_valid_mask(map_x, map_y, W, H):
    """255 where the map samples inside the (W, H) source image, 0 elsewhere."""
    valid = (
        (map_x >= 0) & (map_x <= W - 1) &
        (map_y >= 0) & (map_y <= H - 1)
    )
    return np.where(valid, 255, 0).astype(np.uint8)


def write_calibration(path, W, H, fx, fy, cx, cy, zoom, center_z=0.0):
    with open(path, "w") as f:
        f.write(f"focal_length: [{fx:.6f}, {fy:.6f}]\n")
        f.write(f"principal_point: [{cx:.6f}, {cy:.6f}]\n")
        f.write("distortion_coefficients: [0, 0, 0, 0]\n")
        f.write(f"image_dimension: [{W}, {H}]\n")
        f.write(f"zoom: {zoom:.6f}\n")
        # Optical-axis position of the corrected pinhole relative to the camera centre.
        f.write(f"virtual_center_z: {center_z:.6f}\n")
    print(f"Saved calibration: {path}")


def main():
    parser = argparse.ArgumentParser(description="Refraction removal from a YAML config")
    parser.add_argument("config", help="Path to config YAML file")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute images whose output already exists")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="Threads for image reading, warping and PNG writing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    W, H, fx, fy, cx, cy = camera_params(cfg)
    distortion = distortion_params(cfg)
    housing = housing_params(cfg)

    paths = cfg["paths"]
    rgb_dir, output_dir, mask_dir = paths["rgb_dir"], paths["output_dir"], paths["mask_dir"]
    depth_dir = paths.get("depth_dir")
    calib_path = paths.get("calib_path",
                           os.path.join(os.path.dirname(output_dir), "new_calibration.yaml"))

    corr = cfg["correction"]
    z0_fixed = corr["z0_fixed"]
    step_size = corr.get("step_size", 1)
    crop_valid_bbox = corr.get("crop_valid_bbox", False)
    fit = corr.get("fit_canvas", True)
    zoom = corr.get("zoom")
    depth_crop = corr.get("depth_crop", "reference")
    if depth_crop not in ("reference", "per_image"):
        raise ValueError(f"correction.depth_crop must be 'reference' or 'per_image', not {depth_crop!r}")
    center_z = corr.get("virtual_center", 0.0)
    if center_z == "auto":
        z_range = corr.get("depth_range", [0.5, 8.0])
        center_z, rms_c, rms_0 = optimal_center_z(W, H, fx, fy, cx, cy, housing, z_range=z_range)
        print(f"Virtual centre z = {center_z * 1000:.2f} mm (depth-dependence RMS over "
              f"{z_range}: {rms_c:.2f} px vs {rms_0:.2f} px at the camera centre)")
    center_z = float(center_z)

    if zoom is None:
        print(f"Finding optimal zoom at Z0={z0_fixed}...")
        zoom, zoom_rmse = select_zoom(z0_fixed, W, H, fx, fy, cx, cy, housing,
                                      in_bounds=not fit, center_z=center_z)
        print(f"Best scale{'' if fit else ' in bounds'} = {zoom:.4f}")
        print(f"RMSE = {zoom_rmse:.4f} px")
    else:
        print(f"Using zoom from config = {zoom:.4f}")

    canvas, fixed_map = (W, H, (0.0, 0.0)), None
    if fit:
        canvas, fit_maps = _fit_canvas(W, H, fx, fy, cx, cy, housing, distortion, zoom, z0_fixed,
                                       center_z)
        fixed_map = fit_maps
        print(f"Fitted canvas: {canvas[0]}x{canvas[1]}, origin=({canvas[2][0]:.0f}, "
              f"{canvas[2][1]:.0f}) in the source frame")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    build_map = make_map_builder(W, H, fx, fy, cx, cy, housing, distortion, zoom, canvas,
                                 center_z)

    rgb_paths = find_rgb_paths(rgb_dir, step_size)
    print(f"Found {len(rgb_paths)} images (step={step_size}), "
          f"fit_canvas={fit}, crop_valid_bbox={crop_valid_bbox}, "
          f"radtan={any(d != 0.0 for d in distortion)}")

    # The map at z0_fixed defines the output frame (crop and intrinsics). In depth mode
    # with depth_crop: reference, every image keeps this frame, so all corrected images
    # share one pinhole calibration (as COLMAP expects); pixels the depth map leaves
    # without a source are black and 0 in the per-image mask.
    reference_frame = depth_dir is None or depth_crop == "reference"
    fixed_box = None
    if depth_dir is None:
        print(f"Single depth mode (Z0={z0_fixed}) — computing undistortion map once")
    if reference_frame:
        if fixed_map is None:
            fixed_map = build_map(z0_fixed)
        fixed_mask = compute_valid_mask(*fixed_map, W, H)
    if depth_dir is None:
        cv2.imwrite(f"{mask_dir}/mask.png", fixed_mask)
        print(f"Saved shared mask: {mask_dir}/mask.png")

    if reference_frame:
        out_W, out_H, (ox, oy) = canvas
        out_fx, out_fy = fx * zoom, fy * zoom
        out_cx, out_cy = cx - ox, cy - oy
        if crop_valid_bbox:
            fixed_box = largest_valid_rectangle(fixed_mask)
            y0, y1, x0, x1 = fixed_box
            out_W, out_H = x1 - x0, y1 - y0
            out_cx, out_cy = out_cx - x0, out_cy - y0
            print(f"Inscribed valid rectangle: rows[{y0}:{y1}], cols[{x0}:{x1}] "
                  f"({out_W}x{out_H}, no invalid pixels)")
            print(f"Adjusted intrinsics for cropped output: "
                  f"fx={out_fx:.2f}, fy={out_fy:.2f}, cx={out_cx:.2f}, cy={out_cy:.2f}")

        write_calibration(calib_path, out_W, out_H, out_fx, out_fy, out_cx, out_cy, zoom,
                          center_z)

    def warp_and_save(job):
        """Reads, warps and writes one image (runs in a worker thread; cv2 releases the GIL)."""
        rgb_path, name, out_path, maps, mask, box, mask_path = job
        img = cv2.imread(rgb_path)
        if img is None:
            return f"Unreadable: {rgb_path} → skip"
        if img.shape[:2] != (H, W):
            return f"Size {img.shape[1]}x{img.shape[0]} != camera {W}x{H}: {name} → skip"
        if mask_path is not None:
            if box is not None and depth_crop == "reference":
                y0, y1, x0, x1 = box
                cv2.imwrite(mask_path, mask[y0:y1, x0:x1])
            else:
                cv2.imwrite(mask_path, mask)
        corrected = cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if box is not None:
            # Every pixel inside the inscribed rectangle is valid (at the reference depth
            # with depth_crop: reference; see the per-image mask), so no alpha is needed.
            y0, y1, x0, x1 = box
            out = corrected[y0:y1, x0:x1]
        else:
            out = cv2.cvtColor(corrected, cv2.COLOR_BGR2BGRA)
            out[..., 3] = mask
        cv2.imwrite(out_path, out)
        return f"Saved: {name}.png ({out.shape[1]}x{out.shape[0]}, {out.shape[2]} channels)"

    def jobs():
        """Per-image work items, in order. Depth-map mode builds each map here, in the main
        thread (the map kernel is itself parallel), while workers do the image I/O."""
        for rgb_path in rgb_paths:
            name = os.path.splitext(os.path.basename(rgb_path))[0]
            out_path = f"{output_dir}/{name}.png"
            if os.path.exists(out_path) and not args.overwrite:
                yield f"Exists: {name}.png → skip (use --overwrite to recompute)"
                continue
            if depth_dir is None:
                yield (rgb_path, name, out_path, fixed_map, fixed_mask, fixed_box, None)
                continue
            depth = load_depth(name, depth_dir, W, H)
            if depth is None:
                yield f"No depth: {name} → skip"
                continue
            maps = build_map(depth)
            mask = compute_valid_mask(*maps, W, H)
            if depth_crop == "reference":
                box = fixed_box
            else:
                box = largest_valid_rectangle(mask) if crop_valid_bbox else None
            yield (rgb_path, name, out_path, maps, mask, box, f"{mask_dir}/{name}.png")

    # Bounded pipeline: at most 2 * workers images in flight; messages printed in order.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = deque()
        for job in jobs():
            pending.append(job if isinstance(job, str) else pool.submit(warp_and_save, job))
            while pending and (isinstance(pending[0], str) or len(pending) > 2 * args.workers):
                item = pending.popleft()
                print(item if isinstance(item, str) else item.result())
        for item in pending:
            print(item if isinstance(item, str) else item.result())

    print("Done.")


if __name__ == "__main__":
    main()
