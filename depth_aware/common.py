"""Shared helpers for the depth-aware refraction experiments (docs/explore_depth.md).

Coordinate frames used throughout:
  raw    pixel of the recorded (lens-distorted) underwater image
  ideal  raw pixel with the in-air lens distortion removed; this is the grid on which
         Refrax traces rays (core.undistort.compute_housing_geometry) and on which
         depth maps for paths.depth_dir are defined
  out    pixel of the corrected image Refrax writes (zoomed, on the fitted canvas,
         cropped to the reference valid rectangle)
Depth Z is along the optical axis from the camera centre, in housing units (metres).
"""
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from core.config import load_config, camera_params, distortion_params, housing_params  # noqa: E402
from core.optics import matrix_K, normalize_batch, refract_batch, intersect_plane_batch  # noqa: E402
from core.scale import select_zoom  # noqa: E402
from core.undistort import InverseMapper  # noqa: E402
import remove_refraction as rr  # noqa: E402

WORK = os.environ.get("REFRAX_WORK", "/Users/fischert/robot/refrax-work/depth")


def trace_rays(u, v, K_inv, n_port, d0, tglass, mu_a, mu_g, mu_w):
    """(P2, ray_water) for arbitrary ideal pixel coordinates (any shape)."""
    pix = np.stack([u, v, np.ones_like(u)], axis=-1).astype(np.float64)
    rays = normalize_batch(pix @ K_inv.T)
    P1 = intersect_plane_batch(np.zeros(3), rays, n_port * d0, n_port)
    rg = refract_batch(rays, n_port, mu_a, mu_g)
    P2 = intersect_plane_batch(P1, rg, n_port * (d0 + tglass), n_port)
    rw = refract_batch(rg, n_port, mu_g, mu_w)
    return P2, rw


class Corrector:
    """Refrax correction with an explicit output frame, for points and whole images.

    The output frame (zoom, canvas, crop, intrinsics) is fixed by the reference depth
    z0, exactly as remove_refraction.py does, so fixed-depth and depth-aware outputs
    share one calibration.
    """

    def __init__(self, cfg, z0=None, center_z=0.0, zoom=None):
        self.cfg = cfg
        self.W, self.H, self.fx, self.fy, self.cx, self.cy = camera_params(cfg)
        self.dist = distortion_params(cfg)
        self.housing = housing_params(cfg)
        corr = cfg["correction"]
        self.z0 = float(z0 if z0 is not None else corr["z0_fixed"])
        self.center_z = float(center_z)
        self.K, self.K_inv = matrix_K(self.fx, self.fy, self.cx, self.cy)
        W, H, fx, fy, cx, cy = self.W, self.H, self.fx, self.fy, self.cx, self.cy
        if zoom is None:
            zoom, _ = select_zoom(self.z0, W, H, fx, fy, cx, cy, self.housing,
                                  in_bounds=False, center_z=self.center_z)
        self.zoom = float(zoom)
        canvas = rr.fit_canvas(W, H, fx, fy, cx, cy, self.housing, self.dist, self.zoom,
                               self.z0, self.center_z)
        self.canvas = canvas
        self.build_map = rr.make_map_builder(W, H, fx, fy, cx, cy, self.housing, self.dist,
                                             self.zoom, canvas, self.center_z)
        ref = self.build_map(self.z0)
        self.ref_mask = rr.compute_valid_mask(*ref, W, H)
        self.box = rr.largest_valid_rectangle(self.ref_mask)
        y0, y1, x0, x1 = self.box
        ox, oy = canvas[2]
        self.out_W, self.out_H = x1 - x0, y1 - y0
        self.out_fx, self.out_fy = fx * self.zoom, fy * self.zoom
        self.out_cx, self.out_cy = cx - ox - x0, cy - oy - y0
        self.mapper = InverseMapper(W, H, fx, fy, cx, cy, self.housing, self.zoom, canvas,
                                    center_z=self.center_z)
        # Dense geometry on the ideal grid (float64 for accuracy).
        u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        self.P2, self.rw = trace_rays(u, v, self.K_inv, **self.housing)

    # ---------------------------------------------------------------- points
    def raw_to_ideal(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        if not any(self.dist):
            return pts.reshape(-1, 2)
        k1, k2, p1, p2 = self.dist
        und = cv2.undistortPoints(pts, self.K, np.array([k1, k2, p1, p2]), P=self.K,
                                  criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10))
        return und.reshape(-1, 2)

    def project_world(self, Pw):
        """Out-frame pixel of camera-frame 3D points Pw (N, 3)."""
        Zc = Pw[:, 2] - self.center_z
        x = self.out_fx * Pw[:, 0] / Zc + self.out_cx
        y = self.out_fy * Pw[:, 1] / Zc + self.out_cy
        return np.stack([x, y], 1)

    def world_points_at_depth(self, ideal, Z):
        P2, rw = trace_rays(ideal[:, 0], ideal[:, 1], self.K_inv, **self.housing)
        t = (np.asarray(Z, np.float64) - P2[:, 2]) / rw[:, 2]
        return P2 + t[:, None] * rw

    def world_points_on_plane(self, ideal, n, d):
        """Intersection of the water rays with the plane n . X = d (camera frame)."""
        P2, rw = trace_rays(ideal[:, 0], ideal[:, 1], self.K_inv, **self.housing)
        t = (d - P2 @ n) / (rw @ n)
        return P2 + t[:, None] * rw

    def correct_points(self, raw_pts, Z):
        """Out-frame position of raw pixels whose scene points lie at depth Z."""
        ideal = self.raw_to_ideal(raw_pts)
        return self.project_world(self.world_points_at_depth(ideal, Z))

    # ---------------------------------------------------------------- images
    def out_to_ideal_map(self, depth):
        """Dense out-frame -> ideal-pixel map for a depth map on the ideal grid (or scalar).

        Pixel-exact per-pixel inverse (core.undistort.InverseMapper); NaN where the output
        pixel has no source. (The first version of this branch interpolated a step-4
        subsampled forward map, so depth detail below 4 px was lost.)
        """
        mx, my = self.mapper.build(depth)
        y0, y1, x0, x1 = self.box
        return mx[y0:y1, x0:x1], my[y0:y1, x0:x1]

    def ideal_to_raw(self, ix, iy):
        if not any(self.dist):
            return ix, iy
        from core.optics import apply_radtan_distortion
        xd, yd = apply_radtan_distortion((ix - self.cx) / self.fx, (iy - self.cy) / self.fy,
                                         *self.dist)
        return xd * self.fx + self.cx, yd * self.fy + self.cy

    def correct_image(self, img, depth):
        """Corrected image (out frame) and validity mask for a depth map/scalar."""
        ix, iy = self.out_to_ideal_map(depth)
        rx, ry = self.ideal_to_raw(ix, iy)
        rx = np.nan_to_num(rx, nan=-1).astype(np.float32)
        ry = np.nan_to_num(ry, nan=-1).astype(np.float32)
        out = cv2.remap(img, rx, ry, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0)
        mask = rr.compute_valid_mask(rx, ry, self.W, self.H)
        return out, mask

    def ideal_to_out_grid(self, depth):
        """Out-frame position of every ideal pixel (forward map), for a depth map/scalar."""
        Z = np.broadcast_to(np.asarray(depth, np.float64), (self.H, self.W))
        t = (Z - self.P2[..., 2]) / self.rw[..., 2]
        Pw = self.P2 + t[..., None] * self.rw
        xy = self.project_world(Pw.reshape(-1, 3)).reshape(self.H, self.W, 2)
        return xy[..., 0], xy[..., 1]

    def write_calibration(self, path):
        rr.write_calibration(path, self.out_W, self.out_H, self.out_fx, self.out_fy,
                             self.out_cx, self.out_cy, self.zoom, self.center_z)


def load_cfg(path=None):
    return load_config(path or os.path.join(REPO, "configs", "dataset.yaml"))
