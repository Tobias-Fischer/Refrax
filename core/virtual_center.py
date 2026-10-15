"""Axial position of the corrected (virtual) pinhole that minimises depth dependence.

A flat port has no single viewpoint: the water-side rays do not meet in one point,
so any pinhole correction is exact at one depth only. How depth-dependent it is,
however, depends on where the corrected pinhole is placed. Refrax projects from the
camera centre (center_z = 0). Paraxial water rays, extended backwards, meet the axis
roughly d0 * (1 - mu_w / mu_a) behind the port (about -7 mm for d0 = 2 cm), and
oblique rays further back still; projecting from there leaves a much smaller
depth-dependent residual (cf. the Pinax model, Luczynski et al. 2017).

optimal_center_z returns the center_z that minimises the RMS displacement, over the
whole image, between the maps at depths spread over [z_min, z_max] and the map at the
reference depth z_ref.
"""
import numpy as np
from scipy.optimize import minimize_scalar

from core.optics import matrix_K
from core.undistort import compute_housing_geometry, forward_map


def depth_residual_rms(P2, ray_water, fx, fy, cx, cy, center_z, depths, z_ref, zoom=1.0):
    """RMS (px, scaled by zoom) of map(Z) - map(z_ref) over all pixels and the given depths."""
    ref_x, ref_y = forward_map(P2, ray_water, z_ref, fx, fy, cx, cy, center_z)
    sq = []
    for Z in depths:
        mx, my = forward_map(P2, ray_water, Z, fx, fy, cx, cy, center_z)
        sq.append(np.nanmean((mx - ref_x) ** 2 + (my - ref_y) ** 2))
    return float(zoom * np.sqrt(np.mean(sq)))


def optimal_center_z(W, H, fx, fy, cx, cy, housing, z_range=(0.5, 8.0), z_ref=None,
                     n_depths=9, subsample=8):
    """center_z (same unit as d0) minimising the depth dependence over z_range.

    Depths are sampled log-uniformly, since the depth-dependent part of the map
    scales with 1 / Z. Returns (center_z, rms_at_center_z, rms_at_zero), the RMS in
    px of the unzoomed map.
    """
    _, K_inv = matrix_K(fx, fy, cx, cy)
    P2, ray_water = compute_housing_geometry(H, W, K_inv, **housing)
    P2, ray_water = P2[::subsample, ::subsample], ray_water[::subsample, ::subsample]
    depths = np.geomspace(z_range[0], z_range[1], n_depths)
    if z_ref is None:
        z_ref = float(np.sqrt(z_range[0] * z_range[1]))
    d0 = housing["d0"]
    f = lambda c: depth_residual_rms(P2, ray_water, fx, fy, cx, cy, c, depths, z_ref)
    res = minimize_scalar(f, bounds=(-5 * d0, d0), method="bounded",
                          options={"xatol": 1e-5})
    return float(res.x), float(res.fun), f(0.0)
