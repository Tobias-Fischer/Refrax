"""Selection of the zoom applied to the refraction-corrected image.

A flat port magnifies the scene, so the corrected image is rendered as a
pinhole camera with focal length zoom * (fx, fy). The zoom is the value in
the search range that minimises the RMSE between the refracted projection
and the zoomed pinhole projection. With in_bounds, it is additionally required
that every pixel of the underwater image lands inside the source-sized frame;
without it (correction.fit_canvas), the output canvas grows to hold the image.
"""
import numpy as np
from scipy.optimize import minimize_scalar

from core.optics import matrix_K
from core.undistort import compute_housing_geometry, forward_map

ZOOM_SEARCH = (1.3, 1.55, 80)  # (min, max, number of samples)


def sweep_zoom(map_x, map_y, cx, cy, s_values):
    """RMSE (px) between the forward map scaled about (cx, cy) by s and the identity."""
    H, W = map_x.shape
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))
    dx0, dy0 = map_x - cx, map_y - cy

    rmse_values = []
    for s in s_values:
        px = np.sqrt(((dx0 * s + cx) - u_grid) ** 2 + ((dy0 * s + cy) - v_grid) ** 2)
        rmse_values.append(np.sqrt(np.nanmean(px ** 2)))
    return np.array(rmse_values)


def find_in_bounds_scale(map_x, map_y, s_values, rmse_values, W, H, cx, cy):
    """Lowest-RMSE scale for which every valid (non-NaN) pixel maps inside the frame.

    Returns (None, inf) if no scale in s_values qualifies.
    """
    dx0, dy0 = map_x - cx, map_y - cy

    best_s, best_rmse = None, np.inf
    for s, rmse in zip(s_values, rmse_values):
        scaled_x = dx0 * s + cx
        scaled_y = dy0 * s + cy
        in_bounds = (
            (scaled_x >= 0) & (scaled_x <= W - 1) &
            (scaled_y >= 0) & (scaled_y <= H - 1)
        )
        valid = ~np.isnan(scaled_x) & ~np.isnan(scaled_y)

        if np.all(in_bounds[valid]) and rmse < best_rmse:
            best_s, best_rmse = s, rmse
    return best_s, best_rmse


def select_zoom(Z0, W, H, fx, fy, cx, cy, housing, search=ZOOM_SEARCH, in_bounds=True):
    """Zoom for a fronto-parallel plane at depth Z0. Returns (zoom, rmse_px).

    in_bounds=False returns the unconstrained lowest-RMSE zoom, found by a bounded
    scalar minimisation over the search range rather than on its sample grid.
    """
    _, K_inv = matrix_K(fx, fy, cx, cy)
    P2, ray_water = compute_housing_geometry(H, W, K_inv, **housing)
    map_x, map_y = forward_map(P2, ray_water, Z0, fx, fy, cx, cy)

    if not in_bounds:
        res = minimize_scalar(lambda s: sweep_zoom(map_x, map_y, cx, cy, [s])[0],
                              bounds=search[:2], method='bounded')
        return float(res.x), float(res.fun)

    s_values = np.linspace(*search)
    rmse_values = sweep_zoom(map_x, map_y, cx, cy, s_values)
    zoom, rmse = find_in_bounds_scale(map_x, map_y, s_values, rmse_values, W, H, cx, cy)
    if zoom is None:
        raise RuntimeError(
            f"No zoom in [{search[0]}, {search[1]}] keeps the image within {W}x{H}; "
            "widen the search range, set correction.zoom or correction.fit_canvas in the config."
        )
    return zoom, rmse
