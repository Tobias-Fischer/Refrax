"""Selection of the zoom applied to the refraction-corrected image.

A flat port magnifies the scene, so the corrected image is rendered as a
pinhole camera with focal length zoom * (fx, fy). The zoom is the value in
the search range that minimises the RMSE between the refracted projection
and the zoomed pinhole projection. With in_bounds, it is additionally required
that every pixel of the underwater image lands inside the source-sized frame;
without it (correction.fit_canvas), the output canvas grows to hold the image.
"""
import numpy as np

from core.optics import matrix_K
from core.undistort import forward_map_analytic

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


class _ZoomStats:
    """RMSE(s) of sweep_zoom in closed form: with d = map - c and e = grid - c (finite map
    entries only), mean |s d - e|^2 = s^2 <d.d> - 2 s <d.e> + <e.e>, a quadratic in s, so the
    sweep and the scalar minimisation need three sums instead of a pass over the image per s.
    """

    def __init__(self, map_x, map_y, cx, cy):
        H, W = map_x.shape
        # Same rounding as sweep_zoom on the float32 map: d = float32(map) - c in float32.
        dx = (map_x.astype(np.float32) - np.float32(cx)).astype(np.float64)
        dy = (map_y.astype(np.float32) - np.float32(cy)).astype(np.float64)
        ex = np.arange(W, dtype=np.float64)[None, :] - cx
        ey = np.arange(H, dtype=np.float64)[:, None] - cy
        ok = np.isfinite(dx) & np.isfinite(dy)
        dx, dy = np.where(ok, dx, 0.0), np.where(ok, dy, 0.0)
        n = ok.sum()
        self.dd = ((dx * dx).sum() + (dy * dy).sum()) / n
        self.de = ((dx * ex).sum() + (dy * ey).sum()) / n
        self.ee = (np.where(ok, ex * ex, 0).sum() + np.where(ok, ey * ey, 0).sum()) / n
        # Extremes of d over finite entries decide the in-bounds test for any s > 0.
        self.ext = (dx[ok].min(), dx[ok].max(), dy[ok].min(), dy[ok].max())

    def rmse(self, s):
        s = np.asarray(s, dtype=np.float64)
        return np.sqrt(np.maximum(s * s * self.dd - 2 * s * self.de + self.ee, 0.0))

    def in_bounds(self, s, W, H, cx, cy):
        dx_min, dx_max, dy_min, dy_max = self.ext
        return (dx_min * s + cx >= 0) and (dx_max * s + cx <= W - 1) and \
               (dy_min * s + cy >= 0) and (dy_max * s + cy <= H - 1)


def select_zoom(Z0, W, H, fx, fy, cx, cy, housing, search=ZOOM_SEARCH, in_bounds=True,
                center_z=0.0):
    """Zoom for a fronto-parallel plane at depth Z0. Returns (zoom, rmse_px).

    in_bounds=False returns the unconstrained lowest-RMSE zoom over the search range
    rather than on its sample grid. Same result as the per-pixel sweep (sweep_zoom / find_in_bounds_scale) from closed-form
    sums over the analytic forward map.
    """
    map_x, map_y = forward_map_analytic(W, H, fx, fy, cx, cy, housing, Z0, center_z)
    stats = _ZoomStats(map_x, map_y, cx, cy)

    if not in_bounds:
        # RMSE(s)^2 is a convex quadratic: its bounded minimiser is the clipped vertex
        # (what scipy's bounded minimize_scalar converges to, to ~1e-11 here).
        zoom = float(np.clip(stats.de / stats.dd, search[0], search[1]))
        return zoom, float(stats.rmse(zoom))

    s_values = np.linspace(*search)
    rmse_values = stats.rmse(s_values)
    zoom, rmse = None, np.inf
    for s, r in zip(s_values, rmse_values):
        if stats.in_bounds(s, W, H, cx, cy) and r < rmse:
            zoom, rmse = s, r
    if zoom is None:
        raise RuntimeError(
            f"No zoom in [{search[0]}, {search[1]}] keeps the image within {W}x{H}; "
            "widen the search range, set correction.zoom or correction.fit_canvas in the config."
        )
    return zoom, float(rmse)
