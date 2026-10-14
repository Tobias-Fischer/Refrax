"""Frozen copy of the original (pre-optimisation) implementation, commit 3a1d7b5.

Used as the reference for regression tests and benchmarks; do not optimise.
"""
import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import minimize_scalar
from core.optics import matrix_K, apply_radtan_distortion, normalize_batch, refract_batch, intersect_plane_batch

def compute_housing_geometry(H, W, K_inv, n_port, d0, tglass, mu_a, mu_g, mu_w):
    """Depth-independent part of the trace: exit point P2 on the port and water ray per pixel."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pixels = np.stack([u, v, np.ones((H, W), dtype=np.float32)], axis=-1)
    rays   = normalize_batch(pixels @ K_inv.T)

    plane1    = n_port * d0
    P1        = intersect_plane_batch(np.zeros(3), rays, plane1, n_port)
    ray_glass = refract_batch(rays, n_port, mu_a, mu_g)

    plane2    = plane1 + n_port * tglass
    P2        = intersect_plane_batch(P1, ray_glass, plane2, n_port)
    ray_water = refract_batch(ray_glass, n_port, mu_g, mu_w)
    return P2, ray_water


def forward_map(P2, ray_water, depth, fx, fy, cx, cy):
    """Pinhole projection (fx, fy, cx, cy) of the point each underwater pixel sees at `depth`."""
    Z = np.asarray(depth, dtype=np.float32)
    if Z.ndim == 0:
        Z = np.full(P2.shape[:2], float(Z), dtype=np.float32)

    denom      = ray_water[..., 2]
    safe_denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    t          = (Z - P2[..., 2]) / safe_denom

    Pw_x = P2[..., 0] + t * ray_water[..., 0]
    Pw_y = P2[..., 1] + t * ray_water[..., 1]

    map_x = (fx * Pw_x / Z + cx).astype(np.float32)
    map_y = (fy * Pw_y / Z + cy).astype(np.float32)
    return map_x, map_y


def invert_map(map_x, map_y, H, W, step=4, origin=(0.0, 0.0)):
    """Inverts a forward map by linear interpolation over a subsampled grid (every `step` px).

    map_x, map_y give, per source pixel, its position in the corrected image plane. The
    inverse is evaluated on an H x W output canvas whose pixel (0, 0) sits at corrected-plane
    coordinate `origin`; (0, 0) with the source size is the source frame itself. The last
    source row and column are always sites, so the hull reaches the source border.
    Pixels outside the convex hull of the forward map are NaN.
    """
    src_h, src_w = map_x.shape
    u_s = np.unique(np.append(np.arange(0, src_w, step), src_w - 1))
    v_s = np.unique(np.append(np.arange(0, src_h, step), src_h - 1))
    ug_s, vg_s = np.meshgrid(u_s, v_s)

    src = np.stack([map_x[np.ix_(v_s, u_s)].ravel(),
                     map_y[np.ix_(v_s, u_s)].ravel()], axis=1)
    finite = np.all(np.isfinite(src), axis=1)
    src, ug_s, vg_s = src[finite], ug_s.ravel()[finite], vg_s.ravel()[finite]

    x0, y0 = origin
    ug_full, vg_full = np.meshgrid(np.arange(W, dtype=np.float32) + x0,
                                   np.arange(H, dtype=np.float32) + y0)

    undist_x = griddata(src, ug_s.astype(np.float32), (ug_full, vg_full), method='linear')
    undist_y = griddata(src, vg_s.astype(np.float32), (ug_full, vg_full), method='linear')
    return undist_x.astype(np.float32), undist_y.astype(np.float32)


def zoom_forward_map(map_x, map_y, cx, cy, zoom):
    """Scales a forward map about (cx, cy) by zoom (None: unchanged).

    Computed in float64 so the result does not depend on whether zoom is a Python
    or a NumPy float.
    """
    if zoom is None:
        return map_x, map_y
    return ((map_x.astype(np.float64) - cx) * float(zoom) + cx,
            (map_y.astype(np.float64) - cy) * float(zoom) + cy)


def forward_map_canvas(map_x, map_y):
    """Smallest canvas holding every finite point of a forward map, as (W, H, origin)."""
    finite = np.isfinite(map_x) & np.isfinite(map_y)
    if not np.any(finite):
        raise ValueError("Forward map has no finite entries; cannot size a canvas.")
    x_min, x_max = np.floor(map_x[finite].min()), np.ceil(map_x[finite].max())
    y_min, y_max = np.floor(map_y[finite].min()), np.ceil(map_y[finite].max())
    return int(x_max - x_min) + 1, int(y_max - y_min) + 1, (float(x_min), float(y_min))


def build_undistort_map_closed_form(P2, ray_water, depth, fx, fy, cx, cy, H, W, zoom=None,
                                    origin=(0.0, 0.0)):
    """Undistortion map for cv2.remap: corrected pixel -> underwater pixel.

    The map covers an H x W canvas with pixel (0, 0) at corrected-plane coordinate
    `origin` (see invert_map).
    """
    map_x, map_y = forward_map(P2, ray_water, depth, fx, fy, cx, cy)
    map_x, map_y = zoom_forward_map(map_x, map_y, cx, cy, zoom)
    return invert_map(map_x, map_y, H, W, origin=origin)

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

def make_map_builder(W, H, fx, fy, cx, cy, housing, distortion, zoom, canvas=None):
    """Returns build_map(depth) -> (map_x, map_y), float32 maps for cv2.remap.

    The maps cover canvas = (out_W, out_H, origin), the source frame by default.
    Pixels without a source are set to -1. If radtan distortion coefficients are
    non-zero, the map points into the distorted (raw) image.
    """
    _, K_inv = matrix_K(fx, fy, cx, cy)
    P2, ray_water = compute_housing_geometry(H, W, K_inv, **housing)
    apply_radtan = any(d != 0.0 for d in distortion)
    out_W, out_H, origin = canvas if canvas is not None else (W, H, (0.0, 0.0))

    def build_map(depth):
        map_x, map_y = build_undistort_map_closed_form(
            P2, ray_water, depth, fx, fy, cx, cy, out_H, out_W, zoom=zoom, origin=origin)

        if apply_radtan:
            x_dist, y_dist = apply_radtan_distortion(
                (map_x - cx) / fx, (map_y - cy) / fy, *distortion)
            map_x, map_y = x_dist * fx + cx, y_dist * fy + cy

        return (np.nan_to_num(map_x, nan=-1).astype(np.float32),
                np.nan_to_num(map_y, nan=-1).astype(np.float32))

    return build_map


def fit_canvas(W, H, fx, fy, cx, cy, housing, distortion, zoom, z0):
    """Canvas (out_W, out_H, origin) holding the whole image corrected at depth z0.

    Starts from the bounding box of the zoomed forward map, with one pixel of slack
    for the interpolation hull, then shrinks to the valid pixels of the map built on
    it. origin is the source-frame coordinate of the canvas' pixel (0, 0).
    """
    _, K_inv = matrix_K(fx, fy, cx, cy)
    P2, ray_water = compute_housing_geometry(H, W, K_inv, **housing)
    map_x, map_y = zoom_forward_map(*forward_map(P2, ray_water, z0, fx, fy, cx, cy),
                                    cx, cy, zoom)
    out_W, out_H, (x0, y0) = forward_map_canvas(map_x, map_y)
    out_W, out_H, x0, y0 = out_W + 2, out_H + 2, x0 - 1.0, y0 - 1.0
    if out_W > 8 * W or out_H > 8 * H:
        raise ValueError(f"fit_canvas: corrected image would be {out_W}x{out_H} for a "
                         f"{W}x{H} source; check the housing parameters and zoom ({zoom}).")

    build_map = make_map_builder(W, H, fx, fy, cx, cy, housing, distortion, zoom,
                                 canvas=(out_W, out_H, (x0, y0)))
    valid = compute_valid_mask(*build_map(z0), W, H) > 0
    rows, cols = np.flatnonzero(valid.any(axis=1)), np.flatnonzero(valid.any(axis=0))
    if len(rows) == 0:
        raise ValueError("fit_canvas: the corrected image has no valid pixel.")
    return (int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1),
            (x0 + float(cols[0]), y0 + float(rows[0])))


def compute_valid_mask(map_x, map_y, W, H):
    """255 where the map samples inside the (W, H) source image, 0 elsewhere."""
    valid = (
        (map_x >= 0) & (map_x <= W - 1) &
        (map_y >= 0) & (map_y <= H - 1)
    )
    return np.where(valid, 255, 0).astype(np.uint8)


def largest_valid_rectangle(mask):
    """Largest axis-aligned rectangle of non-zero mask pixels, as (y0, y1, x0, x1).

    Row-by-row largest-rectangle-in-histogram; falls back to the whole image.
    """
    H, W = mask.shape
    valid = mask > 0

    height = np.zeros(W, dtype=np.int32)
    best_area = 0
    best_box = (0, H, 0, W)

    for y in range(H):
        height = np.where(valid[y], height + 1, 0)

        stack = []  # (start_x, h)
        for x in range(W + 1):
            h = height[x] if x < W else 0
            start = x
            while stack and stack[-1][1] > h:
                s_x, s_h = stack.pop()
                area = s_h * (x - s_x)
                if area > best_area:
                    best_area = area
                    best_box = (y - s_h + 1, y + 1, s_x, x)
                start = s_x
            stack.append((start, h))

    return best_box


