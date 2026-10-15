"""Closed-form refraction correction: forward map + scattered-data inversion."""
import numpy as np
from core.optics import normalize_batch, refract_batch, intersect_plane_batch


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

    Approximate (~1e-3 px for step=4) and slow; remove_refraction uses the pixel-exact
    InverseMapper below. Kept for visualisation and as a reference.

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

    from scipy.interpolate import LinearNDInterpolator  # slow import; only needed here

    # One Delaunay triangulation for both coordinates (griddata would build it twice).
    interp = LinearNDInterpolator(src, np.stack([ug_s, vg_s], axis=1).astype(np.float32))
    undist = interp(ug_full, vg_full)
    undist_x, undist_y = undist[..., 0], undist[..., 1]
    return undist_x.astype(np.float32), undist_y.astype(np.float32)


_FWD_CACHE = {}


def forward_map_analytic(W, H, fx, fy, cx, cy, housing, depth):
    """forward_map on the full source grid from the analytic model (core.flatport), float64.

    Equal to forward_map(*compute_housing_geometry(...), depth, ...) up to rounding
    (forward_map additionally rounds to float32), without the per-pixel 3-D trace. depth is
    a scalar. The last result is memoised (read-only arrays).
    """
    from core.flatport import FlatPort
    key = (W, H, fx, fy, cx, cy, float(depth),
           tuple((k, tuple(np.ravel(v))) for k, v in sorted(housing.items())))
    if key in _FWD_CACHE:  # select_zoom and fit_canvas both need the map at z0
        return _FWD_CACHE[key]
    x = (np.arange(W, dtype=np.float64) - cx) / fx
    y = (np.arange(H, dtype=np.float64) - cy) / fy
    X, Y = np.meshgrid(x, y)
    Fx, Fy = FlatPort(**housing).forward(X, Y, 1.0 / float(depth))
    out = fx * Fx + cx, fy * Fy + cy
    for a in out:
        a.flags.writeable = False
    _FWD_CACHE.clear()
    _FWD_CACHE[key] = out
    return out


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



def _numba_kernels():
    try:
        from core import _kernels
        return _kernels
    except ImportError:  # numba not installed: NumPy fallback
        return None


class InverseMapper:
    """Pixel-exact undistortion maps (corrected pixel -> underwater pixel) for one camera,
    housing, zoom and output canvas, solved per output pixel (see core.flatport).

    Replaces build_undistort_map_closed_form + invert_map (scattered-data interpolation of
    the forward map). Depth is a scalar or a per-source-pixel map (H x W), sampled
    bilinearly. Depth-independent set-up is done once in the constructor.

    backend: "numba" (fused parallel kernel), "numpy", or "auto" (numba if installed).
    """

    def __init__(self, W, H, fx, fy, cx, cy, housing, zoom=None, canvas=None, backend="auto"):
        from core.flatport import FlatPort
        self.W, self.H, self.fx, self.fy, self.cx, self.cy = W, H, fx, fy, cx, cy
        self.model = FlatPort(**housing)
        self.zoom = 1.0 if zoom is None else float(zoom)
        out_W, out_H, (ox, oy) = canvas if canvas is not None else (W, H, (0.0, 0.0))
        self.out_W, self.out_H = out_W, out_H
        # Normalised target coordinates of the output columns / rows (zoom undone).
        self.Xs = (np.arange(out_W, dtype=np.float64) + ox - cx) / (fx * self.zoom)
        self.Ys = (np.arange(out_H, dtype=np.float64) + oy - cy) / (fy * self.zoom)
        self.frame = (-cx / fx, (W - 1 - cx) / fx, -cy / fy, (H - 1 - cy) / fy)
        self.kernels = _numba_kernels() if backend in ("auto", "numba") else None
        if backend == "numba" and self.kernels is None:
            raise ImportError("backend='numba' requested but numba is not installed")
        self._init = None  # constant-depth solution (NumPy backend initial guess)

    # -- NumPy backend ---------------------------------------------------------------------
    def _solve_numpy(self, depth):
        from core.flatport import InvDepth, invert_radial, invert_general
        X, Y = np.broadcast_arrays(self.Xs[None, :], self.Ys[:, None])
        invd = InvDepth(depth, self.fx, self.fy, self.cx, self.cy)
        if invd.const:
            if self.model.radial:
                return invert_radial(self.model, X, Y, invd, self.frame)
            return invert_general(self.model, X, Y, invd)
        if self._init is None:
            self._init = self._solve_numpy(float(np.median(1.0 / invd.w)))
        x0, y0 = self._init
        if self.model.radial:
            return invert_radial(self.model, X, Y, invd, self.frame, rho0=np.hypot(x0, y0))
        return invert_general(self.model, X, Y, invd, x0=x0, y0=y0)

    def _numpy_maps(self, depth):
        x, y = self._solve_numpy(depth)
        map_x = x * self.fx + self.cx
        map_y = y * self.fy + self.cy
        # Solutions on the frame border may land a rounding error outside it.
        tol = 1e-6
        inside = ((map_x >= -tol) & (map_x <= self.W - 1 + tol) &
                  (map_y >= -tol) & (map_y <= self.H - 1 + tol))
        return (np.where(inside, np.clip(map_x, 0, self.W - 1), np.nan),
                np.where(inside, np.clip(map_y, 0, self.H - 1), np.nan))

    # -- numba backend ---------------------------------------------------------------------
    def _numba_maps(self, depth, distortion, dtype, invalid):
        from core.flatport import TOL_PX, MAX_ITER_1D, MAX_ITER_2D, _radial_lut
        K = self.kernels
        m = self.model
        depth = np.asarray(depth)
        use_map = depth.ndim == 2
        if use_map:
            wmap = np.ascontiguousarray(1.0 / depth.astype(np.float64))
            w_ref = 1.0 / float(np.median(depth))
            w_const = 0.0
        else:
            wmap = np.zeros((2, 2))
            w_ref = w_const = 1.0 / float(depth)
        P = np.array([m.rg, m.rw, m.cg, m.cw, m.d0, m.tglass, m.dz])
        cam = np.array([self.fx, self.fy, self.cx, self.cy], dtype=np.float64)
        frame = np.array(self.frame)
        frame_px = np.array([self.W - 1.0, self.H - 1.0])
        dist = np.zeros(5)
        if distortion is not None and any(d != 0.0 for d in distortion):
            dist[:4] = distortion
            dist[4] = 1.0
        tol = TOL_PX / max(self.fx, self.fy)
        out_x = np.empty((self.out_H, self.out_W), dtype=dtype)
        out_y = np.empty((self.out_H, self.out_W), dtype=dtype)
        if m.radial:
            lut_R, lut_rho = _radial_lut(m, w_ref, self._rho_max())
            K.radial_maps(self.Xs, self.Ys, P, cam, frame, frame_px, w_const, wmap, use_map,
                          lut_R, lut_rho, tol, MAX_ITER_1D, dist, invalid, out_x, out_y)
        else:
            Ax, Ay, Bx, By = m.AB(np.zeros(1), np.zeros(1))
            g0, _, _ = m.radial_gain(0.0, w_ref)
            K.general_maps(self.Xs, self.Ys, m.R, P, cam, frame_px, w_const, wmap, use_map,
                           float(Ax[0] + Bx[0] * w_ref), float(Ay[0] + By[0] * w_ref),
                           float(g0), tol, MAX_ITER_2D, dist, invalid, out_x, out_y)
        return out_x, out_y

    def _rho_max(self):
        x_lo, x_hi, y_lo, y_hi = self.frame
        return 1.05 * np.hypot(max(-x_lo, x_hi), max(-y_lo, y_hi)) + 4.0 / min(self.fx, self.fy)

    # -- public API -----------------------------------------------------------------------
    def build(self, depth):
        """(map_x, map_y), float64 source pixel coordinates (no lens distortion), NaN where
        the output pixel has no source."""
        if self.kernels is not None:
            return self._numba_maps(depth, None, np.float64, np.nan)
        return self._numpy_maps(depth)

    def remap_maps(self, depth, distortion=None):
        """float32 maps for cv2.remap into the raw (radtan-distorted) image; -1 = invalid."""
        if self.kernels is not None:
            return self._numba_maps(depth, distortion, np.float32, -1.0)
        map_x, map_y = self._numpy_maps(depth)
        if distortion is not None and any(d != 0.0 for d in distortion):
            from core.optics import apply_radtan_distortion
            x_d, y_d = apply_radtan_distortion((map_x - self.cx) / self.fx,
                                               (map_y - self.cy) / self.fy, *distortion)
            map_x, map_y = x_d * self.fx + self.cx, y_d * self.fy + self.cy
        return (np.nan_to_num(map_x, nan=-1).astype(np.float32),
                np.nan_to_num(map_y, nan=-1).astype(np.float32))
