"""Flat-port refraction *rendering* from a pinhole RGB-D image (inverse of Refrax' removal).

A simulator renders an ordinary pinhole RGB + depth image; `FlatPortRenderer` turns it into
the image a real camera behind a flat port would record. For every output (underwater)
pixel the ray is traced through the housing (core.optics conventions: camera frame x right,
y down, z forward, pixel centres at integer coordinates), leaving the port at P2 along the
water ray d. The water ray is then intersected with the depth map.

Because the water rays do not share a single centre of projection (they leave the port at
different points P2 and their back-extensions form a caustic), a water ray projects into
the pinhole render as a short line segment, not a point. With w = 1/Z' the inverse depth in
the render camera (centre c, same orientation as the camera), a point on the water ray
projects to

    x'/z' = a + b * w,   a = d_xy / d_z,   b = (P2 - c)_xy - (P2 - c)_z * a,

i.e. the projection moves *linearly* in w by fx_r * |b| px per unit inverse depth. The
intersection is found by screen-space ray marching in w (steps <= step_px render pixels,
from 1/z_near down to w = 0, the vanishing point of the ray), followed by Illinois
regula-falsi refinement on the bilinearly interpolated inverse depth (exact for planes,
because 1/Z is affine in screen coordinates on a plane). The first crossing is the visible
surface, so occlusions are handled like a z-buffer of infinitely thick surfaces; pixels the
render camera cannot see (a few-px disocclusion strip at depth edges, from the mm-scale
baseline) are flagged in `near_edge`.

Placing the render camera at the *virtual centre* (the point on the port axis closest, in
the least-squares sense, to all water rays; about d0 * (1 - mu_w) behind the camera centre)
shrinks |b| to a fraction of a millimetre, so parallax is small and few march steps are
needed. `plan_render_camera` returns the pinhole camera (pose offset, focal length,
resolution) a simulator has to render.
"""
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from core.optics import normalize_batch, refract_batch, intersect_plane_batch


# --------------------------------------------------------------------------------------
# Housing geometry for arbitrary (possibly distorted, sub-pixel) output pixels
# --------------------------------------------------------------------------------------

def pixel_rays(u, v, K, distortion=None):
    """Unit air rays of pixels (u, v). With radtan `distortion` (k1, k2, p1, p2) the pixels
    are raw (distorted) image coordinates and are undistorted first."""
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y = (u - cx) / fx, (v - cy) / fy
    if distortion is not None and any(c != 0.0 for c in distortion):
        x, y = undistort_radtan(x, y, *distortion)
    return normalize_batch(np.stack([x, y, np.ones_like(x)], axis=-1))


def undistort_radtan(xd, yd, k1, k2, p1, p2, iters=40):
    """Inverse of core.optics.apply_radtan_distortion (fixed-point iteration, as in
    OpenCV's undistortPoints); exact to ~1e-12 for the mild distortion of linear modes."""
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = (xd - dx) / radial, (yd - dy) / radial
    return x, y


def housing_rays(rays_air, n_port, d0, tglass, mu_a, mu_g, mu_w):
    """Port exit point P2 and unit water ray for unit air rays (same steps as
    core.undistort.compute_housing_geometry)."""
    n_port = np.asarray(n_port, dtype=np.float64)
    plane1 = n_port * d0
    P1 = intersect_plane_batch(np.zeros(3), rays_air, plane1, n_port)
    ray_glass = refract_batch(rays_air, n_port, mu_a, mu_g)
    P2 = intersect_plane_batch(P1, ray_glass, plane1 + n_port * tglass, n_port)
    ray_water = refract_batch(ray_glass, n_port, mu_g, mu_w)
    return P2, ray_water


def output_geometry(W, H, K, housing, distortion=None):
    """P2 and water ray for every pixel of a W x H output image."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    return housing_rays(pixel_rays(u, v, K, distortion), **housing)


def project_underwater(X, K, housing, distortion=None, uv0=None, iters=12, tol=1e-6):
    """Refractive projection: underwater pixel (u, v) whose water ray passes through X.

    Newton iteration on the lateral miss distance of the ray in the plane through X
    perpendicular to n_port (finite-difference Jacobian). Returns (u, v), each X.shape[:-1];
    NaN where it did not converge.
    """
    X = np.asarray(X, dtype=np.float64)
    n = np.asarray(housing["n_port"], dtype=np.float64)
    e1 = normalize_batch(np.cross(n, [0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.cross(n, [1.0, 0, 0]))
    e2 = np.cross(n, e1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mu_ratio = housing["mu_w"] / housing["mu_a"]

    if uv0 is None:  # paraxial guess: central projection with Snell on the viewing angle
        r = normalize_batch(X)
        sin_w = np.linalg.norm(np.cross(r, n), axis=-1)
        sin_a = np.clip(sin_w * mu_ratio, 0, 0.999)
        scale = np.where(sin_w > 1e-12, np.tan(np.arcsin(sin_a)) / np.maximum(sin_w / np.sqrt(1 - sin_w**2), 1e-12), 1.0)
        u = fx * scale * X[..., 0] / X[..., 2] + cx
        v = fy * scale * X[..., 1] / X[..., 2] + cy
    else:
        u, v = (np.asarray(c, dtype=np.float64).copy() for c in uv0)

    def residual(u, v):
        P2, d = housing_rays(pixel_rays(u, v, K, distortion), **housing)
        t = np.einsum('...i,i->...', X - P2, n) / np.einsum('...i,i->...', d, n)
        miss = P2 + t[..., None] * d - X
        return np.stack([miss @ e1, miss @ e2], axis=-1)

    h = 1e-3
    for _ in range(iters):
        r0 = residual(u, v)
        J = np.stack([(residual(u + h, v) - r0) / h, (residual(u, v + h) - r0) / h], axis=-1)
        step = np.linalg.solve(J, -r0[..., None])[..., 0]
        u, v = u + step[..., 0], v + step[..., 1]
        if np.nanmax(np.abs(step)) < tol:
            break
    ok = np.all(np.abs(residual(u, v)) < 1e-9 + 1e-6 * np.abs(X[..., 2:3]), axis=-1)
    return np.where(ok, u, np.nan), np.where(ok, v, np.nan)


# --------------------------------------------------------------------------------------
# The pinhole camera the simulator renders
# --------------------------------------------------------------------------------------

@dataclass
class PinholeSpec:
    """Pinhole RGB-D camera to render in the simulator.

    Same orientation as the real camera; optical centre at `centre` (metres, real-camera
    frame: x right, y down, z forward). Pixel centres at integer coordinates (OpenCV);
    simulators with pixel centres at +0.5 need no change as long as cx = (W - 1) / 2.
    """
    W: int
    H: int
    fx: float
    fy: float
    cx: float
    cy: float
    centre: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def hfov_deg(self):
        return float(np.degrees(2 * np.arctan(self.W / (2 * self.fx))))

    @property
    def vfov_deg(self):
        return float(np.degrees(2 * np.arctan(self.H / (2 * self.fy))))

    def to_dict(self):
        d = asdict(self)
        d = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}
        d["centre"] = [float(c) for c in self.centre]
        d.update(hfov_deg=self.hfov_deg, vfov_deg=self.vfov_deg)
        return d


def virtual_centre(P2, ray_water, n_port):
    """Point s * n_port on the port axis minimising the mean squared lateral offset of the
    water rays (back-extended) in the plane through it; returns the 3-vector."""
    n = np.asarray(n_port, dtype=np.float64)
    P, d = P2.reshape(-1, 3), ray_water.reshape(-1, 3)
    ok = np.all(np.isfinite(d), axis=1)
    P, d = P[ok], d[ok]
    dn, Pn = d @ n, P @ n
    A = P - Pn[:, None] * d / dn[:, None]      # offset at the plane n.x = 0
    B = d / dn[:, None] - n                    # derivative w.r.t. s (lateral part)
    s = -np.sum(A * B) / np.sum(B * B)
    return s * n


def plan_render_camera(W, H, K, housing, distortion=None, z_near=0.2, oversample=1.0,
                       centre="virtual", square=True, centred=True, margin_px=2, grid_step=8):
    """Pinhole camera (`PinholeSpec`) that covers every water ray of the W x H output camera
    for all depths >= z_near (measured from the camera centre along z).

    Focal length: oversample * max(fx, fy) * mu_w / mu_a, i.e. the paraxial magnification of
    the flat port, so the render has (at least) the sampling density of the output image at
    its centre. centre: "virtual" (recommended; minimises parallax), "origin" (the real
    camera centre) or a 3-vector.
    """
    u = np.unique(np.r_[np.arange(0, W, grid_step), W - 1]).astype(np.float64)
    v = np.unique(np.r_[np.arange(0, H, grid_step), H - 1]).astype(np.float64)
    ug, vg = np.meshgrid(u, v)
    P2, d = housing_rays(pixel_rays(ug, vg, K, distortion), **housing)

    if isinstance(centre, str):
        c = virtual_centre(P2, d, housing["n_port"]) if centre == "virtual" else np.zeros(3)
    else:
        c = np.asarray(centre, dtype=np.float64)

    f = float(oversample * max(K[0, 0], K[1, 1]) * housing["mu_w"] / housing["mu_a"])
    fx = fy = f
    if not square:
        fx, fy = (float(oversample * K[i, i] * housing["mu_w"] / housing["mu_a"]) for i in (0, 1))

    a, b = _ray_coefficients(P2 - c, d)
    w_near = 1.0 / (z_near - c[2])
    xs = np.concatenate([a[..., 0].ravel(), (a[..., 0] + b[..., 0] * w_near).ravel()])
    ys = np.concatenate([a[..., 1].ravel(), (a[..., 1] + b[..., 1] * w_near).ravel()])
    xs, ys = fx * xs[np.isfinite(xs)], fy * ys[np.isfinite(ys)]
    if centred:
        hx = np.ceil(np.abs(xs).max() + margin_px)
        hy = np.ceil(np.abs(ys).max() + margin_px)
        Wr, Hr = int(2 * hx + 1), int(2 * hy + 1)
        return PinholeSpec(Wr, Hr, fx, fy, (Wr - 1) / 2, (Hr - 1) / 2, c)
    x0, x1 = np.floor(xs.min() - margin_px), np.ceil(xs.max() + margin_px)
    y0, y1 = np.floor(ys.min() - margin_px), np.ceil(ys.max() + margin_px)
    return PinholeSpec(int(x1 - x0 + 1), int(y1 - y0 + 1), fx, fy, -x0, -y0, c)


def _ray_coefficients(P2c, d):
    """a, b with x'/z' = a + b * w along the water ray (P2c: exit point in render frame)."""
    a = d[..., :2] / d[..., 2:3]
    b = P2c[..., :2] - P2c[..., 2:3] * a
    return a, b


# --------------------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------------------

def _sample(img, mu, mv, exact):
    """Bilinear sample of img (H, W[, C]) at float pixel coords (border: clamp)."""
    if not exact:
        mu32, mv32 = mu.astype(np.float32), mv.astype(np.float32)
        if img.ndim == 3 and img.shape[2] > 4:
            return np.stack([cv2.remap(img[..., i], mu32, mv32, cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
                             for i in range(img.shape[2])], axis=-1)
        return cv2.remap(img, mu32, mv32, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    H, W = img.shape[:2]
    x = np.clip(mu, 0, W - 1)
    y = np.clip(mv, 0, H - 1)
    x0 = np.minimum(np.floor(x).astype(np.int64), W - 2)
    y0 = np.minimum(np.floor(y).astype(np.int64), H - 2)
    fx_, fy_ = x - x0, y - y0
    if img.ndim == 3:
        fx_, fy_ = fx_[..., None], fy_[..., None]
    i00, i01 = img[y0, x0], img[y0, x0 + 1]
    i10, i11 = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return ((1 - fy_) * ((1 - fx_) * i00 + fx_ * i01) + fy_ * ((1 - fx_) * i10 + fx_ * i11))


def _sample_flat(img, mu, mv, cols=4096):
    """cv2 bilinear sampling of a single-channel image at 1-D coordinate arrays (remap needs
    2-D maps with sides < 32767, so the arrays are padded and folded into rows)."""
    n = mu.size
    rows = -(-n // cols)
    pad = rows * cols - n
    fold = lambda x: np.pad(x.astype(np.float32), (0, pad)).reshape(rows, cols)
    return _sample(img, fold(mu), fold(mv), False).ravel()[:n]


def depth_to_z(depth, spec, depth_type="z"):
    """Converts a render depth map to z-depth (distance to the image plane).

    depth_type: "z" (Isaac Sim distance_to_image_plane, most sims), or "range" (Euclidean
    distance to the optical centre, e.g. Isaac Sim distance_to_camera).
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth_type == "z":
        return depth
    if depth_type != "range":
        raise ValueError(depth_type)
    u, v = np.meshgrid(np.arange(spec.W, dtype=np.float64), np.arange(spec.H, dtype=np.float64))
    norm = np.sqrt(((u - spec.cx) / spec.fx) ** 2 + ((v - spec.cy) / spec.fy) ** 2 + 1)
    return depth / norm


def depth_edges(inv_depth, rel=0.03):
    """Pixels next to an inverse-depth discontinuity (relative jump > rel), dilated by 1 px."""
    w = inv_depth
    m = np.zeros(w.shape, bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = w[max(dy, 0):w.shape[0], max(dx, 0):w.shape[1] + min(dx, 0)]
        b = w[:w.shape[0] - dy, max(-dx, 0):w.shape[1] - max(dx, 0)]
        jump = np.abs(a - b) > rel * np.maximum(np.maximum(a, b), 1e-12)
        m[max(dy, 0):, max(dx, 0):w.shape[1] + min(dx, 0)] |= jump
        m[:w.shape[0] - dy, max(-dx, 0):w.shape[1] - max(dx, 0)] |= jump
    return cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


# --------------------------------------------------------------------------------------
# The renderer
# --------------------------------------------------------------------------------------

class FlatPortRenderer:
    """Turns pinhole RGB-D renders into flat-port underwater images.

    Parameters mirror the Refrax config: output camera (W, H, fx, fy, cx, cy), optional
    radtan distortion (output then is the raw, distorted image) and housing dict
    (core.config.housing_params). The pinhole render camera is planned with
    `plan_render_camera` unless `spec` is given.
    """

    def __init__(self, W, H, fx, fy, cx, cy, housing, distortion=None, spec=None,
                 z_near=0.2, oversample=1.0, centre="virtual", step_px=0.5, refine_iters=16):
        self.W, self.H = W, H
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        self.housing, self.distortion = housing, distortion
        self.z_near, self.step_px, self.refine_iters = z_near, step_px, refine_iters
        self.spec = spec or plan_render_camera(W, H, self.K, housing, distortion, z_near,
                                               oversample, centre)
        self.P2, self.d = output_geometry(W, H, self.K, housing, distortion)
        self.valid_ray = np.all(np.isfinite(self.d), axis=-1) & (self.d[..., 2] > 1e-9)
        c = self.spec.centre
        d = np.where(self.valid_ray[..., None], self.d, [0.0, 0.0, 1.0])
        self.a, self.b = _ray_coefficients(self.P2 - c, d)
        self.w_near = 1.0 / (z_near - c[2])
        L = self.spec.fx * np.abs(self.b).max(axis=-1) * self.w_near
        self._line_cache = {}
        self.n_steps = int(np.clip(np.ceil(L[self.valid_ray].max() / step_px), 1, 4096))

    @classmethod
    def from_config(cls, cfg, **kw):
        from core.config import camera_params, distortion_params, housing_params
        W, H, fx, fy, cx, cy = camera_params(cfg)
        dist = distortion_params(cfg)
        return cls(W, H, fx, fy, cx, cy, housing_params(cfg),
                   distortion=dist if any(dist) else None, **kw)

    def _lines(self, ftype):
        """Cached flattened ray lines in the render image: u = U0 + DU * w, v = V0 + DV * w."""
        key = np.dtype(ftype).name
        if key not in self._line_cache:
            s = self.spec
            idx = np.flatnonzero(self.valid_ray)
            arrs = [(s.fx * self.a[..., 0] + s.cx).ravel()[idx],
                    (s.fy * self.a[..., 1] + s.cy).ravel()[idx],
                    (s.fx * self.b[..., 0]).ravel()[idx], (s.fy * self.b[..., 1]).ravel()[idx]]
            self._line_cache[key] = (idx, *(x.astype(ftype) for x in arrs))
        return self._line_cache[key]

    def _uv(self, w):
        s = self.spec
        return (s.fx * (self.a[..., 0] + self.b[..., 0] * w) + s.cx,
                s.fy * (self.a[..., 1] + self.b[..., 1] * w) + s.cy)

    def trace(self, depth, depth_type="z", exact=False):
        """Per output pixel: render-image coordinates (map_u, map_v) of the visible surface
        point and auxiliary outputs. depth: (spec.H, spec.W) render depth in metres;
        <= 0, NaN or inf = no geometry (background at infinity).

        Works on flattened arrays of the still-active pixels; cv2.remap accepts (N, 1) maps.
        """
        z = depth_to_z(depth, self.spec, depth_type)
        inv = np.where(np.isfinite(z) & (z > 0), 1.0 / np.where(z > 0, z, 1.0), 0.0)
        inv = inv.astype(np.float64 if exact else np.float32)
        s = self.spec
        ftype = np.float64 if exact else np.float32
        # u(w) = u0 + du * w along each water ray (flattened, valid rays only)
        idx, U0, V0, DU, DV = self._lines(ftype)

        def g(sel, w):
            mu, mv = U0[sel] + DU[sel] * w, V0[sel] + DV[sel] * w
            if exact:
                return w - _sample(inv, mu, mv, True)
            return w - _sample_flat(inv, mu, mv)

        # 1. March from w_near to 0 in n_steps (<= step_px render px per step). Rays that
        #    reach w = 0 always stop there (g = -inv <= 0): background at infinity if inv == 0.
        N = idx.size
        w_hi = np.full(N, self.w_near, ftype)
        g_hi = g(slice(None), w_hi)
        too_near = g_hi <= 0
        w_lo, g_lo = w_hi.copy(), g_hi.copy()
        act = np.flatnonzero(~too_near)
        for k in range(1, self.n_steps + 1):
            w_k = ftype(self.w_near * (1.0 - k / self.n_steps))
            g_k = g(act, w_k)
            hit = g_k <= 0
            h, m = act[hit], act[~hit]
            w_lo[h], g_lo[h] = w_k, g_k[hit]
            w_hi[m], g_hi[m] = w_k, g_k[~hit]
            act = m
            if act.size == 0:
                break

        # 2. Illinois regula falsi on [w_lo, w_hi] (g(w_hi) > 0 >= g(w_lo)), float64, on the
        #    still-unconverged pixels only (g is linear in w on planar patches: 1 iteration).
        br = np.flatnonzero(~too_near & (g_lo < 0))
        Ub, Vb, DUb, DVb = U0[br], V0[br], DU[br], DV[br]
        a_w, a_g = w_hi[br].astype(np.float64), g_hi[br].astype(np.float64)
        b_w, b_g = w_lo[br].astype(np.float64), g_lo[br].astype(np.float64)
        side = np.zeros(br.size, np.int8)
        w_best = b_w.copy()
        tol = self.w_near * (1e-9 if exact else 1e-6)
        act = np.arange(br.size)
        for _ in range(self.refine_iters):
            if act.size == 0:
                break
            aw, ag, bw, bg, sd = a_w[act], a_g[act], b_w[act], b_g[act], side[act]
            den = ag - bg
            w_new = np.where(den > 0, (bw * ag - aw * bg) / np.where(den > 0, den, 1.0),
                             0.5 * (aw + bw))
            wf = w_new.astype(ftype)
            mu, mv = Ub[act] + DUb[act] * wf, Vb[act] + DVb[act] * wf
            smp = _sample(inv, mu, mv, True) if exact else _sample_flat(inv, mu, mv)
            g_new = w_new - smp.astype(np.float64)
            pos = g_new > 0
            # replace the endpoint of the same sign; Illinois: halve the stale one's g
            bg = np.where(pos & (sd == 1), 0.5 * bg, bg)
            ag = np.where(~pos & (sd == -1), 0.5 * ag, ag)
            a_w[act], a_g[act] = np.where(pos, w_new, aw), np.where(pos, g_new, ag)
            b_w[act], b_g[act] = np.where(pos, bw, w_new), np.where(pos, bg, g_new)
            side[act] = np.where(pos, 1, -1)
            w_best[act] = w_new
            act = act[(np.abs(g_new) > tol) & (a_w[act] - b_w[act] > tol)]
        w_flat = w_lo.astype(np.float64)
        w_flat[br] = w_best

        w = np.zeros((self.H, self.W))
        w.ravel()[idx] = w_flat
        tn = np.zeros((self.H, self.W), bool)
        tn.ravel()[idx] = too_near

        map_u, map_v = self._uv(w)
        in_img = (map_u >= -0.5) & (map_u <= s.W - 0.5) & (map_v >= -0.5) & (map_v <= s.H - 0.5)
        background = w <= 1e-12
        c = s.centre
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.where(background, np.inf, 1.0 / w + c[2])       # camera-frame z
            t = (Z - self.P2[..., 2]) / self.d[..., 2]            # water path length
        edges = depth_edges(inv)
        near_edge = _sample(edges.astype(np.float32), map_u, map_v, False) > 0
        valid = self.valid_ray & in_img & ~tn
        return dict(map_u=map_u, map_v=map_v, w=w, Z=Z, water_range=t, valid=valid,
                    background=background & valid, too_near=tn, near_edge=near_edge & valid)

    def render(self, rgb, depth, depth_type="z", exact=False, invalid_value=0):
        """Underwater image (H, W[, C]) from a pinhole render (spec.H, spec.W[, C]) and its
        depth. Returns (image, info) with info from `trace`."""
        info = self.trace(depth, depth_type, exact)
        img = _sample(rgb if exact else np.ascontiguousarray(rgb), info["map_u"],
                      info["map_v"], exact)
        mask = info["valid"] if img.ndim == 2 else info["valid"][..., None]
        img = np.where(mask, img, invalid_value).astype(rgb.dtype if not exact else img.dtype)
        return img, info


def apply_water(rgb, water_range, beta_D, beta_B, B_inf):
    """Optional simple image-formation model (Akkaynak & Treibitz 2018, revised model with
    range-independent coefficients): I = J exp(-beta_D z) + B_inf (1 - exp(-beta_B z)),
    using the *refracted* water path length from the port (not the pinhole range).
    rgb in [0, 1], coefficients per channel."""
    z = np.where(np.isfinite(water_range), water_range, 1e6)[..., None]
    beta_D, beta_B, B_inf = (np.asarray(x, dtype=np.float64) for x in (beta_D, beta_B, B_inf))
    return rgb * np.exp(-beta_D * z) + B_inf * (1 - np.exp(-beta_B * z))
