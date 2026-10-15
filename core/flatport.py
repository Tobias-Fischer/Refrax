"""Analytic flat-port forward model and its pixel-exact inverse.

Forward model
-------------
For an underwater pixel with normalised in-air coordinates p = (x, y) = K^-1 (u, v, 1),
the scene point at camera depth Z projects (in-air pinhole, normalised) to

    F(p, w) = A(p) + B(p) * w,        w = 1 / Z,

where A is the direction of the water ray (A = ray_xy / ray_z) and B = P2_xy - P2_z * A
with P2 the ray's exit point on the port. A and B depend on the housing only, so the
dependence on depth is affine in inverse depth. In the port frame (z' along n_port) the
refraction is radially symmetric: the tangent of the ray angle is scaled by

    k(rho^2) = r / sqrt(1 + (1 - r^2) rho^2),   r = mu_a / mu,   rho = tan(theta_air),

in the glass (kg) and in the water (kw). For n_port = (0, 0, 1) the camera and port frames
coincide and

    F(p, w) = p * (kw + w * h),    h = d0 + tglass * kg - (d0 + tglass) * kw,

i.e. the map only scales p by a function of |p|^2 and w. It agrees with core.optics /
core.undistort.forward_map (the reference ray tracer) to floating-point precision.

Inverse
-------
The corrected image is rendered by cv2.remap, which needs, for every output pixel q, the
underwater pixel p with F(p, w(p)) = q. Instead of interpolating scattered forward-map
samples (scipy griddata), the equation is solved per output pixel:

* n_port on the optical axis: p lies on the ray through the principal point and q, so the
  problem is 1-D in rho = |p|. It is solved by safeguarded Newton iteration on the bracket
  [0, rho_hi] (Newton steps leaving the bracket fall back to bisection), which converges
  for any continuous depth (the bilinearly interpolated depth map), including slopes and
  depth discontinuities.
* tilted n_port: 2-D damped Newton iteration with a finite-difference Jacobian.

Both converge to ~1e-9 px; pixels whose solution lies outside the source frame are NaN.

Virtual centre
--------------
With the corrected pinhole at (0, 0, c) on the optical axis instead of the camera centre
(correction.virtual_center, see core/virtual_center.py), the point at depth Z projects to
(A Z + B) / (Z - c) = A + (B + c A) w' with w' = 1 / (Z - c): the same model with
B -> B + c A and w -> 1 / (Z - c). FlatPort(center_z=c) applies the first substitution,
and callers pass depth - c as the depth (InverseMapper and forward_map_analytic do this).
"""
import numpy as np

# Convergence tolerance in pixels and iteration caps.
TOL_PX = 1e-6
MAX_ITER_1D = 60
MAX_ITER_2D = 40


def _rotation_to_z(n):
    """Rotation matrix R with R @ n = e_z (n unit)."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s, c = np.linalg.norm(v), float(np.dot(n, z))
    if s < 1e-15:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


class FlatPort:
    """Analytic flat-port refraction model F(p, w) = A(p) + B(p) w (normalised coordinates)."""

    def __init__(self, n_port, d0, tglass, mu_a, mu_g, mu_w, center_z=0.0):
        n = np.asarray(n_port, dtype=np.float64)
        self.n = n / np.linalg.norm(n)
        self.R = _rotation_to_z(self.n)
        self.radial = bool(self.n[0] == 0.0 and self.n[1] == 0.0 and self.n[2] > 0)
        self.d0, self.tglass = float(d0), float(tglass)
        self.dz = self.d0 + self.tglass
        # Virtual centre: B -> B + c A, i.e. the port exit is measured from z = c.
        self.cz = float(center_z)
        self.dzc = self.dz - self.cz
        self.rg, self.rw = mu_a / mu_g, mu_a / mu_w
        self.cg, self.cw = 1.0 - self.rg ** 2, 1.0 - self.rw ** 2

    # -- radial building blocks (functions of s = rho^2) ----------------------------------
    def _k(self, s):
        """(kg, kw): ray-tangent scale factors in glass and water. NaN beyond TIR."""
        with np.errstate(invalid="ignore"):
            kg = self.rg / np.sqrt(1.0 + self.cg * s)
            kw = self.rw / np.sqrt(1.0 + self.cw * s)
        return kg, kw

    def radial_gain(self, s, w):
        """g(s, w) with F = p * g for n_port on the axis, and d(rho * g)/d rho."""
        kg, kw = self._k(s)
        h = self.d0 + self.tglass * kg - self.dzc * kw
        g = kw + w * h
        # d(rho k)/d rho = k / (1 + c s)
        dkw = kw / (1.0 + self.cw * s)
        dkg = kg / (1.0 + self.cg * s)
        dg = dkw + w * (self.d0 + self.tglass * dkg - self.dzc * dkw)
        return g, dg, h

    # -- forward model ------------------------------------------------------------------
    def AB(self, x, y):
        """A(p), B(p) for normalised source coordinates (x, y). Returns (Ax, Ay, Bx, By)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.radial:
            kg, kw = self._k(x * x + y * y)
            h = self.d0 + self.tglass * kg - self.dzc * kw
            return x * kw, y * kw, x * h, y * h
        R = self.R
        # Ray direction (x, y, 1) in the port frame, as tangent vector t' = d'_xy / d'_z.
        dz = R[2, 0] * x + R[2, 1] * y + R[2, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            tx = (R[0, 0] * x + R[0, 1] * y + R[0, 2]) / dz
            ty = (R[1, 0] * x + R[1, 1] * y + R[1, 2]) / dz
        tx = np.where(dz > 0, tx, np.nan)
        kg, kw = self._k(tx * tx + ty * ty)
        a = self.d0 + self.tglass * kg  # P2' = (a t', dz)
        # Back to the camera frame (R^T): water direction (kw t', 1), exit point P2.
        Rt = R.T
        wx = Rt[0, 0] * kw * tx + Rt[0, 1] * kw * ty + Rt[0, 2]
        wy = Rt[1, 0] * kw * tx + Rt[1, 1] * kw * ty + Rt[1, 2]
        wz = Rt[2, 0] * kw * tx + Rt[2, 1] * kw * ty + Rt[2, 2]
        px = Rt[0, 0] * a * tx + Rt[0, 1] * a * ty + Rt[0, 2] * self.dz
        py = Rt[1, 0] * a * tx + Rt[1, 1] * a * ty + Rt[1, 2] * self.dz
        pz = Rt[2, 0] * a * tx + Rt[2, 1] * a * ty + Rt[2, 2] * self.dz
        with np.errstate(invalid="ignore", divide="ignore"):
            Ax, Ay = wx / wz, wy / wz
        pz = pz - self.cz
        return Ax, Ay, px - pz * Ax, py - pz * Ay

    def forward(self, x, y, w):
        Ax, Ay, Bx, By = self.AB(x, y)
        return Ax + Bx * w, Ay + By * w


# -- bilinear sampling of inverse depth ------------------------------------------------------
class InvDepth:
    """Inverse depth w = 1/Z: a constant, or a per-source-pixel map sampled bilinearly
    (float64 weights, border-clamped), with its gradient (per pixel, normalised units)."""

    def __init__(self, depth, fx, fy, cx, cy):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        d = np.asarray(depth, dtype=np.float64)
        self.const = d.ndim == 0
        if self.const:
            self.value = 1.0 / float(d)
        else:
            self.w = np.ascontiguousarray(1.0 / d)
            self.H, self.W = d.shape

    def sample(self, x, y, grad=False):
        """w at normalised (x, y); with grad, also (dw/dx, dw/dy) in normalised units."""
        if self.const:
            v = np.full(np.shape(x), self.value)
            return (v, 0.0, 0.0) if grad else v
        u_raw = self.fx * x + self.cx
        v_raw = self.fy * y + self.cy
        u = np.clip(u_raw, 0.0, self.W - 1.0)
        v = np.clip(v_raw, 0.0, self.H - 1.0)
        i0 = np.minimum(np.floor(v).astype(np.intp), self.H - 2)
        j0 = np.minimum(np.floor(u).astype(np.intp), self.W - 2)
        a, b = u - j0, v - i0
        idx = i0 * self.W + j0
        wf = self.w.ravel()
        w00, w01 = wf[idx], wf[idx + 1]
        w10, w11 = wf[idx + self.W], wf[idx + self.W + 1]
        top = w00 + a * (w01 - w00)
        bot = w10 + a * (w11 - w10)
        val = top + b * (bot - top)
        if not grad:
            return val
        dwdu = (w01 - w00) + b * ((w11 - w10) - (w01 - w00))
        dwdv = bot - top
        dwdu = np.where((u_raw > 0) & (u_raw < self.W - 1), dwdu, 0.0)
        dwdv = np.where((v_raw > 0) & (v_raw < self.H - 1), dwdv, 0.0)
        return val, dwdu * self.fx, dwdv * self.fy


# -- inverse solvers ----------------------------------------------------------------------
def _frame_rho_max(ex, ey, x_lo, x_hi, y_lo, y_hi):
    """Distance from the principal point to the frame border along unit direction (ex, ey)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(ex > 0, x_hi / ex, np.where(ex < 0, x_lo / ex, np.inf))
        ty = np.where(ey > 0, y_hi / ey, np.where(ey < 0, y_lo / ey, np.inf))
    return np.minimum(tx, ty)


def _radial_lut(model, w, rho_max, n=4096):
    """Table (R_u, rho(R_u)) uniform in R of the constant-depth radial inverse."""
    rho_t = np.linspace(0.0, rho_max, n)
    g, _, _ = model.radial_gain(rho_t * rho_t, w)
    R_t = rho_t * g
    R_u = np.linspace(0.0, R_t[-1], n)
    if not np.all(np.diff(R_t) > 0):  # not monotone: paraxial guess
        return R_u, R_u / g[0]
    return R_u, np.interp(R_u, R_t, rho_t)


def _radial_lut_inverse(model, w, R, rho_max, n=4096):
    """Initial guess for the constant-depth radial inverse: rho(R) from a dense 1-D table
    of R(rho) = rho g(rho^2, w), which is monotone for physical housings."""
    rho_t = np.linspace(0.0, rho_max, n)
    g, _, _ = model.radial_gain(rho_t * rho_t, w)
    R_t = rho_t * g
    if not np.all(np.diff(R_t) > 0):  # not monotone: fall back to the paraxial guess
        return R / g[0]
    return np.interp(R, R_t, rho_t)


def invert_radial(model, X, Y, invdepth, frame, rho0=None, tol=None):
    """Solves rho * g(rho^2, w(rho e)) = |X| along e = X / |X| (n_port on the axis).

    X, Y: normalised (unzoomed) target coordinates, any shape. frame = (x_lo, x_hi, y_lo,
    y_hi) normalised source frame; roots further than 2 px beyond it are not searched and
    returned as NaN. rho0: optional initial guess. Returns normalised source (x, y).
    """
    shape = np.shape(X)
    X = np.asarray(X, np.float64).ravel()
    Y = np.asarray(Y, np.float64).ravel()
    Rt = np.hypot(X, Y)
    with np.errstate(invalid="ignore", divide="ignore"):
        ex = np.where(Rt > 0, X / Rt, 1.0)
        ey = np.where(Rt > 0, Y / Rt, 0.0)
    fscale = max(invdepth.fx, invdepth.fy)
    tol = TOL_PX / fscale if tol is None else tol
    x_lo, x_hi, y_lo, y_hi = frame
    margin = 2.0 / min(invdepth.fx, invdepth.fy)
    hi = _frame_rho_max(ex, ey, x_lo - margin, x_hi + margin, y_lo - margin, y_hi + margin)
    exit_ = _frame_rho_max(ex, ey, x_lo, x_hi, y_lo, y_hi)
    lo = np.zeros_like(hi)

    def f_and_df(rho, idx):
        exi, eyi = ex[idx], ey[idx]
        xs, ys = rho * exi, rho * eyi
        if invdepth.const:
            w, dwr = invdepth.value, 0.0
        else:
            w, gx, gy = invdepth.sample(xs, ys, grad=True)
            dwr = gx * exi + gy * eyi
        g, dg, h = model.radial_gain(rho * rho, w)
        return rho * g - Rt[idx], dg + rho * h * dwr

    # The root lies in [0, hi] (f(0) = -Rt <= 0). If f(hi) <= 0 there is no root inside
    # the frame; the iteration then converges to hi, which is outside it (marked invalid).
    rho = np.zeros_like(Rt)
    idx = np.flatnonzero(Rt > 0)
    if rho0 is not None:
        r = np.asarray(rho0, np.float64).ravel()[idx]
        r = np.where(np.isfinite(r), r, 0.5 * hi[idx])
    elif invdepth.const:
        r = _radial_lut_inverse(model, invdepth.value, Rt[idx], hi.max())
    else:
        g0, _, _ = model.radial_gain(0.0, 0.0)
        r = Rt[idx] / g0
    lo_i, hi_i = lo[idx], hi[idx]
    r = np.clip(r, lo_i, hi_i)
    dx = hi_i - lo_i
    dxold = dx
    for _ in range(MAX_ITER_1D):
        f, df = f_and_df(r, idx)
        pos = f > 0
        hi_i = np.where(pos, r, hi_i)
        lo_i = np.where(pos, lo_i, r)
        with np.errstate(invalid="ignore", divide="ignore"):
            step = f / df
        r_new = r - step
        conv = (np.abs(step) < tol) & (np.abs(f) < tol)  # |f| too: steep depth ramps
        # Bisect if Newton leaves the bracket (or is NaN) or is not converging fast (rtsafe).
        bad = (~((r_new >= lo_i) & (r_new <= hi_i)) | (np.abs(2 * step) > np.abs(dxold))) & ~conv
        r_new = np.where(bad, 0.5 * (lo_i + hi_i), r_new)
        dxold, dx = dx, r_new - r
        done = conv | (hi_i - lo_i < tol)
        rho[idx[done]] = r_new[done]
        # A root beyond the frame border is invalid anyway: stop early.
        outside = lo_i > exit_[idx] + tol
        rho[idx[outside]] = np.nan
        done |= outside
        keep = ~done
        if not keep.any():
            break
        idx, r, lo_i, hi_i = idx[keep], r_new[keep], lo_i[keep], hi_i[keep]
        dx, dxold = dx[keep], dxold[keep]
    else:
        rho[idx] = r
    xs, ys = rho * ex, rho * ey
    return xs.reshape(shape), ys.reshape(shape)


def invert_general(model, X, Y, invdepth, x0=None, y0=None, tol=None):
    """2-D damped Newton solve of F(p, w(p)) = (X, Y) for a tilted port.

    Initial guess: (x0, y0) if given, else the paraxial guess. Pixels that do not converge
    are NaN (the caller marks solutions outside the source frame invalid).
    """
    shape = np.shape(X)
    X = np.asarray(X, np.float64).ravel()
    Y = np.asarray(Y, np.float64).ravel()
    fscale = max(invdepth.fx, invdepth.fy)
    tol = TOL_PX / fscale if tol is None else tol
    if x0 is None:
        w_ref = invdepth.value if invdepth.const else float(np.median(invdepth.w))
        Ax, Ay, Bx, By = model.AB(np.zeros(1), np.zeros(1))
        g, _, _ = model.radial_gain(0.0, w_ref)
        x = (X - (Ax[0] + Bx[0] * w_ref)) / g
        y = (Y - (Ay[0] + By[0] * w_ref)) / g
    else:
        x = np.asarray(x0, np.float64).ravel().copy()
        y = np.asarray(y0, np.float64).ravel().copy()
        bad = ~(np.isfinite(x) & np.isfinite(y))
        x[bad], y[bad] = X[bad], Y[bad]
    out_x = np.full_like(X, np.nan)
    out_y = np.full_like(X, np.nan)
    idx = np.arange(len(X))
    eps = 1e-6

    def residual(xi, yi, idx):
        w = invdepth.sample(xi, yi)
        Fx, Fy = model.forward(xi, yi, w)
        return Fx - X[idx], Fy - Y[idx]

    rx, ry = residual(x, y, idx)
    for _ in range(MAX_ITER_2D):
        if invdepth.const:
            w, gx, gy = invdepth.value, 0.0, 0.0
        else:
            w, gx, gy = invdepth.sample(x, y, grad=True)
        Ax, Ay, Bx, By = model.AB(x, y)
        Axu, Ayu, Bxu, Byu = model.AB(x + eps, y)
        Axv, Ayv, Bxv, Byv = model.AB(x, y + eps)
        j11 = ((Axu - Ax) + (Bxu - Bx) * w) / eps + Bx * gx
        j12 = ((Axv - Ax) + (Bxv - Bx) * w) / eps + Bx * gy
        j21 = ((Ayu - Ay) + (Byu - By) * w) / eps + By * gx
        j22 = ((Ayv - Ay) + (Byv - By) * w) / eps + By * gy
        det = j11 * j22 - j12 * j21
        with np.errstate(invalid="ignore", divide="ignore"):
            dx = -(j22 * rx - j12 * ry) / det
            dy = -(-j21 * rx + j11 * ry) / det
        # Damping: halve the step until the residual decreases (up to 8 times).
        lam = np.ones_like(x)
        r0 = np.hypot(rx, ry)
        nx, ny = x + dx, y + dy
        nrx, nry = residual(nx, ny, idx)
        for _ in range(8):
            worse = ~(np.hypot(nrx, nry) <= r0)
            if not worse.any():
                break
            lam = np.where(worse, lam * 0.5, lam)
            sub = np.flatnonzero(worse)
            nx[sub], ny[sub] = x[sub] + lam[sub] * dx[sub], y[sub] + lam[sub] * dy[sub]
            srx, sry = residual(nx[sub], ny[sub], idx[sub])
            nrx[sub], nry[sub] = srx, sry
        step = np.hypot(nx - x, ny - y)
        x, y, rx, ry = nx, ny, nrx, nry
        done = (step < tol) & (np.hypot(rx, ry) < tol)
        out_x[idx[done]], out_y[idx[done]] = x[done], y[done]
        keep = ~done & np.isfinite(step)
        if not keep.any():
            break
        idx, x, y, rx, ry = idx[keep], x[keep], y[keep], rx[keep], ry[keep]
    failed = np.flatnonzero(np.isnan(out_x))
    if not invdepth.const and len(failed):
        # Depth discontinuity: bisection on w, the root of w_map(p(w)) - w, where p(w) solves
        # the constant-w problem (always bracketed by the depth range).
        fx_, fy_ = _bisect_w(model, X[failed], Y[failed], invdepth, tol)
        out_x[failed], out_y[failed] = fx_, fy_
    return out_x.reshape(shape), out_y.reshape(shape)


def _newton_const_w(model, X, Y, w, x, y, tol, max_iter=MAX_ITER_2D):
    """Vectorised 2-D Newton for F(p, w) = (X, Y) at per-pixel constant w."""
    eps = 1e-6
    for _ in range(max_iter):
        Ax, Ay, Bx, By = model.AB(x, y)
        Axu, Ayu, Bxu, Byu = model.AB(x + eps, y)
        Axv, Ayv, Bxv, Byv = model.AB(x, y + eps)
        rx, ry = Ax + Bx * w - X, Ay + By * w - Y
        j11 = ((Axu - Ax) + (Bxu - Bx) * w) / eps
        j12 = ((Axv - Ax) + (Bxv - Bx) * w) / eps
        j21 = ((Ayu - Ay) + (Byu - By) * w) / eps
        j22 = ((Ayv - Ay) + (Byv - By) * w) / eps
        det = j11 * j22 - j12 * j21
        with np.errstate(invalid="ignore", divide="ignore"):
            dx = -(j22 * rx - j12 * ry) / det
            dy = -(-j21 * rx + j11 * ry) / det
        x, y = x + dx, y + dy
        if np.all(np.hypot(dx, dy) < tol):
            break
    return x, y


def _bisect_w(model, X, Y, invdepth, tol, n_iter=200):
    lo = np.full_like(X, invdepth.w.min())
    hi = np.full_like(X, invdepth.w.max())
    g, _, _ = model.radial_gain(0.0, 0.5 * (lo[0] + hi[0]))
    x, y = X / g, Y / g
    for _ in range(n_iter):
        w = 0.5 * (lo + hi)
        x, y = _newton_const_w(model, X, Y, w, x, y, tol)
        up = invdepth.sample(x, y) > w
        lo, hi = np.where(up, w, lo), np.where(up, hi, w)
        if np.all(hi - lo < 1e-15 * np.maximum(1.0, hi)):
            break
    return x, y
