"""Numba kernels: per-output-pixel inverse of the flat-port model (see core.flatport).

Same algorithms as core.flatport.invert_radial / invert_general, fused into one parallel
loop that also applies the radial-tangential distortion and writes cv2.remap-ready maps.
Imported lazily; core.flatport's NumPy implementation is used when numba is missing.
"""
import math

import numba as nb
import numpy as np

_OPTS = dict(nogil=True, cache=True)


@nb.njit(**_OPTS)
def _sample_w(wmap, u, v, need_grad):
    """Bilinear inverse depth at pixel (u, v), border-clamped; gradient per pixel."""
    H, W = wmap.shape
    uc = min(max(u, 0.0), W - 1.0)
    vc = min(max(v, 0.0), H - 1.0)
    j0 = min(int(math.floor(uc)), W - 2)
    i0 = min(int(math.floor(vc)), H - 2)
    a = uc - j0
    b = vc - i0
    w00 = wmap[i0, j0]
    w01 = wmap[i0, j0 + 1]
    w10 = wmap[i0 + 1, j0]
    w11 = wmap[i0 + 1, j0 + 1]
    top = w00 + a * (w01 - w00)
    bot = w10 + a * (w11 - w10)
    val = top + b * (bot - top)
    gu = 0.0
    gv = 0.0
    if need_grad:
        if u > 0.0 and u < W - 1.0:
            gu = (w01 - w00) + b * ((w11 - w10) - (w01 - w00))
        if v > 0.0 and v < H - 1.0:
            gv = bot - top
    return val, gu, gv


@nb.njit(**_OPTS)
def _radial_f(rho, ex, ey, Rt, w_const, wmap, use_map, P, cam):
    """f(rho) = rho g(rho^2, w(rho e)) - Rt and its derivative."""
    rg, rw, cg, cw, d0, tg, dz = P[0], P[1], P[2], P[3], P[4], P[5], P[6]
    fx, fy, cx, cy = cam[0], cam[1], cam[2], cam[3]
    s = rho * rho
    kg = rg / math.sqrt(1.0 + cg * s)
    kw = rw / math.sqrt(1.0 + cw * s)
    h = d0 + tg * kg - dz * kw
    dkw = kw / (1.0 + cw * s)
    dkg = kg / (1.0 + cg * s)
    if use_map:
        w, gu, gv = _sample_w(wmap, fx * rho * ex + cx, fy * rho * ey + cy, True)
        dwr = gu * fx * ex + gv * fy * ey
    else:
        w = w_const
        dwr = 0.0
    g = kw + w * h
    dg = dkw + w * (d0 + tg * dkg - dz * dkw)
    return rho * g - Rt, dg + rho * h * dwr


@nb.njit(**_OPTS)
def _frame_exit(ex, ey, x_lo, x_hi, y_lo, y_hi):
    tx = np.inf
    ty = np.inf
    if ex > 0:
        tx = x_hi / ex
    elif ex < 0:
        tx = x_lo / ex
    if ey > 0:
        ty = y_hi / ey
    elif ey < 0:
        ty = y_lo / ey
    return min(tx, ty)


@nb.njit(**_OPTS)
def _finish(x, y, cam, frame_px, dist, out_x, out_y, i, j, invalid):
    """Writes the source pixel of normalised (x, y) (radtan applied) or `invalid`."""
    fx, fy, cx, cy = cam[0], cam[1], cam[2], cam[3]
    u = fx * x + cx
    v = fy * y + cy
    W1, H1 = frame_px[0], frame_px[1]
    tol = 1e-6
    if not (u >= -tol and u <= W1 + tol and v >= -tol and v <= H1 + tol):
        out_x[i, j] = invalid
        out_y[i, j] = invalid
        return
    u = min(max(u, 0.0), W1)
    v = min(max(v, 0.0), H1)
    if dist[4] != 0.0:
        xn = (u - cx) / fx
        yn = (v - cy) / fy
        k1, k2, p1, p2 = dist[0], dist[1], dist[2], dist[3]
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        u = xd * fx + cx
        v = yd * fy + cy
    out_x[i, j] = u
    out_y[i, j] = v


@nb.njit(parallel=True, **_OPTS)
def radial_maps(Xs, Ys, P, cam, frame, frame_px, w_const, wmap, use_map,
                lut_R, lut_rho, tol, max_iter, dist, invalid, out_x, out_y):
    """n_port on the axis: safeguarded 1-D Newton along the ray through q (see flatport)."""
    fmin = min(cam[0], cam[1])
    margin = 2.0 / fmin
    nR = lut_R.shape[0]
    dR = lut_R[1] - lut_R[0]
    for i in nb.prange(Ys.shape[0]):
        Y = Ys[i]
        for j in range(Xs.shape[0]):
            X = Xs[j]
            Rt = math.sqrt(X * X + Y * Y)
            if Rt == 0.0:
                _finish(0.0, 0.0, cam, frame_px, dist, out_x, out_y, i, j, invalid)
                continue
            ex = X / Rt
            ey = Y / Rt
            exit_ = _frame_exit(ex, ey, frame[0], frame[1], frame[2], frame[3])
            hi = _frame_exit(ex, ey, frame[0] - margin, frame[1] + margin,
                             frame[2] - margin, frame[3] + margin)
            lo = 0.0
            # Initial guess: constant-depth inverse from the uniform-in-R table.
            t = (Rt - lut_R[0]) / dR
            k = int(t)
            if k >= nR - 1:
                r = hi
            else:
                t -= k
                r = lut_rho[k] + t * (lut_rho[k + 1] - lut_rho[k])
            r = min(max(r, lo), hi)
            ok = False
            dx = hi - lo
            dxold = dx
            for _ in range(max_iter):
                f, df = _radial_f(r, ex, ey, Rt, w_const, wmap, use_map, P, cam)
                if f > 0:
                    hi = r
                else:
                    lo = r
                step = f / df
                r_new = r - step
                if abs(step) < tol and abs(f) < tol:
                    r = r_new
                    ok = True
                    break
                # Bisect if Newton leaves the bracket or is not converging fast (rtsafe).
                if not (r_new >= lo and r_new <= hi) or abs(2.0 * step) > abs(dxold):
                    r_new = 0.5 * (lo + hi)
                dxold = dx
                dx = r_new - r
                r = r_new
                if hi - lo < tol:
                    ok = True
                    break
                if lo > exit_ + tol:
                    break
            if not ok:
                if lo > exit_ + tol:
                    out_x[i, j] = invalid
                    out_y[i, j] = invalid
                    continue
            _finish(r * ex, r * ey, cam, frame_px, dist, out_x, out_y, i, j, invalid)


@nb.njit(**_OPTS)
def _AB(x, y, R, P):
    """A(p), B(p) for a tilted port (R: camera -> port frame)."""
    rg, rw, cg, cw, d0, tg, dz = P[0], P[1], P[2], P[3], P[4], P[5], P[6]
    dzp = R[2, 0] * x + R[2, 1] * y + R[2, 2]
    if not dzp > 0:
        return np.nan, np.nan, np.nan, np.nan
    tx = (R[0, 0] * x + R[0, 1] * y + R[0, 2]) / dzp
    ty = (R[1, 0] * x + R[1, 1] * y + R[1, 2]) / dzp
    s = tx * tx + ty * ty
    kg = rg / math.sqrt(1.0 + cg * s)
    kw = rw / math.sqrt(1.0 + cw * s)
    a = d0 + tg * kg
    wx = R[0, 0] * kw * tx + R[1, 0] * kw * ty + R[2, 0]
    wy = R[0, 1] * kw * tx + R[1, 1] * kw * ty + R[2, 1]
    wz = R[0, 2] * kw * tx + R[1, 2] * kw * ty + R[2, 2]
    px = R[0, 0] * a * tx + R[1, 0] * a * ty + R[2, 0] * dz
    py = R[0, 1] * a * tx + R[1, 1] * a * ty + R[2, 1] * dz
    pz = R[0, 2] * a * tx + R[1, 2] * a * ty + R[2, 2] * dz
    Ax = wx / wz
    Ay = wy / wz
    return Ax, Ay, px - pz * Ax, py - pz * Ay


@nb.njit(**_OPTS)
def _resid(x, y, X, Y, R, P, cam, w_const, wmap, use_map):
    Ax, Ay, Bx, By = _AB(x, y, R, P)
    if use_map:
        w, gu, gv = _sample_w(wmap, cam[0] * x + cam[2], cam[1] * y + cam[3], False)
    else:
        w = w_const
    return Ax + Bx * w - X, Ay + By * w - Y


@nb.njit(**_OPTS)
def _jac_AB(x, y, R, P):
    """Finite-difference Jacobians of A and B at (x, y): (dAx/dx, dAx/dy, dAy/dx, dAy/dy, same for B)."""
    eps = 1e-6
    Ax, Ay, Bx, By = _AB(x, y, R, P)
    Axu, Ayu, Bxu, Byu = _AB(x + eps, y, R, P)
    Axv, Ayv, Bxv, Byv = _AB(x, y + eps, R, P)
    return ((Axu - Ax) / eps, (Axv - Ax) / eps, (Ayu - Ay) / eps, (Ayv - Ay) / eps,
            (Bxu - Bx) / eps, (Bxv - Bx) / eps, (Byu - By) / eps, (Byv - By) / eps)


@nb.njit(parallel=True, **_OPTS)
def general_maps(Xs, Ys, R, P, cam, frame_px, w_const, wmap, use_map, x0c, y0c, g0,
                 tol, max_iter, dist, invalid, out_x, out_y):
    """Tilted port: per output pixel, Newton iteration on F(p, w(p)) = q.

    Pixels are swept along rows; each starts from its left neighbour's solution and reuses
    the neighbour's Jacobians of A and B (refreshed every 8 pixels or when convergence is
    slow), so an iteration costs one evaluation of the model. Steps that increase the
    residual are halved. Pixels that still fail (depth discontinuities) fall back to
    bisection on inverse depth.
    """
    w_lo = wmap.min()
    w_hi = wmap.max()
    fx, fy = cam[0], cam[1]
    for i in nb.prange(Ys.shape[0]):
        Y = Ys[i]
        have_prev = False
        px = 0.0
        py = 0.0
        age = 1 << 30
        jA0 = jA1 = jA2 = jA3 = jB0 = jB1 = jB2 = jB3 = 0.0
        for j in range(Xs.shape[0]):
            X = Xs[j]
            if have_prev:
                x = px + (X - Xs[j - 1]) / g0
                y = py
            else:
                x = (X - x0c) / g0
                y = (Y - y0c) / g0
            if age >= 8:
                jA0, jA1, jA2, jA3, jB0, jB1, jB2, jB3 = _jac_AB(x, y, R, P)
                age = 0
            age += 1
            ok = False
            r_prev = np.inf
            xp = x
            yp = y
            for it in range(max_iter):
                Ax, Ay, Bx, By = _AB(x, y, R, P)
                if use_map:
                    w, gu, gv = _sample_w(wmap, fx * x + cam[2], fy * y + cam[3], True)
                    gx = gu * fx
                    gy = gv * fy
                else:
                    w = w_const
                    gx = 0.0
                    gy = 0.0
                rx = Ax + Bx * w - X
                ry = Ay + By * w - Y
                rn = math.hypot(rx, ry)
                if not math.isfinite(rn):
                    break
                if rn < 0.1 * tol:  # tol is in source px; output px are zoom x larger
                    ok = True
                    break
                if rn > r_prev:  # overshoot: halve the last step
                    x = 0.5 * (x + xp)
                    y = 0.5 * (y + yp)
                    continue
                if it == 3:  # slow: refresh the Jacobians here
                    jA0, jA1, jA2, jA3, jB0, jB1, jB2, jB3 = _jac_AB(x, y, R, P)
                    age = 1
                r_prev = rn
                j11 = jA0 + w * jB0 + Bx * gx
                j12 = jA1 + w * jB1 + Bx * gy
                j21 = jA2 + w * jB2 + By * gx
                j22 = jA3 + w * jB3 + By * gy
                det = j11 * j22 - j12 * j21
                xp = x
                yp = y
                x -= (j22 * rx - j12 * ry) / det
                y -= (-j21 * rx + j11 * ry) / det
            have_prev = ok
            if ok:
                px = x
                py = y
            else:
                age = 1 << 30
                if use_map:
                    # Depth discontinuity: bisection on w, the root of w_map(p(w)) - w, where
                    # p(w) solves the constant-w problem (bracketed by the depth range).
                    x = (X - x0c) / g0
                    y = (Y - y0c) / g0
                    x, y, ok = _bisect_w(X, Y, x, y, R, P, cam, wmap, w_lo, w_hi, tol, max_iter)
            if ok:
                _finish(x, y, cam, frame_px, dist, out_x, out_y, i, j, invalid)
            else:
                out_x[i, j] = invalid
                out_y[i, j] = invalid


@nb.njit(**_OPTS)
def _newton_const_w(X, Y, w, x, y, R, P, tol, max_iter):
    """2-D Newton for F(p, w) = (X, Y) at constant w (smooth; no damping needed)."""
    eps = 1e-6
    for _ in range(max_iter):
        Ax, Ay, Bx, By = _AB(x, y, R, P)
        Axu, Ayu, Bxu, Byu = _AB(x + eps, y, R, P)
        Axv, Ayv, Bxv, Byv = _AB(x, y + eps, R, P)
        rx = Ax + Bx * w - X
        ry = Ay + By * w - Y
        j11 = ((Axu - Ax) + (Bxu - Bx) * w) / eps
        j12 = ((Axv - Ax) + (Bxv - Bx) * w) / eps
        j21 = ((Ayu - Ay) + (Byu - By) * w) / eps
        j22 = ((Ayv - Ay) + (Byv - By) * w) / eps
        det = j11 * j22 - j12 * j21
        dx = -(j22 * rx - j12 * ry) / det
        dy = -(-j21 * rx + j11 * ry) / det
        x += dx
        y += dy
        if not (math.isfinite(x) and math.isfinite(y)):
            return x, y, False
        if math.hypot(dx, dy) < tol:
            return x, y, True
    return x, y, False


@nb.njit(**_OPTS)
def _bisect_w(X, Y, x, y, R, P, cam, wmap, w_lo, w_hi, tol, max_iter):
    fx, fy, cx, cy = cam[0], cam[1], cam[2], cam[3]
    lo, hi = w_lo, w_hi
    for _ in range(200):
        w = 0.5 * (lo + hi)
        x, y, conv = _newton_const_w(X, Y, w, x, y, R, P, tol, max_iter)
        if not conv:
            return x, y, False
        wp, gu, gv = _sample_w(wmap, fx * x + cx, fy * y + cy, False)
        if wp > w:
            lo = w
        else:
            hi = w
        if hi - lo < 1e-15 * max(1.0, hi):
            break
    # Converged in w; the residual |F(p, w_map(p)) - q| is ~|B| * (hi - lo).
    return x, y, True
