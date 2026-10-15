"""Refraction removal on the GPU (PyTorch; device "cpu", "mps" or "cuda").

Same algorithm as the numba kernels (core._kernels), vectorised over all output pixels of
a batch of frames:

* on-axis port: safeguarded 1-D Newton (rtsafe) along the ray through the principal
  point, bracketed by the frame;
* tilted port: 2-D Newton with an *analytic* Jacobian of A(p), B(p) (the numba kernel's
  finite differences are useless in float32), with step halving on overshoot and, for the
  pixels that do not converge (depth discontinuities), bisection on inverse depth;
* radial-tangential distortion folded into the map, remap with `grid_sample` (bilinear,
  zero padding, `align_corners=False`, i.e. cv2.remap's pixel-centre convention).

Everything that does not depend on the depth (target rays, frame brackets, the
constant-depth solution used as initial guess, the fixed-depth sampling grid) is
precomputed once per camera in float64 on the CPU and uploaded. Per frame, the solve
runs a fixed number of dense iterations without host synchronisation; the few pixels that
are still unconverged (depth steps) are then compacted and finished on the subset.

float32 is sufficient: the residual F(p) - q is evaluated to ~1e-7 (normalised), i.e.
~1e-4 px, two orders below cv2.remap's 1/32 px coordinate quantisation (see
docs/explore_gpu_remove.md). float64 can be requested on cpu/cuda for validation.

Streaming use (e.g. a ROS 2 node)::

    corr = Corrector(cfg, device="mps")
    out, mask = corr.correct(img_bgr_uint8, depth)   # depth=None: fixed depth z0
"""
import logging
import types

import numpy as np
import torch
import torch.nn.functional as F

from core.config import camera_params, distortion_params, housing_params
from core.flatport import FlatPort, MAX_ITER_1D, MAX_ITER_2D, TOL_PX
from core.undistort import InverseMapper

# Inductor's Metal backend warns about every Python float scalar it sees ("float64 cast").
logging.getLogger("torch._inductor.codegen.mps").setLevel(logging.ERROR)
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)

# Convergence tolerance (output-independent source px) per dtype. float32 cannot resolve
# the numba path's 1e-6 px; 1e-3 px is ~10 ulp of the normalised coordinates.
TOL_PX_DTYPE = {torch.float32: 1e-3, torch.float64: TOL_PX}
DENSE_ITERS_1D = 8   # dense (sync-free) iterations before compacting unconverged pixels
DENSE_ITERS_2D = 8
UNROLL_2D = 2        # tilted-port iterations per compiled call (Metal: <= 31 buffers/kernel)
LUT_SIZE = 4096      # constant-depth radial table for the initial guess
BUCKET_MIN, BUCKET_GROWTH = 4096, 8  # compacted subset sizes: 4096 * 8^k
SUB_CHUNK = 8        # iterations per compiled call on the compacted subset
BISECT_W_ITERS = 40  # outer bisection steps on inverse depth (tilted port, depth steps)


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchInverse:
    """Per-camera precomputation + per-frame depth-aware inverse map on a torch device.

    Arguments as core.undistort.InverseMapper, plus distortion (k1, k2, p1, p2) and
    z_ref (depth of the precomputed initial guess, e.g. correction.z0_fixed).
    """

    def __init__(self, W, H, fx, fy, cx, cy, housing, zoom=None, canvas=None, distortion=None,
                 device="cpu", dtype=torch.float32, z_ref=1.0, compile="auto"):
        self.device = torch.device(device)
        if dtype == torch.float64 and self.device.type == "mps":
            raise ValueError("MPS has no float64")
        self.dtype = dtype
        self.W, self.H, self.fx, self.fy, self.cx, self.cy = W, H, fx, fy, cx, cy
        self.model = m = FlatPort(**housing)
        self.housing = housing
        self.zoom = 1.0 if zoom is None else float(zoom)
        self.canvas = canvas if canvas is not None else (W, H, (0.0, 0.0))
        out_W, out_H, (ox, oy) = self.canvas
        self.out_W, self.out_H = out_W, out_H
        self.distortion = None
        if distortion is not None and any(d != 0.0 for d in distortion):
            self.distortion = tuple(float(d) for d in distortion)
        self.tol = TOL_PX_DTYPE[dtype] / max(fx, fy)
        self.frame = (-cx / fx, (W - 1 - cx) / fx, -cy / fy, (H - 1 - cy) / fy)
        self.z_ref = float(z_ref)

        Xs = (np.arange(out_W, dtype=np.float64) + ox - cx) / (fx * self.zoom)
        Ys = (np.arange(out_H, dtype=np.float64) + oy - cy) / (fy * self.zoom)
        X, Y = np.meshgrid(Xs, Ys)
        X, Y = X.ravel(), Y.ravel()
        t = self._t
        w_ref = 1.0 / self.z_ref
        if m.radial:
            Rt = np.hypot(X, Y)
            with np.errstate(invalid="ignore", divide="ignore"):
                ex = np.where(Rt > 0, X / Rt, 1.0)
                ey = np.where(Rt > 0, Y / Rt, 0.0)
            x_lo, x_hi, y_lo, y_hi = self.frame
            margin = 2.0 / min(fx, fy)
            exit_ = _frame_exit(ex, ey, x_lo, x_hi, y_lo, y_hi)
            hi = _frame_exit(ex, ey, x_lo - margin, x_hi + margin, y_lo - margin, y_hi + margin)
            # Initial guess per frame (as the numba kernel): the constant-depth inverse at
            # the frame's median depth, from a 1-D table R(rho) = rho (kw + w h) built on
            # the device (kw, h depend on rho only).
            mapper = InverseMapper(W, H, fx, fy, cx, cy, housing, zoom, canvas)
            rho_t = np.linspace(0.0, mapper._rho_max(), LUT_SIZE)
            kg, kw = m._k(rho_t * rho_t)
            h = m.d0 + m.tglass * kg - m.dz * kw
            self.c = dict(Rt=t(Rt), ex=t(ex), ey=t(ey), exit=t(exit_), hi=t(hi))
            self.lut = dict(rho=t(rho_t), kw=t(kw), h=t(h))
        else:
            # Initial guess per frame: the constant-depth solution p(w) at the frame's
            # median inverse depth, to first order about w_ref: p0 + (w - w_ref) dp/dw
            # (numba / NumPy, float64; paraxial guess where there is none).
            mapper = InverseMapper(W, H, fx, fy, cx, cy, housing, zoom, canvas)
            sol = []
            for f in (1.0, 1.0 / 1.05, 1.0 / 0.95):  # w_ref, w_ref (1 +- 0.05)
                xs, ys = mapper.build(self.z_ref * f)
                sol.append((((xs - cx) / fx).ravel(), ((ys - cy) / fy).ravel()))
            (x0, y0), (xa, ya), (xb, yb) = sol
            dw = 0.05 * w_ref
            xw, yw = (xa - xb) / (2 * dw), (ya - yb) / (2 * dw)
            Ax, Ay, Bx, By = m.AB(np.zeros(1), np.zeros(1))
            g0, _, _ = m.radial_gain(0.0, w_ref)
            px0 = (X - (Ax[0] + Bx[0] * w_ref)) / g0
            py0 = (Y - (Ay[0] + By[0] * w_ref)) / g0
            bad = ~(np.isfinite(x0) & np.isfinite(y0))
            x0[bad], y0[bad] = px0[bad], py0[bad]
            bad = ~(np.isfinite(xw) & np.isfinite(yw))
            xw[bad], yw[bad] = 0.0, 0.0
            self.c = dict(X=t(X), Y=t(Y), x0=t(x0), y0=t(y0), xw=t(xw), yw=t(yw),
                          px0=t(px0), py0=t(py0))
            self.w_ref = w_ref
            self.R = [[float(v) for v in row] for row in m.R]
        self.P = (m.rg, m.rw, m.cg, m.cw, m.d0, m.tglass, m.dz)
        # Fused kernels (static shapes). The compacted subsets are padded to 4096 * 8^k
        # sizes, so only a handful of variants is ever compiled (see warmup()).
        self.compiled = _use_compile(compile, self.device)
        cmp = _compiler(self.compiled)
        self._dense_radial = cmp(self._radial_iterate, dynamic=False)
        self._sub_radial = cmp(_rename(self._radial_iterate), dynamic=False)
        self._dense_general = cmp(self._general_iterate, dynamic=False)
        self._sub_general = cmp(_rename(self._general_iterate), dynamic=False)
        self._sub_bisect = cmp(self._bisect_step, dynamic=False)
        self._finish_c = cmp(self._finish, dynamic=False)
        self._init_c = cmp(self._radial_init if m.radial else self._general_init, dynamic=False)

    def _t(self, a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=self.dtype, device=self.device)

    # -- inverse depth --------------------------------------------------------------------
    def _sample_w(self, wflat, boff, u, v, grad):
        """Bilinear w at pixel (u, v), border-clamped (as core._kernels._sample_w)."""
        H, W = self.H, self.W
        uc = u.clamp(0.0, W - 1.0)
        vc = v.clamp(0.0, H - 1.0)
        j0 = torch.floor(uc).clamp_(max=W - 2)
        i0 = torch.floor(vc).clamp_(max=H - 2)
        a = uc - j0
        b = vc - i0
        idx = boff + i0.long() * W + j0.long()
        w00 = wflat[idx]
        w01 = wflat[idx + 1]
        w10 = wflat[idx + W]
        w11 = wflat[idx + (W + 1)]
        top = w00 + a * (w01 - w00)
        bot = w10 + a * (w11 - w10)
        val = top + b * (bot - top)
        if not grad:
            return val, None, None
        gu = (w01 - w00) + b * ((w11 - w10) - (w01 - w00))
        gv = bot - top
        gu = torch.where((u > 0) & (u < W - 1.0), gu, 0.0)
        gv = torch.where((v > 0) & (v < H - 1.0), gv, 0.0)
        return val, gu, gv

    # -- on-axis port ---------------------------------------------------------------------
    def _radial_f(self, r, ex, ey, Rt, wflat, boff):
        rg, rw, cg, cw, d0, tg, dz = self.P
        s = r * r
        qg = 1.0 / (1.0 + cg * s)
        qw = 1.0 / (1.0 + cw * s)
        kg = rg * torch.sqrt(qg)
        kw = rw * torch.sqrt(qw)
        h = d0 + tg * kg - dz * kw
        dkw = kw * qw
        dkg = kg * qg
        w, gu, gv = self._sample_w(wflat, boff, self.fx * r * ex + self.cx,
                                   self.fy * r * ey + self.cy, True)
        dwr = gu * (self.fx * ex) + gv * (self.fy * ey)
        g = kw + w * h
        dg = dkw + w * (d0 + tg * dkg - dz * dkw)
        return r * g - Rt, dg + r * h * dwr

    def _radial_iterate(self, n, st, ex, ey, Rt, exit_, wflat, boff):
        """n rtsafe iterations on state st = dict(r, lo, hi, dx, dxold, active, valid)."""
        tol = self.tol
        r, lo, hi, dx, dxold, active, valid = (st[k] for k in
                                               ("r", "lo", "hi", "dx", "dxold", "active", "valid"))
        for _ in range(n):
            f, df = self._radial_f(r, ex, ey, Rt, wflat, boff)
            pos = f > 0
            hi = torch.where(pos & active, r, hi)
            lo = torch.where(~pos & active, r, lo)
            step = f / df
            r_new = r - step
            conv = (step.abs() < tol) & (f.abs() < tol)
            bad = (~((r_new >= lo) & (r_new <= hi)) | ((2.0 * step).abs() > dxold.abs())) & ~conv
            r_new = torch.where(bad, 0.5 * (lo + hi), r_new)
            dxold = torch.where(active, dx, dxold)
            dx = torch.where(active, r_new - r, dx)
            r = torch.where(active, r_new, r)
            outside = active & (lo > exit_ + tol)
            valid = valid & ~outside
            active = active & ~(conv | (hi - lo < tol) | outside)
        return dict(r=r, lo=lo, hi=hi, dx=dx, dxold=dxold, active=active, valid=valid)

    def _radial_init(self, w_med):
        """Initial state: rho from the constant-depth table at w_med (B,), bracket [0, hi]."""
        c, L = self.c, self.lut
        B, N = w_med.shape[0], c["Rt"].numel()
        R_t = L["rho"] * (L["kw"] + w_med[:, None] * L["h"])  # (B, n), increasing
        Rt = c["Rt"].expand(B, N).contiguous()
        k = torch.searchsorted(R_t, Rt).clamp_(1, LUT_SIZE - 1)
        R0 = torch.gather(R_t, 1, k - 1)
        R1 = torch.gather(R_t, 1, k)
        rho0, rho1 = L["rho"][k - 1], L["rho"][k]
        r = rho0 + (Rt - R0) / (R1 - R0) * (rho1 - rho0)
        hi = c["hi"].expand(B, N)
        r = torch.where(Rt > R_t[:, -1:], hi, r)
        r = torch.minimum(r.clamp(min=0.0), hi)
        lo = torch.zeros_like(r)
        dx = hi - lo
        active = (c["Rt"] > 0).expand(B, N).clone()
        return dict(r=r, lo=lo, hi=hi.clone(), dx=dx, dxold=dx.clone(), active=active,
                    valid=torch.ones_like(active))

    def _general_init(self, w_med):
        c = self.c
        B, N = w_med.shape[0], c["X"].numel()
        dw = (w_med - self.w_ref)[:, None]
        x = c["x0"] + dw * c["xw"]
        y = c["y0"] + dw * c["yw"]
        return dict(x=x, y=y, xp=x.clone(), yp=y.clone(),
                    rprev=torch.full_like(x, float("inf")),
                    active=torch.ones_like(x, dtype=torch.bool),
                    ok=torch.zeros_like(x, dtype=torch.bool))

    def _solve_radial(self, wflat, B, w_med):
        c = self.c
        N = c["Rt"].numel()
        boff = (torch.arange(B, device=self.device) * (self.H * self.W)).view(B, 1)
        st = self._init_c(w_med)
        consts = (c["ex"], c["ey"], c["Rt"], c["exit"])
        st = self._dense_radial(DENSE_ITERS_1D, st, *consts, wflat, boff)
        # Finish the unconverged pixels (depth steps, slow cases).
        st = self._finish_subset(
            self._sub_radial, lambda n, s_: self._dense_radial(n, s_, *consts, wflat, boff),
            st, ("r", "valid"), N, B, lambda pix: tuple(v[pix] for v in consts),
            MAX_ITER_1D - DENSE_ITERS_1D, SUB_CHUNK)
        r = st["r"]
        return r * c["ex"], r * c["ey"], st["valid"]

    # -- tilted port ----------------------------------------------------------------------
    def _AB_jac(self, x, y):
        """A, B and their analytic Jacobians at normalised (x, y) (tilted port).

        Returns (Ax, Ay, Bx, By, Axx, Axy, Ayx, Ayy, Bxx, Bxy, Byx, Byy), Axy = dAx/dy.
        """
        R = self.R
        rg, rw, cg, cw, d0, tg, dz = self.P
        nz = R[2][0] * x + R[2][1] * y + R[2][2]
        nz = torch.where(nz > 0, nz, float("nan"))
        iz = 1.0 / nz
        tx = (R[0][0] * x + R[0][1] * y + R[0][2]) * iz
        ty = (R[1][0] * x + R[1][1] * y + R[1][2]) * iz
        # dt/dx = (R[0:2, 0] - t R[2, 0]) / nz, dt/dy likewise with column 1.
        txx = (R[0][0] - tx * R[2][0]) * iz
        txy = (R[0][1] - tx * R[2][1]) * iz
        tyx = (R[1][0] - ty * R[2][0]) * iz
        tyy = (R[1][1] - ty * R[2][1]) * iz
        s = tx * tx + ty * ty
        qg = 1.0 / (1.0 + cg * s)
        qw = 1.0 / (1.0 + cw * s)
        kg = rg * torch.sqrt(qg)
        kw = rw * torch.sqrt(qw)
        # dk/ds = -c k q / 2; ds/dx = 2 t.dt/dx
        sx = 2.0 * (tx * txx + ty * tyx)
        sy = 2.0 * (tx * txy + ty * tyy)
        kwx = -0.5 * cw * kw * qw * sx
        kwy = -0.5 * cw * kw * qw * sy
        a = d0 + tg * kg
        ax = -0.5 * tg * cg * kg * qg * sx
        ay = -0.5 * tg * cg * kg * qg * sy
        # Water direction (kw t, 1) and exit point (a t, dz) back in the camera frame (R^T).
        ux, uy = kw * tx, kw * ty
        uxx, uxy = kwx * tx + kw * txx, kwy * tx + kw * txy
        uyx, uyy = kwx * ty + kw * tyx, kwy * ty + kw * tyy
        wx = R[0][0] * ux + R[1][0] * uy + R[2][0]
        wy = R[0][1] * ux + R[1][1] * uy + R[2][1]
        wz = R[0][2] * ux + R[1][2] * uy + R[2][2]
        vx, vy = a * tx, a * ty
        vxx, vxy = ax * tx + a * txx, ay * tx + a * txy
        vyx, vyy = ax * ty + a * tyx, ay * ty + a * tyy
        px = R[0][0] * vx + R[1][0] * vy + R[2][0] * dz
        py = R[0][1] * vx + R[1][1] * vy + R[2][1] * dz
        pz = R[0][2] * vx + R[1][2] * vy + R[2][2] * dz
        iwz = 1.0 / wz
        Ax = wx * iwz
        Ay = wy * iwz
        Bx = px - pz * Ax
        By = py - pz * Ay
        out = [Ax, Ay, Bx, By]
        dA, dB = [], []
        for (u1, u2), (v1, v2) in (((uxx, uyx), (vxx, vyx)), ((uxy, uyy), (vxy, vyy))):
            dwx = R[0][0] * u1 + R[1][0] * u2
            dwy = R[0][1] * u1 + R[1][1] * u2
            dwz = R[0][2] * u1 + R[1][2] * u2
            dpx = R[0][0] * v1 + R[1][0] * v2
            dpy = R[0][1] * v1 + R[1][1] * v2
            dpz = R[0][2] * v1 + R[1][2] * v2
            dAx = (dwx - Ax * dwz) * iwz
            dAy = (dwy - Ay * dwz) * iwz
            dA.append((dAx, dAy))
            dB.append((dpx - dpz * Ax - pz * dAx, dpy - dpz * Ay - pz * dAy))
        (Axx, Ayx), (Axy, Ayy) = dA
        (Bxx, Byx), (Bxy, Byy) = dB
        return out + [Axx, Axy, Ayx, Ayy, Bxx, Bxy, Byx, Byy]

    def _general_iterate(self, n, st, X, Y, wflat, boff):
        tol = self.tol
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        x, y, xp, yp, rprev, active, ok = (st[k] for k in
                                           ("x", "y", "xp", "yp", "rprev", "active", "ok"))
        for _ in range(n):
            (Ax, Ay, Bx, By, Axx, Axy, Ayx, Ayy, Bxx, Bxy, Byx, Byy) = self._AB_jac(x, y)
            w, gu, gv = self._sample_w(wflat, boff, fx * x + cx, fy * y + cy, True)
            gx, gy = gu * fx, gv * fy
            rx = Ax + Bx * w - X
            ry = Ay + By * w - Y
            rn = torch.sqrt(rx * rx + ry * ry)
            conv = rn < tol
            was_active = active
            ok = ok | (active & conv)
            active = active & ~conv & torch.isfinite(rn)
            back = active & (rn > rprev)  # overshoot: halve the last step
            j11 = Axx + w * Bxx + Bx * gx
            j12 = Axy + w * Bxy + Bx * gy
            j21 = Ayx + w * Byx + By * gx
            j22 = Ayy + w * Byy + By * gy
            idet = 1.0 / (j11 * j22 - j12 * j21)
            nx = x - (j22 * rx - j12 * ry) * idet
            ny = y - (-j21 * rx + j11 * ry) * idet
            # Converged pixels still take this Newton step (quadratic refinement below tol).
            step = (active & ~back) | (conv & was_active)
            xp = torch.where(step, x, xp)
            yp = torch.where(step, y, yp)
            rprev = torch.where(step, rn, rprev)
            x = torch.where(step, nx, torch.where(back, 0.5 * (x + xp), x))
            y = torch.where(step, ny, torch.where(back, 0.5 * (y + yp), y))
        return dict(x=x, y=y, xp=xp, yp=yp, rprev=rprev, active=active, ok=ok)

    def _bisect_step(self, X, Y, x, y, lo, hi, wflat, boff):
        """One bisection step on w of w_map(p(w)) - w, p(w) the constant-w solution (three
        Newton steps warm-started from the previous p)."""
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        w = 0.5 * (lo + hi)
        for _ in range(3):
            (Ax, Ay, Bx, By, Axx, Axy, Ayx, Ayy, Bxx, Bxy, Byx, Byy) = self._AB_jac(x, y)
            rx = Ax + Bx * w - X
            ry = Ay + By * w - Y
            j11, j12 = Axx + w * Bxx, Axy + w * Bxy
            j21, j22 = Ayx + w * Byx, Ayy + w * Byy
            idet = 1.0 / (j11 * j22 - j12 * j21)
            x = x - (j22 * rx - j12 * ry) * idet
            y = y - (-j21 * rx + j11 * ry) * idet
        wp, _, _ = self._sample_w(wflat, boff, fx * x + cx, fy * y + cy, False)
        up = wp > w
        return x, y, torch.where(up, w, lo), torch.where(up, hi, w)

    def _bisect_w(self, X, Y, x, y, lo, hi, wflat, boff):
        """Depth steps (tilted port): bisection on inverse depth, bracketed by its range."""
        for _ in range(BISECT_W_ITERS):
            x, y, lo, hi = self._sub_bisect(X, Y, x, y, lo, hi, wflat, boff)
        return x, y, torch.isfinite(x) & torch.isfinite(y)

    def _solve_general(self, wflat, B, w_med):
        c = self.c
        N = c["X"].numel()
        HW = self.H * self.W
        boff = (torch.arange(B, device=self.device) * HW).view(B, 1)
        st = self._init_c(w_med)
        for _ in range(DENSE_ITERS_2D // UNROLL_2D):
            st = self._dense_general(UNROLL_2D, st, c["X"], c["Y"], wflat, boff)
        st = self._finish_subset(
            self._sub_general,
            lambda n, s_: self._dense_general(n, s_, c["X"], c["Y"], wflat, boff),
            st, ("x", "y", "ok"), N, B, lambda pix: (c["X"][pix], c["Y"][pix]),
            MAX_ITER_2D - DENSE_ITERS_2D, UNROLL_2D)
        # Bisection on w for every pixel that failed (finite or not).
        x, y, ok = st["x"], st["y"], st["ok"]
        idx = torch.nonzero(~ok.view(-1)).squeeze(1)
        if idx.numel():
            idx = _pad_bucket(idx)
            pix = idx % N
            b = idx // N
            wb = wflat.view(B, HW)
            w_lo = wb.amin(dim=1)[b]
            w_hi = wb.amax(dim=1)[b]
            bx, by, bok = self._bisect_w(c["X"][pix], c["Y"][pix], c["px0"][pix],
                                         c["py0"][pix], w_lo, w_hi, wflat, b * HW)
            x.view(-1)[idx] = bx
            y.view(-1)[idx] = by
            ok.view(-1)[idx] = bok
        return x, y, ok

    def _finish_subset(self, fn, dense, st, keys, N, B, consts, n_iter, unroll):
        """Continues the iteration on the still-active pixels until none is active (at most
        n_iter more iterations, checked every SUB_CHUNK).

        Few active pixels (the usual case: depth steps): they are gathered into a compact
        array padded to a power-of-8 bucket and iterated with fn, so that only a handful of
        static shapes is ever compiled (see warmup()). Many: dense(st) continues on the
        full arrays."""
        HW = self.H * self.W
        idx = torch.nonzero(st["active"].view(-1)).squeeze(1)
        if not idx.numel():
            return st
        if idx.numel() > B * N // BUCKET_GROWTH:
            for done in range(0, n_iter, SUB_CHUNK):
                for _ in range(SUB_CHUNK // unroll):
                    st = dense(unroll, st)
                if not bool(st["active"].any()):
                    break
            return st
        idx = _pad_bucket(idx)
        pix = idx % N
        boff = idx // N * HW
        cst = consts(pix)
        sub = {k: v.view(-1)[idx] for k, v in st.items()}
        for done in range(0, n_iter, SUB_CHUNK):
            for _ in range(SUB_CHUNK // unroll):
                sub = fn(unroll, sub, *cst, self._wflat_cur, boff)
            if done + SUB_CHUNK < n_iter and not bool(sub["active"].any()):
                break
        for k in keys:
            st[k].view(-1)[idx] = sub[k]
        return st

    def subset_buckets(self, B=1):
        """Subset sizes the compacted phases can take for a batch of B frames."""
        N = self.out_W * self.out_H
        sizes, size = [], BUCKET_MIN
        while True:
            sizes.append(size)
            if size >= B * N // BUCKET_GROWTH:
                return sizes
            size *= BUCKET_GROWTH

    def warmup(self, B=1, subsets=True):
        """Compiles (torch.compile) every kernel variant a batch of B frames can hit, so that
        no compilation happens mid-stream. Compiled kernels are also cached on disk by
        inductor, which makes later processes start faster."""
        d = torch.full((B, self.H, self.W), self.z_ref, device=self.device)
        d[:, : self.H // 2] *= 2.0  # a depth step: exercises the subset phases
        self.remap_maps(d)
        if not subsets or not self.compiled:
            return
        HW, N = self.H * self.W, self.out_W * self.out_H
        wflat = self._wflat(d)
        self._wflat_cur = wflat
        w_med = torch.full((B,), 1.0 / self.z_ref, device=self.device, dtype=self.dtype)
        st0 = self._init_c(w_med)
        c = self.c
        for size in self.subset_buckets(B):
            idx = torch.arange(size, device=self.device) % (B * N)
            pix, boff = idx % N, idx // N * HW
            sub = {k: v.reshape(-1)[idx] for k, v in st0.items()}
            if self.model.radial:
                self._sub_radial(SUB_CHUNK, sub, c["ex"][pix], c["ey"][pix], c["Rt"][pix],
                                 c["exit"][pix], wflat, boff)
            else:
                self._sub_general(UNROLL_2D, sub, c["X"][pix], c["Y"][pix], wflat, boff)
                self._sub_bisect(c["X"][pix], c["Y"][pix], c["px0"][pix], c["py0"][pix],
                                 sub["x"], sub["x"], wflat, boff)
        synchronize(self.device)

    # -- public ---------------------------------------------------------------------------
    def _wflat(self, depth):
        """depth: (B, H, W) tensor on the device, already clamped."""
        return (1.0 / depth.to(self.dtype)).reshape(-1).contiguous()

    def remap_maps(self, depth):
        """(map_x, map_y), each (B, out_H, out_W), remap-ready source pixels in the raw
        (distorted) image, -1 where invalid. depth: (B, H, W) on the device."""
        B = depth.shape[0]
        wflat = self._wflat(depth)
        self._wflat_cur = wflat
        # Reference inverse depth of the initial guess: 1 / median depth (as numba). Where
        # the solution is not unique (occlusion folds at depth steps) it picks the root.
        w_med = 1.0 / depth.reshape(B, -1).median(dim=1).values.to(self.dtype)
        if self.model.radial:
            x, y, ok = self._solve_radial(wflat, B, w_med)
        else:
            x, y, ok = self._solve_general(wflat, B, w_med)
        return self._finish_c(x, y, ok, B)

    def _finish(self, x, y, ok, B):
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        W1, H1 = self.W - 1.0, self.H - 1.0
        u = fx * x + cx
        v = fy * y + cy
        tol = 1e-4 if self.dtype == torch.float32 else 1e-6
        ok = ok & (u >= -tol) & (u <= W1 + tol) & (v >= -tol) & (v <= H1 + tol)
        u = u.clamp(0.0, W1)
        v = v.clamp(0.0, H1)
        if self.distortion is not None:
            k1, k2, p1, p2 = self.distortion
            xn = (u - cx) / fx
            yn = (v - cy) / fy
            r2 = xn * xn + yn * yn
            radial = 1.0 + k1 * r2 + k2 * r2 * r2
            xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
            yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
            u = xd * fx + cx
            v = yd * fy + cy
        u = torch.where(ok, u, -1.0)
        v = torch.where(ok, v, -1.0)
        shape = (B, self.out_H, self.out_W)
        return u.view(shape), v.view(shape)


def _pad_bucket(idx):
    """Pads an index vector (repeating its last entry) to the next bucket size."""
    n = idx.numel()
    size = BUCKET_MIN
    while size < n:
        size *= BUCKET_GROWTH
    if size == n:
        return idx
    return torch.cat((idx, idx[-1:].expand(size - n)))


def _rename(method):
    """A distinct code object for a second torch.compile of the same method, so that its
    compiled variants (other shapes) are cached separately from the dense call's."""
    fn = method.__func__
    code = fn.__code__.replace(co_name=fn.__name__ + "_sub")
    clone = types.FunctionType(code, fn.__globals__, code.co_name, fn.__defaults__,
                               fn.__closure__)
    return types.MethodType(clone, method.__self__)


def _use_compile(compile, device):
    """torch.compile (inductor) fuses each iteration into one kernel: ~10x faster on MPS.
    "auto": on for mps/cuda, off for cpu (inductor's C++ backend needs a compiler and the
    eager CPU path is only a fallback; numba is the CPU path)."""
    if compile == "auto":
        return device.type in ("mps", "cuda")
    return bool(compile)


def _compiler(enabled):
    def cmp(fn, dynamic):
        return torch.compile(fn, dynamic=dynamic) if enabled else fn
    return cmp


def _frame_exit(ex, ey, x_lo, x_hi, y_lo, y_hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(ex > 0, x_hi / ex, np.where(ex < 0, x_lo / ex, np.inf))
        ty = np.where(ey > 0, y_hi / ey, np.where(ey < 0, y_lo / ey, np.inf))
    return np.minimum(tx, ty)


def valid_mask(map_x, map_y, W, H):
    """remove_refraction.compute_valid_mask as a bool tensor."""
    return (map_x >= 0) & (map_x <= W - 1) & (map_y >= 0) & (map_y <= H - 1)


def maps_to_grid(map_x, map_y, W, H):
    """cv2.remap maps (pixel centres at integers) -> grid_sample grid (align_corners=False)."""
    gx = (2.0 * map_x + 1.0) / W - 1.0
    gy = (2.0 * map_y + 1.0) / H - 1.0
    return torch.stack((gx, gy), dim=-1).to(torch.float32)


def remap(img, grid):
    """Bilinear remap with F.grid_sample: uint8 images (B, H, W, C) or float (B, C, H, W).

    Returns the same layout/dtype as the input. Samples outside the source blend with 0
    (zero padding), exactly as cv2.remap with BORDER_CONSTANT; maps at -1 give 0.
    """
    channels_last = img.dtype == torch.uint8
    x = img.permute(0, 3, 1, 2).float() if channels_last else img.float()
    if grid.shape[0] != x.shape[0]:
        grid = grid.expand(x.shape[0], -1, -1, -1)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if channels_last:
        return out.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
    return out


def remap_px(img, map_x, map_y):
    """cv2.remap(INTER_LINEAR, BORDER_CONSTANT=0) of uint8 images (B, H, W, C) with pixel
    maps (B or 1, out_H, out_W), written as four gathers so that torch.compile fuses it into
    one kernel reading the uint8 image directly (~3x faster than grid_sample on MPS; eager it
    is slower, use remap()). Bilinear weights are exact (cv2 quantises them to 1/32)."""
    B, H, W, C = img.shape
    flat = img.reshape(B * H * W, C)
    boff = (torch.arange(B, device=img.device) * (H * W)).view(B, 1, 1)
    x0 = torch.floor(map_x)
    y0 = torch.floor(map_y)
    ax = (map_x - x0).unsqueeze(-1)
    ay = (map_y - y0).unsqueeze(-1)
    x0 = x0.long()
    y0 = y0.long()

    def tap(xi, yi):
        inb = ((xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)).unsqueeze(-1)
        idx = boff + yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1)
        return torch.where(inb, flat[idx].float(), 0.0)

    v00, v01 = tap(x0, y0), tap(x0 + 1, y0)
    v10, v11 = tap(x0, y0 + 1), tap(x0 + 1, y0 + 1)
    top = v00 + ax * (v01 - v00)
    bot = v10 + ax * (v11 - v10)
    return (top + ay * (bot - top)).round().clamp(0, 255).to(torch.uint8)


class Corrector:
    """Streaming refraction removal for one camera / housing, on a torch device.

    Setup (zoom, canvas, fixed-depth map, depth-independent tensors) runs once in the
    constructor. Per frame::

        out, mask = corr.correct(img, depth=None)

    img: uint8 (H, W, C) or (B, H, W, C) (OpenCV / ROS layout), numpy or torch (any
    device); float (B, C, H, W) also works (grid_sample path). depth: None (fixed depth
    correction.z0_fixed: cached map, one remap), or (H, W) / (B, H, W) metric depth per
    source pixel, as in remove_refraction's depth-map mode (resized bilinearly to H x W if
    needed, clamped >= 1e-3 as load_depth does). Returns torch tensors on the device: the
    corrected images on the full output canvas (out_H, out_W) and the validity mask (bool,
    compute_valid_mask semantics). `K` holds the corrected intrinsics (canvas frame);
    cropping to a valid rectangle is left to the caller (core.rectangle).
    """

    def __init__(self, cfg, device=None, dtype=torch.float32, compile="auto"):
        from remove_refraction import _fit_canvas
        from core.scale import select_zoom
        self.device = torch.device(device or default_device())
        W, H, fx, fy, cx, cy = camera_params(cfg)
        self.W, self.H = W, H
        distortion = distortion_params(cfg)
        housing = housing_params(cfg)
        corr = cfg["correction"]
        z0 = corr["z0_fixed"]
        fit = corr.get("fit_canvas", True)
        zoom = corr.get("zoom")
        if zoom is None:
            zoom, _ = select_zoom(z0, W, H, fx, fy, cx, cy, housing, in_bounds=not fit)
        self.zoom = float(zoom)
        if fit:
            canvas, fixed = _fit_canvas(W, H, fx, fy, cx, cy, housing, distortion, zoom, z0)
        else:
            canvas = (W, H, (0.0, 0.0))
            fixed = InverseMapper(W, H, fx, fy, cx, cy, housing, zoom, canvas).remap_maps(
                z0, distortion)
        self.canvas = canvas
        out_W, out_H, (ox, oy) = canvas
        self.K = np.array([[fx * zoom, 0, cx - ox], [0, fy * zoom, cy - oy], [0, 0, 1.0]])
        self.inverse = TorchInverse(W, H, fx, fy, cx, cy, housing, zoom, canvas, distortion,
                                    self.device, dtype, z_ref=z0, compile=compile)
        self.compiled = self.inverse.compiled
        self._remap_px = torch.compile(remap_px, dynamic=False) if self.compiled else None
        # Fixed depth (z0): the CPU kernel's map (float64 solve), uploaded once.
        mx = torch.as_tensor(fixed[0], device=self.device)[None]
        my = torch.as_tensor(fixed[1], device=self.device)[None]
        self.fixed_maps = (mx, my, valid_mask(mx, my, W, H))
        self.fixed_grid = maps_to_grid(mx, my, W, H)

    def warmup(self, B=1, channels=3):
        """Triggers every compilation (solver variants, remap) for batches of B frames."""
        self.inverse.warmup(B)
        img = torch.zeros((B, self.H, self.W, channels), dtype=torch.uint8, device=self.device)
        self.correct(img)
        self.correct(img, torch.full((B, self.H, self.W), self.inverse.z_ref,
                                     device=self.device))
        synchronize(self.device)

    def _depth(self, depth):
        d = torch.as_tensor(depth, device=self.device, dtype=torch.float32)
        if d.dim() == 2:
            d = d[None]
        if d.shape[-2:] != (self.H, self.W):
            d = F.interpolate(d[:, None], size=(self.H, self.W), mode="bilinear",
                              align_corners=False)[:, 0]
        return d.clamp(min=1e-3)

    def maps(self, depth=None):
        """(map_x, map_y, mask): cv2.remap-ready float32 maps (B, out_H, out_W) on the
        device (-1 = invalid) and the validity mask; the cached fixed-depth maps (B = 1) if
        depth is None."""
        if depth is None:
            return self.fixed_maps
        mx, my = self.inverse.remap_maps(self._depth(depth))
        return mx, my, valid_mask(mx, my, self.W, self.H)

    def remap_maps(self, depth):
        """cv2.remap-ready float32 maps (B, out_H, out_W) on the device (-1 = invalid)."""
        return self.inverse.remap_maps(self._depth(depth))

    def correct(self, img, depth=None):
        x = torch.as_tensor(img).to(self.device, non_blocking=True)
        single = x.dim() == 3
        if single:
            x = x[None]
        mx, my, mask = self.maps(depth)
        if self.compiled and x.dtype == torch.uint8:
            out = self._remap_px(x, mx, my)
        else:
            grid = self.fixed_grid if depth is None else maps_to_grid(mx, my, self.W, self.H)
            out = remap(x, grid)
        if single:
            return out[0], mask[0]
        return out, mask


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

