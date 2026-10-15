"""Flat-port projection residual for bundle adjustment with the housing (port tilt, d0) and
the in-air intrinsics as unknowns.

Observation: a keypoint u = (u, v) on the *raw* (refracted, distorted) image, Refrax pixel
convention (pixel (c, r) has its centre at (c, r)). Unknowns: the camera-frame point
P = R X + t and the global parameter vector theta (core.uncertainty.PARAMS order:
d0, tglass, mu_g, mu_w, tilt_x, tilt_y, fx, fy, cx, cy, k1, k2, p1, p2; any subset free).

Residual ("back-projection form", cheapest exact form):

    w      = 1 / P_z                              (the 3-D point gives the depth)
    q(u)   = c + f * F(p(u; K, k), w; housing)    (core.uncertainty.corrected_xy: undo the
                                                    lens, refract, intersect the water ray
                                                    with the plane z = P_z, project)
    pi(P)  = c + f * P_xy / P_z                   (pinhole projection of the 3-D point)
    r      = J_u^-1 (q(u) - pi(P)),   J_u = dq/du  (2 x 2)

c, f are *declared* constants (any values; the corrected-image intrinsics of Refrax are
used so that q is the Refrax-corrected keypoint). q(u) - pi(P) = 0 iff the water ray of the
raw keypoint passes through P, so the residual is exact; the factor J_u^-1 maps it back to
raw pixels to first order, so isotropic keypoint noise on the raw image gives an isotropic,
unit-weight residual (the statistically right weighting). The forward direction
(P -> raw pixel) would need the iterative/quartic flat-port forward solve per residual and
per Jacobian evaluation; this form is closed form apart from the radtan undistortion (8
Newton steps, differentiable).

J_u is evaluated at the current estimate and held fixed for the derivative (its own
derivative multiplies the residual, i.e. it is second order at the solution).

All derivatives come from one batched forward-mode AD pass (torch.func.jvp, vmapped over
the tangent directions theta (14), w, u, v).
"""
import numpy as np
import torch

from core import uncertainty as U

PARAMS = U.PARAMS
IDX = {k: i for i, k in enumerate(PARAMS)}


class Decl:
    """Declared corrected-image pinhole (f, c) + nominal housing frame (R0, mu_a)."""

    def __init__(self, f_out, c_out, R0=np.eye(3), mu_a=1.0, center_z=0.0):
        self.f_out = (float(f_out[0]), float(f_out[1]))
        self.c_out = (float(c_out[0]), float(c_out[1]))
        self.R0 = np.asarray(R0, np.float64)
        self.mu_a = float(mu_a)
        self.center_z = float(center_z)
        self.vc_scale = 1.0


def _q(th, w, ur, vr, decl):
    qx, qy = U.corrected_xy(th, ur, vr, w, decl)
    return torch.stack([qx, qy], -1)


def residual_and_jac(theta, u, Pc, decl, chunk=1 << 16, threads=None):
    """Residuals and derivatives for N observations.

    theta: (14,) numpy; u: (N, 2) raw keypoints; Pc: (N, 3) camera-frame points.
    Returns r (N, 2) [raw px], dr/dPc (N, 2, 3), dr/dtheta (N, 2, 14), the corrected
    residual e = q - pi (N, 2) [corrected px] and J_u^-1 (N, 2, 2).
    """
    if threads:
        torch.set_num_threads(threads)
    N = len(u)
    P = len(PARAMS)
    r_all = np.empty((N, 2))
    e_all = np.empty((N, 2))
    JP_all = np.empty((N, 2, 3))
    JT_all = np.empty((N, 2, P))
    Jui_all = np.empty((N, 2, 2))
    th = torch.as_tensor(np.asarray(theta, np.float64))
    fx, fy = decl.f_out
    cx, cy = decl.c_out
    # tangents: theta (P), w, ur, vr
    nt = P + 3
    T_th = torch.zeros(nt, P, dtype=torch.float64)
    T_th[:P, :P] = torch.eye(P, dtype=torch.float64)
    T_s = torch.zeros(nt, 3, dtype=torch.float64)
    T_s[P:, :] = torch.eye(3, dtype=torch.float64)
    for s in range(0, N, chunk):
        sl = slice(s, min(N, s + chunk))
        ur = torch.as_tensor(u[sl, 0], dtype=torch.float64)
        vr = torch.as_tensor(u[sl, 1], dtype=torch.float64)
        P3 = torch.as_tensor(Pc[sl], dtype=torch.float64)
        w = 1.0 / P3[:, 2]
        f = lambda t, ww, uu, vv: _q(t, ww, uu, vv, decl)
        q = f(th, w, ur, vr)                                     # (n, 2)

        def jv(dt, ds):
            return torch.func.jvp(f, (th, w, ur, vr),
                                  (dt, ds[0].expand_as(w), ds[1].expand_as(ur),
                                   ds[2].expand_as(vr)))[1]
        J = torch.func.vmap(jv)(T_th, T_s).permute(1, 2, 0)      # (n, 2, nt)
        J = J.numpy()
        q = q.numpy()
        Pn = P3.numpy()
        z = Pn[:, 2]
        pi = np.c_[cx + fx * Pn[:, 0] / z, cy + fy * Pn[:, 1] / z]
        e = q - pi
        Ju = J[:, :, P + 1:P + 3]
        Jui = np.linalg.inv(Ju)
        # d e / d P: dq/dw * dw/dP - dpi/dP
        dqdw = J[:, :, P]                                        # (n, 2)
        dE = np.zeros((len(z), 2, 3))
        dE[:, :, 2] = dqdw * (-1.0 / z ** 2)[:, None]
        dE[:, 0, 0] -= fx / z
        dE[:, 0, 2] += fx * Pn[:, 0] / z ** 2
        dE[:, 1, 1] -= fy / z
        dE[:, 1, 2] += fy * Pn[:, 1] / z ** 2
        r_all[sl] = np.einsum("nij,nj->ni", Jui, e)
        e_all[sl] = e
        JP_all[sl] = Jui @ dE
        JT_all[sl] = Jui @ J[:, :, :P]
        Jui_all[sl] = Jui
    return r_all, JP_all, JT_all, e_all, Jui_all


def raw_from_corrected(q, theta, decl, z):
    """Exact raw keypoint (Refrax convention) of corrected-image keypoints q (N, 2) that a
    fixed-depth (z) Refrax correction with parameters theta (on-axis or tilted port) put at
    q: solves c + f F(p, 1/z) = q for p, then applies the in-air lens."""
    from core.flatport import FlatPort, _newton_const_w
    d0, tg, mg, mw, ax, ay, fx, fy, cx, cy, k1, k2, p1, p2 = theta
    n = decl.R0.T @ n_port_of(ax, ay)
    m = FlatPort(n, d0, tg, decl.mu_a, mg, mw)
    X = (q[:, 0] - decl.c_out[0]) / decl.f_out[0]
    Y = (q[:, 1] - decl.c_out[1]) / decl.f_out[1]
    w = 1.0 / z
    g, _, _ = m.radial_gain(0.0, w)
    x, y = _newton_const_w(m, X, Y, w, X / g, Y / g, 1e-14, 60)
    xd, yd = U.distort_radtan(x, y, k1, k2, p1, p2)
    return np.c_[fx * xd + cx, fy * yd + cy]


def n_port_of(tilt_x, tilt_y):
    """Port normal of tilts (rad) in the convention of core.uncertainty (R = Rx(ax) Ry(ay)
    takes the port normal to e_z): n = R^T e_z = (-sin ay cos ax, sin ax, cos ay cos ax)."""
    a, b = tilt_x, tilt_y
    return np.array([-np.sin(b) * np.cos(a), np.sin(a), np.cos(b) * np.cos(a)])


def tilts_of_n(n):
    """Inverse of n_port_of."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    ax = np.arcsin(n[1])
    ay = np.arctan2(-n[0], n[2])
    return ax, ay
