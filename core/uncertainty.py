"""Per-pixel uncertainty of the refraction correction.

A corrected pixel q is the pinhole projection (output intrinsics K_out = zoom * K, which
are *declared*, i.e. fixed) of the scene point that the raw pixel u it was resampled from
sees. With the estimated parameters theta_hat the correction is exact by construction;
with the true parameters theta the same raw pixel u would have been put at

    q_true = c_out + f_out * F_proj(p(u; intr), w; housing) ,

where p(u; intr) undoes the in-air lens (K, radtan) and F_proj is the flat-port model
F = A(p) + B(p) w of core.flatport, projected from the camera centre (center_z = 0) or
from a point on the axis at center_z (the "virtual centre", F / (1 - center_z w),
rescaled to agree with the camera-centre image at z_ref). The error of a corrected pixel
is dq = q_true - q, to first order dq = J(q) dtheta with J the 2 x P Jacobian evaluated
at the source p_hat(q) of the output pixel.

Two implementations of J:
* torch forward-mode autodiff of `corrected_xy` (any port, all parameters, cpu/mps/cuda),
* closed-form derivatives for the on-axis port (`analytic_jacobian`, NumPy float64),
which agree to rounding (tests/test_uncertainty.py).

The part of dq that a pinhole camera absorbs (fx, fy, cx, cy; optionally a constant
rotation and, with a depth map, a translation of the projection centre) is removed by a
least-squares fit over the image (`absorb`), leaving the non-absorbable residual that can
bend a reconstruction.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import torch

# Order of the parameter vector theta. Tilts in radians (rotation of the port frame about
# the camera x / y axis: n_port ~ (-tilt_y, tilt_x, 1)). "w" (inverse depth) is per pixel
# and handled separately.
PARAMS = ("d0", "tglass", "mu_g", "mu_w", "tilt_x", "tilt_y",
          "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")
LABELS = {
    "d0": "d0 (port distance)", "tglass": "glass thickness", "mu_g": "mu_g (glass type)",
    "mu_w": "mu_w (salinity/T/λ)", "tilt_x": "port tilt about x", "tilt_y": "port tilt about y",
    "fx": "in-air fx", "fy": "in-air fy", "cx": "in-air cx", "cy": "in-air cy",
    "k1": "radtan k1", "k2": "radtan k2", "p1": "radtan p1", "p2": "radtan p2",
}


# -- refractive index of (sea)water ----------------------------------------------------------
def water_index(S, T, lam_nm):
    """Quan & Fry (1995), Appl. Opt. 34(18): index of seawater at salinity S (PSU),
    temperature T (deg C) and wavelength lam (nm), atmospheric pressure.
    Valid 0-30 C, 0-35 PSU, 400-700 nm (used here up to 40 PSU)."""
    S, T, L = np.asarray(S, float), np.asarray(T, float), np.asarray(lam_nm, float)
    return (1.31405 + (1.779e-4 - 1.05e-6 * T + 1.6e-8 * T * T) * S - 2.02e-6 * T * T
            + (15.868 + 0.01155 * S - 0.00423 * T) / L - 4382.0 / L ** 2 + 1.1455e6 / L ** 3)


# Effective wavelengths of the R, G, B channels of a Bayer camera (nm; approximate band
# centroids) and the band used for grey-level features underwater (blue-green).
RGB_NM = {"R": 600.0, "G": 530.0, "B": 465.0}


def water_index_range(S=(0.0, 40.0), T=(0.0, 30.0), lam=(450.0, 650.0), n=21):
    """Min / max / mean / std of water_index over a uniform box of (S, T, lam)."""
    g = np.meshgrid(np.linspace(*S, n), np.linspace(*T, n), np.linspace(*lam, n))
    v = water_index(*g)
    return dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
                std=float(v.std()))


# -- parameter priors ----------------------------------------------------------------------
@dataclass
class Prior:
    """1-sigma uncertainty of every parameter, plus the depth model.

    sigma: dict param -> 1 sigma (same units as the config; tilts in radians).
    depth_global: 1-sigma of log(scene depth / assumed depth), shared by all pixels of a
        frame (fixed-depth mode: how well z0 is known).
    depth_local: 1-sigma relative per-pixel depth error (scene relief around z0 in
        fixed-depth mode; depth-map noise in depth-map mode), independent per pixel.
    """
    sigma: dict
    depth_global: float = 0.25
    depth_local: float = 0.10
    notes: dict = field(default_factory=dict)

    def vector(self):
        return np.array([self.sigma.get(k, 0.0) for k in PARAMS], dtype=np.float64)


def default_prior(cfg=None, overrides=None):
    """Default 1-sigma priors (see docs/explore_uncertainty.md, section 1, for sources).

    overrides: dict param -> sigma (tilt in degrees under the key 'tilt_deg'), or
    'depth_global' / 'depth_local'. A config `uncertainty:` section is read the same way.
    """
    w = water_index_range()
    sig = dict(
        d0=2e-3,            # entrance-pupil position + seating: +-2 mm (1 sigma)
        tglass=0.2e-3,      # +-10 % of a 2 mm port
        mu_g=0.03,          # glass type unknown (fused silica 1.458 .. BK7 1.517, PMMA 1.49)
        mu_w=w["std"],      # uniform S 0-40 PSU, T 0-30 C, lambda 450-650 nm
        tilt_x=math.radians(0.5), tilt_y=math.radians(0.5),  # seating / port bonding
        fx=1.0, fy=1.0, cx=1.5, cy=1.5,     # in-air calibration (px)
        k1=2e-3, k2=1e-3, p1=2e-4, p2=2e-4,
    )
    prior = Prior(sigma=sig)
    ov = dict((cfg or {}).get("uncertainty", {}) or {})
    ov.update(overrides or {})
    for k, v in ov.items():
        if k == "tilt_deg":
            sig["tilt_x"] = sig["tilt_y"] = math.radians(float(v))
        elif k in ("depth_global", "depth_local"):
            setattr(prior, k, float(v))
        elif k in sig:
            sig[k] = float(v)
    return prior


# -- the model in torch (differentiable in theta and w) ---------------------------------------
class Geometry:
    """Fixed (declared) quantities: nominal parameters, output intrinsics, projection centre."""

    def __init__(self, cfg_or_parts, zoom, canvas=None, center_z=0.0, z_ref=1.0, crop=None):
        from core.config import camera_params, distortion_params, housing_params
        cfg = cfg_or_parts
        self.W, self.H, fx, fy, cx, cy = camera_params(cfg)
        k = distortion_params(cfg)
        hs = housing_params(cfg)
        self.housing = hs
        self.distortion = k
        n = np.asarray(hs["n_port"], float)
        from core.flatport import _rotation_to_z
        self.R0 = _rotation_to_z(n / np.linalg.norm(n))
        self.mu_a = float(hs["mu_a"])
        self.theta0 = np.array([hs["d0"], hs["tglass"], hs["mu_g"], hs["mu_w"], 0.0, 0.0,
                                fx, fy, cx, cy, *k], dtype=np.float64)
        self.zoom = float(zoom)
        out_W, out_H, (ox, oy) = canvas if canvas is not None else (self.W, self.H, (0.0, 0.0))
        x0 = y0 = 0
        if crop is not None:  # (y0, y1, x0, x1) on the canvas
            y0, y1, x0, x1 = crop
            out_W, out_H = x1 - x0, y1 - y0
        self.out_W, self.out_H = int(out_W), int(out_H)
        self.origin = (ox + x0, oy + y0)        # source-frame coordinate of output (0, 0)
        self.f_out = (fx * self.zoom, fy * self.zoom)
        self.c_out = (cx - self.origin[0], cy - self.origin[1])
        self.center_z = float(center_z)
        self.z_ref = float(z_ref)
        # Virtual centre: rescale so that the image agrees with the camera-centre one at z_ref.
        self.vc_scale = 1.0 - self.center_z / self.z_ref

    def out_grid(self, stride=1):
        """Output pixel coordinates (qx, qy), each (h, w)."""
        xs = np.arange(0, self.out_W, stride, dtype=np.float64)
        ys = np.arange(0, self.out_H, stride, dtype=np.float64)
        return np.meshgrid(xs, ys)


def _rot_xy(ax, ay):
    """Rx(ax) @ Ry(ay) as nested lists of tensors (differentiable)."""
    ca, sa, cb, sb = torch.cos(ax), torch.sin(ax), torch.cos(ay), torch.sin(ay)
    z = torch.zeros_like(ax)
    o = torch.ones_like(ax)
    Rx = [[o, z, z], [z, ca, -sa], [z, sa, ca]]
    Ry = [[cb, z, sb], [z, o, z], [-sb, z, cb]]
    return [[sum(Rx[i][k] * Ry[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def undistort_radtan(xd, yd, k1, k2, p1, p2, iters=8):
    """Ideal normalised (x, y) with radtan(x, y) = (xd, yd), by Newton (differentiable)."""
    x, y = xd, yd
    for _ in range(iters):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2
        fx_ = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) - xd
        fy_ = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y - yd
        drad = k1 + 2 * k2 * r2  # d rad / d r2
        j11 = rad + 2 * x * x * drad + 2 * p1 * y + 6 * p2 * x
        j12 = 2 * x * y * drad + 2 * p1 * x + 2 * p2 * y
        j21 = 2 * x * y * drad + 2 * p1 * x + 2 * p2 * y
        j22 = rad + 2 * y * y * drad + 6 * p1 * y + 2 * p2 * x
        det = j11 * j22 - j12 * j21
        x = x - (j22 * fx_ - j12 * fy_) / det
        y = y - (-j21 * fx_ + j11 * fy_) / det
    return x, y


def distort_radtan(x, y, k1, k2, p1, p2):
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    return (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def flatport_forward(x, y, w, d0, tg, mu_g, mu_w, tilt_x, tilt_y, mu_a, R0):
    """F(p, w) = A(p) + B(p) w of core.flatport.FlatPort, in torch, differentiable in all
    arguments. R0: nominal rotation (3x3 numpy) taking n_port to e_z."""
    Rt = _rot_xy(tilt_x, tilt_y)
    R0t = torch.as_tensor(np.asarray(R0, np.float64), dtype=x.dtype, device=x.device)
    R = [[Rt[i][0] * R0t[0, j] + Rt[i][1] * R0t[1, j] + Rt[i][2] * R0t[2, j]
          for j in range(3)] for i in range(3)]
    dz = R[2][0] * x + R[2][1] * y + R[2][2]
    tx = (R[0][0] * x + R[0][1] * y + R[0][2]) / dz
    ty = (R[1][0] * x + R[1][1] * y + R[1][2]) / dz
    s = tx * tx + ty * ty
    rg, rw = mu_a / mu_g, mu_a / mu_w
    kg = rg / torch.sqrt(1 + (1 - rg * rg) * s)
    kw = rw / torch.sqrt(1 + (1 - rw * rw) * s)
    a = d0 + tg * kg
    hz = d0 + tg
    ux, uy = kw * tx, kw * ty
    wx = R[0][0] * ux + R[1][0] * uy + R[2][0]
    wy = R[0][1] * ux + R[1][1] * uy + R[2][1]
    wz = R[0][2] * ux + R[1][2] * uy + R[2][2]
    vx, vy = a * tx, a * ty
    px = R[0][0] * vx + R[1][0] * vy + R[2][0] * hz
    py = R[0][1] * vx + R[1][1] * vy + R[2][1] * hz
    pz = R[0][2] * vx + R[1][2] * vy + R[2][2] * hz
    Ax, Ay = wx / wz, wy / wz
    return Ax + (px - pz * Ax) * w, Ay + (py - pz * Ay) * w


def corrected_xy(theta, ur, vr, w, geo):
    """Corrected-image position (qx, qy) of raw pixel (ur, vr) seeing inverse depth w, for
    true parameters theta (tensor (P,) in PARAMS order) and declared geometry geo."""
    d0, tg, mu_g, mu_w, tx, ty, fx, fy, cx, cy, k1, k2, p1, p2 = theta.unbind(0)
    x, y = undistort_radtan((ur - cx) / fx, (vr - cy) / fy, k1, k2, p1, p2)
    mu_a = torch.full((), geo.mu_a, dtype=x.dtype, device=x.device)
    X, Y = flatport_forward(x, y, w, d0, tg, mu_g, mu_w, tx, ty, mu_a, geo.R0)
    if geo.center_z != 0.0:
        s = geo.vc_scale / (1.0 - geo.center_z * w)
        X, Y = X * s, Y * s
    return geo.c_out[0] + geo.f_out[0] * X, geo.c_out[1] + geo.f_out[1] * Y


def raw_from_ideal(xs, ys, geo):
    """Raw (distorted) pixel of ideal source pixel coordinates (xs, ys) (numpy, nominal)."""
    fx, fy, cx, cy = geo.theta0[6:10]
    k1, k2, p1, p2 = geo.theta0[10:14]
    x, y = (xs - cx) / fx, (ys - cy) / fy
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return fx * xd + cx, fy * yd + cy


def _jac_chunk(geo, th, u, v, ww):
    """(J (n, 2, P+1), q (n, 2)) by forward-mode AD, all P+1 tangents in one vmapped pass."""
    P = len(PARAMS)
    f = lambda t, wv: torch.stack(corrected_xy(t, u, v, wv, geo), -1)
    DT = torch.zeros(P + 1, P, dtype=th.dtype, device=th.device)
    DT[:P, :P] = torch.eye(P, dtype=th.dtype, device=th.device)
    DW = torch.zeros(P + 1, 1, dtype=th.dtype, device=th.device)
    DW[P] = 1.0
    jv = lambda dt, dw: torch.func.jvp(f, (th, ww), (dt, dw.expand_as(ww)))[1]
    J = torch.func.vmap(jv)(DT, DW)                     # (P+1, n, 2)
    return J.permute(1, 2, 0), f(th, ww)


def jacobian_torch(geo, ur, vr, w, device="cpu", dtype=torch.float64, chunk=1 << 19):
    """Per-pixel Jacobian of the corrected position w.r.t. theta and w (forward-mode
    autodiff). ur, vr, w: numpy arrays of the same shape (raw pixel, inverse depth).
    Returns (J, q): J (N, 2, P+1) numpy (last column d/dw), q (N, 2) corrected position at
    theta0 (should reproduce the output grid)."""
    ur, vr, w = (np.ravel(np.asarray(a, np.float64)) for a in np.broadcast_arrays(ur, vr, w))
    th = torch.as_tensor(geo.theta0, dtype=dtype, device=device)
    P = len(PARAMS)
    Js, qs = [], []
    for s in range(0, len(ur), chunk):
        u = torch.as_tensor(ur[s:s + chunk], dtype=dtype, device=device)
        v = torch.as_tensor(vr[s:s + chunk], dtype=dtype, device=device)
        ww = torch.as_tensor(w[s:s + chunk], dtype=dtype, device=device)

        J, q = _jac_chunk(geo, th, u, v, ww)
        Js.append(J.cpu().numpy())
        qs.append(q.cpu().numpy())
    return np.concatenate(Js).astype(np.float64), np.concatenate(qs).astype(np.float64)


# -- closed-form Jacobian (on-axis port) -------------------------------------------------------
def analytic_jacobian(geo, xs, ys, w):
    """Closed-form d q / d (theta, w) for an on-axis port, at ideal source pixels (xs, ys)
    with inverse depth w (numpy float64). Tilt columns are zero (not modelled here).
    Returns J (N, 2, P+1)."""
    d0, tg, mu_g, mu_w = geo.theta0[:4]
    fx, fy, cx, cy, k1, k2, p1, p2 = geo.theta0[6:]
    mu_a = geo.mu_a
    x = (np.ravel(xs) - cx) / fx
    y = (np.ravel(ys) - cy) / fy
    w = np.broadcast_to(np.asarray(w, float), np.shape(xs)).ravel()
    s = x * x + y * y
    rg, rw = mu_a / mu_g, mu_a / mu_w
    cg, cw = 1 - rg * rg, 1 - rw * rw
    kg = rg / np.sqrt(1 + cg * s)
    kw = rw / np.sqrt(1 + cw * s)
    dz = d0 + tg
    h = d0 + tg * kg - dz * kw
    g = kw + w * h
    # dk/dr at fixed s: (1 + s) / (1 + c s)^1.5 ; dr/dmu = -mu_a / mu^2
    dkw_dmu = (1 + s) / (1 + cw * s) ** 1.5 * (-mu_a / mu_w ** 2)
    dkg_dmu = (1 + s) / (1 + cg * s) ** 1.5 * (-mu_a / mu_g ** 2)
    dg = dict(d0=w * (1 - kw), tglass=w * (kg - kw), mu_g=w * tg * dkg_dmu,
              mu_w=(1 - w * dz) * dkw_dmu)
    # dg/ds for dF/dp = g I + 2 g' p p^T
    dkw_ds = -0.5 * cw * kw / (1 + cw * s)
    dkg_ds = -0.5 * cg * kg / (1 + cg * s)
    gs = dkw_ds + w * (tg * dkg_ds - dz * dkw_ds)
    Fp = np.empty((len(x), 2, 2))
    Fp[:, 0, 0] = g + 2 * gs * x * x
    Fp[:, 0, 1] = Fp[:, 1, 0] = 2 * gs * x * y
    Fp[:, 1, 1] = g + 2 * gs * y * y
    # Implicit derivative of p(u) for fixed raw u: radtan d(p) with Jacobian D,
    # u = f d + c  ->  D dp = -(d df/f + dc/f + dd/dk dk)
    r2 = s
    rad = 1 + k1 * r2 + k2 * r2 * r2
    drad = k1 + 2 * k2 * r2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    D = np.empty((len(x), 2, 2))
    D[:, 0, 0] = rad + 2 * x * x * drad + 2 * p1 * y + 6 * p2 * x
    D[:, 0, 1] = 2 * x * y * drad + 2 * p1 * x + 2 * p2 * y
    D[:, 1, 0] = D[:, 0, 1]
    D[:, 1, 1] = rad + 2 * y * y * drad + 6 * p1 * y + 2 * p2 * x
    rhs = {  # d(raw-normalised residual)/d param
        "fx": (xd / fx, 0 * x), "fy": (0 * x, yd / fy), "cx": (1 / fx + 0 * x, 0 * x),
        "cy": (0 * x, 1 / fy + 0 * x), "k1": (x * r2, y * r2), "k2": (x * r2 ** 2, y * r2 ** 2),
        "p1": (2 * x * y, r2 + 2 * y * y), "p2": (r2 + 2 * x * x, 2 * x * y),
    }
    Dinv = np.linalg.inv(D)
    J = np.zeros((len(x), 2, len(PARAMS) + 1))
    sc = np.ones_like(w)
    if geo.center_z != 0.0:
        sc = geo.vc_scale / (1 - geo.center_z * w)
    f_out = np.array(geo.f_out)
    for i, k in enumerate(PARAMS):
        if k in dg:
            col = np.stack([x * dg[k], y * dg[k]], -1)
        elif k in rhs:
            b = np.stack(rhs[k], -1)
            dp = -np.einsum("nij,nj->ni", Dinv, b)
            col = np.einsum("nij,nj->ni", Fp, dp)
        else:
            continue
        J[:, :, i] = col * sc[:, None] * f_out
    dFw = np.stack([x * h, y * h], -1)
    if geo.center_z != 0.0:
        F = np.stack([x * g, y * g], -1)
        dFw = (dFw + F * (geo.center_z / (1 - geo.center_z * w))[:, None])
    J[:, :, -1] = dFw * sc[:, None] * f_out
    return J


# -- absorbable part ---------------------------------------------------------------------
def basis(qx, qy, geo, sets=("P4",), w=None):
    """Displacement fields (N, 2, K) in output px that a pinhole camera absorbs.

    P4: fx, fy (relative), cx, cy of the output intrinsics; R3: a small constant rotation
    (absorbed by the poses); T3: a translation of the projection centre (fields scale with
    the inverse depth w; constant w makes them collinear with P4)."""
    fxo, fyo = geo.f_out
    x = (np.ravel(qx) - geo.c_out[0]) / fxo
    y = (np.ravel(qy) - geo.c_out[1]) / fyo
    o, z = np.ones_like(x), np.zeros_like(x)
    cols = []
    if "P4" in sets:
        cols += [(fxo * x, z), (z, fyo * y), (o, z), (z, o)]
    if "Z1" in sets:  # isotropic zoom + principal point (3 dof)
        cols += [(fxo * x, fyo * y), (o, z), (z, o)]
    if "R3" in sets:
        cols += [(-fxo * x * y, -fyo * (1 + y * y)), (fxo * (1 + x * x), fyo * x * y),
                 (-fxo * y, fyo * x)]
    if "T3" in sets:
        ww = np.broadcast_to(np.asarray(w, float), x.shape)
        cols += [(-fxo * ww, z), (z, -fyo * ww), (fxo * x * ww, fyo * y * ww)]
    return np.stack([np.stack(c, -1) for c in cols], -1)


def absorb(J, Bs, mask=None):
    """Least-squares split of fields J (N, 2, M) into Bs @ beta + residual over the pixels
    in mask. Returns (residual (N, 2, M), beta (K, M))."""
    N, _, M = J.shape
    m = np.isfinite(J).all(axis=(1, 2)) & np.isfinite(Bs).all(axis=(1, 2))
    if mask is not None:
        m &= np.ravel(mask)
    A = Bs[m].transpose(0, 1, 2).reshape(-1, Bs.shape[-1])     # (2n, K)
    b = J[m].reshape(-1, M)                                      # (2n, M)
    beta, *_ = np.linalg.lstsq(A, b, rcond=1e-10)
    res = J - np.einsum("nck,km->ncm", Bs, beta)
    return res, beta


def covariance(J, sig):
    """sum_i sig_i^2 J_i J_i^T per pixel: (N, 3) = (sxx, sxy, syy)."""
    Js = J * np.asarray(sig)[None, None, :]
    return np.stack([(Js[:, 0] ** 2).sum(-1), (Js[:, 0] * Js[:, 1]).sum(-1),
                     (Js[:, 1] ** 2).sum(-1)], -1)


def field_stats(F, mask=None):
    """RMS and max of |F| (N, 2) over finite (and masked) pixels."""
    n = np.hypot(F[:, 0], F[:, 1])
    m = np.isfinite(n) if mask is None else (np.isfinite(n) & np.ravel(mask))
    return float(np.sqrt(np.mean(n[m] ** 2))), float(n[m].max())


# -- set-up shared by the CLI and the analysis scripts -------------------------------------
def optimal_center_z(cfg, z_range=(0.5, 8.0), n_depths=9, stride=16):
    """Axial projection centre minimising the depth dependence of the corrected image over
    z_range (the "virtual centre" of explore/depth, core/virtual_center.py there)."""
    from scipy.optimize import minimize_scalar
    from core.config import camera_params, housing_params
    from core.flatport import FlatPort
    W, H, fx, fy, cx, cy = camera_params(cfg)
    m = FlatPort(**housing_params(cfg))
    X, Y = np.meshgrid((np.arange(0, W, stride) - cx) / fx, (np.arange(0, H, stride) - cy) / fy)
    ws = 1.0 / np.geomspace(*z_range, n_depths)
    w_ref = 1.0 / np.sqrt(z_range[0] * z_range[1])
    Fs = [np.stack(m.forward(X, Y, w)) for w in ws]
    Fr = np.stack(m.forward(X, Y, w_ref))

    def cost(c):
        return np.sqrt(np.mean([np.nanmean(((F / (1 - c * w) - Fr / (1 - c * w_ref)) ** 2)
                                           .sum(0)) for F, w in zip(Fs, ws)]))
    d0 = m.d0
    return float(minimize_scalar(cost, bounds=(-5 * d0, d0), method="bounded",
                                 options={"xatol": 1e-7}).x)


def setup(cfg, z0=None, center_z=0.0):
    """Zoom and canvas exactly as remove_refraction.py chooses them, the Geometry, and an
    InverseMapper that returns ideal source pixels for a depth (scalar or map).
    center_z != 0: virtual-centre projection on the same canvas (zoom rescaled so the image
    agrees with the camera-centre one at z0)."""
    from core.config import camera_params, distortion_params, housing_params
    from core.scale import select_zoom
    from core.undistort import InverseMapper
    W, H, fx, fy, cx, cy = camera_params(cfg)
    hs, dist = housing_params(cfg), distortion_params(cfg)
    corr = cfg.get("correction", {})
    z0 = float(corr.get("z0_fixed", 1.0) if z0 is None else z0)
    fit = corr.get("fit_canvas", True)
    zoom = corr.get("zoom")
    if zoom is None:
        zoom, _ = select_zoom(z0, W, H, fx, fy, cx, cy, hs, in_bounds=not fit)
    canvas = (W, H, (0.0, 0.0))
    if fit:
        import remove_refraction as rr
        canvas, _ = rr._fit_canvas(W, H, fx, fy, cx, cy, hs, dist, zoom, z0)
    geo = Geometry(cfg, zoom, canvas, center_z=center_z, z_ref=z0)
    return geo, canvas


def source_pixels(geo, canvas, depth, crop=None):
    """Ideal source pixel (xs, ys) of every output pixel (NaN where invalid), for a scalar
    depth or a per-source-pixel depth map (camera-centre projection for maps)."""
    from core.config import camera_params
    from core.undistort import InverseMapper
    zoom = geo.zoom
    if geo.center_z != 0.0:
        if np.ndim(depth) != 0:
            raise NotImplementedError("virtual centre with depth maps")
        zoom = zoom * geo.vc_scale / (1.0 - geo.center_z / float(depth))
    fx, fy, cx, cy = geo.theta0[6:10]
    mp = InverseMapper(geo.W, geo.H, fx, fy, cx, cy, geo.housing, zoom, canvas)
    xs, ys = mp.build(depth)
    if crop is not None:
        y0, y1, x0, x1 = crop
        xs, ys = xs[y0:y1, x0:x1], ys[y0:y1, x0:x1]
    return xs, ys


def w_rms(log_sigma):
    """RMS of w - w0 (relative to w0) when Z is log-normal with log-sigma log_sigma about
    z0 (w = w0 exp(-g)). The model is affine in w, so with this value the depth part of
    the propagated mean-square error is exact for any spread (a plain first-order
    propagation, log_sigma itself, underestimates large spreads: x1.5 at 0.7)."""
    s2 = float(log_sigma) ** 2
    return math.sqrt(math.exp(2 * s2) - 2 * math.exp(0.5 * s2) + 1)


def propagate(geo, ur, vr, w, prior, qx, qy, theta_sets=("P4",), nodes=3, device="cpu",
              dtype=torch.float64, return_parts=False):
    """First-order per-pixel covariances (full, non-absorbable), each (N, 3) px^2.

    ur, vr: raw pixel of each output pixel; w: inverse depth the correction used there
    (scalar or per pixel); qx, qy: the output pixel.

    * Parameters theta (prior.sigma): J Sigma J^T. Their residual after the least-squares
      fit of `theta_sets` (default P4: fx, fy, cx, cy) over these pixels is the
      non-absorbable part (shared by all frames: one intrinsics set per sequence).
      With nodes > 1 the theta term is averaged over the global depth uncertainty by
      Gauss-Hermite quadrature in log-depth (the sensitivities scale with 1/Z).
    * Global depth error (prior.depth_global, log-normal, per frame): J_w w s_g with the
      exact log-normal RMS of w; absorbable by a per-frame translation of the camera
      (T3 fields at the actual inverse depth; isotropic zoom + shift at constant depth).
    * Local depth error (prior.depth_local, independent per pixel, relative to the true
      depth, so it grows by exp(s_g^2) in RMS): never absorbable.
    """
    ur = np.ravel(ur)
    w = np.broadcast_to(np.asarray(w, np.float64), np.shape(ur)).ravel()
    qx, qy = np.ravel(qx), np.ravel(qy)
    sig = prior.vector()
    if nodes > 1 and prior.depth_global > 0:
        x, a = np.polynomial.hermite_e.hermegauss(nodes)
        a = a / a.sum()
    else:
        x, a = np.zeros(1), np.ones(1)
    Bt = basis(qx, qy, geo, theta_sets)
    full_t = np.zeros((len(ur), 3))
    np_t = np.zeros((len(ur), 3))
    Jw = None
    for xk, ak in zip(x, a):
        J, _ = jacobian_torch(geo, ur, vr, w * math.exp(-prior.depth_global * xk),
                              device=device, dtype=dtype)
        if xk == 0.0:
            Jw = J[:, :, -1:]
        Ft = J[:, :, :-1] * sig
        rt, _ = absorb(Ft, Bt)
        full_t += ak * covariance(Ft, np.ones(len(sig)))
        np_t += ak * covariance(rt, np.ones(len(sig)))
    Fg = Jw * (w * w_rms(prior.depth_global))[:, None, None]
    # Relative relief is relative to the true scene depth: E[(w0 e^-g (e^-l - 1))^2]
    Fl = Jw * (w * w_rms(prior.depth_local) * math.exp(prior.depth_global ** 2))[:, None, None]
    rg, _ = absorb(Fg, basis(qx, qy, geo, ("T3",), w=w))
    cg, cgn, cl = (covariance(F, [1.0]) for F in (Fg, rg, Fl))
    full, npc = full_t + cg + cl, np_t + cgn + cl
    if return_parts:
        return full, npc, dict(theta=full_t, theta_np=np_t, zg=cg, zg_np=cgn, zl=cl)
    return full, npc


# -- full-resolution maps on a torch device (CLI) -------------------------------------------
def basis_torch(qx, qy, geo, sets, w=None):
    """basis() on torch tensors (N,) -> (N, 2, K)."""
    fxo, fyo = geo.f_out
    x = (qx - geo.c_out[0]) / fxo
    y = (qy - geo.c_out[1]) / fyo
    o, z = torch.ones_like(x), torch.zeros_like(x)
    cols = []
    if "P4" in sets:
        cols += [(fxo * x, z), (z, fyo * y), (o, z), (z, o)]
    if "R3" in sets:
        cols += [(-fxo * x * y, -fyo * (1 + y * y)), (fxo * (1 + x * x), fyo * x * y),
                 (-fxo * y, fyo * x)]
    if "T3" in sets:
        cols += [(-fxo * w, z), (z, -fyo * w), (fxo * x * w, fyo * y * w)]
    return torch.stack([torch.stack(c, -1) for c in cols], -1)


def _quad(x, y):
    """Coefficients (C0, C1, C2), each (N, 3), of sum_i (x_i + w y_i)(x_i + w y_i)^T."""
    def c(u, v):
        return np.stack([(u[:, 0] * v[:, 0]).sum(-1),
                         0.5 * ((u[:, 0] * v[:, 1]).sum(-1) + (v[:, 0] * u[:, 1]).sum(-1)),
                         (u[:, 1] * v[:, 1]).sum(-1)], -1)
    return c(x, x), 2 * c(x, y), c(y, y)


class FastCov:
    """Per-frame covariance maps in a few elementwise operations.

    F = A(p) + B(p) w is affine in w, so at a fixed source p every parameter column is
    J_i(w) = a_i + w b_i, and sum_i J_i J_i^T = M0 + w M1 + w^2 M2. The absorbed part is
    fitted once per sequence on the reference image (depth z0), separately for the a- and
    b-parts: the a-part is a change of the intrinsics; the b-part, proportional to w, is
    (to first order) a shift of the projection centre, which the per-frame pose takes.
    The source p of an output pixel moves only slightly with the depth, so the terms are
    precomputed at the reference sources; a frame then needs only its inverse depth per
    output pixel. Camera-centre projection.

    ur, vr, qx, qy: raw source pixel and output pixel of every reference-valid pixel.
    """

    def __init__(self, geo, ur, vr, qx, qy, prior, theta_sets=("P4",), device="cpu",
                 dtype=torch.float32, fit_stride=4):
        if geo.center_z != 0.0:
            raise NotImplementedError("FastCov: camera-centre projection only")
        self.geo, self.prior, self.sets = geo, prior, theta_sets
        self.device, self.dtype = torch.device(device), dtype
        sl = slice(None, None, fit_stride)
        a, b, _ = self._ab(ur[sl], vr[sl])
        Bs = basis(qx[sl], qy[sl], geo, theta_sets)
        _, self.beta_a = absorb(a, Bs)
        _, self.beta_b = absorb(b, Bs)
        self.M, self.N, self.Jw = self.terms(ur, vr, qx, qy)
        self.qx, self.qy = self._t(qx), self._t(qy)

    def _t(self, z):
        return torch.as_tensor(np.ascontiguousarray(z), dtype=self.dtype, device=self.device)

    def _ab(self, ur, vr):
        sig = self.prior.vector()
        kw = dict(device=self.device, dtype=self.dtype)
        J0, _ = jacobian_torch(self.geo, ur, vr, np.zeros(len(ur)), **kw)
        J1, _ = jacobian_torch(self.geo, ur, vr, np.ones(len(ur)), **kw)
        return J0[:, :, :-1] * sig, (J1 - J0)[:, :, :-1] * sig, J0[:, :, -1]

    def terms(self, ur, vr, qx, qy):
        """(M, N, Jw) device tensors for the given pixels (full / non-absorbable
        quadratic-in-w coefficients, dq/dw)."""
        a, b, Jw = self._ab(ur, vr)
        Bs = basis(qx, qy, self.geo, self.sets)
        ra = a - np.einsum("nck,kp->ncp", Bs, self.beta_a)
        rb = b - np.einsum("nck,kp->ncp", Bs, self.beta_b)
        return ([self._t(m) for m in _quad(a, b)], [self._t(m) for m in _quad(ra, rb)],
                self._t(Jw))

    def __call__(self, w, terms=None, qxy=None, fit_stride=16):
        """w: (N,) inverse depth per output pixel. Returns (full, np) tensors (N, 3).
        terms/qxy: (M, N, Jw) and (qx, qy) tensors to use instead of the precomputed ones."""
        M, Nn, Jw = terms if terms is not None else (self.M, self.N, self.Jw)
        qx, qy = qxy if qxy is not None else (self.qx, self.qy)
        w = torch.as_tensor(w, dtype=self.dtype, device=self.device)
        pr = self.prior
        W1 = w[:, None]
        full = M[0] + W1 * M[1] + W1 * W1 * M[2]
        npc = Nn[0] + W1 * Nn[1] + W1 * W1 * Nn[2]
        sg = w_rms(pr.depth_global)
        sl = w_rms(pr.depth_local) * math.exp(pr.depth_global ** 2)
        Fg = Jw * (w * sg)[:, None]
        Fl = Jw * (w * sl)[:, None]
        # Per-frame depth scale: absorbed by a translation of this frame's camera.
        Bg = basis_torch(qx, qy, self.geo, ("T3",), w=w)
        A = Bg[::fit_stride].reshape(-1, 3)        # normal equations on the device
        b = Fg[::fit_stride].reshape(-1, 1)
        AtA, Atb = (A.T @ A).cpu().double(), (A.T @ b).cpu().double()
        bg = torch.linalg.lstsq(AtA, Atb).solution
        rg = Fg - (Bg @ bg.to(dtype=self.dtype, device=self.device))[..., 0]
        sq = lambda F: torch.stack([F[:, 0] ** 2, F[:, 0] * F[:, 1], F[:, 1] ** 2], -1)
        return full + sq(Fg) + sq(Fl), npc + sq(rg) + sq(Fl)
