"""Synthetic multi-view test of the corrections: does any of them bend a bundle adjustment?

    python depth_aware/synthetic_ba.py [--config configs/dataset.yaml] [--scale 0.5]

A camera behind the configured flat port flies a three-leg lawn-mower pattern at constant
altitude over a synthetic scene (a flat floor, as in the gopro10 tank, or a reef with
bommies). Scene points are projected exactly through the housing (pixel = inverse of the
analytic flat-port model at the point's own depth) and get 0.3 px of noise. The
observations are then corrected the way Refrax would correct them:

  fixed       one depth z0 for every pixel (the paper's method)
  median      one depth per image: the median true depth of the image
  dense       the true depth of every observation (depth-aware, exact)
  dense10     dense, with a smooth 10 % relative depth error, independent per image
              (what per-image mono-depth fits give)

plus two positive controls that must bend: "none" (no refraction correction, the ideal
pixels used as a pinhole with focal mu_w * f) and "fixed_z0/3" (a wrong constant depth).

each with the corrected pinhole at the camera centre (c = 0) and at the virtual centre
(c = core.virtual_center.optimal_center_z). A pinhole bundle adjustment (fixed
intrinsics) is solved from a perturbed start, and the result is compared with the truth:
reprojection RMS, curvature kappa of the camera trajectory and of the scene (paper,
Eq. 17-18, the same as evaluation/planarity.py) and the point error after a similarity
alignment. This isolates the geometry of the correction from COLMAP's feature matching
and initialisation. Writes docs/figures/synthetic_ba.json.
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from common import REPO, load_cfg
from core.config import camera_params, housing_params
from core.flatport import FlatPort
from core.scale import select_zoom
from core.virtual_center import optimal_center_z

FIG = os.path.join(REPO, "docs", "figures")


# ------------------------------------------------------------------ scene and cameras
def surface(kind):
    """Seabed height above the datum (m) as a function of world (X, Y)."""
    if kind == "flat":
        return lambda X, Y: np.zeros_like(X)
    if kind == "reef":
        def h(X, Y):
            z = 0.08 * np.sin(2.1 * X) * np.cos(1.7 * Y)
            z += 0.40 * np.exp(-((X - 0.4) ** 2 + (Y - 0.1) ** 2) / (2 * 0.25 ** 2))
            z += 0.30 * np.exp(-((X + 0.8) ** 2 + (Y + 0.3) ** 2) / (2 * 0.20 ** 2))
            return z
        return h
    raise ValueError(kind)


def lawnmower(altitude, n_per_leg=15, spacing=0.15, leg_gap=0.4, n_legs=3, seed=0):
    """World-to-camera rotations and centres; camera z looks down, small attitude noise."""
    rng = np.random.default_rng(seed)
    R0 = np.diag([1.0, -1.0, -1.0])  # camera x = world x, camera z = world -Z
    Rs, Cs = [], []
    for leg in range(n_legs):
        xs = (np.arange(n_per_leg) - (n_per_leg - 1) / 2) * spacing
        if leg % 2:
            xs = xs[::-1]
        for x in xs:
            d = Rotation.from_euler("xyz", rng.normal(0, 3, 3), degrees=True).as_matrix()
            Rs.append(d @ R0)
            Cs.append(np.array([x, (leg - (n_legs - 1) / 2) * leg_gap, altitude]))
    return np.array(Rs), np.array(Cs)


# ------------------------------------------------------------------ refraction
def project_through_port(model, Pc):
    """Normalised ideal pixel p with F(p, 1/Z) = (X/Z, Y/Z) (port on the optical axis)."""
    X, Y, Z = Pc[:, 0] / Pc[:, 2], Pc[:, 1] / Pc[:, 2], Pc[:, 2]
    R = np.hypot(X, Y)
    w = 1.0 / Z
    g0, _, _ = model.radial_gain(0.0, w)
    rho = R / g0
    for _ in range(50):
        g, dg, _ = model.radial_gain(rho * rho, w)
        step = (rho * g - R) / dg
        rho -= step
        if np.abs(step).max() < 1e-14:
            break
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.where(R[:, None] > 0, np.stack([X, Y], 1) / R[:, None], 0.0)
    return rho[:, None] * e


def smooth_field(rng, amp):
    """Smooth relative error field e(x, y) on normalised image coordinates, RMS ~ amp."""
    k = rng.normal(0, 1.2, (4, 2))
    ph = rng.uniform(0, 2 * np.pi, 4)
    a = rng.normal(0, 1, 4)
    a *= amp / np.sqrt(0.5 * np.sum(a ** 2))

    def f(p):
        return sum(a[i] * np.sin(p @ k[i] + ph[i]) for i in range(4))
    return f


# ------------------------------------------------------------------ bundle adjustment
def ba(obs_cam, obs_pt, obs_xy, R_init, C_init, X_init, f_out, c):
    """Pinhole BA from the virtual centre c (camera centres = virtual centres)."""
    nc, npnt = len(R_init), len(X_init)
    r0 = Rotation.from_matrix(R_init).as_rotvec()
    x0 = np.concatenate([r0.ravel(), C_init.ravel(), X_init.ravel()])

    def unpack(x):
        r = x[:3 * nc].reshape(nc, 3)
        C = x[3 * nc:6 * nc].reshape(nc, 3)
        P = x[6 * nc:].reshape(npnt, 3)
        return r, C, P

    def resid(x):
        r, C, P = unpack(x)
        Rm = Rotation.from_rotvec(r[obs_cam]).as_matrix()
        Pc = np.einsum("nij,nj->ni", Rm, P[obs_pt] - C[obs_cam])
        uv = f_out * Pc[:, :2] / Pc[:, 2:3]
        return (uv - obs_xy).ravel()

    m = len(obs_cam)
    A = lil_matrix((2 * m, len(x0)), dtype=int)
    i = np.arange(m)
    for k in range(3):
        for s in (0, 1):
            A[2 * i + s, 3 * obs_cam + k] = 1
            A[2 * i + s, 3 * nc + 3 * obs_cam + k] = 1
            A[2 * i + s, 6 * nc + 3 * obs_pt + k] = 1
    sol = least_squares(resid, x0, jac_sparsity=A, method="trf", x_scale="jac", ftol=1e-12,
                        xtol=1e-12, gtol=1e-12, max_nfev=200)
    r, C, P = unpack(sol.x)
    return C, P, float(np.sqrt(np.mean(sol.fun ** 2)))


def umeyama(src, dst):
    """Similarity (s, R, t) with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / a.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def kappa(P):
    """Dimensionless radial curvature of a point set (paper Eq. 15-18)."""
    mu = P.mean(0)
    _, _, Vt = np.linalg.svd(P - mu)
    n, u, v = Vt[2], Vt[0], Vt[1]
    r, a, b = (P - mu) @ n, (P - mu) @ u, (P - mu) @ v
    D = np.hypot(np.ptp(a), np.ptp(b))
    a, b = a / D, b / D  # normalised in-plane coordinates
    A = np.stack([np.ones_like(a), a, b, a * a + b * b], 1)
    coef = np.linalg.lstsq(A, r / D, rcond=None)[0]
    return float(coef[3])


# ------------------------------------------------------------------ experiment
def run(cfg, scale, scene, altitude, seed=0):
    W, H, fx, fy, cx, cy = camera_params(cfg)
    W, H, fx, fy = int(W * scale), int(H * scale), fx * scale, fy * scale
    cx, cy = (cx + 0.5) * scale - 0.5, (cy + 0.5) * scale - 0.5
    housing = housing_params(cfg)
    rng = np.random.default_rng(seed)
    h = surface(scene)
    Rs, Cs = lawnmower(altitude, seed=seed)
    # scene points over the surveyed area
    n_pts = 2500
    Xw = np.stack([rng.uniform(-1.6, 1.6, n_pts), rng.uniform(-0.9, 0.9, n_pts)], 1)
    Pw = np.concatenate([Xw, h(Xw[:, 0], Xw[:, 1])[:, None]], 1)

    model0 = FlatPort(**housing)
    obs = []  # (cam, pt, ideal normalised p (noisy), true depth Z)
    sigma = 0.3 / fx
    for k, (R, C) in enumerate(zip(Rs, Cs)):
        Pc = (Pw - C) @ R.T
        p = project_through_port(model0, Pc)
        u, v = fx * p[:, 0] + cx, fy * p[:, 1] + cy
        ok = (Pc[:, 2] > 0.05) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        for j in np.flatnonzero(ok):
            obs.append((k, j, p[j] + rng.normal(0, sigma, 2), Pc[j, 2]))
    cam = np.array([o[0] for o in obs])
    pt = np.array([o[1] for o in obs])
    p_obs = np.array([o[2] for o in obs])
    Z_true = np.array([o[3] for o in obs])
    cnt = np.bincount(pt, minlength=n_pts)
    keep_pt = cnt >= 3
    sel = keep_pt[pt]
    remap = -np.ones(n_pts, int)
    remap[keep_pt] = np.arange(keep_pt.sum())
    cam, pt, p_obs, Z_true = cam[sel], remap[pt[sel]], p_obs[sel], Z_true[sel]
    Pw = Pw[keep_pt]

    z0 = float(np.median(Z_true))
    c_opt, _, _ = optimal_center_z(W, H, fx, fy, cx, cy, housing, z_range=(0.5, 8.0))
    fields = [smooth_field(np.random.default_rng(1000 + seed * 100 + k), 0.10)
              for k in range(len(Rs))]
    med_img = np.array([np.median(Z_true[cam == k]) if np.any(cam == k) else z0
                        for k in range(len(Rs))])
    depth_modes = {
        "none": None,
        "fixed_z0/3": lambda: np.full(len(cam), z0 / 3),
        "fixed": lambda: np.full(len(cam), z0),
        "median": lambda: med_img[cam],
        "dense": lambda: Z_true,
        "dense10": lambda: Z_true * (1 + np.array([fields[k](p[None])[0]
                                                    for k, p in zip(cam, p_obs)])),
    }
    rows = []
    for c in (0.0, c_opt):
        zoom, _ = select_zoom(z0, W, H, fx, fy, cx, cy, housing, in_bounds=False, center_z=c)
        f_out = fx * zoom  # fx == fy up to 0.04 % here; normalised coordinates use fx
        model = FlatPort(**housing, center_z=c)
        axis = np.einsum("kij,j->ki", Rs.transpose(0, 2, 1), np.array([0, 0, 1.0]))
        Cv = Cs + c * axis  # virtual centres
        for name, dfun in depth_modes.items():
            if dfun is None:
                if c != 0.0:
                    continue
                obs_xy = fx * housing["mu_w"] * p_obs / housing["mu_a"]
                fo = fx * housing["mu_w"] / housing["mu_a"]
            else:
                D = dfun()
                q = np.stack(model.forward(p_obs[:, 0], p_obs[:, 1], 1.0 / (D - c)), 1)
                obs_xy = f_out * q
                fo = f_out
            rng2 = np.random.default_rng(7)
            R_init = np.array([Rotation.from_rotvec(rng2.normal(0, 0.005, 3)).as_matrix() @ R
                               for R in Rs])
            C_init = Cv + rng2.normal(0, 0.01, Cv.shape)
            X_init = Pw + rng2.normal(0, 0.01, Pw.shape)
            C_est, P_est, rms = ba(cam, pt, obs_xy, R_init, C_init, X_init, fo, c)
            s, Rr, t = umeyama(C_est, Cv)
            P_al = s * P_est @ Rr.T + t
            C_al = s * C_est @ Rr.T + t
            ext = np.ptp(Pw[:, :2], 0).max()
            rows.append(dict(
                scene=scene, altitude=altitude, center_mm=round(c * 1000, 2), depth=name,
                z0=round(z0, 3), reproj_px=round(rms, 4),
                kappa_traj=round(kappa(C_al), 4), kappa_traj_true=round(kappa(Cv), 4),
                kappa_scene=round(kappa(P_al), 4), kappa_scene_true=round(kappa(Pw), 4),
                traj_flat_rms_mm=round(1000 * float(np.std(C_al[:, 2] - Cv[:, 2])), 2),
                pt_err_rel=round(float(np.sqrt(np.mean(np.sum((P_al - Pw) ** 2, 1)))) / ext, 5),
                n_obs=int(len(cam)), n_pts=int(len(Pw)), n_cams=int(len(Rs))))
            print(rows[-1], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--scale", type=float, default=0.5, help="camera resolution scale")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    rows = []
    for scene, alt in (("flat", 1.0), ("reef", 1.4)):
        rows += run(cfg, args.scale, scene, alt)
    os.makedirs(FIG, exist_ok=True)
    json.dump(rows, open(os.path.join(FIG, "synthetic_ba.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
