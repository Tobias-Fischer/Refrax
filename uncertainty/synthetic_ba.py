"""Downstream check on synthetic scenes: does covariance weighting help bundle adjustment
when the housing parameters are mis-specified? (docs/explore_uncertainty.md, section 5)

    python uncertainty/synthetic_ba.py [--trials 8] [--scenes tank reef reef_depth]

Per trial the *true* parameters are drawn from the prior; images are simulated exactly
through the true flat-port model (raw pixels + 0.35 px raw keypoint noise, i.e. ~0.5 px in
the corrected image) and corrected with the *nominal* parameters, as Refrax would. A
pinhole BA (shared fx, fy, cx, cy; poses; points; scale fixed by the first baseline) is
then run with
    unweighted        identity covariance (what COLMAP does today),
    weighted np       Sigma_np + s^2 I    (non-absorbable covariance + keypoint noise),
    weighted full     Sigma_full + s^2 I,
    oracle            unweighted, corrected with the true parameters (noise floor),
    refractive BA     the raw observations with d0, mu_w, tilt_x, tilt_y as extra BA
                      parameters (the "estimate it" alternative; exact depth from the
                      current points),
each with intrinsics refined; unweighted / weighted np also with fixed intrinsics.

Scenes: tank (planar board 1.2 m at 0.7-1.3 m, hand-held views), reef (3-D terrain
0.6-2 m, down-looking survey) in fixed-depth mode (z0 = median depth), and reef_depth
(reef, depth-aware correction with 3 % per-pixel depth noise).
Writes docs/figures/synthetic_ba.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evaluation"))
from core.config import load_config  # noqa: E402
from core.flatport import FlatPort  # noqa: E402
from core import uncertainty as U  # noqa: E402
import planarity as pl  # noqa: E402

SIG_RAW = 0.35   # keypoint noise in the raw image (px)
SIG_KP = 0.5     # assumed keypoint noise in the corrected image (px) for the weights


# -- scenes ------------------------------------------------------------------------------
def look_at(C, target, rng, roll_deg=10.0):
    z = target - C
    z /= np.linalg.norm(z)
    up = np.array([0.0, -1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0, 0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])               # camera-from-world rotation
    R = Rotation.from_euler("z", rng.uniform(-roll_deg, roll_deg), degrees=True).as_matrix() @ R
    return R, -R @ C


def make_scene(kind, rng):
    if kind == "tank":
        X = np.c_[rng.uniform(-0.6, 0.6, (1500, 2)), np.zeros(1500)]
        cams = []
        for _ in range(40):
            d = rng.uniform(0.7, 1.3)
            az, el = rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(0, 35))
            C = d * np.array([np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), -np.cos(el)])
            cams.append(look_at(C, rng.uniform(-0.15, 0.15, 3) * [1, 1, 0], rng))
        return X, cams
    # reef: terrain z (down = +z in the world; cameras above at z < 0)
    n = 2500
    xy = rng.uniform(-1.6, 1.6, (n, 2))
    bumps = [(rng.uniform(-1.4, 1.4, 2), rng.uniform(0.2, 0.5), rng.uniform(-0.35, 0.35))
             for _ in range(8)]
    z = 0.15 * xy[:, 0] + sum(a * np.exp(-((xy - c) ** 2).sum(1) / (2 * s * s))
                              for c, s, a in bumps)
    X = np.c_[xy, z]
    cams = []
    for i, yline in enumerate(np.linspace(-0.9, 0.9, 5)):
        for x in np.linspace(-1.0, 1.0, 9)[:: 1 if i % 2 == 0 else -1]:
            C = np.array([x, yline, -1.1]) + rng.normal(0, 0.03, 3)
            tgt = C + np.array([*rng.normal(0, 0.1, 2), 1.0])
            cams.append(look_at(C, tgt, rng, roll_deg=5))
    return X, cams


# -- simulation --------------------------------------------------------------------------
def n_port_of(tilt_x, tilt_y):
    a, b = tilt_x, tilt_y
    return np.array([-np.sin(b) * np.cos(a), np.sin(a), np.cos(b) * np.cos(a)])


def raw_of(Xn, w, th, geo):
    """Raw pixels of normalised pinhole directions Xn (n, 2) at inverse depth w for true
    parameters th (exact inverse of the flat-port model + true lens)."""
    d0, tg, mg, mw, ax, ay, fx, fy, cx, cy, k1, k2, p1, p2 = th
    m = FlatPort(n_port_of(ax, ay), d0, tg, geo.mu_a, mg, mw)
    out = np.full((len(Xn), 2), np.nan)
    from core.flatport import _newton_const_w
    g, _, _ = m.radial_gain(0.0, w)
    x, y = _newton_const_w(m, Xn[:, 0], Xn[:, 1], w, Xn[:, 0] / g, Xn[:, 1] / g, 1e-13, 60)
    xd, yd = U.distort_radtan(x, y, k1, k2, p1, p2)
    out[:, 0], out[:, 1] = fx * xd + cx, fy * yd + cy
    return out


def corrected(th, u, w, geo):
    T = lambda a: torch.as_tensor(np.asarray(a, np.float64))
    qx, qy = U.corrected_xy(T(th), T(u[:, 0]), T(u[:, 1]), T(w), geo)
    return np.c_[qx.numpy(), qy.numpy()]


def simulate(X, cams, th_true, geo, rng, depth_mode, z0):
    """Observations: list of (cam, point, q_obs, u_obs, w_corr, q_true_pinhole)."""
    obs = []
    W, H = geo.W, geo.H
    for ci, (R, t) in enumerate(cams):
        Xc = X @ R.T + t
        ok = Xc[:, 2] > 0.2
        idx = np.flatnonzero(ok)
        Xn = Xc[idx, :2] / Xc[idx, 2:]
        fov = np.hypot(*Xn.T) < 1.0      # well beyond the corrected field of view (~0.85)
        idx, Xn = idx[fov], Xn[fov]
        w = 1.0 / Xc[idx, 2]
        u = raw_of(Xn, w, th_true, geo)
        q_pin = np.c_[geo.c_out[0] + geo.f_out[0] * Xn[:, 0],
                      geo.c_out[1] + geo.f_out[1] * Xn[:, 1]]
        inside = np.isfinite(u).all(1) & (u[:, 0] > 2) & (u[:, 0] < W - 3) & \
            (u[:, 1] > 2) & (u[:, 1] < H - 3)
        idx, u, w, q_pin = idx[inside], u[inside], w[inside], q_pin[inside]
        # Consistency of the simulator with the model at the truth; directions far outside
        # the field of view can make the Newton solve wander into the frame: drop them.
        chk = np.abs(corrected(th_true, u, w, geo) - q_pin).max(1)
        good = chk < 1e-6
        if (~good).sum() > 0.01 * len(good):
            raise RuntimeError(f"simulator: {(~good).sum()} of {len(good)} inconsistent")
        idx, u, w = idx[good], u[good], w[good]
        u_obs = u + rng.normal(0, SIG_RAW, u.shape)
        if depth_mode:
            w_corr = w * np.exp(rng.normal(0, 0.03, len(w)))
        else:
            w_corr = np.full(len(w), 1.0 / z0)
        for k in range(len(idx)):
            obs.append((ci, idx[k], u_obs[k], w_corr[k]))
    cam = np.array([o[0] for o in obs])
    pt = np.array([o[1] for o in obs])
    u_obs = np.array([o[2] for o in obs])
    w_corr = np.array([o[3] for o in obs])
    return cam, pt, u_obs, w_corr


# -- bundle adjustment ----------------------------------------------------------------------
class BA:
    """Pinhole BA in the corrected image: poses (rotation vector, translation), points,
    optional shared (fx, fy, cx, cy); cam 0 fixed, |C1 - C0| fixed (scale gauge).
    Residuals whitened with per-observation 2x2 covariances (None = identity).
    refr: optional dict(u=raw obs, geo, th0, idx=param indices) -> refractive BA where the
    observation is re-corrected with the current refraction parameters at the current depth.
    """

    def __init__(self, cams, X, cam, pt, q, intr, cov=None, refine_intr=True, baseline=None,
                 refr=None):
        self.cam, self.pt, self.q = cam, pt, q
        self.nc, self.npnt = len(cams), len(X)
        self.intr0 = np.asarray(intr, float)
        self.refine = refine_intr
        self.baseline = baseline
        self.refr = refr
        if cov is None:
            self.Lt = None
        else:
            Si = np.linalg.inv(cov)
            L = np.linalg.cholesky(Si)              # Si = L L^T
            self.Lt = np.transpose(L, (0, 2, 1))    # whitened r = L^T e
        self.x0 = self.pack(cams, X, self.intr0,
                            None if refr is None else refr["th0"][refr["idx"]])

    def pack(self, cams, X, intr, rp=None):
        poses = [np.r_[Rotation.from_matrix(R).as_rotvec(), t] for R, t in cams[1:]]
        parts = [np.concatenate(poses), X.ravel()]
        if self.refine:
            parts.append(intr)
        if rp is not None:
            parts.append(rp)
        self.R0, self.t0 = cams[0]
        return np.concatenate(parts)

    def unpack(self, x):
        nc = self.nc - 1
        P = x[:6 * nc].reshape(nc, 6)
        X = x[6 * nc:6 * nc + 3 * self.npnt].reshape(-1, 3)
        o = 6 * nc + 3 * self.npnt
        intr = x[o:o + 4] if self.refine else self.intr0
        o += 4 if self.refine else 0
        rp = x[o:] if self.refr is not None else None
        Rs = np.concatenate([self.R0[None], Rotation.from_rotvec(P[:, :3]).as_matrix()])
        ts = np.concatenate([self.t0[None], P[:, 3:]])
        return Rs, ts, X, intr, rp

    def residual(self, x, whiten=True):
        Rs, ts, X, intr, rp = self.unpack(x)
        Xc = np.einsum("nij,nj->ni", Rs[self.cam], X[self.pt]) + ts[self.cam]
        fx, fy, cx, cy = intr
        pred = np.c_[cx + fx * Xc[:, 0] / Xc[:, 2], cy + fy * Xc[:, 1] / Xc[:, 2]]
        q = self.q
        if self.refr is not None:
            th = self.refr["th0"].copy()
            th[self.refr["idx"]] = rp
            q = corrected(th, self.refr["u"], 1.0 / Xc[:, 2], self.refr["geo"])
        e = q - pred
        if whiten and self.Lt is not None:
            e = np.einsum("nij,nj->ni", self.Lt, e)
        r = e.ravel()
        if self.baseline is not None:
            C0 = -Rs[0].T @ ts[0]
            C1 = -Rs[1].T @ ts[1]
            r = np.r_[r, 1e4 * (np.linalg.norm(C1 - C0) - self.baseline)]
        return r

    def sparsity(self):
        n = len(self.cam)
        nc = self.nc - 1
        rows, cols = [], []
        ar = np.arange(n)
        for k in range(2):
            rr = 2 * ar + k
            c = self.cam - 1
            m = c >= 0
            for j in range(6):
                rows.append(rr[m]); cols.append(6 * c[m] + j)
            for j in range(3):
                rows.append(rr); cols.append(6 * nc + 3 * self.pt + j)
            o = 6 * nc + 3 * self.npnt
            extra = (4 if self.refine else 0) + (len(self.refr["idx"]) if self.refr else 0)
            for j in range(extra):
                rows.append(rr); cols.append(np.full(n, o + j))
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        nr = 2 * n + (1 if self.baseline is not None else 0)
        S = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nr, len(self.x0))).tolil()
        if self.baseline is not None:
            S[nr - 1, :6] = 1
        return S.tocsr()

    def solve(self):
        res = least_squares(self.residual, self.x0, jac_sparsity=self.sparsity(),
                            method="trf", x_scale="jac", xtol=1e-12, ftol=1e-12,
                            gtol=1e-10, max_nfev=60)
        self.x = res.x
        return self.unpack(res.x)


# -- evaluation ------------------------------------------------------------------------
def umeyama(A, B):
    """s, R, t with B ~ s R A + t."""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U_, S, Vt = np.linalg.svd(b.T @ a / len(A))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U_ @ Vt))
    R = U_ @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    return s, R, mb - s * R @ ma


def evaluate(sol, cams_gt, X_gt, ba, planar):
    Rs, ts, X, intr, rp = sol
    C = np.einsum("nji,nj->ni", Rs, -ts)
    Cg = np.array([-R.T @ t for R, t in cams_gt])
    s, R, t = umeyama(C, Cg)
    Ca = s * C @ R.T + t
    Xa = s * X @ R.T + t
    rot = [np.degrees(np.linalg.norm(Rotation.from_matrix(Rg @ (Re @ R.T).T).as_rotvec()))
           for (Rg, _), Re in zip(cams_gt, Rs)]
    # Rotation error modulo a constant camera-side rotation (what a tilted port or a
    # cx/cy error turns into; harmless for the structure, absorbed by the extrinsics)
    Errs = np.stack([Rg @ (Re @ R.T).T for (Rg, _), Re in zip(cams_gt, Rs)])
    Uq, _, Vq = np.linalg.svd(Errs.sum(0))
    Q = Uq @ np.diag([1, 1, np.linalg.det(Uq @ Vq)]) @ Vq
    rot_rel = [np.degrees(np.linalg.norm(Rotation.from_matrix(E @ Q.T).as_rotvec()))
               for E in Errs]
    e = ba.residual(ba.x, whiten=False)[: 2 * len(ba.cam)].reshape(-1, 2)
    out = dict(pos_mm=1e3 * float(np.sqrt(((Ca - Cg) ** 2).sum(1).mean())),
               rot_deg=float(np.sqrt(np.mean(np.square(rot)))),
               rot_rel_deg=float(np.sqrt(np.mean(np.square(rot_rel)))),
               pts_mm=1e3 * float(np.sqrt(((Xa - X_gt) ** 2).sum(1).mean())),
               reproj_px=float(np.sqrt((e ** 2).sum(1).mean())),
               intr=[float(v) for v in intr])
    if rp is not None:
        out["refr_params"] = [float(v) for v in rp]
    if planar:
        r = pl.fit_quadric(pl.analyze(Xa))
        out["plan_rel_rmse_pct"] = float(np.sqrt(np.mean(r["dist"] ** 2)) / r["D"] * 100)
        out["kappa"] = float(r["c3"] * r["D"])
    return out


# -- covariances for the observations ------------------------------------------------------
class ObsCov:
    def __init__(self, cfg, prior, depth_mode, z0, device):
        from refraction_uncertainty import CovarianceMaps
        cfg = dict(cfg)
        cfg["correction"] = dict(cfg["correction"], z0_fixed=z0)
        self.cm = CovarianceMaps(cfg, prior, device=device)
        self.depth_mode = depth_mode
        if not depth_mode:
            self.full, self.np_, _ = self.cm.maps(z0)

    def __call__(self, q, u, w, cam):
        """(Sigma_full, Sigma_np) (n, 2, 2) at corrected positions q (n, 2)."""
        if not self.depth_mode:
            j = np.clip(np.round(q[:, 0]).astype(int), 0, self.full.shape[1] - 1)
            i = np.clip(np.round(q[:, 1]).astype(int), 0, self.full.shape[0] - 1)
            F, N = self.full[i, j], self.np_[i, j]
        else:
            fc = self.cm.fc
            F = np.zeros((len(q), 3))
            N = np.zeros((len(q), 3))
            for c in np.unique(cam):
                m = cam == c
                terms = fc.terms(u[m, 0], u[m, 1], q[m, 0], q[m, 1])
                f_, n_ = fc(w[m], terms=terms, qxy=(fc._t(q[m, 0]), fc._t(q[m, 1])),
                            fit_stride=1)
                F[m], N[m] = f_.cpu().numpy(), n_.cpu().numpy()
        mk = lambda C: np.stack([np.stack([C[:, 0], C[:, 1]], -1),
                                 np.stack([C[:, 1], C[:, 2]], -1)], -2)
        return mk(F), mk(N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=os.path.join(REPO, "configs/dataset.yaml"))
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--scenes", nargs="*", default=["tank", "reef", "reef_depth"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "figures", "synthetic_ba.json"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    results = {}
    for scene in args.scenes:
        depth_mode = scene.endswith("_depth")
        kind = scene.replace("_depth", "")
        rng = np.random.default_rng(10)
        X, cams = make_scene(kind, rng)
        Zs = np.concatenate([(X @ R.T + t)[:, 2] for R, t in cams])
        z0 = float(np.median(Zs[Zs > 0.2]))
        prior = U.default_prior(None, {"depth_global": 0.10, "depth_local": 0.03}
                                if depth_mode else {"depth_global": 0.25, "depth_local": 0.10})
        geo, _ = U.setup(dict(cfg, correction=dict(cfg["correction"], z0_fixed=z0)))
        oc = ObsCov(cfg, prior, depth_mode, z0, args.device)
        intr0 = [geo.f_out[0], geo.f_out[1], geo.c_out[0], geo.c_out[1]]
        Cg = np.array([-R.T @ t for R, t in cams])
        baseline = float(np.linalg.norm(Cg[1] - Cg[0]))
        sig = prior.vector()
        rows = []
        for tr in range(args.trials):
            t0 = time.time()
            th_true = geo.theta0 + sig * rng.standard_normal(len(sig))
            cam, pt, u_obs, w_corr = simulate(X, cams, th_true, geo, rng, depth_mode, z0)
            q_obs = corrected(geo.theta0, u_obs, w_corr, geo)
            q_orc = corrected(th_true, u_obs, w_corr, geo)
            Sf, Sn = oc(q_obs, u_obs, w_corr, cam)
            I = SIG_KP ** 2 * np.eye(2)
            used = np.unique(pt)
            remap = -np.ones(len(X), int)
            remap[used] = np.arange(len(used))
            Xu, ptu = X[used], remap[pt]
            # perturbed initialisation
            cams_i = [(Rotation.from_rotvec(rng.normal(0, 0.005, 3)).as_matrix() @ R,
                       t + rng.normal(0, 0.005, 3)) if i else (R, t)
                      for i, (R, t) in enumerate(cams)]
            X_i = Xu + rng.normal(0, 0.005, Xu.shape)
            runs = {
                "unweighted": dict(q=q_obs, cov=None),
                "weighted np": dict(q=q_obs, cov=Sn + I),
                "weighted full": dict(q=q_obs, cov=Sf + I),
                "unweighted, fixed K": dict(q=q_obs, cov=None, refine=False),
                "weighted np, fixed K": dict(q=q_obs, cov=Sn + I, refine=False),
                "oracle": dict(q=q_orc, cov=None),
                "refractive BA": dict(q=q_obs, cov=None, refr=dict(
                    u=u_obs, geo=geo, th0=geo.theta0.copy(), idx=np.array([0, 3, 4, 5]))),
            }
            row = dict(trial=tr, theta_true=[float(v) for v in th_true], n_obs=len(cam))
            for name, r in runs.items():
                if name == "refractive BA" and depth_mode:
                    continue
                ba = BA(cams_i, X_i, cam, ptu, r["q"], intr0, cov=r["cov"],
                        refine_intr=r.get("refine", True), baseline=baseline,
                        refr=r.get("refr"))
                sol = ba.solve()
                row[name] = evaluate(sol, cams, Xu, ba, planar=(kind == "tank"))
            rows.append(row)
            print(scene, tr, f"{time.time() - t0:.0f}s",
                  {k: (round(v["pos_mm"], 2), round(v["rot_rel_deg"], 3), round(v["reproj_px"], 3),
                       round(v.get("kappa", 0), 4)) for k, v in row.items()
                   if isinstance(v, dict)}, flush=True)
        results[scene] = dict(z0=z0, n_cams=len(cams), n_points=len(X), rows=rows)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    summarize(results)


def summarize(results):
    for scene, d in results.items():
        print(f"\n{scene} (z0 {d['z0']:.2f} m): median over {len(d['rows'])} trials")
        names = [k for k, v in d["rows"][0].items() if isinstance(v, dict)]
        keys = ["pos_mm", "rot_deg", "rot_rel_deg", "pts_mm", "reproj_px", "plan_rel_rmse_pct", "kappa"]
        print(f"{'method':24s}" + "".join(f"{k:>18s}" for k in keys))
        for n in names:
            vals = []
            for k in keys:
                v = [r[n][k] for r in d["rows"] if k in r[n]]
                vals.append(f"{np.median(np.abs(v)) if k == 'kappa' else np.median(v):18.4f}"
                            if v else f"{'-':>18s}")
            print(f"{n:24s}" + "".join(vals))


if __name__ == "__main__":
    main()
