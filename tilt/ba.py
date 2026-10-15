"""Refractive bundle adjustment (Levenberg-Marquardt with Schur complement) with the
flat-port housing (port tilt, optionally d0) and in-air intrinsics as global unknowns.

Residual: tilt/model.py (raw-pixel, back-projection form). Unknowns: camera poses
(rotation vector increment applied on the left, translation), 3-D points, and a free
subset of theta (core.uncertainty.PARAMS). Points are eliminated (3x3 blocks); the reduced
system over poses + globals is dense (6 n_cams + k) and solved directly, which also gives
the posterior covariance of the globals (gauge: camera 0 fixed + one translation
coordinate for the scale; the globals are gauge invariant so their marginal covariance
does not depend on that choice).

Robust loss: Huber (IRLS). Optional Gaussian priors on the globals.
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation

from tilt.model import PARAMS, IDX, residual_and_jac


def skew(v):
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], -1),
                     np.stack([v[:, 2], z, -v[:, 0]], -1),
                     np.stack([-v[:, 1], v[:, 0], z], -1)], 1)


class RefracBA:
    def __init__(self, R, t, X, ci, pj, u, theta, decl, free=("tilt_x", "tilt_y"),
                 fixed_cams=(0,), huber=2.0, prior=None, sigma_px=1.0, threads=2,
                 fix_scale=True, verbose=True):
        """R (nc,3,3) camera-from-world, t (nc,3), X (np,3), observations (ci, pj, u raw
        px), theta (14,), decl (tilt.model.Decl). free: names of free globals.
        prior: dict name -> (mean, sigma) Gaussian prior. sigma_px: keypoint noise used to
        scale the covariance (None -> estimated from the residuals)."""
        self.R, self.t, self.X = R.copy(), t.copy(), X.copy()
        self.ci, self.pj, self.u = np.asarray(ci), np.asarray(pj), np.asarray(u, float)
        self.theta = np.asarray(theta, float).copy()
        self.decl = decl
        self.free = [IDX[k] for k in free]
        self.free_names = list(free)
        self.nc, self.np_ = len(R), len(X)
        self.fixed = set(fixed_cams)
        self.slot = -np.ones(self.nc, int)
        fc = [i for i in range(self.nc) if i not in self.fixed]
        self.slot[fc] = np.arange(len(fc))
        self.nf = len(fc)
        self.k = len(self.free)
        self.huber = huber
        self.prior = prior or {}
        self.sigma_px = sigma_px
        self.threads = threads
        self.verbose = verbose
        # The metric scale is (weakly) observable through refraction only (d0, tglass in
        # metres); by default it is held at the initial value (gauge-like constraint).
        self.fix_scale = fix_scale
        self._drop = self._scale_coord() if self.nf else None

    def _scale_coord(self):
        """Index (in the reduced vector) of the translation coordinate most affected by a
        scaling of the scene about camera 0; fixing it removes the scale freedom."""
        C = np.einsum("nji,nj->ni", self.R, -self.t)
        c0 = C[min(self.fixed)]
        idx = np.flatnonzero(self.slot >= 0)
        dt = -np.einsum("nij,nj->ni", self.R[idx], C[idx] - c0)
        a = np.unravel_index(np.argmax(np.abs(dt)), dt.shape)
        return 6 * a[0] + 3 + a[1]

    # -- evaluation ------------------------------------------------------------------------
    def Pc(self, R=None, t=None, X=None, theta=None):
        R = self.R if R is None else R
        t = self.t if t is None else t
        X = self.X if X is None else X
        return np.einsum("nij,nj->ni", R[self.ci], X[self.pj]) + t[self.ci]

    def _rho(self, a):
        d = self.huber
        if d is None:
            return 0.5 * a * a
        return np.where(a <= d, 0.5 * a * a, d * (a - 0.5 * d))

    def _prior_terms(self, theta):
        r, J = [], []
        for name, (m, s) in self.prior.items():
            if IDX[name] not in self.free:
                continue
            j = self.free.index(IDX[name])
            r.append((theta[IDX[name]] - m) / s)
            row = np.zeros(self.k)
            row[j] = 1.0 / s
            J.append(row)
        return np.array(r), np.array(J).reshape(len(r), self.k)

    def cost(self, R, t, X, theta):
        r = self._res_only(theta, self.Pc(R, t, X))
        a = np.hypot(r[:, 0], r[:, 1])
        pr, _ = self._prior_terms(theta)
        return float(self._rho(a).sum() + 0.5 * (pr ** 2).sum())

    def _res_only(self, theta, Pc):
        import torch
        from tilt.model import _q
        with torch.no_grad():
            th = torch.as_tensor(theta)
            q = _q(th, torch.as_tensor(1.0 / Pc[:, 2]), torch.as_tensor(self.u[:, 0]),
                   torch.as_tensor(self.u[:, 1]), self.decl).numpy()
        fx, fy = self.decl.f_out
        cx, cy = self.decl.c_out
        e = q - np.c_[cx + fx * Pc[:, 0] / Pc[:, 2], cy + fy * Pc[:, 1] / Pc[:, 2]]
        # whiten with the J_u of the last linearisation (held fixed, see model.py)
        return np.einsum("nij,nj->ni", self._Jui, e)

    # -- linear system ---------------------------------------------------------------------
    def linearize(self):
        Pc = self.Pc()
        r, JP, JT, e, self._Jui = residual_and_jac(self.theta, self.u, Pc, self.decl,
                                                    threads=self.threads)
        a = np.hypot(r[:, 0], r[:, 1])
        if self.huber is None:
            s = np.ones_like(a)
        else:
            s = np.sqrt(np.minimum(1.0, self.huber / np.maximum(a, 1e-12)))
        RX = np.einsum("nij,nj->ni", self.R[self.ci], self.X[self.pj])
        Jc = np.concatenate([JP @ (-skew(RX)), JP], axis=2)          # (N, 2, 6)
        Jp = JP @ self.R[self.ci]                                      # (N, 2, 3)
        Jg = JT[:, :, self.free]                                       # (N, 2, k)
        r = r * s[:, None]
        Jc, Jp, Jg = Jc * s[:, None, None], Jp * s[:, None, None], Jg * s[:, None, None]
        self.lin = dict(r=r, Jc=Jc, Jp=Jp, Jg=Jg, a=a)
        return a

    def normal_equations(self, lam):
        L = self.lin
        ci, pj = self.ci, self.pj
        nf, k, npnt = self.nf, self.k, self.np_
        sl = self.slot[ci]
        m = sl >= 0
        nC = 6 * nf + k
        Jc, Jp, Jg, r = L["Jc"], L["Jp"], L["Jg"], L["r"]
        # camera / global block (dense)
        Hcc = np.zeros((nC, nC))
        bc = np.zeros(nC)
        Bcc = np.einsum("nri,nrj->nij", Jc[m], Jc[m])
        blk = np.zeros((nf, 6, 6))
        np.add.at(blk, sl[m], Bcc)
        for i in range(nf):
            Hcc[6 * i:6 * i + 6, 6 * i:6 * i + 6] = blk[i]
        if k:
            Bcg = np.einsum("nri,nrj->nij", Jc[m], Jg[m])
            cg = np.zeros((nf, 6, k))
            np.add.at(cg, sl[m], Bcg)
            Hcc[:6 * nf, 6 * nf:] = cg.reshape(6 * nf, k)
            Hcc[6 * nf:, :6 * nf] = cg.reshape(6 * nf, k).T
            Hcc[6 * nf:, 6 * nf:] = np.einsum("nri,nrj->ij", Jg, Jg)
        bcam = np.zeros((nf, 6))
        np.add.at(bcam, sl[m], -np.einsum("nri,nr->ni", Jc[m], r[m]))
        bc[:6 * nf] = bcam.ravel()
        if k:
            bc[6 * nf:] = -np.einsum("nri,nr->i", Jg, r)
            pr, PJ = self._prior_terms(self.theta)
            if len(pr):
                Hcc[6 * nf:, 6 * nf:] += PJ.T @ PJ
                bc[6 * nf:] -= PJ.T @ pr
        # points
        Hpp = np.zeros((npnt, 3, 3))
        np.add.at(Hpp, pj, np.einsum("nri,nrj->nij", Jp, Jp))
        bp = np.zeros((npnt, 3))
        np.add.at(bp, pj, -np.einsum("nri,nr->ni", Jp, r))
        # cross blocks: E (nC x 3 npnt)
        rows, cols, vals = [], [], []
        Wcp = np.einsum("nri,nrj->nij", Jc[m], Jp[m])                  # (M, 6, 3)
        ii = (6 * sl[m])[:, None, None] + np.arange(6)[None, :, None]
        jj = (3 * pj[m])[:, None, None] + np.arange(3)[None, None, :]
        rows.append(np.broadcast_to(ii, Wcp.shape).ravel())
        cols.append(np.broadcast_to(jj, Wcp.shape).ravel())
        vals.append(Wcp.ravel())
        if k:
            Wgp = np.zeros((npnt, k, 3))
            np.add.at(Wgp, pj, np.einsum("nri,nrj->nij", Jg, Jp))
            ii = 6 * nf + np.arange(k)[None, :, None] + np.zeros((npnt, 1, 1), int)
            jj = (3 * np.arange(npnt))[:, None, None] + np.arange(3)[None, None, :]
            rows.append(np.broadcast_to(ii, Wgp.shape).ravel())
            cols.append(np.broadcast_to(jj, Wgp.shape).ravel())
            vals.append(Wgp.ravel())
        E = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(nC, 3 * npnt))
        # damping
        if lam > 0:
            dC = np.diag(Hcc).copy()
            Hcc = Hcc + lam * np.diag(np.maximum(dC, 1e-9))
            dP = np.einsum("nii->ni", Hpp).copy()
            Hpp = Hpp + lam * np.einsum("ni,ij->nij", np.maximum(dP, 1e-9), np.eye(3))
        Hpi = np.linalg.inv(Hpp)
        Hinv = _block_diag(Hpi)
        M = E @ Hinv
        S = Hcc - (M @ E.T).toarray()
        rhs = bc - M @ bp.ravel()
        return S, rhs, E, Hpi, bp

    def step(self, lam):
        S, rhs, E, Hpi, bp = self.normal_equations(lam)
        if self.fix_scale and self._drop is not None:
            keep = np.setdiff1d(np.arange(len(S)), [self._drop])
            dc = np.zeros(len(S))
            dc[keep] = np.linalg.solve(S[np.ix_(keep, keep)], rhs[keep])
        else:
            dc = np.linalg.solve(S, rhs)
        dp = np.einsum("nij,nj->ni", Hpi, bp - (E.T @ dc).reshape(-1, 3))
        nf = self.nf
        R, t = self.R.copy(), self.t.copy()
        idx = np.flatnonzero(self.slot >= 0)
        d = dc[:6 * nf].reshape(nf, 6)
        R[idx] = Rotation.from_rotvec(d[:, :3]).as_matrix() @ R[idx]
        t[idx] = t[idx] + d[:, 3:]
        X = self.X + dp
        th = self.theta.copy()
        th[self.free] += dc[6 * nf:]
        return R, t, X, th, dc

    def solve(self, max_iter=50, lam=1e-4, tol=1e-7):
        t0 = time.time()
        self.linearize()
        c = self.cost(self.R, self.t, self.X, self.theta)
        self.history = [c]
        for it in range(max_iter):
            R, t, X, th, dc = self.step(lam)
            c_new = self.cost(R, t, X, th)
            if c_new < c:
                rel = (c - c_new) / c
                self.R, self.t, self.X, self.theta = R, t, X, th
                lam = max(lam / 3, 1e-9)
                self.linearize()
                c = self.cost(self.R, self.t, self.X, self.theta)   # with the new J_u
                self.history.append(c)
                if self.verbose:
                    g = ", ".join(f"{n}={self._fmt(n)}" for n in self.free_names)
                    print(f"  it {it:2d} cost {c:.6e} lam {lam:.1e} {g} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                dg = dc[6 * self.nf:]
                small = (np.max(np.abs(dg) / np.maximum(np.abs(th[self.free]), 1e-3))
                         if self.k else 0.0) < 1e-6
                if rel < tol or (rel < 1e-5 and small):
                    break
            else:
                lam *= 10
                if lam > 1e8:
                    break
        return self

    def _fmt(self, n):
        v = self.theta[IDX[n]]
        return f"{np.degrees(v):.4f}deg" if n.startswith("tilt") else f"{v:.6g}"

    # -- covariance --------------------------------------------------------------------------
    def covariance(self):
        """Posterior covariance of the free globals (and of all reduced variables), gauge
        fixed by camera 0 + one translation coordinate. Scaled by sigma_px^2 (or by the
        residual variance estimate if sigma_px is None)."""
        self.linearize()
        S, _, _, _, _ = self.normal_equations(0.0)
        nf, k = self.nf, self.k
        # scale fixed (as in the solve); without fix_scale this still is the gauge choice
        keep = np.setdiff1d(np.arange(len(S)), [self._scale_coord()]) if nf else \
            np.arange(len(S))
        Ci = np.linalg.inv(S[np.ix_(keep, keep)])
        L = self.lin
        a_ = L["a"]
        n_obs = len(a_)
        if self.sigma_px is None:
            dof = 2 * n_obs - (6 * nf + 3 * self.np_ + k - 1)
            s2 = float((L["r"] ** 2).sum() / max(dof, 1))
        else:
            s2 = self.sigma_px ** 2
        Cg = Ci[-k:, -k:] * s2 if k else np.zeros((0, 0))
        self.sigma2 = s2
        return Cg

    def summary(self):
        Cg = self.covariance()
        sd = np.sqrt(np.diag(Cg))
        corr = Cg / np.outer(sd, sd)
        out = {}
        for i, n in enumerate(self.free_names):
            v, s = self.theta[IDX[n]], sd[i]
            if n.startswith("tilt"):
                v, s = np.degrees(v), np.degrees(s)
            out[n] = (float(v), float(s))
        a = self.lin["a"]
        return dict(params=out, corr=corr.tolist(), names=self.free_names,
                    rms_px=float(np.sqrt(np.mean(a ** 2))), sigma2=self.sigma2,
                    n_obs=int(len(a)))


def _block_diag(B):
    n = len(B)
    rows = (3 * np.arange(n))[:, None, None] + np.arange(3)[None, :, None]
    cols = (3 * np.arange(n))[:, None, None] + np.arange(3)[None, None, :]
    rows = np.broadcast_to(rows, B.shape).ravel()
    cols = np.broadcast_to(cols, B.shape).ravel()
    return sp.csr_matrix((B.ravel(), (rows, cols)), shape=(3 * n, 3 * n))
