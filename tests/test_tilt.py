"""Tests of the port-tilt refractive BA (tilt/)."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from core.config import load_config  # noqa: E402
from core import uncertainty as U  # noqa: E402
from tilt.model import Decl, residual_and_jac, raw_from_corrected, n_port_of, tilts_of_n  # noqa: E402
from tilt.ba import RefracBA  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(os.path.join(REPO, "configs/dataset.yaml"))
    geo = U.Geometry(cfg, 1.5)
    decl = Decl((1390.0, 1391.0), (849.0, 484.0), geo.R0, geo.mu_a)
    return cfg, geo, decl


def _raw_of_point(theta, P, decl):
    """Exact raw pixel of camera-frame points P (n, 3) (raw_from_corrected at the point's
    own depth with the pinhole projection as 'corrected' keypoint)."""
    out = []
    for p in P:
        q = np.array([[decl.c_out[0] + decl.f_out[0] * p[0] / p[2],
                       decl.c_out[1] + decl.f_out[1] * p[1] / p[2]]])
        out.append(raw_from_corrected(q, theta, decl, p[2])[0])
    return np.array(out)


def test_tilt_convention_roundtrip():
    for ax, ay in [(0.01, -0.02), (-0.03, 0.005), (0.0, 0.0)]:
        bx, by = tilts_of_n(n_port_of(ax, ay))
        assert abs(bx - ax) < 1e-14 and abs(by - ay) < 1e-14


def test_residual_zero_and_raw_pixel_units(setup):
    _, geo, decl = setup
    th = geo.theta0.copy()
    th[4], th[5] = np.radians(0.7), np.radians(-1.1)
    rng = np.random.default_rng(0)
    P = np.c_[rng.uniform(-0.5, 0.5, 30), rng.uniform(-0.3, 0.3, 30), rng.uniform(0.6, 3, 30)]
    u = _raw_of_point(th, P, decl)
    r, JP, JT, e, Jui = residual_and_jac(th, u, P, decl)
    assert np.abs(r).max() < 1e-7
    # shifting the raw keypoint by (1, 0) px gives a residual of ~(1, 0) raw px
    r2 = residual_and_jac(th, u + [1.0, 0.0], P, decl)[0]
    assert np.abs(r2 - [1.0, 0.0]).max() < 5e-3


def test_jacobians_numeric(setup):
    _, geo, decl = setup
    th = geo.theta0.copy()
    th[4], th[5] = np.radians(0.3), np.radians(0.4)
    P = np.array([[0.2, -0.1, 1.2], [-0.4, 0.25, 0.8]])
    u = _raw_of_point(th, P, decl) + 0.3
    r, JP, JT, e, Jui = residual_and_jac(th, u, P, decl)
    h = 1e-6
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        num = (np.einsum("nij,nj->ni", Jui, residual_and_jac(th, u, P + d, decl)[3]) -
               np.einsum("nij,nj->ni", Jui, residual_and_jac(th, u, P - d, decl)[3])) / (2 * h)
        assert np.allclose(JP[:, :, k], num, rtol=1e-5, atol=1e-6)
    for k in (0, 4, 5, 6, 8):
        d = np.zeros(14)
        d[k] = h * max(1.0, abs(th[k]))
        num = (np.einsum("nij,nj->ni", Jui, residual_and_jac(th + d, u, P, decl)[3]) -
               np.einsum("nij,nj->ni", Jui, residual_and_jac(th - d, u, P, decl)[3])) / (2 * d[k])
        assert np.allclose(JT[:, :, k], num, rtol=1e-4, atol=1e-6)


def test_ba_recovers_tilt(setup):
    _, geo, decl = setup
    th = geo.theta0.copy()
    th[4], th[5] = np.radians(0.8), np.radians(-0.6)
    rng = np.random.default_rng(1)
    X = np.c_[rng.uniform(-0.5, 0.5, 250), rng.uniform(-0.4, 0.4, 250), rng.uniform(-0.1, 0.1, 250)]
    R, t, ci, pj, uu = [], [], [], [], []
    for k in range(10):
        az = 2 * np.pi * k / 10
        C = np.array([0.35 * np.cos(az), 0.35 * np.sin(az), -1.0])
        z = -C / np.linalg.norm(C)
        x = np.cross([0, 1.0, 0], z)
        x /= np.linalg.norm(x)
        Rk = np.stack([x, np.cross(z, x), z])
        R.append(Rk)
        t.append(-Rk @ C)
        P = X @ Rk.T + t[-1]
        u = _raw_of_point(th, P, decl)
        ok = (u[:, 0] > 0) & (u[:, 0] < 1920) & (u[:, 1] > 0) & (u[:, 1] < 1080)
        ci += [k] * ok.sum()
        pj += list(np.flatnonzero(ok))
        uu.append(u[ok])
    R, t, uu = np.array(R), np.array(t), np.concatenate(uu)
    ba = RefracBA(R, t, X + rng.normal(0, 0.003, X.shape), np.array(ci), np.array(pj), uu,
                  geo.theta0.copy(), decl, free=("tilt_x", "tilt_y"), huber=None,
                  sigma_px=0.5, verbose=False)
    ba.solve(max_iter=30)
    assert abs(np.degrees(ba.theta[4]) - 0.8) < 1e-3
    assert abs(np.degrees(ba.theta[5]) + 0.6) < 1e-3
    sm = ba.summary()
    assert sm["params"]["tilt_x"][1] > 0
