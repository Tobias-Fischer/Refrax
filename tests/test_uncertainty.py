"""Tests of core/uncertainty.py and refraction_uncertainty.py (needs torch: pixi run -e gpu
test-uncertainty). Reduced-resolution GoPro 10 camera (320 x 180) as in test_regression.
"""
import copy
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from core import uncertainty as U  # noqa: E402
from core.config import load_config  # noqa: E402
from core.rectangle import largest_valid_rectangle  # noqa: E402

SCALE = 1 / 6


def small_cfg(tilt_deg=0.0):
    cfg = load_config(os.path.join(ROOT, "configs", "dataset.yaml"))
    c = cfg["camera"]
    for k in ("W", "H"):
        c[k] = int(c[k] * SCALE)
    for k in ("fx", "fy"):
        c[k] *= SCALE
    for k in ("cx", "cy"):
        c[k] = (c[k] + 0.5) * SCALE - 0.5
    if tilt_deg:
        a = np.radians(tilt_deg)
        cfg["housing"]["n_port"] = [float(np.sin(a)), 0.0, float(np.cos(a))]
    return cfg


def grid(cfg, z=1.0, stride=4, center_z=0.0):
    geo, canvas = U.setup(cfg, center_z=center_z)
    xs, ys = U.source_pixels(geo, canvas, z)
    xs, ys = xs[::stride, ::stride], ys[::stride, ::stride]
    qx, qy = geo.out_grid(stride)
    ok = np.isfinite(xs)
    ur, vr = U.raw_from_ideal(xs[ok], ys[ok], geo)
    return geo, canvas, xs[ok], ys[ok], ur, vr, qx[ok], qy[ok]


def test_water_index():
    # fresh water 20 C, sodium D line: 1.3330; seawater 35 PSU: ~1.3394
    assert abs(float(U.water_index(0, 20, 589.3)) - 1.3330) < 2e-4
    assert abs(float(U.water_index(35, 20, 589.3)) - 1.3394) < 3e-4


@pytest.mark.parametrize("center_z", [0.0, -0.0103])
def test_forward_reproduces_grid(center_z):
    geo, _, xs, ys, ur, vr, qx, qy = grid(small_cfg(), center_z=center_z)
    J, q = U.jacobian_torch(geo, ur, vr, np.ones(len(ur)))
    assert np.abs(q - np.c_[qx, qy]).max() < 1e-8


@pytest.mark.parametrize("center_z", [0.0, -0.0103])
def test_analytic_equals_autodiff(center_z):
    geo, _, xs, ys, ur, vr, qx, qy = grid(small_cfg(), z=0.7, center_z=center_z)
    J, _ = U.jacobian_torch(geo, ur, vr, np.full(len(ur), 1 / 0.7))
    Ja = U.analytic_jacobian(geo, xs, ys, 1 / 0.7)
    for i, k in enumerate(list(U.PARAMS) + ["w"]):
        if k.startswith("tilt"):
            continue
        scale = np.abs(J[:, :, i]).max()
        assert np.abs(J[:, :, i] - Ja[:, :, i]).max() <= 1e-9 * max(scale, 1e-12), k


@pytest.mark.parametrize("tilt", [0.0, 3.0])
def test_autodiff_equals_finite_differences(tilt):
    geo, _, xs, ys, ur, vr, qx, qy = grid(small_cfg(tilt))
    J, _ = U.jacobian_torch(geo, ur, vr, np.ones(len(ur)))
    sig = U.default_prior().vector()
    T = torch.as_tensor
    th0 = T(geo.theta0)
    for i, k in enumerate(U.PARAMS):
        e = sig[i] * 1e-3
        tp, tm = th0.clone(), th0.clone()
        tp[i] += e
        tm[i] -= e
        f = lambda t: torch.stack(U.corrected_xy(t, T(ur), T(vr), torch.ones(len(ur),
                                                 dtype=torch.float64), geo), -1).numpy()
        fd = (f(tp) - f(tm)) / (2 * e)
        assert np.abs(fd - J[:, :, i]).max() < 1e-6 * np.abs(J[:, :, i]).max() + 1e-12, k


def test_absorb_removes_pinhole_fields():
    geo, _, xs, ys, ur, vr, qx, qy = grid(small_cfg())
    B = U.basis(qx, qy, geo, ("P4", "R3"))
    rng = np.random.default_rng(0)
    F = B @ rng.normal(size=(B.shape[-1], 3))
    r, beta = U.absorb(F, B)
    assert np.abs(r).max() < 1e-9


def test_w_rms_matches_sampling():
    rng = np.random.default_rng(0)
    for s in (0.1, 0.7):
        g = rng.normal(0, s, 2_000_000)
        assert abs(np.sqrt(np.mean((np.exp(-g) - 1) ** 2)) / U.w_rms(s) - 1) < 0.01


def test_linear_covariance_matches_monte_carlo():
    cfg = small_cfg(2.0)
    geo, _, xs, ys, ur, vr, qx, qy = grid(cfg, stride=8)
    prior = U.default_prior(None, {"depth_global": 0.0, "depth_local": 0.0})
    full, npc = U.propagate(geo, ur, vr, 1.0, prior, qx, qy, nodes=1)
    rng = np.random.default_rng(1)
    n = 3000
    th = geo.theta0 + prior.vector() * rng.standard_normal((n, len(geo.theta0)))
    T = torch.as_tensor
    q0 = np.c_[qx, qy]
    qx_, qy_ = U.corrected_xy(T(th).T[:, :, None], T(ur)[None], T(vr)[None],
                              torch.ones(1, len(ur), dtype=torch.float64), geo)
    d = np.stack([qx_.numpy(), qy_.numpy()], -1) - q0[None]
    tr_mc = (d ** 2).sum(-1).mean(0)
    ratio = np.sqrt(tr_mc / (full[:, 0] + full[:, 2]))
    assert abs(np.median(ratio) - 1) < 0.05 and ratio.min() > 0.85 and ratio.max() < 1.15


def test_fastcov_equals_propagate_at_reference():
    cfg = small_cfg()
    geo, _, xs, ys, ur, vr, qx, qy = grid(cfg, stride=2)
    prior = U.default_prior()
    full, npc = U.propagate(geo, ur, vr, 1.0 / geo.z_ref, prior, qx, qy, nodes=1)
    fc = U.FastCov(geo, ur, vr, qx, qy, prior, device="cpu", dtype=torch.float64,
                   fit_stride=1)
    f2, n2 = fc(np.full(len(ur), 1.0 / geo.z_ref), fit_stride=1)
    assert np.allclose(f2.numpy(), full, rtol=1e-6, atol=1e-9)
    assert np.allclose(n2.numpy(), npc, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("mode", ["fixed", "depth"])
def test_cli(tmp_path, mode):
    cfg = small_cfg()
    img_dir = tmp_path / "rgb"
    img_dir.mkdir()
    W, H = cfg["camera"]["W"], cfg["camera"]["H"]
    for i in range(2):
        cv2.imwrite(str(img_dir / f"im{i}.png"), np.full((H, W, 3), 100, np.uint8))
    paths = dict(rgb_dir=str(img_dir), output_dir=str(tmp_path / "out"),
                 mask_dir=str(tmp_path / "mask"), depth_dir=None)
    if mode == "depth":
        dd = tmp_path / "depth"
        dd.mkdir()
        u, v = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
        for i in range(2):
            np.save(dd / f"im{i}.npy", (0.8 + 0.6 * u * v + 0.1 * i).astype(np.float32))
        paths["depth_dir"] = str(dd)
    cfg["paths"] = paths
    cp = tmp_path / "cfg.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    subprocess.run([sys.executable, os.path.join(ROOT, "remove_refraction.py"), str(cp)],
                   check=True, cwd=ROOT, capture_output=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "refraction_uncertainty.py"), str(cp),
                    "--device", "cpu", "--per-image"], check=True, cwd=ROOT,
                   capture_output=True)
    cov = tmp_path / "cov"
    for i in range(2):
        img = cv2.imread(str(tmp_path / "out" / f"im{i}.png"))
        C = np.load(cov / (f"im{i}.npy"))
        assert C.dtype == np.float32 and C.shape == img.shape[:2] + (3,)
        assert np.isfinite(C).all()  # cropped to the valid rectangle
        assert (C[..., 0] > 0).all() and (C[..., 0] * C[..., 2] - C[..., 1] ** 2 > 0).all()
        F = np.load(tmp_path / "cov_full" / f"im{i}.npy")
        assert (F[..., 0] + F[..., 2] >= C[..., 0] + C[..., 2] - 1e-6).mean() > 0.99
    if mode == "fixed":
        assert (cov / "shared.npy").exists()
