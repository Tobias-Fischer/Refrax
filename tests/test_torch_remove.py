"""Tests of the torch (GPU) refraction removal (core.torch_remove) against the numba path.

Runs on cpu and, where available, mps and cuda. Run: pixi run -e gpu test-gpu
"""
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import remove_refraction as rr  # noqa: E402
from core.config import load_config, camera_params, distortion_params, housing_params  # noqa: E402
from core.flatport import FlatPort  # noqa: E402
from core.torch_remove import (Corrector, TorchInverse, maps_to_grid, remap, remap_px,  # noqa: E402
                               valid_mask)
from core.undistort import InverseMapper  # noqa: E402
from test_regression import (SCALE, gt_residual, plane_depth, smooth_depth, step_depth,  # noqa: E402
                             tilted)

DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.fixture(scope="module")
def cam():
    cfg = load_config(os.path.join(ROOT, "configs", "dataset.yaml"))
    W, H, fx, fy, cx, cy = camera_params(cfg)
    s = SCALE
    c = dict(W=int(W * s), H=int(H * s), fx=fx * s, fy=fy * s, cx=(cx + 0.5) * s - 0.5,
             cy=(cy + 0.5) * s - 0.5, dist=distortion_params(cfg), housing=housing_params(cfg),
             z0=cfg["correction"]["z0_fixed"])
    cfg["camera"].update(W=c["W"], H=c["H"], fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"])
    c["cfg"] = cfg
    zoom, _ = rr.select_zoom(c["z0"], c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"],
                             c["housing"], in_bounds=False)
    c["zoom"] = zoom
    return c


_INV = {}


def inverse(c, housing_key, device, dtype=torch.float32, dist=True):
    """Cached TorchInverse (compilation on mps/cuda is the slow part of these tests)."""
    key = (housing_key, device, dtype, dist)
    if key not in _INV:
        housing = c["housing"] if housing_key == (0, 0) else tilted(c["housing"], *housing_key)
        canvas = rr.fit_canvas(c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"], housing,
                               c["dist"], c["zoom"], c["z0"])
        _INV[key] = TorchInverse(c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"], housing,
                                 c["zoom"], canvas, c["dist"] if dist else None, device, dtype,
                                 z_ref=c["z0"])
    return _INV[key]


def depth_of(c, kind):
    W, H = c["W"], c["H"]
    return {"const": np.full((H, W), c["z0"], np.float32), "smooth": smooth_depth(W, H),
            "plane": plane_depth(W, H), "step": step_depth(W, H)}[kind]


def tmaps(inv, depth):
    d = torch.as_tensor(depth, device=inv.device)[None]
    mx, my = inv.remap_maps(d)
    return mx[0].cpu().numpy().astype(np.float64), my[0].cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------------------------
def test_analytic_jacobian(cam):
    """The tilted port's analytic Jacobian of A, B against central differences (float64)."""
    inv = inverse(cam, (-3, 7), "cpu", torch.float64)
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.uniform(-0.9, 0.9, 500), dtype=torch.float64)
    y = torch.as_tensor(rng.uniform(-0.5, 0.5, 500), dtype=torch.float64)
    out = inv._AB_jac(x, y)
    m = FlatPort(**tilted(cam["housing"], -3, 7))
    ref = m.AB(x.numpy(), y.numpy())
    for a, b in zip(out[:4], ref):
        assert np.abs(a.numpy() - b).max() < 1e-12
    e = 1e-6
    fd_x = [(p - q) / (2 * e) for p, q in zip(m.AB(x.numpy() + e, y.numpy()),
                                               m.AB(x.numpy() - e, y.numpy()))]
    fd_y = [(p - q) / (2 * e) for p, q in zip(m.AB(x.numpy(), y.numpy() + e),
                                               m.AB(x.numpy(), y.numpy() - e))]
    Axx, Axy, Ayx, Ayy, Bxx, Bxy, Byx, Byy = (t.numpy() for t in out[4:])
    for an, fd in ((Axx, fd_x[0]), (Axy, fd_y[0]), (Ayx, fd_x[1]), (Ayy, fd_y[1]),
                   (Bxx, fd_x[2]), (Bxy, fd_y[2]), (Byx, fd_x[3]), (Byy, fd_y[3])):
        assert np.abs(an - fd).max() < 1e-7


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tilt", [(0, 0), (5, 0)])
@pytest.mark.parametrize("kind", ["const", "smooth", "plane", "step"])
def test_maps_match_numba(cam, device, tilt, kind):
    """float32 torch maps (radtan, -1 invalid) against the numba kernel's float32 maps."""
    c = cam
    inv = inverse(c, tilt, device)
    depth = depth_of(c, kind)
    ref = InverseMapper(c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"], inv.housing,
                        c["zoom"], inv.canvas).remap_maps(depth, c["dist"])
    mx, my = tmaps(inv, depth)
    vr = rr.compute_valid_mask(*ref, c["W"], c["H"]) > 0
    vt = rr.compute_valid_mask(mx, my, c["W"], c["H"]) > 0
    assert (vr ^ vt).sum() <= 3
    d = np.hypot(mx - ref[0], my - ref[1])[vr & vt]
    if kind == "step" and tilt != (0, 0):
        # Occlusion folds: two valid roots; numba's row-wise warm start may pick the other.
        assert np.mean(d > 5e-3) < 5e-3
    else:
        assert d.max() < 5e-3
    assert np.all(mx[~vt] == -1) or np.all((mx[~vt] < 0) | (mx[~vt] > c["W"] - 1) |
                                           (my[~vt] < 0) | (my[~vt] > c["H"] - 1))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tilt", [(0, 0), (5, 0)])
@pytest.mark.parametrize("kind", ["smooth", "step"])
def test_ground_truth(cam, device, tilt, kind):
    """Every valid float32 map entry forward-traces to its output pixel (< 5e-3 px)."""
    c = cam
    inv = inverse(c, tilt, device, dist=False)
    depth = depth_of(c, kind)
    mx, my = tmaps(inv, depth)
    mx[mx < 0], my[mx < 0] = np.nan, np.nan
    res = gt_residual(FlatPort(**inv.housing), mx, my, depth, c, c["zoom"], inv.canvas)
    assert res.max() < 5e-3


@pytest.mark.parametrize("tilt", [(0, 0), (5, 0)])
def test_float64_matches_numba(cam, tilt):
    """In float64 (cpu) the torch solver reproduces the numba kernel to ~1e-6 px."""
    c = cam
    inv = inverse(c, tilt, "cpu", torch.float64, dist=False)
    depth = smooth_depth(c["W"], c["H"], 3)
    nx, ny = InverseMapper(c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"], inv.housing,
                           c["zoom"], inv.canvas).build(depth)
    mx, my = tmaps(inv, depth)
    ok = np.isfinite(nx)
    assert np.array_equal(ok, mx >= 0)
    assert np.hypot(mx - nx, my - ny)[ok].max() < 1e-5


@pytest.mark.parametrize("device", DEVICES)
def test_remap_matches_cv2(cam, device):
    """grid_sample (align_corners=False) reproduces cv2.remap(INTER_LINEAR, BORDER_CONSTANT)
    up to cv2's 1/32 px weight quantisation."""
    c = cam
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 256, (c["H"], c["W"], 3), dtype=np.uint8), (5, 5), 1)
    mx, my = InverseMapper(c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"], c["housing"],
                           c["zoom"]).remap_maps(smooth_depth(c["W"], c["H"]), c["dist"])
    ref = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    tx = torch.as_tensor(mx, device=device)[None]
    ty = torch.as_tensor(my, device=device)[None]
    timg = torch.as_tensor(img, device=device)[None]
    valid = rr.compute_valid_mask(mx, my, c["W"], c["H"]) > 0
    assert np.array_equal(valid_mask(tx, ty, c["W"], c["H"])[0].cpu().numpy(), valid)
    fused = torch.compile(remap_px, dynamic=False) if device != "cpu" else remap_px
    for out in (remap(timg, maps_to_grid(tx, ty, c["W"], c["H"])), fused(timg, tx, ty)):
        out = out[0].cpu().numpy()
        d = out.astype(float) - ref.astype(float)
        assert np.abs(d).max() <= 3  # every pixel, including the border blend
        psnr = 10 * np.log10(255 ** 2 / max(np.mean(d ** 2), 1e-12))
        assert psnr > 45


@pytest.mark.parametrize("device", DEVICES)
def test_corrector_batch_and_fixed(cam, device):
    c = cam
    corr = Corrector(c["cfg"], device)
    rng = np.random.default_rng(1)
    imgs = rng.integers(0, 256, (2, c["H"], c["W"], 3), dtype=np.uint8)
    depths = np.stack([smooth_depth(c["W"], c["H"], k) for k in (4, 5)])
    out_b, mask_b = corr.correct(imgs, depths)
    for k in range(2):
        out, mask = corr.correct(imgs[k], depths[k])
        assert torch.equal(out, out_b[k]) and torch.equal(mask, mask_b[k])
    # Fixed depth: the cached CLI map + one grid_sample.
    out, mask = corr.correct(imgs[0])
    canvas, (fx_, fy_) = rr._fit_canvas(c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"],
                                        c["housing"], c["dist"], c["zoom"], c["z0"])
    assert out.shape[:2] == (canvas[1], canvas[0])
    ref = cv2.remap(imgs[0], fx_, fy_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    assert np.array_equal(mask.cpu().numpy(), rr.compute_valid_mask(fx_, fy_, c["W"], c["H"]) > 0)
    assert np.abs(out.cpu().numpy().astype(int) - ref).max() <= 3
    # Half-resolution depth is resized on the device, as load_depth does with cv2.resize.
    half = smooth_depth(c["W"] // 2, c["H"] // 2, 4)
    full = np.clip(cv2.resize(half, (c["W"], c["H"]), interpolation=cv2.INTER_LINEAR), 1e-3, None)
    a = corr.remap_maps(half)
    b = corr.remap_maps(full)
    va, vb = a[0] >= 0, b[0] >= 0
    assert (va ^ vb).sum() <= 3
    assert (a[0] - b[0]).abs()[va & vb].max() < 1e-2


@pytest.mark.parametrize("device", DEVICES)
def test_cli_device(cam, tmp_path, device):
    """remove_refraction.py --device <torch device> in depth-map mode against the default."""
    c = cam
    rgb = tmp_path / "rgb"
    rgb.mkdir()
    ddir = tmp_path / "depth"
    ddir.mkdir()
    yy, xx = np.mgrid[0:c["H"], 0:c["W"]]
    for k in range(2):
        img = np.stack([(127 + 100 * np.sin(xx / (5 + k) + ch) * np.cos(yy / 7))
                        for ch in range(3)], axis=-1).astype(np.uint8)
        cv2.imwrite(str(rgb / f"img{k}.png"), img)
        np.save(ddir / f"img{k}.npy", smooth_depth(c["W"] // 2, c["H"] // 2, k))
    outs = {}
    for dev in ("numba", device):
        cfg = dict(c["cfg"])
        cfg["paths"] = dict(rgb_dir=str(rgb), output_dir=str(tmp_path / dev / "corrected"),
                            mask_dir=str(tmp_path / dev / "mask"), depth_dir=str(ddir))
        p = tmp_path / f"cfg_{dev}.yaml"
        p.write_text(yaml.safe_dump(cfg))
        subprocess.run([sys.executable, os.path.join(ROOT, "remove_refraction.py"), str(p),
                        "--workers", "2", "--device", dev], check=True, capture_output=True,
                       cwd=ROOT)
        outs[dev] = [cv2.imread(str(tmp_path / dev / "corrected" / f"img{k}.png"))
                     for k in range(2)]
    for a, b in zip(outs["numba"], outs[device]):
        assert a.shape == b.shape
        assert np.abs(a.astype(int) - b).max() <= 2
