"""Regression tests: the fast pipeline against the reference implementation (tests/reference_impl.py,
the original code) on a reduced-resolution camera, plus ground-truth checks of the inverse maps.

Run: pixi run pytest tests
"""
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import reference_impl as ref  # noqa: E402
import remove_refraction as rr  # noqa: E402
from core.config import load_config, camera_params, distortion_params, housing_params  # noqa: E402
from core.flatport import FlatPort, InvDepth  # noqa: E402
from core.optics import matrix_K  # noqa: E402
from core.rectangle import largest_valid_rectangle  # noqa: E402
from core.scale import select_zoom  # noqa: E402
from core.undistort import InverseMapper, _numba_kernels  # noqa: E402

BACKENDS = ["numpy"] + (["numba"] if _numba_kernels() is not None else [])
SCALE = 1 / 6  # 320 x 180 camera


@pytest.fixture(scope="module")
def cam():
    cfg = load_config(os.path.join(ROOT, "configs", "dataset.yaml"))
    W, H, fx, fy, cx, cy = camera_params(cfg)
    s = SCALE
    return dict(W=int(W * s), H=int(H * s), fx=fx * s, fy=fy * s, cx=(cx + 0.5) * s - 0.5,
                cy=(cy + 0.5) * s - 0.5, dist=distortion_params(cfg), housing=housing_params(cfg),
                z0=cfg["correction"]["z0_fixed"])


def tilted(housing, deg_x, deg_y=0.0):
    h = dict(housing)
    a, b = np.deg2rad(deg_x), np.deg2rad(deg_y)
    n = np.array([np.sin(a), np.sin(b), 1.0])
    h["n_port"] = n / np.linalg.norm(n)
    return h


def smooth_depth(W, H, seed=0):
    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    ph = rng.uniform(0, 2 * np.pi, 2)
    return (1.6 + 0.8 * np.sin(2 * np.pi * u + ph[0]) * np.cos(2 * np.pi * v + ph[1])).astype(
        np.float32)


def step_depth(W, H):
    d = np.full((H, W), 2.0, np.float32)
    d[H // 4: H // 2, W // 3: 2 * W // 3] = 0.6
    return d


def plane_depth(W, H):
    v = np.linspace(0, 1, H)[:, None] * np.ones((1, W))
    return (3.0 - 2.4 * v).astype(np.float32)


def geometry(c, housing):
    zoom, _ = ref.select_zoom(c["z0"], c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"],
                              housing, in_bounds=False)
    canvas = ref.fit_canvas(c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"], housing,
                            c["dist"], zoom, c["z0"])
    return zoom, canvas


def gt_residual(model, map_x, map_y, depth, c, zoom, canvas):
    """|F(map(q), w) - q| (output px) over finite map entries."""
    out_W, out_H, (ox, oy) = canvas
    ok = np.isfinite(map_x)
    x = (map_x[ok] - c["cx"]) / c["fx"]
    y = (map_y[ok] - c["cy"]) / c["fy"]
    w = InvDepth(np.asarray(depth, np.float64), c["fx"], c["fy"], c["cx"], c["cy"]).sample(x, y)
    Fx, Fy = model.forward(x, y, w)
    jj, ii = np.meshgrid(np.arange(out_W), np.arange(out_H))
    return np.hypot(c["fx"] * zoom * Fx + c["cx"] - ox - jj[ok],
                    c["fy"] * zoom * Fy + c["cy"] - oy - ii[ok])


# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tilt", [(0, 0), (5, 0), (-3, 7)])
def test_forward_model_matches_ray_tracer(cam, tilt):
    c = cam
    housing = tilted(c["housing"], *tilt) if tilt != (0, 0) else c["housing"]
    _, K_inv = matrix_K(c["fx"], c["fy"], c["cx"], c["cy"])
    P2, ray_water = ref.compute_housing_geometry(c["H"], c["W"], K_inv, **housing)
    u, v = np.meshgrid(np.arange(c["W"]), np.arange(c["H"]))
    model = FlatPort(**housing)
    for Z in (0.4, 1.0, 5.0):
        t = (Z - P2[..., 2]) / ray_water[..., 2]
        rx = c["fx"] * (P2[..., 0] + t * ray_water[..., 0]) / Z + c["cx"]
        ry = c["fy"] * (P2[..., 1] + t * ray_water[..., 1]) / Z + c["cy"]
        Fx, Fy = model.forward((u - c["cx"]) / c["fx"], (v - c["cy"]) / c["fy"], 1 / Z)
        assert np.abs(c["fx"] * Fx + c["cx"] - rx).max() < 1e-9
        assert np.abs(c["fy"] * Fy + c["cy"] - ry).max() < 1e-9


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tilt", [(0, 0), (5, 0)])
@pytest.mark.parametrize("kind", ["const", "smooth", "plane", "step"])
def test_inverse_map_vs_reference_and_ground_truth(cam, backend, tilt, kind):
    c = cam
    housing = tilted(c["housing"], *tilt) if tilt != (0, 0) else c["housing"]
    zoom, canvas = geometry(c, housing)
    depth = {"const": c["z0"], "smooth": smooth_depth(c["W"], c["H"]),
             "plane": plane_depth(c["W"], c["H"]), "step": step_depth(c["W"], c["H"])}[kind]
    out_W, out_H, origin = canvas
    _, K_inv = matrix_K(c["fx"], c["fy"], c["cx"], c["cy"])
    P2, ray_water = ref.compute_housing_geometry(c["H"], c["W"], K_inv, **housing)
    rx, ry = ref.build_undistort_map_closed_form(P2, ray_water, depth, c["fx"], c["fy"], c["cx"],
                                                 c["cy"], out_H, out_W, zoom=zoom, origin=origin)
    mapper = InverseMapper(c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"], housing, zoom,
                           canvas, backend=backend)
    nx, ny = mapper.build(depth)

    # Ground truth: every new map entry forward-traces back to its output pixel.
    res = gt_residual(FlatPort(**housing), nx, ny, depth, c, zoom, canvas)
    assert res.max() < 1e-6

    # Agreement with the reference (step=4 griddata) inverse.
    vr, vn = np.isfinite(rx), np.isfinite(nx)
    assert (vr ^ vn).mean() < (2e-3 if kind in ("const", "smooth") else 1e-2)
    d = np.hypot(nx - rx, ny - ry)[vr & vn]
    if kind == "step":  # at depth discontinuities the reference interpolates across the gap
        assert np.mean(d > 0.05) < 0.02
    else:
        assert np.percentile(d, 99) < 0.05
        assert d.max() < 0.5


def test_backends_agree(cam):
    if len(BACKENDS) < 2:
        pytest.skip("numba not installed")
    c = cam
    zoom, canvas = geometry(c, c["housing"])
    for housing in (c["housing"], tilted(c["housing"], 4, -2)):
        for depth in (c["z0"], smooth_depth(c["W"], c["H"], 3)):
            maps = [InverseMapper(c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"], housing,
                                  zoom, canvas, backend=b).build(depth) for b in BACKENDS]
            (ax, ay), (bx, by) = maps
            assert np.array_equal(np.isfinite(ax), np.isfinite(bx))
            ok = np.isfinite(ax)
            assert np.hypot(ax - bx, ay - by)[ok].max() < 1e-6


@pytest.mark.parametrize("backend", BACKENDS)
def test_remap_maps_match_reference_builder(cam, backend):
    """Final float32 maps (radtan, -1 = invalid) against the reference make_map_builder."""
    c = cam
    zoom, canvas = geometry(c, c["housing"])
    args = (c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"], c["housing"], c["dist"], zoom,
            canvas)
    Rx, Ry = ref.make_map_builder(*args)(c["z0"])
    mapper = InverseMapper(*args[:7], zoom, canvas, backend=backend)
    Nx, Ny = mapper.remap_maps(c["z0"], c["dist"])
    assert Nx.dtype == np.float32 and Nx.shape == Rx.shape
    mr = rr.compute_valid_mask(Rx, Ry, c["W"], c["H"]) > 0
    mn = rr.compute_valid_mask(Nx, Ny, c["W"], c["H"]) > 0
    assert (mr ^ mn).mean() < 2e-3
    assert np.hypot(Nx - Rx, Ny - Ry)[mr & mn].max() < 0.1
    assert np.all(Nx[~mn] == -1) or np.all((Nx[~mn] < 0) | (Nx[~mn] > c["W"] - 1) |
                                           (Ny[~mn] < 0) | (Ny[~mn] > c["H"] - 1))


@pytest.mark.parametrize("in_bounds", [True, False])
@pytest.mark.parametrize("z0", [0.5, 1.0, 2.5])
def test_select_zoom(cam, in_bounds, z0):
    c = cam
    args = (z0, c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"], c["housing"])
    if in_bounds:
        assert select_zoom(*args, in_bounds=True)[0] == ref.select_zoom(*args, in_bounds=True)[0]
    else:
        # The reference uses scipy's bounded Brent search (xatol 1e-5); the new code returns the
        # exact vertex of the quadratic RMSE(s), whose RMSE can only be lower.
        zn, rn = select_zoom(*args, in_bounds=False)
        zr, rr_ = ref.select_zoom(*args, in_bounds=False)
        assert abs(zn - zr) < 1e-5
        assert rn <= rr_ + 1e-9


def test_fit_canvas(cam):
    c = cam
    for housing in (c["housing"], tilted(c["housing"], 5)):
        zoom, canvas = geometry(c, housing)
        new = rr.fit_canvas(c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"], housing,
                            c["dist"], zoom, c["z0"])
        assert new == canvas


@pytest.mark.parametrize("backend", ["auto", "numpy"])
def test_largest_valid_rectangle(backend):
    rng = np.random.default_rng(1)
    for t in range(200):
        H, W = rng.integers(1, 30, 2)
        if t % 2:
            m = (rng.random((H, W)) < rng.uniform(0.3, 0.97)).astype(np.uint8) * 255
        else:
            m = np.zeros((H, W), np.uint8)
            m[rng.integers(0, H):rng.integers(0, H + 1), rng.integers(0, W):rng.integers(0, W + 1)] = 1
        assert largest_valid_rectangle(m, backend=backend) == ref.largest_valid_rectangle(m)


# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["fixed", "depth", "depth_ref"])
def test_cli_end_to_end(cam, tmp_path, mode):
    """remove_refraction.py on a tiny dataset; outputs compared with the reference pipeline."""
    c = cam
    rng = np.random.default_rng(0)
    rgb = tmp_path / "rgb"
    rgb.mkdir()
    ddir = tmp_path / "depth"
    ddir.mkdir()
    yy, xx = np.mgrid[0:c["H"], 0:c["W"]]
    for k in range(3):
        img = np.stack([(127 + 100 * np.sin(xx / (5 + k) + ch) * np.cos(yy / 7)) for ch in range(3)],
                       axis=-1)
        img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(str(rgb / f"img{k}.png"), img)
        np.save(ddir / f"img{k}.npy", smooth_depth(c["W"] // 2, c["H"] // 2, k))
    cfg = load_config(os.path.join(ROOT, "configs", "dataset.yaml"))
    cfg["camera"].update(W=c["W"], H=c["H"], fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"])
    cfg["paths"] = dict(rgb_dir=str(rgb), output_dir=str(tmp_path / "out" / "corrected"),
                        mask_dir=str(tmp_path / "out" / "mask"),
                        depth_dir=str(ddir) if mode != "fixed" else None)
    cfg["correction"]["depth_crop"] = "reference" if mode == "depth_ref" else "per_image"
    cfg["correction"]["virtual_center"] = 0.0
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    subprocess.run([sys.executable, os.path.join(ROOT, "remove_refraction.py"), str(cfg_path),
                    "--workers", "2"], check=True, capture_output=True, cwd=ROOT)

    zoom, canvas = geometry(c, c["housing"])
    build = ref.make_map_builder(c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"],
                                 c["housing"], c["dist"], zoom, canvas)
    out = tmp_path / "out"
    if mode == "fixed":
        assert (out / "mask" / "mask.png").exists()
    if mode != "depth":  # per-image crops have no common calibration
        assert (out / "new_calibration.yaml").exists()
    mx0, my0 = build(c["z0"])
    ref_box = ref.largest_valid_rectangle(ref.compute_valid_mask(mx0, my0, c["W"], c["H"]))
    for k in range(3):
        depth = c["z0"] if mode == "fixed" else rr.load_depth(f"img{k}", str(ddir), c["W"], c["H"])
        mx, my = build(depth)
        mask = ref.compute_valid_mask(mx, my, c["W"], c["H"])
        # depth_crop: reference keeps the crop of z0 for every image (one calibration)
        y0, y1, x0, x1 = ref_box if mode == "depth_ref" else ref.largest_valid_rectangle(mask)
        img = cv2.imread(str(rgb / f"img{k}.png"))
        expected = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[y0:y1, x0:x1]
        got = cv2.imread(str(out / "corrected" / f"img{k}.png"))
        assert got.shape == expected.shape
        mse = np.mean((got.astype(float) - expected.astype(float)) ** 2)
        assert 10 * np.log10(255 ** 2 / max(mse, 1e-12)) > 45
        if mode != "fixed":
            assert (out / "mask" / f"img{k}.png").exists()


# -- virtual centre (correction.virtual_center) -----------------------------------------------
@pytest.mark.parametrize("tilt", [(0, 0), (5, 0)])
def test_virtual_center_forward_matches_ray_tracer(cam, tilt):
    """FlatPort(center_z=c) with w = 1/(Z - c) equals the ray tracer projected from (0, 0, c)."""
    from core.undistort import forward_map, forward_map_analytic
    c = cam
    housing = tilted(c["housing"], *tilt) if tilt != (0, 0) else c["housing"]
    _, K_inv = matrix_K(c["fx"], c["fy"], c["cx"], c["cy"])
    P2, ray_water = ref.compute_housing_geometry(c["H"], c["W"], K_inv, **housing)
    for cz in (-0.0103, 0.004):
        for Z in (0.4, 1.0, 5.0):
            t = (Z - P2[..., 2]) / ray_water[..., 2]
            rx = c["fx"] * (P2[..., 0] + t * ray_water[..., 0]) / (Z - cz) + c["cx"]
            ry = c["fy"] * (P2[..., 1] + t * ray_water[..., 1]) / (Z - cz) + c["cy"]
            ax, ay = forward_map_analytic(c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"],
                                          housing, Z, cz)
            assert np.abs(ax - rx).max() < 1e-9 and np.abs(ay - ry).max() < 1e-9
            fx_, fy_ = forward_map(P2.astype(np.float32), ray_water.astype(np.float32), Z,
                                   c["fx"], c["fy"], c["cx"], c["cy"], cz)
            assert np.abs(fx_ - rx).max() < 1e-2


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tilt", [(0, 0), (5, 0)])
@pytest.mark.parametrize("kind", ["const", "smooth", "step"])
def test_virtual_center_inverse_ground_truth(cam, backend, tilt, kind):
    """Inverse maps with a virtual centre trace forward (ray tracer, projected from c) onto
    their output pixel, for constant and per-pixel depth measured from the camera centre."""
    c = cam
    cz = -0.0103
    housing = tilted(c["housing"], *tilt) if tilt != (0, 0) else c["housing"]
    zoom, _ = select_zoom(c["z0"], c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"], housing,
                          in_bounds=False, center_z=cz)
    canvas = rr.fit_canvas(c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"], housing,
                           (0, 0, 0, 0), zoom, c["z0"], cz)
    depth = {"const": c["z0"], "smooth": smooth_depth(c["W"], c["H"]),
             "step": step_depth(c["W"], c["H"])}[kind]
    mapper = InverseMapper(c["W"], c["H"], c["fx"], c["fy"], c["cx"], c["cy"], housing, zoom,
                           canvas, backend=backend, center_z=cz)
    nx, ny = mapper.build(depth)
    ok = np.isfinite(nx)
    assert ok.mean() > 0.5
    # Independent check with the 3-D ray tracer at the sampled depth.
    _, K_inv = matrix_K(c["fx"], c["fy"], c["cx"], c["cy"])
    from core.optics import normalize_batch, refract_batch, intersect_plane_batch
    u, v = nx[ok], ny[ok]
    rays = normalize_batch(np.stack([u, v, np.ones_like(u)], -1) @ K_inv.T)
    n = np.asarray(housing["n_port"], float)
    n = n / np.linalg.norm(n)
    P1 = intersect_plane_batch(np.zeros(3), rays, n * housing["d0"], n)
    rg = refract_batch(rays, n, housing["mu_a"], housing["mu_g"])
    P2 = intersect_plane_batch(P1, rg, n * (housing["d0"] + housing["tglass"]), n)
    rw = refract_batch(rg, n, housing["mu_g"], housing["mu_w"])
    # The depth map is interpolated (bilinearly) in 1 / (Z - c), the model's inverse depth.
    Z = cz + 1.0 / InvDepth(np.asarray(depth, np.float64) - cz, c["fx"], c["fy"], c["cx"],
                            c["cy"]).sample((u - c["cx"]) / c["fx"], (v - c["cy"]) / c["fy"])
    t = (Z - P2[:, 2]) / rw[:, 2]
    Pw = P2 + t[:, None] * rw
    out_W, out_H, (ox, oy) = canvas
    jj, ii = np.meshgrid(np.arange(out_W), np.arange(out_H))
    qx = c["fx"] * zoom * Pw[:, 0] / (Pw[:, 2] - cz) + c["cx"] - ox
    qy = c["fy"] * zoom * Pw[:, 1] / (Pw[:, 2] - cz) + c["cy"] - oy
    res = np.hypot(qx - jj[ok], qy - ii[ok])
    assert np.percentile(res, 99.9) < 1e-5, res.max()
