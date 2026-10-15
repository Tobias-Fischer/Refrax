"""torch / fused-Metal renderer vs. the numpy renderer (core/render.py), on a 480x270 output
camera. Runs on every available device (cpu always; mps / cuda skipped if unavailable)."""
import numpy as np
import pytest
import torch

from core.config import load_config
from core.render_torch import TorchFlatPortRenderer
from sim.scenes import standard_scenes
from sim.validate_torch import scaled_renderer, compare, to_np

DEVICES = [
    "cpu",
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")),
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")),
]
SCENES = ["plane_0.3m", "tilted_plane", "steps", "spheres"]


@pytest.fixture(scope="module")
def setup():
    r = scaled_renderer(load_config("configs/dataset.yaml"), 0.25)
    scenes = standard_scenes()
    data = {}
    for nm in SCENES:
        rgb, z, _ = scenes[nm].render_pinhole(r.spec)
        img_x, info_x = r.render(rgb, z, exact=True)
        data[nm] = (rgb, z, img_x, info_x)
    return r, data


def check_depth(info_x, info_n):
    """Camera-frame Z and water path length agree with numpy off depth edges."""
    m = info_x["valid"] & ~info_x["background"] & ~info_x["near_edge"]
    for k in ("Z", "water_range"):
        rel = np.abs(info_n[k][m] - info_x[k][m]) / info_x[k][m]
        assert rel.max() < 1e-5, k


def check(m):
    assert m["valid_mismatch"] == 0
    assert m["background_mismatch"] < 1e-4
    assert m["coord_max_no_edge"] < 1e-3            # float32 rounding only (~2e-4 px)
    assert m["coord_p9999_all"] < 1e-2
    assert m["coord_max_all"] < 0.1                  # 4 Illinois iterations at depth edges
    assert m["psnr_vs_numpy_db"] > 60


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scene", SCENES)
def test_matches_numpy(setup, device, scene):
    r, data = setup
    rgb, z, img_x, info_x = data[scene]
    img, info = TorchFlatPortRenderer(r, device).render(rgb, z)
    info_n = to_np(info)
    assert info_n["near_edge"].sum() == pytest.approx(info_x["near_edge"].sum(), rel=0.02, abs=5)
    check(compare(info_x, img_x, info_n, img[0].cpu().numpy()))
    check_depth(info_x, info_n)


@pytest.mark.parametrize("device", DEVICES)
def test_batch_uint8_and_range_depth(setup, device):
    r, data = setup
    tr = TorchFlatPortRenderer(r, device)
    rgbs = np.stack([(255 * data[nm][0]).astype(np.uint8) for nm in SCENES])
    zs = np.stack([data[nm][1] for nm in SCENES])
    img_b, info_b = tr.render(rgbs, zs)
    assert img_b.dtype == torch.uint8 and img_b.shape == (len(SCENES), r.H, r.W, 3)
    # range depth (distance to the optical centre) gives the same trace as z-depth
    s = r.spec
    v, u = np.mgrid[0:s.H, 0:s.W].astype(np.float64)
    norm = np.sqrt(((u - s.cx) / s.fx) ** 2 + ((v - s.cy) / s.fy) ** 2 + 1)
    _, info_r = tr.render(rgbs, zs * norm, depth_type="range")
    for i, nm in enumerate(SCENES):
        img_s, info_s = tr.render(rgbs[i], zs[i])
        assert torch.equal(info_b["map_u"][i], info_s["map_u"][0])
        assert torch.equal(img_b[i], img_s[0])
        d = (info_r["map_u"][i] - info_s["map_u"][0]).abs()[info_s["valid"][0]]
        assert float(d.max()) < 0.05


def _metal(r):
    from core.render_metal import MetalFlatPortRenderer
    return MetalFlatPortRenderer(r)


def _warp(device):
    def make(r):
        from core.render_warp import WarpFlatPortRenderer
        return WarpFlatPortRenderer(r, device)
    return make


def _has_warp():
    try:
        import warp  # noqa: F401
        return True
    except ImportError:
        return False


FUSED = [
    pytest.param(_metal, id="metal", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")),
    pytest.param(_warp("cpu"), id="warp-cpu", marks=pytest.mark.skipif(not _has_warp(), reason="no warp")),
    pytest.param(_warp("cuda"), id="warp-cuda", marks=pytest.mark.skipif(
        not (_has_warp() and torch.cuda.is_available()), reason="no warp / CUDA")),
]


@pytest.mark.parametrize("make", FUSED)
@pytest.mark.parametrize("scene", SCENES)
def test_fused_kernel(setup, make, scene):
    r, data = setup
    rgb, z, img_x, info_x = data[scene]
    img, info = make(r).render(rgb, z)
    info_n = to_np(info) | dict(near_edge=info_x["near_edge"])
    check(compare(info_x, img_x, info_n, img[0].cpu().numpy()))
    check_depth(info_x, info_n)


@pytest.mark.parametrize("make", FUSED)
def test_fused_kernel_batch_uint8_range(setup, make):
    """Batched uint8 frames and range depth give the torch renderer's result."""
    r, data = setup
    fr = make(r)
    tr = TorchFlatPortRenderer(r, "cpu")
    rgbs = np.stack([(255 * data[nm][0]).astype(np.uint8) for nm in SCENES])
    zs = np.stack([data[nm][1] for nm in SCENES])
    s = r.spec
    v, u = np.mgrid[0:s.H, 0:s.W].astype(np.float64)
    norm = np.sqrt(((u - s.cx) / s.fx) ** 2 + ((v - s.cy) / s.fy) ** 2 + 1)
    img_t, info_t = tr.render(rgbs, zs)
    for depth, dt in ((zs, "z"), ((zs * norm).astype(np.float32), "range")):
        img, info = fr.render(rgbs, depth, depth_type=dt)
        assert img.dtype == torch.uint8 and img.shape == img_t.shape
        valid = info_t["valid"]
        assert torch.equal(info["valid"].cpu(), valid)
        du = (info["map_u"].cpu() - info_t["map_u"]).abs()[valid]
        assert float(du.max()) < (0.05 if dt == "z" else 0.1)
        diff = (img.cpu().int() - img_t.int()).abs()[valid]
        assert float((diff > 1).float().mean()) < 1e-3
