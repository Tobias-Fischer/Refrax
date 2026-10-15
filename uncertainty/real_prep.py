"""Inputs for the gopro10 weighted-BA check (section 5 of docs/explore_uncertainty.md).

    python uncertainty/real_prep.py RUN_DIR Z0 OUT_DIR

RUN_DIR: a fixed-depth pipeline stage of explore/depth (e.g.
refrax-work/depth/gopro10/z1_c0mm/it0_fixed) with colmap/sparse/0_txt and calibration.yaml,
corrected at depth Z0 from the camera centre. Writes to OUT_DIR:
  cov/shared.npy, cov_full/shared.npy   the interchange-format maps for that run's
                                        corrected images (checked against calibration.yaml)
  obs_cov.npz                           per-observation covariances (image name, point2D
                                        index, np (xx, xy, yy), full), sampled bilinearly at
                                        the COLMAP keypoint (pixel centre at +0.5)
  board_hulls.npz                       AprilGrid outline per image (corrected tag hull),
                                        used to select the 3-D points on the board
"""
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from core.config import load_config  # noqa: E402
from core import uncertainty as U  # noqa: E402
from core.rectangle import largest_valid_rectangle  # noqa: E402
from refraction_uncertainty import CovarianceMaps, crop  # noqa: E402

DET = "/Users/fischert/robot/refrax-work/depth/board/gopro10_det.npz"


def read_images(path):
    out = {}
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    for a, b in zip(lines[::2], lines[1::2]):
        name = a.split()[-1]
        v = np.array(b.split(), float).reshape(-1, 3) if b.strip() else np.zeros((0, 3))
        out[name] = v
    return out


def bilinear_nan(M, xy):
    H, W, _ = M.shape
    u = np.clip(xy[:, 0], 0, W - 1.0)
    v = np.clip(xy[:, 1], 0, H - 1.0)
    c0 = np.minimum(np.floor(u).astype(int), W - 2)
    r0 = np.minimum(np.floor(v).astype(int), H - 2)
    a, b = (u - c0)[:, None], (v - r0)[:, None]
    acc = np.zeros((len(xy), 3))
    wsum = np.zeros((len(xy), 1))
    for dr, dc, wt in ((0, 0, (1 - a) * (1 - b)), (0, 1, a * (1 - b)), (1, 0, (1 - a) * b),
                       (1, 1, a * b)):
        val = M[r0 + dr, c0 + dc]
        ok = np.isfinite(val).all(1, keepdims=True)
        acc += np.where(ok, val * wt, 0)
        wsum += np.where(ok, wt, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return acc / wsum


def main():
    run, z0, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    os.makedirs(out, exist_ok=True)
    cfg = load_config(os.path.join(REPO, "configs/dataset.yaml"))
    cfg["correction"]["z0_fixed"] = z0
    prior = U.default_prior(None, {"depth_global": 0.25, "depth_local": 0.10})
    cm = CovarianceMaps(cfg, prior, device="mps")
    full, npc, ok = cm.maps(z0)
    box = largest_valid_rectangle(np.where(ok, 255, 0).astype(np.uint8))
    full, npc = crop(full, box), crop(npc, box)
    calib = load_config(os.path.join(run, "calibration.yaml"))
    y0, y1, x0, x1 = box
    c_out = (cm.geo.c_out[0] - x0, cm.geo.c_out[1] - y0)
    print("map", npc.shape, "calib", calib["image_dimension"], "c_out", c_out,
          calib["principal_point"], "f", cm.geo.f_out, calib["focal_length"])
    assert [npc.shape[1], npc.shape[0]] == calib["image_dimension"]
    assert np.allclose(c_out, calib["principal_point"], atol=1e-3)
    assert np.allclose(cm.geo.f_out, calib["focal_length"], rtol=1e-5)
    for d, M in (("cov", npc), ("cov_full", full)):
        os.makedirs(os.path.join(out, d), exist_ok=True)
        np.save(os.path.join(out, d, "shared.npy"), M)

    imgs = read_images(os.path.join(run, "colmap/sparse/0_txt/images.txt"))
    names, idxs, cn, cf = [], [], [], []
    for name, v in imgs.items():
        xy = v[:, :2] - 0.5            # COLMAP pixel centre (0.5, 0.5) -> Refrax (0, 0)
        names += [name] * len(v)
        idxs.append(np.arange(len(v)))
        cn.append(bilinear_nan(npc, xy))
        cf.append(bilinear_nan(full, xy))
    np.savez_compressed(os.path.join(out, "obs_cov.npz"), name=np.array(names),
                        idx=np.concatenate(idxs), np_=np.concatenate(cn),
                        full=np.concatenate(cf))
    cn = np.concatenate(cn)
    s = np.sqrt(cn[:, 0] + cn[:, 2])
    print(f"observations {len(s)}: sqrt(tr Sigma_np) median {np.nanmedian(s):.2f} px, "
          f"p95 {np.nanpercentile(s, 95):.2f}, NaN {np.isnan(s).sum()}")

    # AprilGrid outline per image in COLMAP coordinates (corrected tag corners, convex hull)
    det = np.load(DET, allow_pickle=True)["det"].item()
    geo = cm.geo
    import torch
    T = torch.as_tensor
    hulls = {}
    for st, (ids, corners) in det.items():
        if len(ids) < 8:
            continue
        raw = corners.reshape(-1, 2).astype(np.float64)
        qx, qy = U.corrected_xy(T(geo.theta0), T(raw[:, 0]), T(raw[:, 1]),
                                T(np.full(len(raw), 1.0 / z0)), geo)
        q = np.c_[qx.numpy() - x0, qy.numpy() - y0] + 0.5   # COLMAP convention
        hulls[st] = cv2.convexHull(q.astype(np.float32))
    np.savez(os.path.join(out, "board_hulls.npz"), hulls=np.array(hulls, dtype=object))
    board = board_points(os.path.join(run, "colmap/sparse/0_txt"), hulls)
    print("board points", len(board))


def board_points(model_txt, hulls):
    """3-D points whose observations fall inside the board outline (>= 5 px) in >= half
    of their images and in at least 2 (explore/depth eval_sfm.board_points)."""
    imgs = read_images(os.path.join(model_txt, "images.txt"))
    inside, total = {}, {}
    for name, v in imgs.items():
        pid = v[:, 2].astype(int)
        sel = pid >= 0
        for p in pid[sel]:
            total[p] = total.get(p, 0) + 1
        hull = hulls.get(os.path.splitext(name)[0])
        if hull is None:
            continue
        for p, (x, y) in zip(pid[sel], v[sel, :2]):
            if cv2.pointPolygonTest(hull, (float(x), float(y)), True) > 5:
                inside[p] = inside.get(p, 0) + 1
    return np.array([p for p, k in inside.items() if k >= 2 and k >= 0.5 * total[p]])


if __name__ == "__main__":
    main()
