"""Feed an estimated housing back into the Refrax correction and re-run pinhole SfM BA
(docs/explore_tilt.md, §4b "consistency").

    python tilt/feedback.py MODEL_TXT CALIB_YAML CONFIG Z0 REAL_JSON OUT_DIR --hulls H.npz
                            [--colmap /path/to/colmap] [--scale s]

For each variant the keypoints of MODEL_TXT (a COLMAP model of Refrax-corrected images,
fixed depth Z0, nominal housing) are mapped back to the raw image exactly and re-corrected
with the variant's parameters (same declared pinhole, same depth Z0), i.e. exactly what
re-running remove_refraction.py with the new n_port / in-air intrinsics would give at
those keypoints (the detector response to the slightly different resampling is ignored).
The model with the new keypoints is then refined by a stock COLMAP global BA (PINHOLE,
focal length and principal point refined, poses and points refined) and the AprilGrid
board planarity is evaluated. Variants:
    nominal          the original keypoints (the current pipeline)
    tilt             n_port from the estimated tilt
    tilt+K           ... and the estimated in-air fx, fy, cx, cy
    K                in-air fx, fy, cx, cy only (from the tilt-free refractive BA)
REAL_JSON: output of tilt/real.py (uses variants "K+tilt" and "K").
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from core.config import load_config  # noqa: E402
from core import uncertainty as U  # noqa: E402
from tilt.model import Decl, raw_from_corrected, IDX  # noqa: E402
from tilt.real import load_model, board_ids, planarity  # noqa: E402

COLMAP = "/Users/fischert/robot/colmap-uncertainty/install/bin/colmap"


def rewrite_images(src, dst, mapper):
    """Copy a text model, mapping every keypoint (x, y) -> mapper(xy array)."""
    os.makedirs(dst, exist_ok=True)
    for f in ("cameras.txt", "points3D.txt"):
        with open(os.path.join(src, f)) as a, open(os.path.join(dst, f), "w") as b:
            b.write(a.read())
    lines = open(os.path.join(src, "images.txt")).read().splitlines()
    out = []
    head = [l for l in lines if l.startswith("#")]
    body = [l for l in lines if not l.startswith("#")]
    for a, b in zip(body[::2], body[1::2]):
        v = np.array(b.split(), float).reshape(-1, 3) if b.strip() else np.zeros((0, 3))
        if len(v):
            v[:, :2] = mapper(v[:, :2])
        out.append(a)
        out.append(" ".join(f"{x:.6f} {y:.6f} {int(p)}" for x, y, p in v))
    with open(os.path.join(dst, "images.txt"), "w") as f:
        f.write("\n".join(head + out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("calib")
    ap.add_argument("config")
    ap.add_argument("z0", type=float)
    ap.add_argument("real_json")
    ap.add_argument("out")
    ap.add_argument("--hulls", required=True)
    ap.add_argument("--colmap", default=COLMAP)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.config)
    cal = load_config(args.calib)
    geo = U.Geometry(cfg, 1.0)
    decl = Decl(cal["focal_length"], cal["principal_point"], geo.R0, geo.mu_a)
    est = json.load(open(args.real_json))["variants"]

    def theta_from(variant, keys):
        th = geo.theta0.copy()
        if variant is None:
            return th
        p = est[variant]["params"]
        for k in keys:
            v = p[k][0]
            th[IDX[k]] = np.radians(v) if k.startswith("tilt") else v
        return th

    variants = {"nominal": theta_from(None, [])}
    if "K+tilt" in est:
        variants["tilt"] = theta_from("K+tilt", ["tilt_x", "tilt_y"])
        variants["tilt+K"] = theta_from("K+tilt", ["tilt_x", "tilt_y", "fx", "fy", "cx", "cy"])
    if "K" in est:
        variants["K"] = theta_from("K", ["fx", "fy", "cx", "cy"])

    m0 = load_model(args.model)
    hulls = np.load(args.hulls, allow_pickle=True)["hulls"].item()
    bids = board_ids(m0, hulls)
    pid_board = set(m0["pid"][bids].tolist())
    res = {}
    for name, th in variants.items():
        def mapper(xy, th=th):
            q = xy - 0.5
            u = raw_from_corrected(q, geo.theta0, decl, args.z0)
            T = lambda a: torch.as_tensor(np.asarray(a, np.float64))
            qx, qy = U.corrected_xy(T(th), T(u[:, 0]), T(u[:, 1]),
                                    T(np.full(len(u), 1.0 / args.z0)), decl)
            qn = np.c_[qx.numpy(), qy.numpy()] + 0.5
            bad = ~np.isfinite(qn).all(1)
            qn[bad] = xy[bad]
            return qn
        d_in = os.path.join(args.out, name, "input")
        d_out = os.path.join(args.out, name, "ba")
        rewrite_images(args.model, d_in, mapper)
        os.makedirs(d_out, exist_ok=True)
        cmd = [args.colmap, "bundle_adjuster", "--input_path", d_in, "--output_path", d_out,
               "--BundleAdjustment.refine_focal_length", "1",
               "--BundleAdjustment.refine_principal_point", "1",
               "--BundleAdjustment.refine_extra_params", "0",
               "--BundleAdjustmentCeres.max_num_iterations", "100",
               "--BundleAdjustment.use_keypoint_covariances", "0"]
        subprocess.run(cmd, check=True, capture_output=True,
                       env=dict(os.environ, OMP_NUM_THREADS=str(args.threads)))
        subprocess.run([args.colmap, "model_converter", "--input_path", d_out, "--output_path",
                        d_out, "--output_type", "TXT"], check=True, capture_output=True)
        m = load_model(d_out)
        sel = np.array([i for i, p in enumerate(m["pid"]) if p in pid_board])
        X = m["X"][sel]
        # reprojection RMS (pinhole) of the refined model
        Pc = np.einsum("nij,nj->ni", m["R"][m["ci"]], m["X"][m["pj"]]) + m["t"][m["ci"]]
        fx, fy, cx, cy = m["K"]
        e = m["q"] - np.c_[fx * Pc[:, 0] / Pc[:, 2] + cx, fy * Pc[:, 1] / Pc[:, 2] + cy]
        r = dict(planarity=planarity(X), K=list(m["K"]),
                 reproj_rms=float(np.sqrt((e ** 2).sum(1).mean())),
                 theta=[float(v) for v in th])
        res[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != "theta"}), flush=True)
        json.dump(res, open(os.path.join(args.out, "feedback.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
