"""Weighted vs unweighted global BA on a real COLMAP model (pycolmap + pyceres).

Runs in an environment with pycolmap >= 4.2 and pyceres (e.g. the "proto" environment of
the colmap-uncertainty workspace) and uses its covariance-weighted BA
(tools/pycolmap_weighted_ba.py: COLMAP's own Ceres problem with the reprojection
residuals whitened by chol(Sigma^-1)):

    <proto python> uncertainty/real_ba.py MODEL_DIR PREP_DIR OUT_DIR [--colmap-unc DIR]

PREP_DIR comes from uncertainty/real_prep.py (obs_cov.npz). Variants (all starting from
the same model; PINHOLE intrinsics refined or fixed):
    unweighted             unit covariance (stock COLMAP)
    np                     Sigma_np + s^2 I, s = 0.5 px
    full                   Sigma_full + s^2 I
Each result is written as a text model to OUT_DIR/<variant>/.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pycolmap

SIG_KP = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prep")
    ap.add_argument("out")
    ap.add_argument("--colmap-unc", default="/Users/fischert/robot/colmap-uncertainty")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.colmap_unc, "tools"))
    from pycolmap_weighted_ba import weighted_bundle_adjustment

    z = np.load(os.path.join(args.prep, "obs_cov.npz"))
    key = {}
    for kind in ("np_", "full"):
        C = z[kind]
        M = np.stack([np.stack([C[:, 0], C[:, 1]], -1), np.stack([C[:, 1], C[:, 2]], -1)], -2)
        M = M + SIG_KP ** 2 * np.eye(2)
        key[kind] = M
    lut = {(n, int(i)): k for k, (n, i) in enumerate(zip(z["name"], z["idx"]))}

    stats = {}
    for variant in ("unweighted", "np", "full"):
        for refine in (True, False):
            name = f"{variant}{'' if refine else '_fixedK'}"
            rec = pycolmap.Reconstruction(args.model)
            opts = pycolmap.BundleAdjustmentOptions()
            opts.refine_focal_length = refine
            opts.refine_principal_point = refine
            opts.refine_extra_params = False
            try:
                opts.ceres.solver_options.num_threads = args.threads
            except AttributeError:
                pass
            try:
                opts.ceres.solver_options.max_num_iterations = 100
            except AttributeError:
                pass
            if variant == "unweighted":
                covs = lambda image, idx: None
            else:
                M = key["np_" if variant == "np" else "full"]
                covs = (lambda M: (lambda image, idx: M[lut[(image.name, idx)]]))(M)
            t0 = time.time()
            summary, nw = weighted_bundle_adjustment(rec, covs, opts)
            d = os.path.join(args.out, name)
            os.makedirs(d, exist_ok=True)
            rec.write_text(d)
            cam = next(iter(rec.cameras.values()))
            stats[name] = dict(seconds=time.time() - t0, weighted=nw,
                               reproj_colmap=float(rec.compute_mean_reprojection_error()),
                               params=[float(p) for p in cam.params])
            print(name, json.dumps(stats[name]), flush=True)
    with open(os.path.join(args.out, "ba_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
