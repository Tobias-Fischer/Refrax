"""Refractive (port-tilt) global BA with the C++ flat-port cost of colmap-uncertainty
(branch port-tilt) through pycolmap, on a COLMAP model of Refrax-corrected images.

Runs in the colmap-uncertainty default environment (patched pycolmap + pyceres):

    pixi run --manifest-path ../colmap-uncertainty/pixi.toml python tilt/pycolmap_flatport_ba.py \
        MODEL_DIR FLATPORT_CONFIG OUT_DIR [--metric-scale 0.11347] [--refine-d0] [--fixed-air]
        [--check-python RESIDUALS.npz]

MODEL_DIR: COLMAP model (bin or txt) whose camera is the declared pinhole of the Refrax
correction described by FLATPORT_CONFIG (configs/flatport/*.txt). The BA is COLMAP's own
DefaultBundleAdjuster with BundleAdjustmentOptions.flatport set: every keypoint is mapped
back to the raw image and the flat-port reprojection error (port tilt, in-air intrinsics,
optionally d0) replaces the pinhole one. Writes the refined model, the estimate and the
to OUT_DIR.
"""
import argparse
import json
import os
import time

import numpy as np
import pycolmap
import pycolmap.cost_functions  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--metric-scale", type=float, default=None)
    ap.add_argument("--refine-d0", action="store_true")
    ap.add_argument("--fixed-air", action="store_true")
    ap.add_argument("--loss", default="HUBER", help="TRIVIAL | HUBER | CAUCHY | SOFT_L1")
    ap.add_argument("--loss-scale", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-iter", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rec = pycolmap.Reconstruction(args.model)
    fp = pycolmap.read_flatport_options(args.config)
    if args.metric_scale is not None:
        fp.metric_scale = args.metric_scale
    fp.refine_d0 = args.refine_d0
    if args.fixed_air:
        fp.refine_air_focal_length = False
        fp.refine_air_principal_point = False
    fp.update_declared_camera = True
    fp.output_path = os.path.join(args.out, "flatport_state.txt")

    opts = pycolmap.BundleAdjustmentOptions()
    opts.flatport = fp
    opts.print_summary = True
    opts.ceres.loss_function_type = getattr(
        pycolmap.LossFunctionType, args.loss) if hasattr(pycolmap, "LossFunctionType") \
        else getattr(type(opts.ceres.loss_function_type), args.loss)
    opts.ceres.loss_function_scale = args.loss_scale
    opts.ceres.solver_options.num_threads = args.threads
    opts.ceres.solver_options.max_num_iterations = args.max_iter
    opts.ceres.solver_options.function_tolerance = 1e-9
    cfg = pycolmap.BundleAdjustmentConfig()
    for image_id in rec.reg_image_ids():
        cfg.add_image(image_id)
    cfg.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)

    t0 = time.time()
    ba = pycolmap.create_default_ceres_bundle_adjuster(opts, cfg, rec)
    summary = ba.solve()
    t_solve = time.time() - t0
    st = fp.state
    out = dict(model=args.model, seconds=t_solve,
               tilt_deg=[float(np.degrees(st.housing[0])), float(np.degrees(st.housing[1]))],
               d0=float(st.housing[2]), air_params=list(map(float, st.air_params)),
               metric_scale=float(st.last_metric_scale),
               declared_camera=list(map(float, next(iter(rec.cameras.values())).params)),
               termination=str(summary.termination_type))
    # (posterior covariance: use tilt/real.py; the housing block is not exposed as a
    #  numpy view by the bindings, so pyceres.Covariance cannot address it)
    rec.write(args.out)
    with open(os.path.join(args.out, "flatport_ba.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
