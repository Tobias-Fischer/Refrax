"""Self-calibration of the GoPro HERO11 + flat port from an underwater COLMAP model.

    python depth_aware/lizard_calib.py <raw model_txt> [--mu-w 1.34] [--out-dir configs]

No in-air calibration exists for the Lizard Island camera. The paper's recipe (git history
of configs/lizardisland.yaml: fx = 383.418 / 1.33 on 593x518 images, cx, cy from the same
self-calibration, distortion 0, d0 = 2 cm, tglass = 2 mm) is:

  1. run COLMAP on the raw underwater images with refined intrinsics (self-calibration);
     this absorbs the refraction at the scene's depth into a perspective + radial model;
  2. in-air focal length = underwater focal length / mu_w (the paraxial magnification of a
     flat port), principal point unchanged, no lens distortion;
  3. housing: nominal d0 = 2 cm, tglass = 2 mm (Z / d0 dominates, tglass is negligible,
     paper Sec. III-E), mu_w of sea water;
  4. z0: the operating depth (paper: planarity sweep; here: median metric depth of the
     sparse points, metric scale from GPS, see lizard_eval.py).

This script implements step 2 in two ways:

  paper   fx_air = fx_w / mu_w, fy_air = fy_w / mu_w, (cx, cy) unchanged, k = p = 0
          (exactly the paper's rule; the GoPro's own lens distortion is dropped)
  snell   the whole underwater model is converted: every pixel's underwater ray direction
          (COLMAP's OPENCV model, undistorted) is refracted back into air with Snell's law
          (sin theta_air = mu_w / mu_a * sin theta_w; parallel interfaces, so the glass
          drops out), and an in-air OPENCV model (fx, fy, cx, cy, k1, k2, p1, p2) is fitted
          to pixel -> air ray by least squares. This keeps the lens distortion the
          self-calibration found and is exact for distant scenes; the remaining
          depth-dependent part (the d0 term) is what Refrax corrects with z0.

The conversion assumes the port normal is the optical axis and the camera centre is where
COLMAP's underwater pinhole is (the 1 cm virtual-centre offset only matters through d0).
Writes the fitted parameters to stdout / JSON.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_camera(model_txt):
    for line in open(f"{model_txt}/cameras.txt"):
        if line.startswith("#"):
            continue
        e = line.split()
        return e[1], int(e[2]), int(e[3]), np.array(list(map(float, e[4:])))
    raise ValueError("no camera")


def opencv_undistort(u, v, p):
    fx, fy, cx, cy, k1, k2, p1, p2 = p
    pts = np.stack([u, v], -1).reshape(-1, 1, 2).astype(np.float64)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    und = cv2.undistortPoints(pts, K, np.array([k1, k2, p1, p2]),
                              criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12))
    return und[:, 0, 0], und[:, 0, 1]


def opencv_distort(x, y, p):
    fx, fy, cx, cy, k1, k2, p1, p2 = p
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return fx * xd + cx, fy * yd + cy


def water_to_air(x, y, mu_w, mu_a=1.0):
    """Normalised underwater ray (x, y, 1) -> normalised in-air ray, Snell at a flat port."""
    r = np.hypot(x, y)
    s_w = r / np.sqrt(1 + r * r)
    s_a = mu_w / mu_a * s_w
    if np.any(s_a >= 1):
        raise ValueError("ray beyond the critical angle")
    r_a = s_a / np.sqrt(1 - s_a * s_a)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(r > 0, r_a / r, mu_w / mu_a)
    return x * g, y * g


def snell_fit(W, H, p_w, mu_w, step=8):
    u, v = np.meshgrid(np.arange(0, W, step, dtype=float), np.arange(0, H, step, dtype=float))
    u, v = u.ravel(), v.ravel()
    xw, yw = opencv_undistort(u, v, p_w)
    xa, ya = water_to_air(xw, yw, mu_w)
    fx0, fy0 = p_w[0] / mu_w, p_w[1] / mu_w

    def resid(q):
        uu, vv = opencv_distort(xa, ya, q)
        return np.concatenate([uu - u, vv - v])

    q0 = np.array([fx0, fy0, p_w[2], p_w[3], 0, 0, 0, 0])
    sol = least_squares(resid, q0, method="lm", xtol=1e-14, ftol=1e-14)
    r = sol.fun.reshape(2, -1)
    err = np.hypot(r[0], r[1])
    return sol.x, dict(rms_px=float(np.sqrt(np.mean(err ** 2))), max_px=float(err.max()))


def radial_profile(W, H, p_w, p_air, mu_w, n=50):
    """Underwater and in-air ray angle along the image diagonal (for the doc)."""
    t = np.linspace(0, 1, n)
    u = p_w[2] + t * (W - 1 - p_w[2])
    v = p_w[3] + t * (H - 1 - p_w[3])
    xw, yw = opencv_undistort(u, v, p_w)
    xa, ya = water_to_air(xw, yw, mu_w)
    return dict(r_px=np.hypot(u - p_w[2], v - p_w[3]).tolist(),
                theta_w_deg=np.degrees(np.arctan(np.hypot(xw, yw))).tolist(),
                theta_air_deg=np.degrees(np.arctan(np.hypot(xa, ya))).tolist())


CONFIG = """# GoPro HERO11 Black (8:7 photo, 5568x4872 downscaled 6x to 928x812) in a flat-port housing
# on the Lizard Island ASV (LIRS Mar 2024, South Palfrey 1, GoPro1). No in-air calibration
# exists: the camera is SELF-CALIBRATED from the underwater images (docs/explore_lizard.md):
#   COLMAP with a refined OPENCV model on the raw images (subset of {n_images} images)
#   -> {how}
# Housing: nominal values of the paper (d0 = 2 cm, tglass = 2 mm), sea water mu_w = {mu_w}.
# z0: median camera-to-scene depth of the raw SfM points, metric scale from GPS ({z0_note}).
{generated}
camera:
  W: {W}
  H: {H}
  fx: {fx:.4f}
  fy: {fy:.4f}
  cx: {cx:.4f}
  cy: {cy:.4f}
  k1: {k1:.6f}
  k2: {k2:.6f}
  p1: {p1:.6f}
  p2: {p2:.6f}

housing:
  n_port: [0.0, 0.0, 1.0]
  mu_a: 1.0
  mu_g: 1.47
  mu_w: {mu_w}
  d0: 0.02
  tglass: 0.002

paths:
  rgb_dir: "{rgb_dir}"
  output_dir: "{work}/corrected/images"
  mask_dir: "{work}/corrected/mask"
  depth_dir: null

correction:
  z0_fixed: {z0}
  zoom: null
  fit_canvas: true
  step_size: 1
  crop_valid_bbox: true
  depth_crop: reference
  virtual_center: 0.0
  depth_range: [1.0, 8.0]
"""


def write_config(path, p, mu_w, z0, how, z0_note, rgb_dir, work, W, H, n_images):
    keys = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]
    txt = CONFIG.format(W=W, H=H, mu_w=mu_w, z0=z0, how=how, z0_note=z0_note, rgb_dir=rgb_dir,
                        work=work, n_images=n_images,
                        generated="# Generated by depth_aware/lizard_calib.py --write-config.",
                        **dict(zip(keys, p)))
    open(path, "w").write(txt)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--mu-w", type=float, default=1.34)
    ap.add_argument("--json", default=None)
    ap.add_argument("--write-config", nargs=2, metavar=("SNELL_YAML", "PAPER_YAML"),
                    help="write Refrax configs for both conversions")
    ap.add_argument("--z0", type=float, default=None)
    ap.add_argument("--z0-note", default="")
    ap.add_argument("--rgb-dir", default="/Users/fischert/robot/refrax-work/lizard/raw928")
    ap.add_argument("--work", default="/Users/fischert/robot/refrax-work/lizard")
    ap.add_argument("--n-images", type=int, default=476)
    args = ap.parse_args()
    model, W, H, p_w = read_camera(args.model)
    assert model == "OPENCV", model
    paper = np.array([p_w[0] / args.mu_w, p_w[1] / args.mu_w, p_w[2], p_w[3], 0, 0, 0, 0])
    snell, fit = snell_fit(W, H, p_w, args.mu_w)
    # how far is the paper's (distortion-free) model from the converted one, in px?
    u, v = np.meshgrid(np.arange(0, W, 8, dtype=float), np.arange(0, H, 8, dtype=float))
    xw, yw = opencv_undistort(u.ravel(), v.ravel(), p_w)
    xa, ya = water_to_air(xw, yw, args.mu_w)
    pu, pv = opencv_distort(xa, ya, paper)
    d_paper = np.hypot(pu - u.ravel(), pv - v.ravel())
    out = dict(W=W, H=H, mu_w=args.mu_w, underwater_opencv=p_w.tolist(),
               paper=paper.tolist(), snell=snell.tolist(), snell_fit=fit,
               paper_vs_selfcal_px=dict(rms=float(np.sqrt(np.mean(d_paper ** 2))),
                                        max=float(d_paper.max())),
               profile=radial_profile(W, H, p_w, snell, args.mu_w))
    names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]
    for k in ("underwater_opencv", "paper", "snell"):
        print(f"{k:18s} " + " ".join(f"{n}={x:.5g}" for n, x in zip(names, out[k])))
    print("snell fit residual", fit, " paper model vs self-calibrated rays",
          out["paper_vs_selfcal_px"])
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
    if args.write_config:
        snell_yaml, paper_yaml = args.write_config
        write_config(snell_yaml, snell, args.mu_w, args.z0,
                     "Snell conversion of the whole underwater model to an in-air OPENCV model "
                     f"(fit RMS {fit['rms_px']:.3f} px)", args.z0_note, args.rgb_dir, args.work,
                     W, H, args.n_images)
        write_config(paper_yaml, paper, args.mu_w, args.z0,
                     "paper's rule: f_air = f_water / mu_w, principal point kept, no distortion",
                     args.z0_note, args.rgb_dir, args.work, W, H, args.n_images)


if __name__ == "__main__":
    main()
