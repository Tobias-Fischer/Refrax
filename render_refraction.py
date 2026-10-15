"""Adds flat-port refraction to simulator renders (the inverse of remove_refraction.py).

Usage:
    python render_refraction.py configs/dataset.yaml plan [--oversample 2] [--z-near 0.2]
        prints / writes the pinhole camera the simulator must render (render_camera.yaml)
    python render_refraction.py configs/dataset.yaml apply SIM_DIR [--depth-type z|range]
        SIM_DIR/pinhole/<name>.png + SIM_DIR/pinhole_depth/<name>.npy (metres) rendered with
        that camera -> SIM_DIR/underwater/<name>.png and SIM_DIR/underwater_depth/<name>.npy

The output camera is the config's camera (W, H, fx, fy, cx, cy, radtan distortion) behind
its housing. The render camera has the same orientation as the real camera, and its optical
centre is offset by `centre` (metres; x right, y down, z forward) -- about 1 cm behind the
real centre at the port's virtual centre. See docs/explore_sim.md.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import yaml

from core.config import load_config
from core.render import FlatPortRenderer, PinholeSpec


def write_spec(path, spec, n_steps):
    d = spec.to_dict()
    d = {k: (float(v) if not isinstance(v, (list, int)) else v) for k, v in d.items()}
    d["march_steps"] = n_steps
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def read_spec(path):
    d = yaml.safe_load(open(path))
    return PinholeSpec(int(d["W"]), int(d["H"]), d["fx"], d["fy"], d["cx"], d["cy"],
                       np.array(d["centre"], float))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("command", choices=["plan", "apply"])
    ap.add_argument("sim_dir", nargs="?", default=".")
    ap.add_argument("--oversample", type=float, default=1.0)
    ap.add_argument("--z-near", type=float, default=0.2)
    ap.add_argument("--centre", default="virtual", help="virtual | origin")
    ap.add_argument("--depth-type", default="z", choices=["z", "range"])
    ap.add_argument("--spec", default=None, help="render_camera.yaml (default: SIM_DIR/render_camera.yaml if present)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec_path = args.spec or os.path.join(args.sim_dir, "render_camera.yaml")
    spec = read_spec(spec_path) if args.command == "apply" and os.path.exists(spec_path) else None
    r = FlatPortRenderer.from_config(cfg, spec=spec, z_near=args.z_near,
                                     oversample=args.oversample, centre=args.centre)

    if args.command == "plan":
        os.makedirs(args.sim_dir, exist_ok=True)
        write_spec(spec_path, r.spec, r.n_steps)
        print(open(spec_path).read())
        print(f"Written {spec_path}")
        return

    out_rgb = os.path.join(args.sim_dir, "underwater")
    out_d = os.path.join(args.sim_dir, "underwater_depth")
    os.makedirs(out_rgb, exist_ok=True)
    os.makedirs(out_d, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(args.sim_dir, "pinhole", "*.png"))):
        name = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        depth = np.load(os.path.join(args.sim_dir, "pinhole_depth", name + ".npy"))
        if img.shape[:2] != (r.spec.H, r.spec.W) or depth.shape[:2] != (r.spec.H, r.spec.W):
            raise ValueError(f"{name}: render must be {r.spec.W}x{r.spec.H} (see plan)")
        uw, info = r.render(img, depth.squeeze(), depth_type=args.depth_type)
        cv2.imwrite(os.path.join(out_rgb, name + ".png"), uw)
        np.save(os.path.join(out_d, name + ".npy"),
                np.where(info["valid"], info["Z"], np.nan).astype(np.float32))
        print(f"{name}: valid={info['valid'].mean():.4f} near_edge={info['near_edge'].mean():.4f}")


if __name__ == "__main__":
    main()
