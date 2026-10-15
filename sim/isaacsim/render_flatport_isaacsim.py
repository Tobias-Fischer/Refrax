"""Flat-port underwater camera for NVIDIA Isaac Sim (and OceanSim), via Replicator.

UNTESTED here (needs Linux/Windows + RTX GPU) -- written against the Replicator annotator
API (rep.create.render_product, AnnotatorRegistry 'rgb' / 'distance_to_image_plane',
rep.orchestrator.step) and plain pxr.UsdGeom, which are stable across Isaac Sim 4.5-6.x.

Pipeline per frame:
  1. A pinhole USD camera ("render camera") is rendered with the intrinsics and resolution
     from core.render.plan_render_camera, mounted as a child of the real camera's pose at
     the virtual centre of the flat port (about 1 cm *behind* the real optical centre).
  2. FlatPortRenderer turns its RGB + z-depth into the image the real camera behind the
     flat port records (and its per-pixel depth / water path length).

Run (Linux + NVIDIA RTX; see sim/isaacsim/README.md):
    cd sim/isaacsim && pixi run render -- --config ../../configs/dataset.yaml --frames 20

Outputs in --out: pinhole/<i>.png, pinhole_depth/<i>.npy, underwater/<i>.png,
underwater_depth/<i>.npy (camera-frame Z per underwater pixel, inf = background),
render_camera.yaml.
"""
import argparse
import math
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "dataset.yaml"))
    ap.add_argument("--usd", default=None, help="USD stage to open (default: demo scene)")
    ap.add_argument("--out", default="out_isaacsim")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--oversample", type=float, default=2.0,
                    help="render focal length / (paraxial magnification * f); 2 = sharp corners")
    ap.add_argument("--z-near", type=float, default=0.2, help="closest scene depth (m)")
    ap.add_argument("--rt-subframes", type=int, default=8)
    ap.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True,
                    help="run without a window (--no-headless to watch the render)")
    return ap.parse_args()


def main():
    args = parse_args()

    # Plan the render camera before Kit starts (pure numpy).
    from core.config import load_config
    from core.render import FlatPortRenderer
    cfg = load_config(args.config)
    renderer = FlatPortRenderer.from_config(cfg, z_near=args.z_near, oversample=args.oversample)
    spec = renderer.spec
    print("render camera:", spec.to_dict(), "march steps:", renderer.n_steps)

    from isaacsim import SimulationApp                     # noqa: E402  (Kit must start first)
    app = SimulationApp({"headless": args.headless, "width": spec.W, "height": spec.H})

    import cv2
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, UsdGeom

    if args.usd:
        omni.usd.get_context().open_stage(args.usd)
        app.update()
    stage = omni.usd.get_context().get_stage()
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0       # metres per stage unit

    if not args.usd:
        build_demo_scene(rep, mpu)

    # --- camera rig: /World/FlatPortRig = pose of the REAL camera (USD camera axes:
    # -Z forward, +Y up, +X right), /World/FlatPortRig/RenderCam = pinhole we render.
    rig = UsdGeom.Xform.Define(stage, "/World/FlatPortRig")
    rig_t = rig.AddTranslateOp()
    rig_r = rig.AddRotateXYZOp()
    cam = UsdGeom.Camera.Define(stage, "/World/FlatPortRig/RenderCam")
    c = spec.centre / mpu                                   # Refrax frame (x r, y down, z fwd)
    cam.AddTranslateOp().Set(Gf.Vec3d(c[0], -c[1], -c[2]))  # -> USD camera frame
    # Pinhole intrinsics through the aperture: fx = W * focalLength / horizontalAperture.
    # Centred principal point (cx = (W-1)/2 in pixel-centre-at-integer convention).
    hap = 20.955
    cam.GetHorizontalApertureAttr().Set(hap)
    cam.GetVerticalApertureAttr().Set(hap * spec.H / spec.W * spec.fx / spec.fy)
    cam.GetFocalLengthAttr().Set(spec.fx * hap / spec.W)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.005 / mpu, 1000.0 / mpu))
    cam.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)

    rp = rep.create.render_product(str(cam.GetPath()), (spec.W, spec.H))
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_annot.attach([rp])
    depth_annot.attach([rp])

    for sub in ("pinhole", "pinhole_depth", "underwater", "underwater_depth"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    from render_refraction import write_spec
    write_spec(os.path.join(args.out, "render_camera.yaml"), spec, renderer.n_steps)

    for i in range(args.frames):
        # demo trajectory: slow forward motion with a little yaw/pitch (stage units)
        rig_t.Set(Gf.Vec3d(0.05 * math.sin(0.3 * i) / mpu, 0.0, -0.05 * i / mpu))
        rig_r.Set(Gf.Vec3f(-5.0 * math.sin(0.2 * i), 6.0 * math.sin(0.25 * i), 0.0))
        rep.orchestrator.step(rt_subframes=args.rt_subframes)

        rgba = np.asarray(rgb_annot.get_data())
        depth = np.asarray(depth_annot.get_data(), dtype=np.float64).squeeze() * mpu
        depth[~np.isfinite(depth) | (depth <= 0)] = np.inf      # no hit -> background
        rgb = np.ascontiguousarray(rgba[..., :3])

        uw, info = renderer.render(rgb, depth, depth_type="z")
        name = f"{i:05d}"
        cv2.imwrite(os.path.join(args.out, "pinhole", name + ".png"), rgb[..., ::-1])
        np.save(os.path.join(args.out, "pinhole_depth", name + ".npy"), depth.astype(np.float32))
        cv2.imwrite(os.path.join(args.out, "underwater", name + ".png"), uw[..., ::-1])
        np.save(os.path.join(args.out, "underwater_depth", name + ".npy"),
                np.where(info["valid"], info["Z"], np.nan).astype(np.float32))
        fin = depth[np.isfinite(depth)]
        print(f"frame {i}: depth {fin.min() if fin.size else np.nan:.3f}-"
              f"{fin.max() if fin.size else np.nan:.3f} m, valid={info['valid'].mean():.4f} near_edge={info['near_edge'].mean():.4f} "
              f"too_near={info['too_near'].mean():.4f}")

    rep.orchestrator.wait_until_complete()
    app.close()


def build_demo_scene(rep, mpu):
    """Textured-ish demo: seabed, boxes and spheres 0.3-4 m in front of the camera
    (USD camera at the origin looks along -Z, +Y up)."""
    s = 1.0 / mpu
    rep.create.light(light_type="Dome", intensity=1000)
    rep.create.light(light_type="Distant", intensity=3000, rotation=(-60, 30, 0))
    sand = rep.create.material_omnipbr(diffuse=(0.76, 0.66, 0.48), roughness=0.9)
    rep.create.plane(position=(0, -0.6 * s, -3 * s), scale=(1.0 * s, 1, 1.0 * s), material=sand)
    rng = np.random.default_rng(1)
    for k in range(25):
        z = -rng.uniform(0.3, 4.0)
        x, y = rng.uniform(-0.7, 0.7) * -z, rng.uniform(-0.4, 0.3) * -z
        mat = rep.create.material_omnipbr(diffuse=tuple(rng.uniform(0.15, 0.95, 3)))
        size = rng.uniform(0.03, 0.12) * -z
        make = rep.create.cube if k % 2 else rep.create.sphere
        make(position=(x * s, y * s, z * s), scale=size * s,
             rotation=tuple(rng.uniform(0, 90, 3)), material=mat)


if __name__ == "__main__":
    main()
