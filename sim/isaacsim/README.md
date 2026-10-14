# Flat-port camera in Isaac Sim / OceanSim (untested adapter)

`render_flatport_isaacsim.py` renders a pinhole RGB + depth image with Replicator and turns
it into the image of a camera behind a flat port with `core.render.FlatPortRenderer`
(see `docs/explore_sim.md`). Requires Linux x86_64 (GLIBC >= 2.35) or Windows, an NVIDIA
RTX GPU and driver supported by Isaac Sim; it does **not** run on macOS.

```bash
cd sim/isaacsim
pixi install                      # pulls isaacsim 6.2.0.0 from pypi.nvidia.com (~15 GB)
pixi run render -- --config ../../configs/dataset.yaml --frames 20 --out out
# your own stage (camera pose: edit the rig in the script or drive /World/FlatPortRig):
pixi run render -- --usd /path/to/reef.usd --frames 100
```

Isaac Sim 5.x: change `python = "3.11.*"` and `isaacsim==5.1.0.0` in `pixi.toml`.
Without pixi: `pip install "isaacsim[all,extscache]==6.2.0.0" --extra-index-url
https://pypi.nvidia.com opencv-python-headless pyyaml scipy` in a Python 3.12 env.

Checks to do on the first run:
1. `render_camera.yaml`: W/H/fx and `centre` (z about -10 mm for the GoPro example).
2. The printed depth range must match the scene (if not, check `metersPerUnit` handling).
3. Run `python -m sim.check_render out` from the repo root: it re-removes refraction from
   `underwater/` with Refrax using `underwater_depth/` and compares with `pinhole/`.
4. The demo scene's primitive sizes assume Replicator primitives of 1 stage unit; if objects
   look 100x too large, divide `size` in `build_demo_scene` by 100.

## OceanSim
OceanSim's `UW_Camera` (isaacsim/oceansim/sensors/UW_Camera.py) reads `LdrColor` and
`distance_to_camera` and launches the `UW_render` Warp kernel. To give it a flat port:
- set the camera prim's intrinsics/resolution/offset to `FlatPortRenderer(...).spec`
  (as in this script),
- before the kernel: `uw_rgb, info = renderer.render(raw_rgba, depth, depth_type="range")`,
- feed the kernel `info["water_range"]` (water path length from the port, NaN/inf
  replaced by a large value) instead of the pinhole range, and output at the real camera
  resolution. `core.render.apply_water` is a NumPy version of the same model.
