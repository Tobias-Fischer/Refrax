# Exploration: depth-aware refraction removal

Branch `explore/depth`. This is a research exploration, not a finished feature. All numbers come
from the scripts in `depth_aware/`, and every figure is in `docs/figures/`.

## TL;DR

1. **Most of Refrax's depth dependence is a choice of viewpoint, not an error.** Refrax projects
   the corrected image from the camera centre. The refracted water rays nearly meet about 10 mm
   *behind* the camera (−10.3 mm for the GoPro 10 config, `d0` = 2 cm). Projecting from there
   (`correction.virtual_center: auto`, new) cuts the depth dependence of the map about 6x:
   - 0.44 px instead of 2.8 px RMS over 0.5–8 m;
   - the "virtual centre" rows in the sensitivity table.

   It costs nothing, needs no depth, and the corrected images are still exact pinhole images. The
   camera simply sits 1 cm further back.
2. **The error that can bend a reconstruction is the non-pinhole part, and it is small.** Synthetic
   reef, correct median depth as z0:
   - near reef (0.43–1.1 m): fixed-depth Refrax is **0.22 px RMS / 1.1 px max** away from the
     nearest pinhole camera;
   - far reef (1.7–4.4 m): **0.06 / 0.29 px**.

   Per-pixel depth can only remove this.
3. **Per-pixel depth can make things worse.** With 10 % smooth depth error (typical for aligned mono
   depth) at the camera centre, the depth-aware correction is 1.03 px RMS, about 5x *worse* than
   fixed depth. It only pays off together with the virtual centre (0.17 px) or with depth better
   than a few %.
4. **The constant depth matters far more than per-pixel depth.**
   - A plane at 0.3 m corrected with the shipped `z0_fixed: 1.0` keeps 2.4 px RMS (8.3 px max) of
     non-pinhole error.
   - At 4–8 m it keeps 0.8–0.9 px RMS.
   - Getting the per-sequence or per-image depth within ±25 % is the main lever. Per-pixel depth
     is a second-order refinement.
5. **Metric scale can be self-calibrated, coarsely.** The needed accuracy is modest: ±25 % gives
   ≤0.25 px RMS at ~1 m with the virtual centre. A refractive-BA proxy (`scale_by_ba`, §5) finds a
   consistent minimum at a median depth of **0.94–1.02 m** in three independent runs, one of them
   started from a deliberately wrong 0.4 m prior. That is ±4 %, and suggests the shipped
   `z0_fixed = 1.0` was right for this sequence. The minimum is very shallow, though: it is only
   0.002–0.006 px below the unmodified model, and flat to 0.03 px between 0.7 and 2 m. Treat it as
   a sanity check on the prior, or use it for near, 3D scenes. A known altitude, laser scaler or
   stereo baseline is still the robust route. The AprilGrid cannot fix the scale: its per-image
   refractive plane fit is unobservable under a 1.9 px corner-noise floor.
6. **Real data (gopro10, 128 images, a flat AprilGrid in a tank) shows no gain, sometimes a loss.**
   The dense depth-aware pipeline (COLMAP + Depth Anything V2) runs end to end. Compared with
   fixed-depth Refrax at z0 = 1 m:
   - reprojection error is **0.516 vs 0.483 px**;
   - track length is 13.80 vs 13.91;
   - registered images are 128/128 in both;
   - board planarity is unchanged: 0.159 vs 0.162 %, |κ| ≤ 0.003.

   Per-image median depth is neutral (0.482 px). With the virtual centre the dense run broke the
   reconstruction twice (bent board, κ = 1.8), even though every per-image check passes. This is
   unexplained; see §6. When started from a wrong prior (z0 = 0.4 m), the dense run did fix the
   planarity (κ −0.009 → +0.002, rel. RMSE 0.172 → 0.155 %), but reprojection got worse (0.488 →
   0.520 px). The scene is planar with ~5 % depth variation per image, so on this data
   depth-awareness can only repair a wrong z0.

Recommendation: ship the virtual centre (or at least evaluate it on the paper's sequences), and
choose `z0_fixed` per sequence (or per image from SfM). Keep the dense depth path for near-range,
strongly 3D scenes, run it with the virtual centre, and evaluate it on data with ground truth.

---

## 1. Background: how depth enters Refrax

A pixel's water ray starts at the port exit point `P2` with direction `r_w`. Refrax intersects it
with the plane `Z = z0` and projects the hit point with the in-air intrinsics, zoomed:
`forward_map` → `invert_map` → `cv2.remap`.

- The paper (§III-D) uses one constant `Z0` per sequence: the mean scene depth, chosen by a
  planarity sweep (Fig. 7).
- It names per-pixel depth, from stereo or monocular prediction, as future work.
- Its Lizard Island data (HERO11, 1–8 m, ASV) is the varied-depth case. Malaysia (HERO13) is
  near-planar at 0.5–1.5 m.

The code already accepts per-image depth maps (`paths.depth_dir`). This branch adds three
things:

| change | where | why |
| --- | --- | --- |
| `correction.virtual_center: <metres> \| auto` (default 0 = old behaviour) | `core/undistort.forward_map(center_z=)`, `core/virtual_center.py`, `remove_refraction.py` | puts the corrected pinhole where the water rays nearly converge; see §3 |
| `correction.depth_crop: reference` (new default) or `per_image` (old) | `remove_refraction.py` | Depth mode used to crop every image to its own valid rectangle, so the principal point changed per image and no calibration was written. Now all images keep the crop and intrinsics of the reference depth `z0_fixed`, one `new_calibration.yaml` is written, and pixels without a source are black and 0 in the per-image mask. COLMAP then gets one camera, which is what SfM needs. |
| `osx-arm64` platform, `depth` environment (COLMAP 3.11.1, PyTorch with MPS, transformers) | `pixi.toml` | runs on Apple silicon; `pixi run -e depth ...` |

**Depth convention, a subtle point.** The depth map is indexed on the grid that
`compute_housing_geometry` traces. That is the *lens-undistorted* in-air pixel grid ("ideal"
grid). It is not the recorded, distorted raw image. The README says "raw underwater pixel grid".
For the GoPro 10 config the two differ by up to **25 px** at the corners (median 3.8 px). The
offset is irrelevant for smooth depth but matters at depth edges. `depth_aware/common.py` documents the
frames, and the pipeline writes depth on the ideal grid.

## 2. Pipeline (`depth_aware/pipeline.py`, `pixi run -e depth depth-pipeline`)

```
raw images ──(a) fixed-depth Refrax at z0 (reference frame, one calibration)──► corrected_0
corrected_0 ──(b) COLMAP: SIFT, exhaustive matching, mapper with fixed PINHOLE──► model_0
model_0 ──(c) per image: points3D -> camera frame Z_c (track >= 3, err <= 2 px)──► sparse depth (SfM units)
corrected_0 ──(d) Depth Anything V2-Large (MPS) ─► relative disparity
              robust per-image fit 1/Z_c ~ a*disp + b (Huber IRLS + 3 sigma trim) ─► dense Z_c
(scale) metres per SfM unit s: given, or refractive-BA self-calibration (scale_by_ba)
(e) Z = s*Z_c + center_z, resampled to the ideal raw grid through the z0 forward map
    (each ideal pixel -> its position in corrected_0 -> bilinear lookup) ─► depth/<name>.npy
(f) depth-aware Refrax in the same reference frame ─► corrected_1 ─► COLMAP ─► model_1   (iterate)
```

Design decisions:

- **Fit in the corrected frame, write on the raw grid.**
  - SfM keypoints and mono depth both live in the corrected (pinhole-like) image, and DA is trained
    on pinhole images. So the fit is done there.
  - The dense map is carried to the raw grid with the exact forward map of the correction that
    produced the image.
  - Sampling the forward map avoids inverting it.
- **Inverse-depth affine fit.** DAv2 (relative) predicts affine-invariant disparity. On gopro10
  (128 images, ~4.4k sparse points each) the median held-out relative depth error (5-fold, untrimmed)
  is:

  | model | median held-out error |
  | --- | --- |
  | constant depth (median) | 5.2 % |
  | scale-only `1/Z = a·disp` | 18.1 % (DA's disparity has a large offset) |
  | **affine `1/Z = a·disp + b`** | **0.69 %** |
  | spatially varying gain (bilinear in x, y) + shift | 0.26 % |

  - The affine fit is used. `--fit spatial` is available; the spatial model overfits more easily
    on frames with few points.
  - In 2/128 frames the affine gain came out ≤ 0, and those fell back to scale-only.
  - This mirrors ORCA_splat's lessons: fit in the model's native space; affine fits can flip sign
    on flat frames, so gate them; score on held-out points, untrimmed.
  - Dense depth is clamped to `[0.5·p2, 2·p98]` of the frame's sparse depth.
- **Why Depth Anything V2-Large (`depth-anything/Depth-Anything-V2-Large-hf`).**
  - It runs on MPS through plain `transformers`: ~0.9 s per 1697x955 frame on the M5 after warm-up,
    12 s load.
  - It has no extra dependencies, and relative disparity is exactly what a per-image sparse
    alignment needs.
  - DA3 was considered:
    - ORCA_splat vendors it, and its multi-view mode would be the natural next step;
    - but ORCA found that DA3METRIC's metric head saturates underwater (gain off by 1.4–2x) and
      that affine fits of DA3 break on flat seabed frames;
    - installing it needs a vendored, trimmed package; and only DA3METRIC is Apache-2.0 (the other
      checkpoints are CC BY-NC).
  - **License:** DAv2-Large is CC BY-NC-4.0. Only the Small model is Apache-2.0. Set
    `REFRAX_MONO_MODEL=depth-anything/Depth-Anything-V2-Small-hf` for a permissive pipeline.
  - Mono quality is not the bottleneck here: 0.7 % held-out error is far below what matters (§4).
- **Reference output frame.** Fixed-depth and depth-aware images share the crop and intrinsics, so
  COLMAP runs on both are directly comparable and keep one camera.
- Large outputs (COLMAP DBs, depth maps, corrected images) are written to
  `/Users/fischert/robot/refrax-work/depth/gopro10/<z0>_<centre>/it*`.

Commands:

```bash
pixi run -e depth depth-sensitivity           # §4 tables and figures (seconds)
pixi run -e depth depth-synthetic             # §4 synthetic experiment (~1 min)
pixi run -e depth depth-pipeline --z0 1.0 --step 2 --iters 1              # §6 (~40 min on CPU COLMAP)
pixi run -e depth depth-pipeline --z0 1.0 --center-z -0.01034 --step 2    # with the virtual centre
pixi run -e depth depth-pipeline --z0 1.0 --mode median                    # one depth per image
pixi run -e depth python depth_aware/eval_sfm.py <run dir>[@centre_mm] ...      # SfM stats + board planarity
pixi run -e depth python depth_aware/eval_board.py --runs name=<z0 or depth dir>[@centre_mm] ...
```

## 3. The virtual centre

`P2` lies on the port, about 2 cm in front of the camera, and `r_w` is bent towards the axis.
Extended backwards, the water rays cross the optical axis behind the camera centre:

- paraxially at `d0·(1 − μw/μa) ≈ −6.8 mm`;
- oblique rays further back.

Projecting the scene from a point on the axis at `c` gives a depth-independent image for every ray
that passes through that point. `core/virtual_center.optimal_center_z` picks the `c` that
minimises the RMS change of the map over a depth range. For the GoPro 10 config the result is
**c = −10.34 mm, independent of the range** (0.3–1, 0.5–2, 1–8 and 0.5–8 m all give −10.34 mm).
This is the Pinax idea (Łuczyński et al. 2017) applied inside Refrax.

Consequences:

- Fixed-depth Refrax at the camera centre is already, approximately, a *pinhole at c with focal
  `zoom·(z0−c)/z0`*. Its large apparent depth sensitivity (paper Fig. 3) is mostly an isotropic
  magnification that SfM absorbs as a 1 cm shift of the camera centre.
- The paper's sensitivity plots (RMSE of the displacement field vs Z, d0) therefore overstate the
  depth sensitivity relevant for SfM. The quantity that matters is the distance to the *nearest*
  pinhole ("non-pinhole residual" below).
- With per-pixel depth any centre is exact, but the virtual centre makes the result ~6x less
  sensitive to *depth errors* (§4).

## 4. How accurate must depth be? (forward model, GoPro 10 config, zoom 1.512)

Displacement in corrected-image px when the depth is wrong by a relative error `e` at true depth Z.
"np" is the non-pinhole part, after removing the best isotropic zoom. Full table:
`docs/figures/sensitivity.json`. Figures: `sens_relative_error.png`, `sens_fixed_z0.png`,
`sens_spatial.png`.

| true Z | depth error | camera centre: RMS / max | camera centre: np RMS / max | virtual centre: RMS / max |
| --- | --- | --- | --- | --- |
| 0.3 m | 10 % | 1.99 / 4.14 | 0.31 / 1.06 | 0.30 / 1.02 |
| 0.3 m | 25 % | 4.39 / 9.11 | 0.68 / 2.34 | 0.66 / 2.26 |
| 0.5 m | 10 % | 1.20 / 2.48 | 0.19 / 0.64 | 0.18 / 0.63 |
| 0.5 m | 25 % | 2.63 / 5.47 | 0.42 / 1.42 | 0.41 / 1.39 |
| 1 m | 10 % | 0.60 / 1.24 | 0.10 / 0.33 | 0.09 / 0.32 |
| 1 m | 25 % | 1.32 / 2.73 | 0.21 / 0.72 | 0.21 / 0.71 |
| 1 m | 50 % | 2.19 / 4.55 | 0.35 / 1.20 | 0.35 / 1.18 |
| 2 m | 25 % | 0.66 / 1.37 | 0.11 / 0.36 | 0.10 / 0.36 |
| 4 m | 25 % | 0.33 / 0.68 | 0.05 / 0.18 | 0.05 / 0.18 |
| 8 m | 50 % | 0.27 / 0.57 | 0.04 / 0.15 | 0.04 / 0.15 |

- The effect scales as `d0/Z` and grows towards the image corners (`sens_spatial.png`).
- At the camera centre, ~85 % of it is the isotropic part. The virtual centre removes exactly that
  part, which is why its column equals the "np" column.
- Rule of thumb:
  - to keep the non-pinhole error below ~0.3 px max, a 1 m scene needs depth within ~10 %, a 0.5 m
    scene within ~5 %, and a ≥2 m scene within ~25 %;
  - beyond ~4 m, depth errors of 50 % are harmless.

**Wrong constant depth** (shipped `z0_fixed = 1 m`, fronto-parallel plane at Z; non-pinhole
residual, `sens_fixed_z0.png`):

| true Z | 0.3 m | 0.5 m | 0.7 m | 1.5 m | 2 m | 4 m | 8 m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RMS / max [px] | 2.44 / 8.34 | 1.04 / 3.57 | 0.45 / 1.53 | 0.35 / 1.19 | 0.52 / 1.79 | 0.78 / 2.68 | 0.91 / 3.13 |

**Synthetic scenes with known depth** (`depth_aware/synthetic.py`). Tilted seabed with two Gaussian
bommies, traced exactly through the housing:

- near reef: 0.43–1.1 m, median 0.83;
- far reef: the same scene 4x larger, 1.7–4.4 m.

"Non-pinhole" is the residual against the best pinhole (centre and focal searched). Full table:
`synthetic.json`, `synthetic_near_reef.png`, `synthetic_scale_sweep.png`.

| method (non-pinhole RMS / max px) | near reef, cam. centre | near reef, virtual centre | far reef, cam. centre | far reef, virtual centre |
| --- | --- | --- | --- | --- |
| fixed z0 = 1 m (config) | 0.28 / 1.38 | 0.28 / 1.38 | 0.70 / 2.46 | 0.70 / 2.46 |
| fixed z0 = median depth | 0.22 / 1.14 | 0.22 / 1.14 | 0.06 / 0.29 | 0.06 / 0.29 |
| depth-aware, exact | 0 | 0.05 / 0.22 (search grid) | 0 | 0.01 / 0.06 |
| depth-aware, scale ×1.25 | 0.24 / 1.04 | 0.24 / 0.97 | 0.06 / 0.26 | 0.06 / 0.25 |
| depth-aware, scale ×2 | 0.58 / 2.56 | 0.58 / 2.56 | 0.15 / 0.64 | 0.15 / 0.64 |
| depth-aware, smooth 10 % error | **1.03 / 5.53** | **0.17 / 1.14** | 0.22 / 1.44 | 0.04 / 0.39 |
| depth-aware, per-pixel 10 % noise | 0.83 / 10.05 | 0.15 / 1.57 | 0.21 / 2.85 | 0.04 / 0.60 |
| depth-aware, ×1.25 + smooth 10 % | 0.79 / 4.11 | 0.25 / 1.43 | 0.21 / 1.40 | 0.08 / 0.55 |

Reading:

- With exact depth, depth-aware correction is perfect. That is a gain of 0.22 px RMS / 1.1 px max
  over the best constant depth on the near reef, and of only 0.06 / 0.3 px on the far reef.
- A *wrong* constant (config 1 m on the far reef) costs more than all depth variation: 0.70 px RMS.
- At the camera centre, realistic depth errors turn the gain into a loss (1.03 vs 0.22 px). With
  the virtual centre the dense correction stays at or below the fixed-depth error for 10 % noise.
- A metric-scale error of ×1.25 roughly cancels the benefit. ×2 doubles the error.

## 5. Metric scale

Refraction depends on `Z/d0`, so the depth fed to the correction must be metric. COLMAP is
up-to-scale. Options considered:

1. **Prior on scene distance** (the paper's `z0` / altitude / operating distance). Choose `s` so
   that the median sparse depth equals the prior. Simple, and by §4 the result tolerates ±25 %
   (≤0.25 px np-RMS at ~1 m; with the virtual centre, depth enters only through the small
   non-pinhole term). This is the de-facto choice below.
2. **Self-calibration from the refraction residual** (implemented: `scale_by_ba`). For each
   candidate `s`:
   - every observation of the fixed-depth model is mapped back to its ideal raw pixel;
   - it is re-corrected with the exact flat-port model at its own depth `s·Z_c + c`;
   - COLMAP's `bundle_adjuster` (fixed intrinsics) is re-run.

   A wrong `s` should leave a non-pinhole residual, so the minimum should mark the true scale.
3. **Metric mono depth** (DA metric / DA3METRIC / UniDepth). Not pursued: ORCA_splat measured
   DA3METRIC's metric gain to be off by 1.4–2x underwater, and indoor/outdoor metric heads are not
   trained on refracted water scenes. It is worth a check against a calibrated target, but not as
   the scale source.
4. **Known object size / external sensor** (board tag size, laser scaler, DVL, stereo baseline,
   GNSS of an ASV with known draft). This is the reliable route.

**Result of option 2 on gopro10** (`real_scale_selfcal.png`). Each candidate grid spans 16x
around the prior.

| run | unmodified model RMS | best re-corrected RMS | minimum at median depth |
| --- | --- | --- | --- |
| z0 = 1 m, camera centre | 0.794 px | 0.797 px | 1.02 m (prior 1.0) |
| z0 = 1 m, virtual centre | 0.800 px | 0.794 px | 0.94 m (prior 1.0) |
| z0 = 0.4 m, camera centre | 0.791 px | 0.789 px | **0.97 m** (prior 0.4) |

What the three curves show:

- All three have the same minimum, near 1 m. In the z0 = 0.4 m run it is far from the prior, so
  the estimate is not just the prior echoed back.
- Scales implying < 0.5 m are strongly rejected: > 5 px at 0.1 m.
- Above 0.7 m the curve is flat to within 0.03 px over a factor of 3. Re-correcting at the best
  scale lowers the BA error by only 0.002–0.006 px against the unmodified model.

So the scale *is* observable from the refraction residual. It is coarse (roughly ±10–20 % judging
by the curvature, ±4 % between runs) and it depends on the scene: a near-planar tank scene at
1 m with ~0.5 px features is near the limit, since §4 predicts ≤0.2 px non-pinhole signal for
25 % scale error at 1 m. That precision is sufficient for the correction (§4), but not for
metrology.

The AprilGrid does not help with absolute depth here:

- a per-image refractive plane fit (homography residual over plane normal and distance) returns
  0.07–20 m;
- tag sizes come out inconsistent;
- the floor is 1.9 px RMS of non-repeatable corner noise.

Recommended order:

1. a platform prior or sensor (altitude, laser scaler, stereo, ASV GNSS + draft);
2. otherwise `scale_by_ba` as a sanity check of that prior;
3. never a metric mono-depth head on its own underwater.

## 6. Real data: gopro10 (refrax-dataset)

**What the data is.** The HF `refrax-dataset` sequences `gopro10_linear` (255 images) and
`gopro11_linear` (305) are a 6x6 AprilGrid (AprilTag 36h11, tag spacing ratio 0.50) lying flat in a
tank or pool. They are not the paper's Malaysia/Lizard Island surveys:

- the paper used HERO13 and HERO11 for those;
- no gopro11 calibration is in the repo, so only gopro10 is used.

The scene is planar, the trajectory hand-held, and per-image depth varies by only ~5 % (median
spread of the sparse depths). Trajectory flatness, the paper's Lizard Island metric, does not
apply.

Two ground-truth-free metrics are used besides the SfM statistics:

- **board planarity** of the COLMAP points on the board, using `evaluation/planarity.py`'s fit
  after a 5-MAD outlier rejection;
- **board homography residual** of the corrected tag corners (`depth_aware/eval_board.py`, no SfM). This
  is scale-free and pinhole-invariant, and it is evaluated on 225 images with a self-calibrated
  board layout.

Every 2nd image (128) is used, with COLMAP CPU, exhaustive matching, and fixed PINHOLE intrinsics
from the reference calibration.

| run (z0 = 1 m) | reg. | points | track | reproj [px] | board rel. RMSE | board κ |
| --- | --- | --- | --- | --- | --- | --- |
| fixed depth, camera centre (Refrax) | 128/128 | 42 875 | 13.91 | 0.483 | 0.162 % | −0.003 |
| fixed depth, virtual centre | 128/128 | 42 497 | 13.92 | 0.486 | 0.155 % | −0.001 |
| per-image median depth (SfM), camera centre | 128/128 | 42 610 | 13.89 | 0.482 | 0.161 % | +0.004 |
| dense depth (SfM + DAv2), camera centre | 128/128 | 42 855 | 13.80 | 0.516 | 0.159 % | +0.002 |
| dense depth (SfM + DAv2), virtual centre ⚠ | 128/128 | 47 181 | 11.82 | 0.632 | 6.98 % | +1.78 (bent) |
| fixed depth z0 = 0.4 m (wrong prior), camera centre | 128/128 | 42 965 | 13.86 | 0.488 | 0.172 % | −0.009 |
| dense depth from the z0 = 0.4 m model (scale self-calibrated to 0.97 m) | 128/128 | 44 035 | 13.85 | 0.520 | 0.155 % | +0.002 |

Board homography residual, the mean over images of the per-image RMS (px):

| method | value |
| --- | --- |
| uncorrected (lens-undistorted only) | ~6 |
| fixed z0 = 1 m | 1.908 |
| dense depth | 1.907 (paired Δ −0.001 ± 0.000) |
| fixed z0 0.3 / 0.5 / 0.7 / 2 m | 1.98 / 1.94 / 1.93 / 1.93 |
| virtual centre, z0 = 0.5 | 1.917 |

Refraction correction itself matters (6 → 1.9 px). The choice of depth changes the residual by
less than 0.05 px, under a random 1.9 px floor.

**Interpretation.**

- **The dense depth-aware run is consistently a bit worse in reprojection** (+0.03 px) and has
  slightly shorter tracks. The board check sees no change in per-image geometry. This matches the
  synthetic finding that per-pixel depth injects per-image inconsistencies that a constant depth
  does not have: noise in the per-image fits, mono artefacts at the board edges and the tank frame,
  and differing estimates of the same surface in overlapping images.
- **Per-image median depth is neutral**, which is expected: depth hardly changes between frames.
- **The dense run is useful when the prior is wrong.** With z0 = 0.4 m the board is slightly bowled
  (κ −0.009). After the self-calibrated dense correction it is as flat as the best fixed-depth run.
  This mirrors the paper's Fig. 7 z0 sweep, but automatically.
- **The virtual-centre dense run is an unexplained failure.** Two mapper runs on the same database
  give reprojection 0.63 / 0.76 px, 47–54k points, track 10–12, a strongly bent board, and
  hundreds of "Linear solver failure" messages in global BA. Yet:
  - the corrected images differ from the fixed-depth ones by ~1 grey level on average;
  - the board homography residual is unchanged (1.905 vs 1.908 px);
  - point/image round trips through the depth-map correction are exact to < 0.01 px;
  - the fixed-depth virtual-centre run is fine (0.486 px, κ −0.001).

  Verified two-view pairs drop from 6492 to 6020 (camera centre: 6919 → 6675). This needs a
  look before anyone relies on the virtual centre in depth mode. Suspects are COLMAP's
  initialisation on these near-planar, near-duplicate views, or an interaction with the
  per-image masks.
- Board planarity was already at the noise level with fixed depth at a good z0: |κ| ≈ 0.003,
  against the paper's |κ| up to 0.08 for "Ours" on real reefs. A planar tank target at ~1 m
  cannot show a gain from per-pixel depth.

## 7. Limitations

- No real dataset with strong depth variation and ground truth was available. The paper's
  Lizard Island sequences (1–8 m) and Malaysia `p1_s01` at 0.5 m are the right test beds. §4
  predicts small gains at ≥1 m and visible gains only below ~0.7 m or for strongly 3D scenes in
  the image periphery.
- The self-calibrated scale (≈1 m) is not verified against ground truth. The true board distance
  and tag size are unknown.
- One iteration only. With no gain at iteration 1, iterating is pointless on this data. It is
  supported (`--iters`).
- Occlusion boundaries: a depth discontinuity makes the forward map fold. `invert_map`'s linear
  interpolation then blends across it. This is harmless for smooth reef surfaces, but the mask
  does not flag folded regions.
- The board corner noise floor (1.9 px) is unexplained: blur, ripple, or possibly a water surface
  in the optical path. Treat the board metric as relative only.

- The depth-mode warps here use this branch's `depth_aware/common.py:Corrector`: a forward map, a
  step-4 subsampled grid, and one shared Delaunay triangulation. Depth is therefore effectively
  sampled every 4 px, which is fine for the smooth mono-fit maps used here. Branch
  `explore/speed` has an exact per-pixel inverse (`core/flatport.py`, `F = A(p) + B(p)/Z`, which
  is the same structure that makes the virtual centre work) and fixes depth-mode validity bugs. It
  was not used for these runs.

## 8. Next steps

1. Evaluate `virtual_center: auto` on the paper's sequences (κ, reprojection, registration). It is
   free, and §3/§4 predict it removes most of the z0 sensitivity. That would also explain why the
   Fig. 7 optimum is broad.
2. Replace the per-sequence z0 by a per-image constant depth from a first SfM pass
   (`--mode median`). Do it with the virtual centre on Lizard Island, where altitude changes along
   the survey.
3. Run the dense mode only for near-range, strongly 3D scenes, and with the virtual centre.
   Consider DA3 multi-view or MASt3R depth for cross-image consistency; per-image mono fits are
   the main noise source.
4. Get metric scale from the platform (ASV GNSS + known camera draft, DVL altitude, stereo rig on
   the Malaysia data) instead of the residual.
5. A tank dataset with a 3D target of known geometry at 0.3–1 m would make the depth effect
   measurable.

## 9. Notes on the paper and code

- **Sensitivity framing.** Paper Fig. 3 / §III-E report the displacement RMSE against Z and d0.
  Most of that displacement is an isotropic, depth-dependent magnification, which is what a 1 cm
  shift of the projection centre produces (§3). The SfM-relevant non-pinhole part is ~6x smaller.
  Quoting both would make the "Z/d0 dominates" message more precise. It also suggests that the
  Fig. 7 z0 optimum is largely a focal-length (zoom·(z0−c)/z0) effect.
- **Method description vs code.** The paper (Eq. 14) solves the backward map per pixel as
  nonlinear least squares. The code inverts a step-4 forward map by scattered-data interpolation.
  Worth stating, especially for depth maps, which are then effectively sampled every 4 px.
- **Depth mode.**
  - Depth maps are defined on the lens-undistorted grid, not on raw pixels (up to 25 px apart for
    this camera).
  - The per-image crop made calibration impossible; fixed here with `depth_crop: reference`.
- **Dataset.**
  - The public refrax-dataset (gopro10/11_linear) is a tank AprilGrid sequence, not the paper's
    Malaysia (HERO13) or Lizard Island (HERO11) data.
  - `configs/dataset.yaml` covers only gopro10, and gopro11 has no calibration in the repo.
  - Its corner residual floor of ~1.9 px after correction (6 px before) is worth understanding
    if it is used for validation.
- **Shipped z0.** `z0_fixed: 1.0` is consistent with this sequence: self-calibration gives
  0.94–1.02 m. For other data, a wrong z0 is the largest single error source (§4: 2.4 px RMS at
  0.3 m).
