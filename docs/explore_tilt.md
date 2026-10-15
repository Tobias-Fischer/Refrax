# Exploration: estimating the port tilt in bundle adjustment

Branch `explore/tilt` (on `explore/uncertainty` → `explore/gpu-remove` → `explore/speed`).
Read `docs/explore_uncertainty.md` first: it found that **port tilt** (the port normal
`n_port` deviating from the optical axis) is the only housing parameter that is both large
(0.5° → ~4 px) and not absorbed by refitting the pinhole intrinsics (0.6–1.0 px RMS
non-absorbable per 0.5° at 1 m; 0.3–0.5 px if a constant camera rotation is also allowed).
The PI asked to estimate the two tilt angles (optionally d0) jointly in bundle adjustment
on real data.

Code: `tilt/` (Python prototype), `tests/test_tilt.py`, `configs/flatport/`; C++ in
`colmap-uncertainty` branch `port-tilt` (inner `colmap-src` branch `port-tilt`, commits
`08b3493`, `6e1859b`). Large outputs: `/Users/fischert/robot/refrax-work/tilt/`.

## TL;DR

@@TLDR@@

---

## 1. Prior art

**Upstream COLMAP has no refractive camera models.** The COLMAP 4.2.1 source in
`colmap-uncertainty/colmap-src` (and its pycolmap bindings) contains no flat-/dome-port
model (`grep -ri "refrac\|flatport\|domeport\|underwater"` over the tree finds nothing), and
there is no pull request or issue for it on `colmap/colmap` (GitHub search, 2026-10-15).

**Refractive COLMAP** (She, Seegräber, Nakath, Köser, IROS 2024, arXiv 2403.08640) is a
*fork*: `cau-git.rz.uni-kiel.de/inf-ag-koeser/colmap_underwater` (mirror
`github.com/MDSKiel/colmap_underwater`), based on COLMAP 3.10-dev, last commit 2024-10-29,
never upstreamed. What it does (read from `src/colmap/sensor/models_refrac.h`,
`estimators/cost_functions.h`, `estimators/bundle_adjustment.cc`):

* `FLATPORT` refractive model with parameters `Nx, Ny, Nz, int_dist, int_thick, na, ng, nw`
  layered on any in-air camera model (and `DOMEPORT`). The whole SfM pipeline is refractive:
  generalised relative pose, refractive triangulation / absolute pose (virtual cameras).
* BA residual: per keypoint a *virtual pinhole camera* on the refraction axis; the residual
  is the pinhole reprojection error on that virtual image plane (focal = in-air fx), i.e. an
  angular error of the water ray scaled by fx. The forward projection (`ImgFromCam`) needs
  100 Newton iterations with numeric derivatives and is not used in BA.
* `refine_refrac_params` refines the port normal (sphere manifold) **and** `int_dist` (d0)
  together — tilt and d0 cannot be refined separately; the mapper can fix them until N
  images are registered.

**The Refrax paper already compares against it** (arXiv 2610.07788, Sec. IV–V, Lizard
Island / Malaysia table): full joint refractive SfM does not consistently beat plain COLMAP
— κ ∈ [+0.10, +0.69] on Malaysia, two failed reconstructions, dome artefacts — which the
paper attributes to over-parameterisation hindering convergence (local minima). So Refractive
COLMAP is a cautionary baseline, not the solution: what is needed is a **minimal,
well-conditioned refinement** — only θx, θy (optionally d0) on top of Refrax's image-space
correction, with the rest of SfM unchanged.

**What we used it for:** an independent cross-check of the tilt estimate. I built the fork
with the colmap-uncertainty toolchain (two trivial compile fixes for Boost 1.90 /
clang: `refrax-work/tilt/kiel_build.patch`, plus exposing `BundleAdjustment.enable_refraction`
/ `refine_refrac_params` on the `bundle_adjuster` CLI) and ran **only its bundle adjuster**
on exactly our observations (raw keypoints, metric poses/points, `tilt/kiel_export.py`), not
its mapper. Its estimate agrees with ours (§4b).

Older related work (camera-side calibration of the interface normal from a target or in
BA): Treibitz et al. (PAMI 2012), Agrawal et al. (CVPR 2012, axial camera, multi-layer flat
refraction with normal estimation), Jordt-Sedlazeck & Koch (ICCV 2013, refractive SfM with
interface estimation), Chadebecq et al. (ICCV 2017, refractive two-view geometry). None is
available as a maintained COLMAP component.

**Decision:** do not duplicate a refractive SfM. Keep Refrax's pinhole pipeline (feature
matching, registration and triangulation on corrected images) and add a refractive *BA* term
with 2–3 extra global parameters. This is (a) a Python prototype with an exact Schur-LM and
posterior covariance, and (b) a C++ cost in the patched COLMAP that `bundle_adjuster`,
pycolmap and the incremental mapper's local/global BAs can use.

## 2. Formulation

**Parameters.** In-air intrinsics (fx, fy, cx, cy; radtan k1, k2, p1, p2 optional), port tilt
θx, θy (n_port = Rᵀe_z with R = Rx(θx) Ry(θy), i.e. n ≈ (−θy, θx, 1), the convention of
`core/uncertainty.py`), optionally d0. Fixed: μ_a, μ_g, μ_w, tglass (the uncertainty study
showed glass is irrelevant and μ_w is a pure zoom, absorbed by fx). Per image a pose, per
track a 3-D point.

**Observations: raw keypoints.** The SfM model is built on Refrax-corrected images (fixed
depth z0, nominal housing). Every corrected keypoint q maps back *exactly* to the raw image:
solve c + f F(p, 1/z0) = q for the in-air normalised ray p (2-D Newton on the closed-form
model of `core/flatport.py`), then apply the in-air lens (`tilt.model.raw_from_corrected`;
C++ `FlatPortBundleAdjustmentOptions::RawFromKeypoint`). No re-extraction is needed and the
matches/tracks stay valid. (Depth-map mode would need the per-pixel depth for this step.)

**Residual: back-projection form, closed form.** The 3-D point gives the depth exactly. For
P = R X + t (metric: × the scale in metres per model unit):

    w     = 1 / P_z
    F(u)  = A(p(u)) + B(p(u)) · w          (water ray of the raw keypoint at depth P_z)
    e     = P_xy / P_z − F(u)               (= 0 iff the refracted ray passes through P)
    r     = J_u⁻¹ e,   J_u = ∂F/∂u (2×2)     (≈ raw-pixel units, isotropic for raw noise)

`A, B` are the affine-in-inverse-depth flat-port coefficients of `core/flatport.py` (tilted
case). The forward direction (point → raw pixel) would need the iterative/quartic flat-port
forward solve inside every residual and Jacobian evaluation (Refractive COLMAP's
`ImgFromCam` does 100 numeric-derivative Newton steps); the back-projection form needs one
closed-form refraction plus the in-air undistortion (8 Newton steps). `J_u` is evaluated at
the current estimate and held fixed in the derivative (its derivative multiplies e, so it is
second order at the solution); the C++ unit test confirms the Jacobians against numeric
differentiation at the solution, and the 1-px test confirms the raw-pixel scaling.

**Jacobians.** Python: one batched forward-mode AD pass (`torch.func.jvp`, vmapped over the
14 + 3 tangent directions: θ, w, u, v), chained with the pose/point derivatives. C++:
Ceres autodiff (`FlatPortReprojErrorCostFunctor<CameraModel>`), with the undistortion done
in double and one Newton correction in the Jet type (exact first derivatives by the
implicit-function theorem, valid for every COLMAP camera model).

**Solver (Python, `tilt/ba.py`).** Levenberg–Marquardt with the points eliminated (3×3
blocks, Schur complement); the reduced system over poses + globals is dense (6·n_img + k ≈
770) and solved directly, which also gives the **posterior covariance** of the globals
(points marginalised). Huber loss (2 px), IRLS. Gauge: camera 0 fixed; the **metric scale is
held fixed** (one translation coordinate) — refraction makes it weakly observable (d0, tglass
are metric), but explore/depth found self-calibrating it is coarse; the gopro10 scale is
explore/depth's 0.11347 m/unit (median depth 1.02 m), Lizard's is the GPS scale. The
globals are gauge-invariant, so their marginal covariance does not depend on the gauge
choice. Cost: ~18 s per LM iteration for 596k observations on 2 CPU threads.

## 3. Identifiability

@@IDENT@@

## 4. Results

### 4a. Synthetic

@@SYN@@

### 4b. gopro10 tank

@@REAL@@

### 4c. Lizard Island

@@LIZ@@

## 5. Implementation in COLMAP (colmap-uncertainty, branch `port-tilt`)

Branch `port-tilt` of `/Users/fischert/robot/colmap-uncertainty` (outer repo) and of its
`colmap-src` (inner, on top of the covariance patch `uncertainty`). Not pushed; `pixi run
export-patch` would turn it into `patches/0002-*.patch`.

* `estimators/cost_functions/flatport.h`: `FlatPortReprojErrorCostFunctor<CameraModel>`
  (blocks point3D 3, cam_from_world 7, in-air camera params, housing 3 = θx, θy, d0) and a
  constant-pose variant; `depth_scale` (metres per model unit) as a constant;
  `FlatPortImgFromCam` (forward projection by Newton, for simulation/tests). Templated on the
  in-air camera model (any COLMAP model), factory by `CameraModelId`.
* `estimators/flatport_options.{h,cc}`: `FlatPortBundleAdjustmentOptions` — how keypoints map
  back to the raw image (declared corrected pinhole, keypoint offset −0.5, z0, nominal in-air
  camera and housing), metric scale (given, or per BA "median observation depth = z0"), what
  to refine (tilt, d0, in-air focal / principal point / distortion), `min_num_reg_images`
  before the globals are freed, `update_declared_camera`, and a **shared state** (in-air
  params, housing, raw-keypoint cache) that persists across all BAs of a mapper run; text
  config reader (`configs/flatport/gopro10_z1.txt`) and state writer.
* `DefaultBundleAdjuster`: with `BundleAdjustmentOptions::flatport` set, every reprojection
  residual of a single-camera frame is replaced by the flat-port residual of the raw
  keypoint; the reconstruction's own (declared pinhole) camera is not in the problem; after
  the solve the declared pinhole is refitted to the corrected keypoints by linear least
  squares, so the pinhole parts of the mapper (registration, triangulation, outlier filters)
  absorb the change of the housing estimate (the hybrid stays consistent: corrected
  keypoints + pinhole everywhere except in BA). Multi-camera rigs throw (not supported).
* CLI: `colmap bundle_adjuster --flatport_config FILE` and `colmap mapper --flatport_config
  FILE` (local and global BAs). pycolmap: `FlatPortBundleAdjustmentOptions`,
  `FlatPortState`, `FlatPortConstants`, `read_flatport_options`,
  `BundleAdjustmentOptions.flatport`, `IncrementalPipelineOptions.ba_flatport`,
  `cost_functions.FlatPortReprojErrorCost` (for pyceres problems).
* Tests: `flatport_test` (5: on-axis model = closed form, Snell's law and tilt convention,
  zero residual + 1-px scaling, autodiff vs numeric Jacobians, Ceres BA recovers a
  (0.6°, −1.1°) tilt to < 0.001°) and `flatport_ba_test` (2: RawFromKeypoint inverts the
  Refrax correction to 1e-8 px; DefaultBundleAdjuster with flat-port options recovers
  (0.8°, −1.3°) from corrected keypoints of a synthetic dataset, and stock BA cannot fit it).

@@CPPRES@@

## 6. Recommendations and next steps

@@REC@@

## 7. Reproduce

@@REPRO@@
