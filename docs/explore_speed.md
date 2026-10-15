# Exploration: speeding up Refrax

Branch `explore/speed`. Goal: make `remove_refraction.py` much faster without losing
accuracy, keeping the CLI, config and output files unchanged.

**Summary.** The undistortion map is now solved exactly per output pixel instead of
interpolated from a step-4 subsampled forward map with `scipy.griddata`. Building a map
went from about 1.1 s to about 25 ms, roughly 40x faster. The new maps are about 10^9x
closer to ground truth: the residual is 1e-12 px instead of 1e-3 px. End to end, the
fixed-depth mode is ~4x faster and is now limited by PNG I/O. The depth-map mode is
~25x faster.

## Where the time went (baseline profile)

Baseline profile: GoPro 10 config, 1920x1080 input, 1782x1138 fitted canvas, M5 Mac.
This run was made under light load (load average about 4):

| stage | baseline | note |
| --- | ---: | --- |
| zoom selection | 420 ms | full-res ray trace, then 80-sample sweep or Brent over 2 M pixels |
| fit_canvas | 1630 ms | recomputes the geometry, then builds a **full map** just to size the canvas |
| map-builder setup | 180 ms | recomputes the housing geometry a third time |
| map build | 1070 ms / map | `griddata` called twice, so the **Delaunay triangulation is built twice** |
| largest_valid_rectangle | 280 ms / map | pure-Python stack loop over 2 M pixels |
| imread / remap / PNG encode | 25 / 3 / 30 ms per image | |

In fixed-depth mode the fixed cost is ~3.6 s, after which the run is I/O-bound. In
depth-map mode every image pays ~1.35 s for the map and rectangle. The PI's suspicion
was right: the same work is redone. The geometry is computed 3x and the full map 2x in
fixed mode, and the triangulation is built 2x per map. However, the bigger win comes
from removing the scattered-data inversion altogether.

## What changed

1. **Closed-form forward model** (`core/flatport.py`). In normalised coordinates, the
   point seen at depth Z projects to `F(p, w) = A(p) + B(p)·w` with `w = 1/Z`. This is
   affine in inverse depth. In the port frame the refraction only scales the ray
   tangent, by `k(ρ²) = r / sqrt(1 + (1 - r²) ρ²)` for glass and water. For
   `n_port = (0,0,1)` this simplifies to `F = p · (k_w + w·h(ρ²))`. The model agrees
   with the original 3-D ray tracer to 1e-12 px for on-axis and tilted ports (tested).
2. **Exact per-pixel inverse** instead of `griddata`:
   * **On-axis port.** The source pixel lies on the ray from the principal point through
     the output pixel, so the problem is 1-D in ρ. It is solved with safeguarded Newton
     (rtsafe): a Newton step that leaves the bracket or converges slowly falls back to
     bisection. The initial guess comes from a 1-D table of the constant-depth inverse.
     The depth map is sampled bilinearly in float64. Because f(ρ) is continuous and
     bracketed, this converges for every pixel, including slopes, noise and depth steps.
   * **Tilted port (general path).** Uses Newton per pixel:
     * Each pixel is warm-started from its left neighbour, and the neighbour's Jacobian
       is reused (chord method), so one model evaluation is needed per iteration.
     * Steps that increase the residual are halved.
     * Pixels that still fail (depth discontinuities) fall back to bisection on w of
       `w_map(p(w)) - w`, which is always bracketed by the depth range.
   * Validity uses the exact criterion "source inside `[0,W-1]×[0,H-1]`" instead of the
     convex hull of the forward map.
3. **Fused numba kernel** (`core/_kernels.py`). It solves each pixel, applies the radtan
   distortion and writes cv2.remap-ready float32 maps with -1 for invalid pixels, all
   in one parallel pass. `cache=True` stores the compiled kernel on disk. A cold
   compile takes ~1–2 s once; afterwards loading it takes <0.3 s. Without numba, a
   vectorised NumPy implementation of the same algorithm is used: same results, about
   0.25–0.7 s per map.
4. **Zoom selection** (`core/scale.py`). RMSE(s)² is a quadratic in s, so three sums
   over the analytic forward map replace the 80 full-image passes or the Brent
   evaluations. The in-bounds test only needs the extremes of the map. The `in_bounds`
   zoom is bit-identical to before. The unconstrained zoom is now the exact vertex, not
   Brent's result with `xatol=1e-5`. The difference is ≤1e-6 in zoom, and the new value
   has a lower RMSE. This also avoids importing `scipy.optimize`, which took 0.5 s at
   start-up.
5. **fit_canvas** reuses the memoised forward map. The map it builds on the enlarged
   canvas is cropped and reused as the fixed-depth map, which is exact because the crop
   is an integer offset of the same grid.
6. **largest_valid_rectangle** (`core/rectangle.py`). The original stack algorithm is
   compiled with numba. It returns identical boxes, including tie-breaking, in ~3 ms
   instead of 280 ms. A vectorised NumPy DP fallback (~20 ms) is tested to return
   identical boxes too.
7. **Pipelined I/O** (`--workers`, default `min(8, cpu)`). Depth maps are built in the
   main thread while a thread pool reads, warps and writes PNGs. cv2 releases the GIL.
   Messages are still printed in order. PNG compression is unchanged: OpenCV's default
   was already the fastest setting, since levels 0–9 are no faster or much slower.
8. `invert_map` (kept for reference) builds one triangulation for both coordinates.
   `visualise_refraction.py` uses the new mapper and prints the same statistics.
9. `pixi.toml`: added the `osx-arm64` platform and the `numba` and `pytest`
   dependencies, plus a `pixi run test` task.

I did not implement map caching on disk keyed by a config hash: the whole fixed-depth
set-up now takes ~0.15 s, so a cache would not pay for itself. I did not use torch/MPS
because numba already brings a map to ~25 ms, which is below the per-image PNG I/O
cost.

## Speed

Measured with `benchmarks/bench.py`, in-process, median of 3–5 repeats. Image stages
run serially here so the two versions are comparable. **Caveat:** other agents (COLMAP)
were running on the machine. The first table was measured at load ≈ 4–7, the later ones
at load 15–90, so absolute numbers vary by up to ~2x between runs. The ratios are
stable.

### Per stage (load ≈ 4–7; 10 images fixed, 8 images depth-map)

| stage | fixed: baseline | fixed: new | depth-map: baseline | depth-map: new |
| --- | ---: | ---: | ---: | ---: |
| zoom selection | 494 ms | 73 ms | 401 ms | 59 ms |
| fit_canvas | 1768 ms | 76 ms* | 1583 ms | 67 ms |
| map-builder setup | 209 ms | 0.1 ms | 183 ms | 0.1 ms |
| map build, per map | 1337 ms | 0 (reused) | 1060 ms | **27 ms** |
| mask + crop rectangle, per map | 292 ms | 9 ms | 271 ms | 7 ms |
| image read + remap + PNG encode, per image | 53 ms | 52 ms | 53 ms | 51 ms |
| **total** | **4.6 s** | **0.69 s** | **13.3 s** | **0.83 s** |

\* fit_canvas has since been reduced further by memoising the forward map.

Isolated map build (`InverseMapper.remap_maps`, median of 7 runs, at load ≈ 20):

| case | baseline (griddata) | new numba | new NumPy fallback |
| --- | ---: | ---: | ---: |
| on-axis port, constant depth | 1.07 s | 13–22 ms | 0.24 s |
| on-axis port, depth map | 1.07 s | 25–46 ms | 0.45–0.7 s |
| 5° tilted port, constant depth | 1.1 s | 65–115 ms | ~1 s |
| 5° tilted port, depth map | 1.1 s | 70–135 ms | 1–3.7 s |

### End to end (CLI, 3 repeats, load 15–90)

| run | baseline | new, 8 workers | new, 4 workers | new, 1 worker |
| --- | ---: | ---: | ---: | ---: |
| fixed depth, all 255 gopro10 images | 22–34 s | 4.5–6.8 s | 6.3–8.0 s | 21–24 s |
| depth maps, 40 images | 109–157 s | 5.2–8.0 s | 4.7–7.2 s | 5.5–9.1 s |

* **Fixed-depth mode** is now pure I/O: about 50 ms of PNG decode and encode per image,
  so the gain comes from the worker threads. On a quiet machine the earlier 20-image run
  went from 6.0 s to 2.9 s.
* **Depth-map mode** goes from ~3 s to ~0.15 s per image, a ~20–25x speed-up.

## Accuracy

The script is `benchmarks/accuracy.py`. Both versions are given the same zoom and canvas
(GoPro 10 config). "GT residual" is `|F(map(q), w(map(q))) - q|`: each inverse map is
traced forward and compared with the output pixel it was computed for. The depth map is
sampled bilinearly in this check.

### On-axis port (the dataset config)

| depth | map new vs ref: max / mean / p99 (px) | valid-mask disagreement | GT residual ref: max / mean | GT residual new: max | image PSNR new vs ref |
| --- | --- | ---: | --- | ---: | ---: |
| z0 = 1 m | 4.0e-3 / 1.4e-3 / 3.3e-3 | 1 px | 2.8e-3 / 1.3e-3 | 6.9e-13 | 71.4 dB |
| smooth 0.5–3 m | 5.1e-3 / 1.5e-3 / 3.6e-3 | 1 px | 3.5e-3 / 1.3e-3 | 6.8e-13 | 71.4 dB |
| tilted plane 0.6–3 m | 4.1e-3 / 1.4e-3 / 3.2e-3 | 173 px (0.009 %) | **1.32** / 1.2e-3 | 7.2e-13 | 71.6 dB |
| smooth + 1 % noise | 0.62 / 2.8e-2 / 0.17 | 237 px | **0.46** / 2.2e-2 | 7.2e-13 | 59.1 dB |
| depth steps | 6.2 / 9.5e-3 / 3.6e-3 | 3 px | **4.4** / 6.3e-3 | 2e-12 | 52.9 dB |
| half-res smooth (resized) | 1.7e-2 / 1.4e-3 / 3.4e-3 | 2 px | 9.5e-3 / 1.3e-3 | 7.2e-13 | 71.4 dB |

### 5° tilted port (general solver)

| depth | map new vs ref: max / mean | GT residual ref: max / mean | GT residual new: max |
| --- | --- | --- | ---: |
| z0 = 1 m | 4.3e-3 / 1.4e-3 | 2.9e-3 / 1.3e-3 | 9e-13 |
| smooth | 5.5e-3 / 1.5e-3 | 3.3e-3 / 1.3e-3 | 1.5e-7 |
| tilted plane | 4.6e-3 / 1.4e-3 | **1.98** / 1.3e-3 | 1.5e-7 |
| depth steps | 7.9 / 1.0e-2 | **5.4** / 7.3e-3 | 1.5e-7 |

For the tilted port, the final iteration stops at a residual of 0.1·tol (1.5e-7 px)
instead of continuing to machine precision.

**Final outputs.** For the 255 real gopro10 images in fixed mode:
* All crop sizes and principal points are identical.
* PSNR new vs baseline is ≥ 70.2 dB, with a maximum difference of 1 grey level.
* The shared mask differs by 11 pixels.
* In the calibration file only `focal_length` changes, by 1.2e-4 px, because of the
  exact zoom vertex. `zoom` itself prints identically to 6 decimals.

For 40 images in depth-map mode:
* Smooth, plane and half-res depth: ~71 dB.
* Noisy depth: ~59 dB. One noisy image is at 29 dB because its inscribed rectangle moved
  by a pixel.
* Depth steps: ~50 dB, from differences at the discontinuities.
* 2 of 40 crops differ by 1 px in width.

In short:
* **Where the depth is smooth, the two versions agree to within the baseline's
  interpolation error** (≤ 5e-3 px). That error is far below cv2.remap's own 1/32-px
  coordinate quantisation.
* **The new maps are exact** (≤ 1.5e-7 px).
* **Where they differ more, the baseline is the one that is wrong.** This happens at
  noise, depth steps and non-convex image borders; see below.

## Correctness issues found in the original code

1. **Convex-hull validity.** `griddata(linear)` fills the convex hull of the forward-map
   samples. When the image of the frame is non-convex, output pixels outside the real
   image are interpolated from border pixels and marked valid. This happens with tilted
   or sloped depth maps, for example. The error was up to 1.3–2 px on 170–280 pixels per
   image in the tests. The new code uses the exact frame test.
2. **Depth is subsampled.** In depth-map mode only every 4th depth sample enters the map,
   so depth detail finer than 4 px is ignored. This gives up to 0.46 px error with 1 %
   depth noise, and up to 4–5 px at depth edges, where the gap is interpolated across.
3. **Non-unit `n_port` breaks the geometry silently.** The plane is placed at `n·d0` and
   Snell's law assumes a unit normal. With `[0, 0, 2]`, d0 doubles and the water ray is
   no longer unit length. The new model normalises `n_port`.
4. **Unconstrained zoom.** It was a Brent estimate with `xatol=1e-5`, not the optimum.
   The difference is ≤1e-6 in zoom, which is harmless.
5. **Occlusion folds (not fixed).** At occlusions, two surfaces map to the same output
   pixel. Neither version picks the nearer, visible surface: the old one blends them,
   the new one returns one valid root. A 1-D root scan along the ray could pick the
   nearest root for the on-axis port if this matters.
6. **Edge content is dropped (modelling remark, behaviour kept).** The source domain is
   the *undistorted* rectangle `[0,W-1]×[0,H-1]`, which is then mapped through radtan
   into the raw image. Any raw-image content outside that rectangle is never used.

## Tests and tools

* `pixi run test` runs `tests/test_regression.py`, 33 tests on a 320x180 camera:
  * The forward model is checked against the ray tracer (on-axis and tilted ports).
  * Inverse maps are compared with the reference and with ground truth for the numba
    and NumPy backends, an on-axis and a tilted port, and constant, smooth, plane and
    step depth.
  * The two backends are checked against each other.
  * remap maps with radtan, `select_zoom`, `fit_canvas`, and `largest_valid_rectangle`
    on random masks (both backends).
  * An end-to-end CLI run in both modes is compared with the reference pipeline.
* `tests/reference_impl.py` is a frozen copy of the original implementation.
* `benchmarks/bench.py` (per stage), `benchmarks/accuracy.py`,
  `benchmarks/compare_outputs.py` and `benchmarks/make_synthetic_depth.py` (synthetic
  depth maps: smooth, plane, noisy, step, half-res).
