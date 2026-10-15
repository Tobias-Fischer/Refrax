# Exploration: real-time depth-aware refraction removal on the GPU

Branch `explore/gpu-remove`, branched from `explore/speed`. The goal is per-frame,
depth-aware refraction removal fast enough for real time, prototyped with PyTorch on an
Apple M5 (24 GB, MPS) and meant to be verified later on the lab's NVIDIA machines. The
code is device-agnostic (`cpu` / `mps` / `cuda`). Read `docs/explore_speed.md` first: this
port keeps its algorithm.

**Summary.**
* **Speed.** On the M5 GPU, a 1920x1080 frame with its own dense depth map is corrected in
  **~9 ms**: the per-pixel exact inverse plus the remap, on a 1782x1138 canvas. The numba
  CPU kernel takes ~34 ms and torch-CPU ~440 ms. Fixed-depth mode, a cached map plus one
  remap, takes 0.7 ms.
* **Bottleneck.** Depth Anything V2 Small itself takes 39–53 ms at 518x518 on the same
  GPU, so end to end the M5 runs at ~15–20 fps. The refraction removal is 15–20 % of that.
* **Accuracy.** float32 is enough. The maps agree with numba to ≤ 9e-4 px (p99 2.5e-4 px)
  and trace back to their output pixel to ≤ 5e-4 px. Corrected real images are at
  ≥ 83.6 dB PSNR against cv2.remap with the numba maps, with a maximum difference of 1–3
  grey levels.

## Design (`core/torch_remove.py`)

Same mathematics as `core/_kernels.py`. F(p, w) = A(p) + B(p)·w is solved per output pixel,
vectorised over all pixels of a batch of frames.

* **Per-camera precompute.** `TorchInverse.__init__` runs once, in float64 on the CPU, and
  uploads the results.
  * On-axis port: per output pixel, the target radius `Rt`, the ray direction `(ex, ey)`,
    the frame exit and the bracket `hi`. Also a 4096-entry table of kw(ρ) and h(ρ) for the
    initial guess.
  * Tilted port: the constant-depth solution p(w_ref) and dp/dw, by central differences of
    two numba solves, plus the paraxial guess.
  * `Corrector` also uploads the fixed-depth map, built by the numba kernel exactly as the
    CLI builds it.
* **Per frame (depth-aware).** Inputs are `depth (B, H, W)` on the source grid, as in the
  CLI's depth-map mode. An image of a different size is resized bilinearly on the device
  and clamped ≥ 1e-3 m.
  1. w = 1/depth. Bilinear sampling and its gradient are written as four gathers, the same
     arithmetic as `_sample_w`.
  2. Initial guess at the frame's **median** inverse depth, as numba does:
     * on-axis: a per-frame table R(ρ) = ρ(kw + w·h), then `searchsorted`;
     * tilted: p0 + (w − w_ref)·dp/dw.

     At occlusion folds, where two valid roots exist, this choice decides which root is
     returned. With the median depth, the on-axis roots are identical to numba's.
  3. Solve.
     * On-axis port: 8 dense iterations of safeguarded Newton (rtsafe) with no host sync.
       The still-active pixels (depth steps) are then compacted and continued for up to 60
       iterations in total, like the numba loop.
     * Tilted port: full 2-D Newton with an **analytic Jacobian** of A and B (tested
       against central differences). The numba kernel's finite differences with eps = 1e-6
       are useless in float32. Overshoots are halved. A pixel that converges still takes
       its last Newton step (quadratic refinement below tol). Unconverged pixels are
       compacted and finished, and those that still fail fall back to bisection on inverse
       depth (40 steps, each 3 constant-w Newton steps).
  4. Frame test, radtan folded in, and remap-ready float32 maps with −1 for invalid
     pixels. `valid_mask` matches `compute_valid_mask`.
  5. Remap. `remap_px` is cv2.remap(INTER_LINEAR, BORDER_CONSTANT=0) written as four
     gathers from the uint8 BGR image. It is compiled into one kernel and is ~3x faster on
     MPS than `F.grid_sample`, which inductor decomposes badly. `remap()` (grid_sample,
     `align_corners=False`, grid = (2u+1)/W − 1) is used in eager mode and for float
     inputs. Both match cv2.remap at every pixel, including the partial blend with the
     zero border just outside the frame, to ≤ 1–3 grey levels. cv2 quantises bilinear
     weights to 1/32, while the torch remap uses exact weights.
* **Fixed-depth mode.** Cached maps plus one `remap_px`.
* **torch.compile is essential.**
  * Eager MPS took ~200 ms per 1080p map: each of ~60 elementwise ops per iteration is a
    separate pass over 2 M pixels. Inductor fuses each iteration into one Metal kernel,
    which brings this to ~8 ms.
  * Metal allows at most 31 buffers per kernel. The tilted iteration is therefore compiled
    2 iterations at a time and the bisection 1 step at a time. With 8 unrolled steps the
    shader fails to build.
  * Dynamic-shape compilation of the compacted subsets also exceeds the limit. Subsets are
    therefore padded to static sizes of 4096·8^k. Subsets larger than 1/8 of the frame
    continue on the dense arrays. A 1080p stream thus uses at most 3 subset variants, which
    `warmup()` compiles up front.
  * `compile="auto"` turns compilation on for mps and cuda. On cpu torch runs eagerly,
    since the CPU path is numba.
* **Streaming API.**

  ```python
  from core.torch_remove import Corrector
  corr = Corrector(cfg, device="mps")      # zoom, canvas, fixed map, tensors (~0.2-1.5 s)
  corr.warmup(B=1)                         # torch.compile of all variants (see below)
  out, mask = corr.correct(img_bgr, depth) # (H,W,3) uint8 + (H,W) depth -> canvas + bool
  out, mask = corr.correct(img_bgr)        # fixed depth z0
  corr.K                                   # corrected intrinsics (canvas frame)
  ```

  Inputs may be numpy or tensors on any device, either single frames or batches
  `(B, H, W, 3)`. `corr.maps(depth)` returns the maps and mask, for example for a node
  that remaps several streams. Cropping to the inscribed valid rectangle is left to the
  caller (`core.rectangle`, ~3 ms on CPU). A real-time node would rather publish the full
  canvas with a fixed CameraInfo and the mask.
* **CLI.** `remove_refraction.py --device {numba,cpu,mps,cuda}`, default `numba`, which is
  unchanged. A torch device only changes how the depth-map mode builds its maps. Fixed
  mode, cv2.remap, cropping and PNG I/O stay the same, so outputs remain directly
  comparable. On 20 real gopro10 images with synthetic depth, `--device mps` and numba
  give:
  * identical crop sizes;
  * PSNR 83.8–84.2 dB, maximum difference 3 grey levels;
  * masks that differ by ≤ 1 px.

  The CLI stays I/O-bound. With torch import and compile it is slower overall: 10 s
  instead of 4.7 s for 20 images. The GPU path is for streaming.
* **pixi.**
  * The `gpu` environment adds pytorch, with tasks `test-gpu` and `bench-remove-gpu`.
  * The `depth` environment adds transformers, with task `bench-depth-anything`.
  * The default environment and `pixi run test` are unchanged.

## Accuracy (`benchmarks/accuracy_gpu.py`, GoPro 10 config, 1920x1080)

Columns:
* map columns: torch float32 maps (radtan applied, −1 = invalid) against the numba
  kernel's, over pixels valid in both;
* GT: the torch maps without distortion, traced forward with the analytic model (float64,
  depth sampled bilinearly), as `|F(map(q), w) − q|` in output px;
* PSNR: a real gopro10 image corrected by `Corrector.correct` against cv2.remap with the
  numba maps, over all canvas pixels.

The depth kinds are those of `make_synthetic_depth.py`, plus z0 as a constant map.

### On-axis port (the dataset config), MPS float32

| depth | max / p99 / mean (px) | > 0.01 px | mask xor | GT max (px) | PSNR (dB) | max Δ grey |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| const 1 m | 6.4e-4 / 2.5e-4 / 7.6e-5 | 0 | 0 | 2.9e-4 | 83.8 | 1 |
| smooth | 6.2e-4 / 2.5e-4 / 7.6e-5 | 0 | 0 | 2.9e-4 | 84.4 | 1 |
| plane | 6.4e-4 / 2.5e-4 / 7.6e-5 | 0 | 0 | 2.9e-4 | 84.3 | 1 |
| noisy | 9.2e-4 / 2.5e-4 / 7.6e-5 | 0 | 0 | 3.3e-4 | 84.3 | 1 |
| step | 8.7e-4 / 2.5e-4 / 7.6e-5 | 0 | 0 | 4.7e-4 | 84.4 | 1 |
| halfres | 5.5e-4 / 2.5e-4 / 7.6e-5 | 0 | 0 | 2.8e-4 | 84.1 | 1 |

torch-CPU float32 gives the same numbers within ±1 in the last digit. torch-CPU
**float64** reproduces numba to 8.6e-5 px max, with GT ≤ 1e-12 px. That 8.6e-5 px is the
rounding of numba's own float32 output maps at |u| ≈ 1900. In other words, float32 adds
~5e-4 px. This is two orders of magnitude below cv2.remap's 1/32-px coordinate
quantisation and well below any calibration error.

### 5° tilted port, MPS float32

| depth | max / p99 (px) | > 0.01 px | mask xor | GT max (px) | PSNR (dB) |
| --- | --- | ---: | ---: | ---: | ---: |
| const | 7.1e-4 / 2.8e-4 | 0 | 1 | 4.1e-4 | 83.1 |
| smooth | 7.3e-4 / 2.8e-4 | 0 | 0 | 3.7e-4 | 83.5 |
| plane | 7.6e-4 / 2.8e-4 | 0 | 0 | 3.8e-4 | 83.7 |
| noisy | 9.1e-2 / 2.9e-4 | 1 | 0 | 1.9e-2 | 83.8 |
| step | 4.2 / 2.9e-4 | 1101 | 1 | 2.2e-2 | 51.7 |
| halfres | 7.3e-4 / 2.8e-4 | 0 | 0 | 3.7e-4 | 83.2 |

* **Step depth.** The 1101 px (0.05 %) are occlusion folds where numba and torch return
  *different valid roots*. Numba warm-starts each pixel from its left neighbour, which
  cannot be reproduced in parallel. torch float64 gives the same 1099 px with GT 6e-11, so
  this is root choice, not precision. Both versions are wrong in the same way there: the
  visible, nearer surface is not chosen (see explore_speed, "occlusion folds").
* **float32 outliers.** The 1.9–2.2e-2 px GT maxima are a handful of pixels on the
  depth-discontinuity bisection fallback, where float32 limits the constant-w Newton.
  This is still below cv2.remap's 1/32 px.

Tests (`tests/test_torch_remove.py`, 33 tests on a 320x180 camera, cpu and mps, cuda when
present) check:
* the analytic Jacobian;
* maps against numba (4 depth kinds x 2 ports);
* ground truth;
* float64 against numba to < 1e-5 px;
* both remaps against cv2.remap;
* `Corrector` batch == single frames, fixed mode == CLI map, half-resolution depth;
* the CLI with `--device` against the default.

Run them with `pixi run -e gpu test-gpu`, in ~25 s with a warm inductor cache.

## Speed (`pixi run -e gpu bench-remove-gpu`, M5 24 GB)

The table gives the median ms per frame with the p10–p90 range, over 20 frames after 3
warm-up frames, and 3 frames for torch-CPU. Frames cycle through the smooth, plane, noisy
and step depths and 4 real images. Inputs are device-resident and the device is
synchronised around every call. Transfers are listed separately. Load average was ≈ 7
during the run, though other agents were active on the machine (CPU and possibly the GPU),
so expect ±30 % between runs.
* **fixed**: cached map plus remap.
* **depth**: solve the map from this frame's depth, then remap.
* **map**: the solve alone.

On-axis port, batch 1:

| input (canvas) | backend | fixed | depth-aware | map only | H→D / D→H |
| --- | --- | ---: | ---: | ---: | ---: |
| 960x540 (891x569) | numba CPU (+cv2.remap) | 0.3 (0.2–0.4) | 8.6 (6.8–9.8) | 7.9 | – |
| | torch CPU (eager) | 8.1 | 146 | 126 | – |
| | **torch MPS** | 1.0 (0.8–1.0) | **7.4 (5.3–8.2)** | 2.1 | 0.4 / 0.4 |
| 1920x1080 (1782x1138) | numba CPU | 1.0 (1.0–1.3) | 33.6 (25–41) | 29.9 | – |
| | torch CPU (eager) | 16.8 | 442 | 403 | – |
| | **torch MPS** | 0.7 (0.7–1.0) | **9.3 (8.6–11.0)** | 7.7 | 0.9 / 0.7 |
| 3840x2160 (3565x2278) | numba CPU | 4.1 (3.9–4.3) | 130 (95–184) | 119 | – |
| | torch CPU (eager) | 106 | 2147 | 2081 | – |
| | **torch MPS** | 1.9 (1.8–2.1) | **31.4 (30–35)** | 27.5 | 2.6 / 2.2 |

**Other configurations (MPS).**
* 5° tilted port at 1080p: depth-aware 16.5 ms (13.8–18.5) on MPS, 51 ms with numba.
* Batch of 4, per frame:
  * 960x540: 3.0 ms. The single-frame 540p time is overhead-dominated: a few host syncs per
    frame for compaction, plus kernel launches.
  * 1080p: 8.6 ms.
  * 4K: 33 ms.
* Fixed per-frame overhead: GPU memory (driver-allocated) is 0.4 GB at 1080p and 2.4 GB at
  4K for B = 1, and 6.5 GB at 4K for B = 4.
* Setup: `Corrector()` takes 0.2–0.6 s, mostly the numba fixed map and the float64
  precompute. `warmup()` (torch.compile) takes 1.5–15 s with a warm on-disk inductor
  cache, and ~10–20 s per resolution cold. Run it once at node start-up.

**Where the time goes.**
* The 1080p map takes ~8 compiled kernel launches for the dense phase, plus nonzero and
  `.any()` syncs, plus the bucketed subset.
* The kernels are memory- and latency-bound. Each iteration reads and writes ~10 float32
  state arrays.
* A hand-written Metal or CUDA kernel with per-thread early exit, mirroring the numba loop,
  should be several times faster still, but was not needed for the budget.

### End-to-end real-time budget (1080p, M5 MPS)

Depth Anything V2 Small was timed with `pixi run -e depth bench-depth-anything`.
* **Weights.** The network is built from the Depth-Anything-V2-Small-hf configuration
  (24.8 M parameters) with random weights. Timing does not depend on the weights, and
  nothing is downloaded; `--pretrained` uses the hub weights.
* **Setup.** Batch 1, transformers 5.19, torch 2.14.1.
* **fp16.** fp16 needs eager attention: the fused fp16 SDPA Metal kernel fails to build
  with torch 2.14 on this Mac.
* **Included in the timing:** bilinear upsampling of the prediction to 1080p.

| stage | ms |
| --- | ---: |
| image H→D (uint8 BGR) | 0.9 |
| DA-V2-S preprocessing (resize/normalise to 518) | ~1 (not measured) |
| DA-V2-S inference, 518x518: fp16 / fp32 | 39 / 53 |
| (DA-V2-S at the processor's 16:9 size 518x924: fp16 / fp32) | (77 / 105) |
| depth-aware refraction removal (map + remap) | 9.3 |
| output + mask D→H (only if a CPU consumer needs them) | 0.7 |
| **total, 518x518 fp16 / fp32** | **≈ 51 / 65 → 15–20 fps** |

On the M5, real time at 30 fps is therefore limited by the depth network, not by
refraction removal. Refraction removal alone runs at > 100 fps at 1080p.

**Caveat.** DA-V2-Small (`-hf`, relative) predicts affine-invariant *relative inverse
depth*. The refraction model needs metric depth in the same units as the housing (m). Use
the metric fine-tunes, which have the same architecture and therefore the same timing, or
a per-frame scale/shift fit, for example to sparse SLAM or DVL ranges. In-water metric
accuracy of DA is a separate question. The correction's sensitivity to depth is small:
the depth-induced shift is ~h·w·ρ, a few px from 0.5 m to 3 m for this housing.

## CUDA: estimates and what to verify

**Not measured — estimates only**, by scaling with memory bandwidth and published
DA-V2 numbers:
* **RTX 4090 / A6000-class GPU.** Depth-aware map and remap at 1080p ≈ 1–2 ms, and 4K
  ≈ 4–8 ms. DA-V2-S at 518 ≈ 5–10 ms in PyTorch fp16, and less with TensorRT. 30–60 fps
  end to end looks comfortable.
* **Jetson AGX Orin.** Its bandwidth is comparable to the M5, so expect roughly M5-class
  numbers for refraction removal: ≈ 8–15 ms at 1080p. DA-V2-S needs TensorRT fp16 to stay
  within ~30 ms.

To verify on the lab's Linux and NVIDIA machines:
1. `pixi run -e gpu test-gpu` with cuda. The tests parametrise over cuda automatically.
   This needs a CUDA build of pytorch from conda-forge (`pytorch-gpu`, or `cuda-version`
   pinning) in the `gpu` feature on linux-64.
2. `pixi run -e gpu bench-remove-gpu --devices cuda` and
   `pixi run -e depth bench-depth-anything --device cuda`.
3. Inductor on CUDA generates Triton kernels. Metal's 31-buffer limit does not apply, so
   `UNROLL_2D` can be raised to 8, and possibly the subset compilation can be made
   `dynamic=True`. Check the compile times.
4. Check that the `torch.compile` gather fusion (`remap_px`) beats `grid_sample` on CUDA
   as well. cuDNN's grid_sample is much better than MPS's, and `remap()` is the fallback.
5. If the host syncs (compaction) dominate at 540p/1080p, use CUDA graphs
   (`mode="reduce-overhead"`) for the dense phase. Alternatively, write one fused per-pixel
   kernel with early exit (Triton, or CuPy RawKernel) that mirrors `core/_kernels.py`.
6. float32 accuracy on CUDA should match the cpu/mps tables. The tests assert it.

## Plugging into ROS 2 / Isaac ROS

* **Plain ROS 2 node (rclpy).**
  * Construct `Corrector(cfg, "cuda")` and call `warmup()` at start-up.
  * Subscribe to the image (and the depth from a DA node) with an approximate time
    synchronizer.
  * Call `correct(img, depth)`.
  * Publish the corrected image, the mask (`mono8`) and a constant CameraInfo built from
    `corr.K` and the canvas size. The canvas is fixed and the zoom is chosen at z0, so the
    intrinsics do not change per frame. This is the main advantage of not cropping per
    frame.
  * cv_bridge conversion is a host copy. At 1080p that is ~1 ms each way, which is fine.
* **Running DA in the same process** (torch) keeps the depth on the GPU. Only the camera
  image is uploaded, and only the outputs are downloaded if a CPU consumer needs them.
* **Isaac ROS (NITROS).** NITROS passes GPU buffers between nodes in the same process
  (type adaptation).
  * A Python node can wrap such a CUDA buffer with `torch.from_dlpack` (zero copy). The
    corrector needs no host round trip when it sits between a GPU image source, a
    DNN-based depth node and a GPU consumer such as visual SLAM or nvblox.
  * The fixed-depth mode is just a static remap. For it, the existing Isaac ROS
    rectification nodes (precomputed maps) are an alternative: export `corr.maps()` once.
  * Not yet verified: which Isaac ROS packages accept an external remap or LUT; check
    against the installed release.
* **Latency.** For lower latency, run DA on frame t+1 and the correction on frame t on
  separate CUDA streams. On MPS this gains little, since there is one queue.

## Files

* `core/torch_remove.py`: `TorchInverse` (solver), `Corrector` (streaming API),
  `remap_px` / `remap` / `maps_to_grid` / `valid_mask`.
* `remove_refraction.py`: `--device`.
* `tests/test_torch_remove.py`.
* `benchmarks/bench_gpu.py`, `benchmarks/accuracy_gpu.py`,
  `benchmarks/bench_depth_anything.py`.
* `pixi.toml`: `gpu` and `depth` environments, `test-gpu`, `bench-remove-gpu` and
  `bench-depth-anything`.
