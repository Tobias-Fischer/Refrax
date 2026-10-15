# Exploration: real-time flat-port refraction rendering on the GPU

Branch `explore/gpu-render` (from `explore/sim`). This branch ports the RGB-D refraction renderer of
`docs/explore_sim.md` (`core/render.py`) to the GPU. The prototype runs on the Apple M5 (MPS). The code
is written so that it moves to the lab's NVIDIA machines (CUDA) without changes.

| Part | Files |
|---|---|
| Portable torch renderer (cpu / mps / cuda), optional `torch.compile` | `core/render_torch.py` |
| Fused single kernel, Metal (MPS only, via `torch.mps.compile_shader`) | `core/render_metal.py` |
| Fused single kernel, NVIDIA Warp (CPU here, CUDA target) | `core/render_warp.py` |
| Accuracy against the numpy renderer | `sim/validate_torch.py` → `docs/figures/gpu_validation.json` |
| Benchmark | `sim/bench_render_gpu.py` → `docs/figures/gpu_bench.json` |
| Tests (cpu, mps, cuda; skipped if unavailable) | `tests/test_render_torch.py` |
| CLI | `render_refraction.py ... apply SIM_DIR --device auto\|cpu\|mps\|cuda` |

```bash
pixi run test-render-gpu       # pytest, ~10 s (environment "gpu": pytorch 2.14, warp-lang 1.17)
pixi run validate-render-gpu   # all validation scenes, all variants, ~3 min
pixi run bench-render-gpu      # 960x540 / 1920x1080 / 3840x2160, ~10 min
pixi run -e cuda bench-render-gpu   # Linux + NVIDIA: CUDA builds of pytorch and warp (untested)
```

**Summary.** The fused kernel renders a 1920×1080 frame in **1.5 ms** on the M5 GPU (8.7 ms at 4K). That is
600× faster than the numpy renderer. It matches numpy to **≤ 2e-4 px** away from depth edges and to
≤ 0.001 px overall. The portable torch path is also exact, but it is bound by per-op overhead: 73 ms
eager, 24 ms with `torch.compile`. Real time in a simulator therefore needs the fused kernel: Warp on
CUDA (Isaac Sim / OceanSim) or a fragment/compute shader elsewhere.

## 1. Design

The algorithm is the one in `core/render.py`, and so are the masks and their semantics. Each output
pixel's water ray projects into the pinhole render as a line in inverse depth w, written
u = U0 + DU·w, v = V0 + DV·w. The renderer marches w from 1/z_near to 0 in fixed steps of
≤ 0.5 render px. The first crossing with the bilinear inverse-depth texture is refined with Illinois
regula falsi, and the RGB is then gathered bilinearly at the result.

**Per camera / housing (once).** `FlatPortRenderer` computes in float64 numpy the planned render
camera, P2, the water ray, and (U0, V0, DU, DV) per output pixel. These are uploaded once as float32.
MPS has no float64, so nothing depth-independent is computed on the device. Setup takes 0.5 s at
1080p and 2.5–3 s at 4K, almost all of it numpy. It could be moved to the GPU if cameras change at
run time, but it does not need to be.

**Per frame (`TorchFlatPortRenderer`, portable).**
1. Inverse-depth texture (`depth_type` z or range).
2. **March.** The w values w_k are the same for all pixels. A chunk of 4 steps is therefore one
   `grid_sample` call, with frames as channels and steps stacked along the grid height. Only the
   first-hit index is tracked: chunks run in reverse and use one `torch.where` per step. The two
   bracket values are re-sampled afterwards.
3. **Refinement.** 4 Illinois iterations on all pixels with no compaction, so there is no host
   sync. Converged pixels are frozen as in numpy. 4 iterations are exact everywhere except at depth
   edges, where the residual is ≤ 0.02 px. Use `refine_iters=8` for ≤ 0.001 px.
4. One `grid_sample` of the RGB (uint8 or float, HWC in and out). Then the masks, Z and the water
   path length. `near_edge` is optional (`depth_edges_torch`, +4 ms at 1080p on MPS).

Frames are batched along the first axis. Batched results are bit-identical to single-frame
results.

**Fused kernels (`MetalFlatPortRenderer`, `WarpFlatPortRenderer`).**
- One thread per output pixel. The march exits early at the first crossing, refinement stops
  per thread on numpy's tolerance, and the RGB is gathered in the same thread.
- A small first kernel turns depth into the inverse-depth texture. Forming 1/z per bilinear tap
  inside the march instead was 3.5× slower.
- The Metal kernel reads uint8 or float32 RGB directly and writes the final uint8/float image,
  w, and a flag byte (valid / too near / background).
- Z and the water path length are derived from w on request (`depth_outputs`). In eager torch
  this costs ~4 ms at 1080p on MPS, so `render(..., info=False)` skips it. In a simulator the
  water model would go into the kernel (see §5).

**Device notes (MPS):**
- No op falls back to the CPU. The tests run with `PYTORCH_ENABLE_MPS_FALLBACK` unset, so a
  fallback would raise. The ops used are `grid_sample` (bilinear, border, `align_corners=True`),
  elementwise ops, `where`, `addcmul`, integer indexing, `max_pool2d` and `F.pad`. All are native
  on MPS and CUDA.
- An `amin` over an int tensor took 12 ms on MPS, even over a size-1 dimension. The reverse-order
  `where` march avoids it.
- `torch.compile` (inductor) works on MPS. It fuses the march and refinement into generated Metal
  kernels. A cold compile takes 12–20 s per (resolution, batch); warm-cache start-up takes 0.4–6 s.
- `grid_sample` is exact bilinear, unlike `cv2.remap`, which uses 1/32 px fixed-point weights. The
  torch path is therefore slightly closer to numpy's `exact=True` path than numpy's own default.

**float16 is not usable for the march.**
- `grid_sample` needs the grid and the texture in the same dtype. Half-precision normalised
  coordinates quantise positions to ~0.4 px on a 1600 px render.
- Measured: **0.45 px max error off edges, 2.7 px at edges, image PSNR 34–48 dB** against numpy,
  for only 25 % less time (55 ms vs 73 ms).
- A kernel could keep the coordinates in float32 and only store the inverse-depth texture in half
  (relative error 5e-4, i.e. < 0.01 px here). That halves texture traffic but is not implemented.

## 2. Accuracy (`pixi run validate-render-gpu`, 1920×1080 output, all 7 validation scenes)

The reference is the numpy renderer with `exact=True`: float64 bilinear sampling and 16 refinement
iterations. "coord" is |Δ(map_u, map_v)| in render pixels, i.e. the sample positions in the pinhole
render. Off edge means not flagged `near_edge`. The ground-truth error (GT err) is the same metric as
`sim/validate.py`: the exact refractive reprojection of the shown scene point, on 10k random pixels.

| Variant | coord max, off edges | coord p99, off edges | coord max, all pixels | valid / background mask mismatch | image PSNR vs numpy | GT err max off edges (spheres / steps / tilted) |
|---|---|---|---|---|---|---|
| numpy default (cv2.remap) | 1.1e-5 | 5.5e-7 | 0.001 | 0 / 0 | 108–123 dB | 9.4e-3 / 6.8e-3 / 1.8e-3 |
| torch cpu, fp32 | 2.1e-4 | 1.3e-4 | 0.021 (edges) | 0 / 0 | 104–119 dB | 9.7e-3 / 6.9e-3 / 1.9e-3 |
| torch mps, fp32 | 2.1e-4 | 1.3e-4 | 0.021 (edges) | 0 / 0 | 104–119 dB | 9.8e-3 / 7.0e-3 / 1.8e-3 |
| torch mps, `torch.compile` | 2.1e-4 | 1.3e-4 | 0.021 (edges) | 0 / 0 | 104–119 dB | 9.2e-3 / 6.9e-3 / 1.9e-3 |
| torch mps, **fp16** | **0.46** | 0.35 | **2.76** | 0 / 6e-5 | 34–48 dB | — |
| Metal fused kernel | 1.3e-4 | 1.0e-4 | 0.001 | 0 / 0 | 105–120 dB | 9.4e-3 / 6.7e-3 / 2.0e-3 |
| Warp fused kernel (CPU) | 1.3e-4 | 1.0e-4 | 0.001 | 0 / 0 | 105–120 dB | 9.4e-3 / 6.9e-3 / 1.9e-3 |

- **All float32 variants agree with numpy to ≤ 2e-4 px.** This is float32 rounding of
  coordinates around 1600 px, whose ulp is 1.2e-4.
- **The ground-truth error is unchanged:** p99 ≤ 4e-3 px and max ≤ 0.01 px off edges, as for numpy.
- The larger "all pixels" values (≤ 0.021 px) are depth-edge pixels in the torch path, which uses a
  fixed 4 refinement iterations.
- Batched rendering is bit-identical to single-frame rendering for every variant.
- `render_refraction.py apply --device mps` on the `check_render --make-demo` scenes gives uint8
  images within 1 LSB of numpy on every pixel except one, and identical validity masks.
- Camera-frame Z agrees to < 1e-5 relative off edges. Pixels straddling a depth edge can differ by
  up to 5 %, because Z is very sensitive there; this also comes from the 4 refinement iterations.

The tests (`tests/test_render_torch.py`) run all of this on a 480×270 output camera:
- 4 scenes (near plane, tilted plane, steps, spheres);
- torch on cpu / mps / cuda, plus the Metal and Warp kernels;
- uint8 batches, `depth_type="range"`, and Z / water range.

They assert coord ≤ 1e-3 px off edges, p99.99 ≤ 0.01 px, max ≤ 0.1 px, identical valid masks and
PSNR > 60 dB.

## 3. Benchmark (`pixi run bench-render-gpu`)

**Setup.**
- Apple M5 (10-core GPU, 24 GB), torch 2.14.1, warp 1.17.0, macOS 27.
- Scene "spheres" (occlusions + background), uint8 RGB + float32 z-depth, as a simulator delivers
  them.
- Median and p25–p75 over 30 repeats after 3 warm-up runs (CPU rows: 3–7 repeats).

**The machine was heavily shared.** Load average was 120–155 during the run (other agents, colmap),
and torch used 4 CPU threads. **The CPU rows are therefore pessimistic.** The numpy renderer took
0.5 s per 1080p frame in `explore_sim.md` and 1.1 s here. GPU rows were stable (narrow IQR), but
they may also have shared the GPU. Measured copy bandwidth: MPS 113 GB/s (spec 153), CPU 52 GB/s.

**Columns.**
- *frame:* time per frame with inputs already on the device.
  - Torch rows return all outputs: maps, masks, Z, water range; no `near_edge`.
  - Fused rows return the image + flags + w (`info=False`).
- *setup:* per camera and housing; numpy geometry + upload.
- *H2D / D2H:* numpy → device copy of RGB + depth, and device → numpy copy of the uint8 image.
  On the M5 these are copies within unified memory.

| Output | Method | Batch | Setup (ms) | **Frame (ms/frame)**, median (p25–p75) | H2D (ms/frame) | D2H (ms/frame) | fps (frame) |
|---|---|---|---|---|---|---|---|
| 960×540 | numpy (cv2.remap) | 1 | 87 | 201 (197–213) | – | – | 5 |
| | torch-cpu | 1 | 108 | 166 (164–169) | 0 | 0 | 6 |
| | Warp fused (cpu) | 1 | 159 | 79 (73–87) | 0 | 0 | 13 |
| | torch-mps | 1 / 4 | 122 | 15.7 (15.4–16.1) / 12.8 | 0.6 / 0.1 | 0.3 / 0.1 | 64 / 78 |
| | torch-mps compile | 1 / 4 | 0.9 s first call | 4.8 (4.7–4.9) / 4.0 | 0.4 / 0.1 | 0.3 / 0.1 | 208 / 250 |
| | **Metal fused** | 1 / 4 | 118 | **0.40** (0.40–0.40) / 0.30 | 0.2 / 0.1 | 0.2 / 0.1 | 2500 |
| 1920×1080 | numpy (cv2.remap) | 1 | 489 | 1138 (1108–1166) | – | – | 0.9 |
| | torch-cpu | 1 | 587 | 904 (859–991) | 0 | 0 | 1.1 |
| | Warp fused (cpu) | 1 | 624 | 479 (450–510) | 0 | 0 | 2.1 |
| | torch-mps | 1 / 4 | 711 | 73.0 (72.2–73.3) / 66.3 | 0.6 / 0.4 | 0.6 / 0.3 | 14 / 15 |
| | torch-mps fp16 (inaccurate) | 1 / 4 | 554 | 55.0 (54.6–55.4) / 52.3 | 0.6 / 0.5 | 0.5 / 0.3 | 18 / 19 |
| | torch-mps compile | 1 / 4 | 0.4 s first call | 23.5 (23.3–24.0) / 25.0 | 0.6 / 0.5 | 0.5 / 0.3 | 43 / 40 |
| | **Metal fused** | 1 / 4 | 567 | **1.5** (1.5–1.6) / 1.3 | 0.6 / 0.5 | 0.6 / 0.4 | 670 / 770 |
| 3840×2160 | numpy (cv2.remap) | 1 | 2484 | 7819 (7750–8098) | – | – | 0.1 |
| | torch-cpu | 1 | 2868 | 5664 (5255–5778) | 0 | 0 | 0.2 |
| | Warp fused (cpu) | 1 | 3471 | 2915 (2910–2964) | 0 | 0 | 0.3 |
| | torch-mps | 1 | 2892 | 397 (391–400) | 2.1 | 1.4 | 2.5 |
| | torch-mps compile | 1 | 0.7 s first call | 140 (139–141) | 2.0 | 1.3 | 7 |
| | **Metal fused** | 1 | 2841 | **8.7** (8.4–9.7) | 2.0 | 1.5 | 115 |

Render (pinhole) sizes are 801×513, 1599×1019 and 3191×2035, with 13, 27 and 53 march steps.
Further costs:
- Eager torch `near_edge`: +4–13 ms on MPS.
- Fused kernels with Z / water range unpacked in eager torch: +4.4 ms at 1080p and +16 ms at 4K.
  Fusing them into the kernel would make this ≈ 0.

**Reading.**
- **Eager torch is bound by op count, not arithmetic.** About 200 full-frame elementwise passes
  cost ~0.3 ms each at 1080p on MPS, against 0.65 ms for one `grid_sample`.
- **`torch.compile` cuts this 3×.**
- **The fused kernel is 15× faster again.** At 1080p its DRAM traffic is ~30 MB: 6.5 MB depth,
  4.9 MB RGB in, 6.2 MB image + 8 MB w out. That is ~0.3 ms at the measured bandwidth. The
  remaining time goes to the up to ~30 dependent bilinear taps per pixel, served from cache.
- **Batching helps the eager path (launch amortisation) but not the fused kernel.**
- **Cost scales with pixels × march steps.** The step count grows with resolution because steps are
  ≤ 0.5 render px. At 4K, `step_px=1` would halve the march at little cost: rays then cross thin
  occluders less reliably, which still needs to be measured.

### Estimated CUDA performance (**estimate, not measured**)

Scaling assumptions:
- The fused kernel is bound by cache/ALU throughput. Assume it scales with GPU throughput relative
  to the M5's 10-core GPU (~4–5 FP32 TFLOPS, 153 GB/s):
  - RTX 3090: 35.6 TFLOPS, 936 GB/s, 82 SMs; assume ×4–8.
  - Quadro RTX 6000: 16.3 TFLOPS, 672 GB/s, 72 SMs; assume ×3–5.
- Eager and compiled torch are bound by bandwidth and launches. Assume ×5–8 bandwidth, plus
  ~2 ms of CPU-side launch overhead for eager.
- PCIe 4.0 x16 (3090) ≈ 20 GB/s pinned. PCIe 3.0 x16 (RTX 6000) ≈ 11 GB/s pinned. Pageable host
  memory is about half that.

| 1920×1080 (4K in brackets) | RTX 3090 (estimate) | Quadro RTX 6000 (estimate) |
|---|---|---|
| Warp fused kernel | 0.2–0.4 ms (1–2 ms) | 0.3–0.5 ms (1.7–3 ms) |
| torch eager | 10–17 ms (55–80 ms) | 12–20 ms (70–100 ms) |
| torch.compile | 3–6 ms (18–30 ms) | 4–8 ms (25–40 ms) |
| H2D RGB + depth (11.4 MB; 4K: 45 MB) + D2H uint8 image (6.2 MB; 4K: 25 MB) | 0.9 ms pinned, ~2 ms pageable (3.5 / 7 ms) | 1.6 ms pinned, ~3 ms pageable (6 / 12 ms) |

The transfers disappear in Isaac Sim, where the annotators deliver device arrays.

## 4. What to verify on CUDA

1. **Environment.** `pixi run -e cuda test-render-gpu`. The `cuda` environment
   (linux-64, `cuda = "12"` virtual package) resolves to pytorch 2.14.1 `cuda129` and warp-lang 1.17
   `cuda129`. It needs a driver that supports CUDA 12.9. Check that the build includes sm_75
   (Quadro RTX 6000, Turing) and sm_86 (3090). The `torch-cuda` and `warp-cuda` tests are written
   but have never run.
2. **Accuracy.** `pixi run -e cuda validate-render-gpu` must show the same ≤ 2e-4 px. CUDA
   `grid_sample` and Warp use the same float32 formulas. TF32 does not affect these ops.
3. **Benchmark.** `pixi run -e cuda bench-render-gpu` measures the real numbers behind the
   estimates above, including `torch.compile` with Triton.
   - Add a pinned-memory H2D variant (`tensor.pin_memory()` + `non_blocking=True`).
   - Add CUDA graphs for the eager path (`torch.cuda.graphs`), which removes the launch overhead.
4. **Warp kernel on CUDA.**
   - Check that `WarpFlatPortRenderer(r, "cuda")` compiles and matches the CPU result.
   - Time it with `wp.synchronize`.
   - Check divergence cost: early-exit loops are fine on SIMT, but warps whose rays hit at very
     different depths serialise.
   - Try storing the inverse-depth texture as a `wp.Texture2D` (hardware bilinear) or in half
     precision. Hardware filtering on NVIDIA uses 8-bit fractional weights (1/256), so check the
     sub-pixel accuracy; it should still be well below 0.01 px.
5. **Isaac Sim end to end.** Run the `sim/isaacsim` adapter with `WarpFlatPortRenderer` on the
   annotator device arrays (see §5). Measure per-frame overhead against the render itself.
6. **Memory.** The eager path holds a few (B, 4·H, W) float32 buffers per march chunk: ~250 MB at
   1080p B=4 and ~400 MB at 4K B=1. Lower `chunk` to reduce this. The fused kernels hold only the
   inputs and outputs.

## 5. Path to in-simulator real time

The fused kernel body is about 60 lines (`KERNEL_SRC` in `core/render_metal.py`, `flatport_render`
in `core/render_warp.py`). Per frame it needs only three inputs:
- a per-pixel float4 table (U0, V0, DU, DV), constant per camera;
- the depth texture and the RGB texture;
- the scalars w_near, n_steps and tol.

It therefore maps directly onto any shading language:

- **Isaac Sim / OceanSim (recommended, CUDA).**
  - Replicator annotators (`rgb`, `distance_to_image_plane`) can return device arrays when created
    with `AnnotatorRegistry.get_annotator(name, device="cuda")`, i.e. Warp arrays (API to check on
    the installed Isaac Sim version).
  - `WarpFlatPortRenderer.launch` consumes these zero-copy via `wp.from_torch` / `wp.from_dlpack`.
    Alternatively, the kernel can be registered as a Replicator augmentation
    (`rep.annotators.Augmentation`, which accepts Warp kernels; to check). The refracted image is
    then itself an annotator.
  - OceanSim's water model runs after it. Fusing that model into the same kernel would remove the
    w → Z → water-range pass:
    I = J·exp(−β_D·t) + B∞·(1 − exp(−β_B·t)), with t computed from w in-thread.
  - `sim/isaacsim/render_flatport_isaacsim.py` currently calls the numpy renderer. Swapping in
    `WarpFlatPortRenderer(r, "cuda")` is a two-line change, left for when it can be tested on a
    CUDA machine.
- **Gazebo / Ogre2 (GLSL) and Unity (HLSL).** Use a post-process pass on the render camera's
  colour + depth targets:
  - store the line table as an RGBA32F texture (output resolution);
  - store the inverse depth as R32F with linear filtering;
  - the loop count is constant per camera, so it can be a uniform or a compile-time constant.
  The output has the size of the real camera, not of the render, so the pass needs its own render
  target. Depth must be linearised from the projection's non-linear z-buffer first.
- **Unreal (HoloOcean).** Use a SceneCapture2D with the planned camera → render target → a material
  with a Custom HLSL node holding the same loop, drawn to a canvas at the output resolution. The
  `SceneDepth` units (cm) need checking.

Frame budgets in all cases:
- The pinhole render itself is a normal raster/RT pass at ~0.7× the output pixel count; ~2× with
  oversample 2 for sharp corners.
- The post-process costs < 1 ms at 1080p on a desktop GPU (estimate above).
- Refraction therefore adds little to a simulator's frame time. Disocclusion holes at depth edges
  (`docs/explore_sim.md` §5) remain the accuracy limit, not the speed.

## 6. Notes and limitations

- The fused kernels skip the `near_edge` mask. It is cheap to add: a 3×3 jump test on the inverse
  depth at the final sample position. The torch path has it.
- `--device` in `render_refraction.py` uses the eager torch path (all outputs). For a file-based
  pipeline the bottleneck is PNG/NPY I/O, not the renderer.
- The torch path's fixed 4 refinement iterations give ≤ 0.02 px at depth edges. Pass
  `refine_iters=8` for ≤ 0.001 px. Each extra iteration costs ~7 ms eager at 1080p on MPS. The
  fused kernels iterate per pixel until converged and need no such choice.
- The CPU benchmark rows were measured under heavy load and are not representative of an idle
  M5.
