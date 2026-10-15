"""Fused single-kernel variant of the flat-port renderer (Metal via torch.mps.compile_shader).

Prototype of the shader path (one thread per output pixel: march, Illinois refinement and
RGB gather in one kernel, nothing but the result written to memory). It is MPS-only by
construction; the same kernel body maps 1:1 onto CUDA / Warp / GLSL / HLSL (see
docs/explore_gpu_render.md and `KERNEL_SRC`). Per-camera data are those of
TorchFlatPortRenderer: one float4 (U0, V0, DU, DV) per output pixel, in render pixels.

Differences to the torch path: the march stops at the first crossing (early exit per thread)
and the refinement stops per thread on numpy's convergence criterion, so the result equals
the numpy renderer's up to float32 rounding (no fixed iteration count is needed). The kernel
reads the simulator's buffers directly: depth (z or range, float32; a first tiny kernel turns
it into the inverse-depth texture, cheaper than forming 1/z per bilinear tap) and RGB(A) as
uint8 or float32 HWC, and writes the final image (invalid pixels = invalid_value), w and a flag
byte (1 = valid, 2 = too near, 4 = background).
"""
import numpy as np
import torch

from core.render_torch import TorchFlatPortRenderer, _upload

KERNEL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

struct Params {          // must match MetalFlatPortRenderer._params
    int HW, Wr, Hr, n, iters, C, B, range_depth;
    float w_near, tol, fx, fy, cx, cy, invalid;
};

// pass 1: inverse z-depth texture; 0 = no geometry (<= 0, inf, nan)
kernel void inverse_depth(device const float* depth [[buffer(0)]], device float* inv [[buffer(1)]],
                          constant Params& P [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    const int n = P.Wr * P.Hr;
    if (int(gid) >= n * P.B) return;
    float z = depth[gid];
    if (P.range_depth) {
        const int q = int(gid) % n, y = q / P.Wr, x = q - y * P.Wr;
        float a = (float(x) - P.cx) / P.fx, b = (float(y) - P.cy) / P.fy;
        z = z * rsqrt(a * a + b * b + 1.0f);
    }
    inv[gid] = (isfinite(z) && z > 0.0f) ? 1.0f / z : 0.0f;
}

// bilinear sample, border clamp, pixel centres at integers
// (= grid_sample align_corners=True, padding_mode="border" = cv2.remap BORDER_REPLICATE)
inline float bil_inv(device const float* d, float u, float v, constant Params& P) {
    float x = clamp(u, 0.0f, float(P.Wr - 1)), y = clamp(v, 0.0f, float(P.Hr - 1));
    int x0 = min(int(x), P.Wr - 2), y0 = min(int(y), P.Hr - 2);
    float fx = x - float(x0), fy = y - float(y0);
    device const float* q = d + y0 * P.Wr + x0;
    return mix(mix(q[0], q[1], fx), mix(q[P.Wr], q[P.Wr + 1], fx), fy);
}

inline float to_out(float x, float)  { return x; }
inline uchar to_out(float x, uchar)  { return uchar(clamp(rint(x), 0.0f, 255.0f)); }

#define FLATPORT_KERNEL(NAME, T)                                                              \
kernel void NAME(device const float4* line [[buffer(0)]], device const uchar* valid_ray [[buffer(1)]], \
                 device const float* inv [[buffer(2)]], device const T* rgb [[buffer(3)]],      \
                 device const float* wsteps [[buffer(4)]], device float* w_out [[buffer(5)]], \
                 device T* img [[buffer(6)]], device uchar* flags [[buffer(7)]],              \
                 constant Params& P [[buffer(8)]], uint gid [[thread_position_in_grid]]) {   \
    if (int(gid) >= P.HW * P.B) return;                                                       \
    const int b = int(gid) / P.HW, p = int(gid) - b * P.HW;                                   \
    const float4 L = line[p];                                                                 \
    device const float* d = inv + size_t(b) * P.Wr * P.Hr;                                    \
    /* 1. march w_k = w_near (1 - k/n), k = 0..n: first k with g = w - inv(u(w)) <= 0 */      \
    float w_hi = wsteps[0];                                                                   \
    float g_hi = w_hi - bil_inv(d, L.x + L.z * w_hi, L.y + L.w * w_hi, P);                    \
    float w_lo = w_hi, g_lo = g_hi;                                                           \
    bool too_near = g_hi <= 0.0f;                                                             \
    if (!too_near) {                                                                          \
        for (int k = 1; k <= P.n; ++k) {                                                      \
            float w = wsteps[k];                                                              \
            float g = w - bil_inv(d, L.x + L.z * w, L.y + L.w * w, P);                        \
            if (g <= 0.0f) { w_lo = w; g_lo = g; break; }                                     \
            w_hi = w; g_hi = g;                                                               \
        }                                                                                     \
    }                                                                                         \
    /* 2. Illinois regula falsi on [w_lo, w_hi], g(w_hi) > 0 >= g(w_lo) */                    \
    float w = w_lo;                                                                           \
    if (!too_near && g_lo < 0.0f) {                                                           \
        float aw = w_hi, ag = g_hi, bw = w_lo, bg = g_lo;                                     \
        int side = 0;                                                                         \
        for (int it = 0; it < P.iters; ++it) {                                                \
            float den = ag - bg;                                                              \
            float wn = den > 0.0f ? (bw * ag - aw * bg) / den : 0.5f * (aw + bw);             \
            float gn = wn - bil_inv(d, L.x + L.z * wn, L.y + L.w * wn, P);                    \
            w = wn;                                                                           \
            if (gn > 0.0f) { if (side == 1) bg *= 0.5f; aw = wn; ag = gn; side = 1; }        \
            else           { if (side == -1) ag *= 0.5f; bw = wn; bg = gn; side = -1; }       \
            if (fabs(gn) <= P.tol || aw - bw <= P.tol) break;                                 \
        }                                                                                     \
    }                                                                                         \
    w_out[gid] = w;                                                                           \
    /* 3. colour gather (HWC) */                                                              \
    float u = L.x + L.z * w, v = L.y + L.w * w;                                               \
    bool in_img = u >= -0.5f && u <= float(P.Wr) - 0.5f && v >= -0.5f && v <= float(P.Hr) - 0.5f; \
    bool valid = valid_ray[p] && in_img && !too_near;                                         \
    flags[gid] = uchar((valid ? 1 : 0) | (too_near ? 2 : 0) | (valid && w <= 1e-12f ? 4 : 0)); \
    device T* o = img + size_t(gid) * P.C;                                                    \
    if (!valid) { for (int c = 0; c < P.C; ++c) o[c] = to_out(P.invalid, T()); return; }     \
    float x = clamp(u, 0.0f, float(P.Wr - 1)), y = clamp(v, 0.0f, float(P.Hr - 1));           \
    int x0 = min(int(x), P.Wr - 2), y0 = min(int(y), P.Hr - 2);                               \
    float fx = x - float(x0), fy = y - float(y0);                                             \
    device const T* q = rgb + (size_t(b) * P.Wr * P.Hr + size_t(y0) * P.Wr + x0) * P.C;       \
    const int rs = P.Wr * P.C;                                                                \
    for (int c = 0; c < P.C; ++c)                                                             \
        o[c] = to_out(mix(mix(float(q[c]), float(q[P.C + c]), fx),                            \
                          mix(float(q[rs + c]), float(q[rs + P.C + c]), fx), fy), T());       \
}

FLATPORT_KERNEL(flatport_render_f32, float)
FLATPORT_KERNEL(flatport_render_u8, uchar)
"""


class MetalFlatPortRenderer(TorchFlatPortRenderer):
    """Fused-kernel renderer; requires device 'mps'. refine_iters is an upper bound."""

    _lib = None

    def __init__(self, base, refine_iters=16):
        super().__init__(base, "mps", torch.float32, refine_iters)
        if MetalFlatPortRenderer._lib is None:
            MetalFlatPortRenderer._lib = torch.mps.compile_shader(KERNEL_SRC)
        s = self.spec
        line = np.stack([s.fx * base.a[..., 0] + s.cx, s.fy * base.a[..., 1] + s.cy,
                         s.fx * base.b[..., 0], s.fy * base.b[..., 1]], -1)
        self.line = torch.as_tensor(line.astype(np.float32), device=self.device).contiguous()
        self.valid_u8 = self.valid_ray.to(torch.uint8).contiguous()
        self._params = {}

    def _param_buf(self, B, C, depth_type, invalid_value):
        key = (B, C, depth_type, float(invalid_value))
        if key not in self._params:
            s = self.spec
            i = np.array([self.H * self.W, s.W, s.H, self.n_steps, self.refine_iters, C, B,
                          int(depth_type == "range")], np.int32)
            f = np.array([self.w_near, self.w_near * 1e-6, s.fx, s.fy, s.cx, s.cy,
                          invalid_value], np.float32)
            # one byte buffer holding the Params struct (8 int32 + 7 float32)
            self._params[key] = torch.from_numpy(np.concatenate([i.view(np.uint8), f.view(np.uint8)])).to(self.device)
        return self._params[key]

    def launch(self, depth, x, depth_type="z", invalid_value=0):
        """The kernel alone: depth (B, Hr, Wr) float32, x (B, Hr, Wr, C) uint8 or float32,
        contiguous on the device -> w (B, H, W), img (B, H, W, C) (dtype of x), flags."""
        B, Hr, Wr, C = x.shape
        if depth_type not in ("z", "range"):
            raise ValueError(depth_type)
        w = torch.empty((B, self.H, self.W), device=self.device)
        img = torch.empty((B, self.H, self.W, C), dtype=x.dtype, device=self.device)
        flags = torch.empty((B, self.H, self.W), dtype=torch.uint8, device=self.device)
        P = self._param_buf(B, C, depth_type, invalid_value)
        inv = torch.empty_like(depth)
        self._lib.inverse_depth(depth, inv, P, threads=inv.numel())
        fn = self._lib.flatport_render_u8 if x.dtype == torch.uint8 else self._lib.flatport_render_f32
        fn(self.line, self.valid_u8, inv, x, self.w_steps, w, img, flags, P,
           threads=B * self.H * self.W)
        return w, img, flags

    def render(self, rgb, depth, depth_type="z", invalid_value=0, edges=False, info=True):
        """Same as TorchFlatPortRenderer.render (no near_edge). info=False skips the few
        elementwise ops that unpack the flags and map coordinates (returns raw flags)."""
        z = _upload(depth, self.device)
        x = _upload(rgb, self.device)
        if z.ndim == 2:
            z, x = z[None], x[None]
        if x.dtype not in (torch.uint8, torch.float32):
            x = x.float()
        w, img, flags = self.launch(z.float().contiguous(), x.contiguous(), depth_type, invalid_value)
        if not info:
            return img, dict(w=w, flags=flags)
        Z, water_range = self.depth_outputs(w)
        return img, dict(w=w, Z=Z, water_range=water_range, flags=flags, valid=(flags & 1) > 0, too_near=(flags & 2) > 0,
                         background=(flags & 4) > 0,
                         map_u=self.line[..., 0] + self.line[..., 2] * w,
                         map_v=self.line[..., 1] + self.line[..., 3] * w)
