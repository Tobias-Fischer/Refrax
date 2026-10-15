"""Fused flat-port renderer as an NVIDIA Warp kernel (same algorithm as core/render_metal.py).

This is the intended CUDA / Isaac Sim path: Isaac Sim (Replicator annotators) hands out
RGB and depth as Warp/torch arrays on the GPU, so the kernel can run in the same process
without a host round trip. On macOS Warp only has a CPU backend (LLVM); the kernel is
validated there against the numpy renderer (tests/test_render_torch.py) and is expected to
run unchanged with device="cuda" (not yet verified, see docs/explore_gpu_render.md).

Interop: inputs/outputs are torch tensors (cpu or cuda) shared zero-copy via wp.from_torch.
"""
import numpy as np
import torch
import warp as wp

from core.render_torch import TorchFlatPortRenderer, _upload

wp.config.quiet = True


@wp.kernel
def inverse_depth(depth: wp.array3d(dtype=float), rng: int, K: wp.vec4, inv: wp.array3d(dtype=float)):
    """Inverse z-depth texture; 0 = no geometry (<= 0, inf, nan). K = (fx, fy, cx, cy)."""
    b, y, x = wp.tid()
    z = depth[b, y, x]
    if rng != 0:
        a = (float(x) - K[2]) / K[0]
        c = (float(y) - K[3]) / K[1]
        z = z / wp.sqrt(a * a + c * c + 1.0)
    r = float(0.0)
    if wp.isfinite(z) and z > 0.0:
        r = 1.0 / z
    inv[b, y, x] = r


@wp.func
def _bil(img: wp.array3d(dtype=float), b: int, u: float, v: float):
    """Bilinear sample, border clamp, pixel centres at integers."""
    Hr = img.shape[1]
    Wr = img.shape[2]
    x = wp.clamp(u, 0.0, float(Wr - 1))
    y = wp.clamp(v, 0.0, float(Hr - 1))
    x0 = wp.min(int(x), Wr - 2)
    y0 = wp.min(int(y), Hr - 2)
    fx = x - float(x0)
    fy = y - float(y0)
    top = wp.lerp(img[b, y0, x0], img[b, y0, x0 + 1], fx)
    bot = wp.lerp(img[b, y0 + 1, x0], img[b, y0 + 1, x0 + 1], fx)
    return wp.lerp(top, bot, fy)


@wp.kernel
def flatport_render(line: wp.array2d(dtype=wp.vec4), inv: wp.array3d(dtype=float),
                    rgb: wp.array4d(dtype=float), wsteps: wp.array(dtype=float),
                    n: int, iters: int, tol: float,
                    w_out: wp.array3d(dtype=float), img: wp.array4d(dtype=float),
                    flags: wp.array3d(dtype=wp.uint8)):
    b, i, j = wp.tid()
    L = line[i, j]
    Hr = inv.shape[1]
    Wr = inv.shape[2]

    # 1. march w_k = w_near (1 - k/n); first k with g = w - inv(u(w)) <= 0
    w_hi = float(wsteps[0])
    g_hi = float(w_hi - _bil(inv, b, L[0] + L[2] * w_hi, L[1] + L[3] * w_hi))
    w_lo = float(w_hi)
    g_lo = float(g_hi)
    too_near = g_hi <= 0.0
    if not too_near:
        for k in range(1, n + 1):
            w = wsteps[k]
            g = w - _bil(inv, b, L[0] + L[2] * w, L[1] + L[3] * w)
            if g <= 0.0:
                w_lo = w
                g_lo = g
                break
            w_hi = w
            g_hi = g

    # 2. Illinois regula falsi on [w_lo, w_hi]
    wb = float(w_lo)
    if (not too_near) and g_lo < 0.0:
        aw = float(w_hi)
        ag = float(g_hi)
        bw = float(w_lo)
        bg = float(g_lo)
        side = int(0)
        for it in range(iters):
            den = ag - bg
            wn = 0.5 * (aw + bw)
            if den > 0.0:
                wn = (bw * ag - aw * bg) / den
            gn = wn - _bil(inv, b, L[0] + L[2] * wn, L[1] + L[3] * wn)
            wb = wn
            if gn > 0.0:
                if side == 1:
                    bg = 0.5 * bg
                aw = wn
                ag = gn
                side = 1
            else:
                if side == -1:
                    ag = 0.5 * ag
                bw = wn
                bg = gn
                side = -1
            if wp.abs(gn) <= tol or aw - bw <= tol:
                break
    w_out[b, i, j] = wb

    # 3. colour gather
    u = L[0] + L[2] * wb
    v = L[1] + L[3] * wb
    in_img = u >= -0.5 and u <= float(Wr) - 0.5 and v >= -0.5 and v <= float(Hr) - 0.5
    f = int(0)
    if too_near:
        f = f + 1
    if in_img:
        f = f + 2
    flags[b, i, j] = wp.uint8(f)
    x = wp.clamp(u, 0.0, float(Wr - 1))
    y = wp.clamp(v, 0.0, float(Hr - 1))
    x0 = wp.min(int(x), Wr - 2)
    y0 = wp.min(int(y), Hr - 2)
    fx = x - float(x0)
    fy = y - float(y0)
    for c in range(rgb.shape[3]):
        top = wp.lerp(rgb[b, y0, x0, c], rgb[b, y0, x0 + 1, c], fx)
        bot = wp.lerp(rgb[b, y0 + 1, x0, c], rgb[b, y0 + 1, x0 + 1, c], fx)
        img[b, i, j, c] = wp.lerp(top, bot, fy)


class WarpFlatPortRenderer(TorchFlatPortRenderer):
    """Fused Warp kernel; device 'cpu' or 'cuda' (torch device naming)."""

    def __init__(self, base, device="cpu", refine_iters=16):
        super().__init__(base, device, torch.float32, refine_iters)
        wp.init()
        self.wp_device = "cpu" if self.device.type == "cpu" else f"cuda:{self.device.index or 0}"
        s = self.spec
        line = np.stack([s.fx * base.a[..., 0] + s.cx, s.fy * base.a[..., 1] + s.cy,
                         s.fx * base.b[..., 0], s.fy * base.b[..., 1]], -1).astype(np.float32)
        self.line_t = torch.as_tensor(line, device=self.device).contiguous()
        self.line = wp.from_torch(self.line_t, dtype=wp.vec4)
        self.wsteps_wp = wp.from_torch(self.w_steps)
        self.tol = self.w_near * 1e-6

    def launch(self, depth, x, depth_type="z"):
        """depth (B, Hr, Wr) float32 (z or range), x (B, Hr, Wr, C) float32; contiguous."""
        B, Hr, Wr, C = x.shape
        w = torch.empty((B, self.H, self.W), device=self.device)
        img = torch.empty((B, self.H, self.W, C), device=self.device)
        flags = torch.empty((B, self.H, self.W), dtype=torch.uint8, device=self.device)
        inv = torch.empty_like(depth)
        wp.launch(inverse_depth, dim=depth.shape, device=self.wp_device,
                  inputs=[wp.from_torch(depth), int(depth_type == "range"),
                          wp.vec4(self.spec.fx, self.spec.fy, self.spec.cx, self.spec.cy)],
                  outputs=[wp.from_torch(inv)])
        wp.launch(flatport_render, dim=(B, self.H, self.W), device=self.wp_device,
                  inputs=[self.line, wp.from_torch(inv), wp.from_torch(x), self.wsteps_wp,
                          self.n_steps, self.refine_iters, self.tol],
                  outputs=[wp.from_torch(w), wp.from_torch(img), wp.from_torch(flags, dtype=wp.uint8)])
        return w, img, flags

    def render(self, rgb, depth, depth_type="z", invalid_value=0, edges=False, info=True):
        if depth_type not in ("z", "range"):
            raise ValueError(depth_type)
        z = _upload(depth, self.device)
        x = _upload(rgb, self.device)
        if z.ndim == 2:
            z, x = z[None], x[None]
        out_dtype = x.dtype
        w, img, flags = self.launch(z.float().contiguous(), x.float().contiguous(), depth_type)
        too_near = (flags & 1) > 0
        valid = self.valid_ray & ((flags & 2) > 0) & ~too_near
        if out_dtype == torch.uint8:
            img = img.round().clamp(0, 255)
        img = torch.where(valid[..., None], img, torch.full_like(img, invalid_value)).to(out_dtype)
        if not info:
            return img, dict(w=w, valid=valid, flags=flags)
        Z, water_range = self.depth_outputs(w)
        info = dict(w=w, Z=Z, water_range=water_range, valid=valid, too_near=too_near, background=(w <= 1e-12) & valid,
                    map_u=self.line_t[..., 0] + self.line_t[..., 2] * w,
                    map_v=self.line_t[..., 1] + self.line_t[..., 3] * w)
        return img, info
