"""PyTorch port of the flat-port refraction renderer (core.render.FlatPortRenderer).

Device-agnostic (cpu / mps / cuda); only standard ops are used (grid_sample, elementwise,
min/gather, max_pool2d). No float64 on the device (MPS has none): the per-camera geometry is
computed once in float64 numpy by `FlatPortRenderer` and uploaded as float32.

Split of the work:

* per camera / housing (`TorchFlatPortRenderer.__init__`, once): the water ray of every
  output pixel projected into the pinhole render as a line in inverse depth w,
  u = U0 + DU * w, v = V0 + DV * w, stored directly in grid_sample's normalised
  coordinates (align_corners=True), plus P2_z and d_z for depth / water path length;
* per frame (`trace`, `render`): march every line against the inverse-depth texture
  (n_steps + 1 fixed w values shared by all pixels, so a chunk of steps is a single
  grid_sample with frames as channels), Illinois regula-falsi refinement with a fixed number
  of iterations (no compaction, no host sync), one grid_sample of the RGB render.

Frames are batched along the first axis. Semantics (masks, background at w = 0, first
crossing = visible surface) are those of the numpy renderer; see core/render.py.
"""
import numpy as np
import torch
import torch.nn.functional as F

from core.render import FlatPortRenderer


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _upload(x, device):
    """numpy / tensor -> tensor on device (float64 is cast to float32 on the host first:
    MPS has no float64, and it halves the transfer)."""
    if isinstance(x, np.ndarray):
        if x.dtype == np.float64:
            x = x.astype(np.float32)
        x = torch.from_numpy(np.ascontiguousarray(x))
    elif x.dtype == torch.float64:
        x = x.float()
    return x.to(device, non_blocking=True)


def _grid_sample(img, grid):
    """Bilinear, border clamp, pixel centres at integer coordinates (= cv2.remap with
    BORDER_REPLICATE, without cv2's 1/32-px fixed-point weights)."""
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def depth_edges_torch(inv, rel=0.03):
    """core.render.depth_edges for inv (B, 1, H, W): bool mask of pixels next to a relative
    inverse-depth jump > rel, dilated by 1 px."""
    B, _, H, W = inv.shape
    m = torch.zeros((B, 1, H, W), dtype=torch.bool, device=inv.device)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = inv[..., dy:, max(dx, 0):W + min(dx, 0)]
        b = inv[..., :H - dy, max(-dx, 0):W - max(dx, 0)]
        jump = (a - b).abs() > rel * torch.clamp(torch.maximum(a, b), min=1e-12)
        # F.pad order: (left, right, top, bottom)
        m |= F.pad(jump, (max(dx, 0), max(-dx, 0), dy, 0))
        m |= F.pad(jump, (max(-dx, 0), max(dx, 0), 0, dy))
    return F.max_pool2d(m.to(inv.dtype), 3, stride=1, padding=1) > 0


class TorchFlatPortRenderer:
    """GPU/CPU renderer with the same interface as FlatPortRenderer.trace / render.

    base: a FlatPortRenderer (provides spec, P2, d, a, b, w_near, n_steps).
    dtype: torch.float32 (default) or torch.float16 (measured: not accurate enough for
        the march coordinates, see docs/explore_gpu_render.md).
    refine_iters: fixed number of Illinois iterations (converged pixels are frozen).
    chunk: march steps evaluated per grid_sample call (memory ~ chunk * B * H * W * 16 B).
    compile: wrap march + refinement in torch.compile (inductor fuses the elementwise ops
        and grid_sample; first call per batch size compiles for ~10 s).
    """

    def __init__(self, base: FlatPortRenderer, device=None, dtype=torch.float32,
                 refine_iters=4, chunk=4, compile=False):
        self.base = base
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype
        self.refine_iters, self.chunk = refine_iters, chunk
        s = self.spec = base.spec
        self.W, self.H = base.W, base.H
        self.n_steps, self.w_near = base.n_steps, float(base.w_near)
        self.sx, self.sy = 2.0 / (s.W - 1), 2.0 / (s.H - 1)
        gx0 = (s.fx * base.a[..., 0] + s.cx) * self.sx - 1.0
        gy0 = (s.fy * base.a[..., 1] + s.cy) * self.sy - 1.0
        gdx = s.fx * base.b[..., 0] * self.sx
        gdy = s.fy * base.b[..., 1] * self.sy
        t = lambda x, dt=dtype: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=self.device)
        self.g0 = t(np.stack([gx0, gy0], -1))                                   # (H, W, 2)
        self.gd = t(np.stack([gdx, gdy], -1))
        self.valid_ray = torch.as_tensor(base.valid_ray, device=self.device)
        dz = np.where(base.valid_ray, base.d[..., 2], 1.0)
        self.p2z, self.dz = t(base.P2[..., 2], torch.float32), t(dz, torch.float32)
        # march schedule: w_k = w_near * (1 - k / n), k = 0..n (same values as numpy)
        k = np.arange(self.n_steps + 1)
        w_steps = (self.w_near * (1.0 - k / self.n_steps)).astype(np.float32)
        self.w_steps = t(w_steps)
        self._range_norm = None
        self._solve = torch.compile(self._solve_eager, dynamic=False) if compile else self._solve_eager

    @classmethod
    def from_config(cls, cfg, device=None, dtype=torch.float32, refine_iters=4, chunk=4,
                    compile=False, **kw):
        return cls(FlatPortRenderer.from_config(cfg, **kw), device, dtype, refine_iters, chunk,
                   compile)

    # ----------------------------------------------------------------------------------
    def inverse_depth(self, depth, depth_type="z"):
        """(B, Hr, Wr) or (Hr, Wr) depth (numpy or tensor) -> (B, 1, Hr, Wr) inverse z-depth
        on the device (0 = no geometry)."""
        z = _upload(depth, self.device)
        if z.ndim == 2:
            z = z[None]
        z = z.to(torch.float32)
        if depth_type == "range":
            if self._range_norm is None:
                s = self.spec
                v, u = torch.meshgrid(torch.arange(s.H, device=self.device, dtype=torch.float32),
                                      torch.arange(s.W, device=self.device, dtype=torch.float32),
                                      indexing="ij")
                self._range_norm = torch.sqrt(((u - s.cx) / s.fx) ** 2 + ((v - s.cy) / s.fy) ** 2 + 1)
            z = z / self._range_norm
        elif depth_type != "z":
            raise ValueError(depth_type)
        ok = torch.isfinite(z) & (z > 0)
        inv = torch.where(ok, 1.0 / torch.where(ok, z, torch.ones_like(z)), torch.zeros_like(z))
        return inv[:, None].to(self.dtype)

    def _grid(self, w):
        """Normalised sample coordinates (..., H, W, 2) for inverse depth w (..., H, W)."""
        return torch.addcmul(self.g0, self.gd, w[..., None])

    def march(self, inv):
        """Fixed-step march. Only the index of the first step with g = w - inv <= 0 is
        tracked (k = n has w = 0, so every ray stops); the bracket values are re-sampled
        afterwards. Returns (w_lo, g_lo, w_hi, g_hi, too_near), each (B, H, W)."""
        B = inv.shape[0]
        H, W, dev = self.H, self.W, self.device
        inv_c = inv.transpose(0, 1)                       # (1, B, Hr, Wr): frames as channels
        n = self.n_steps + 1
        k_hit = torch.full((B, H, W), n - 1, dtype=torch.int64, device=dev)
        # chunks and steps in reverse order: the last write is the first hit (a plain
        # where per step; reductions such as amin over int are very slow on MPS)
        for k0 in reversed(range(0, n - 1, self.chunk)):  # last step (w = 0) always hits
            wk = self.w_steps[k0:min(k0 + self.chunk, n - 1)]
            S = wk.shape[0]
            grid = self._grid(wk[:, None, None]).reshape(1, S * H, W, 2)
            hit = _grid_sample(inv_c, grid).view(B, S, H, W) >= wk.view(1, S, 1, 1)
            for s in reversed(range(S)):
                k_hit = torch.where(hit[:, s], k0 + s, k_hit)
        too_near = k_hit == 0
        w_lo = self.w_steps[k_hit]
        w_hi = self.w_steps[(k_hit - 1).clamp(min=0)]
        g_lo = w_lo - _grid_sample(inv, self._grid(w_lo))[:, 0]
        g_hi = w_hi - _grid_sample(inv, self._grid(w_hi))[:, 0]
        return w_lo, g_lo, w_hi, g_hi, too_near

    def refine(self, inv, w_lo, g_lo, w_hi, g_hi, too_near):
        """Illinois regula falsi on [w_lo, w_hi] with g(w_hi) > 0 >= g(w_lo); fixed number of
        iterations, pixels that converged (numpy's criterion) are frozen."""
        tol = self.w_near * 1e-6
        br = ~too_near & (g_lo < 0)
        a_w, a_g, b_w, b_g = w_hi, g_hi, w_lo, g_lo
        side = torch.zeros_like(w_lo)                                          # +1 / -1 / 0
        w_best = b_w
        act = br
        for _ in range(self.refine_iters):
            den = a_g - b_g
            pd = den > 0
            w_new = torch.where(pd, (b_w * a_g - a_w * b_g) / torch.where(pd, den, torch.ones_like(den)),
                                0.5 * (a_w + b_w))
            g_new = w_new - _grid_sample(inv, self._grid(w_new))[:, 0]
            pos = g_new > 0
            up_a, up_b = act & pos, act & ~pos
            b_g = torch.where(up_a & (side > 0), 0.5 * b_g, b_g)
            a_g = torch.where(up_b & (side < 0), 0.5 * a_g, a_g)
            a_w, a_g = torch.where(up_a, w_new, a_w), torch.where(up_a, g_new, a_g)
            b_w, b_g = torch.where(up_b, w_new, b_w), torch.where(up_b, g_new, b_g)
            side = torch.where(act, torch.where(pos, 1.0, -1.0).to(side.dtype), side)
            w_best = torch.where(act, w_new, w_best)
            act = act & (g_new.abs() > tol) & (a_w - b_w > tol)
        return torch.where(br, w_best, w_lo)

    def depth_outputs(self, w):
        """Camera-frame Z and water path length |X - P2| from the ray parameter w = 1/Z'."""
        bg = w <= 1e-12
        Z = torch.where(bg, torch.full_like(w, float("inf")),
                        1.0 / torch.where(bg, torch.ones_like(w), w) + float(self.spec.centre[2]))
        return Z, (Z - self.p2z) / self.dz

    def _solve_eager(self, inv):
        w_lo, g_lo, w_hi, g_hi, too_near = self.march(inv)
        return self.refine(inv, w_lo, g_lo, w_hi, g_hi, too_near), too_near

    def trace(self, depth, depth_type="z", edges=True, inv=None):
        """Batched trace. depth: (B, Hr, Wr) or (Hr, Wr). Returns a dict of (B, H, W)
        tensors on the device: map_u, map_v (render px), w, Z, water_range, valid,
        background, too_near and (if edges) near_edge; plus 'grid' for sampling."""
        if inv is None:
            inv = self.inverse_depth(depth, depth_type)
        w, too_near = self._solve(inv)
        w = w.float()
        s = self.spec
        grid = self._grid(w.to(self.dtype))
        gf = grid.float()
        map_u = (gf[..., 0] + 1.0) / self.sx
        map_v = (gf[..., 1] + 1.0) / self.sy
        in_img = (map_u >= -0.5) & (map_u <= s.W - 0.5) & (map_v >= -0.5) & (map_v <= s.H - 0.5)
        background = w <= 1e-12
        Z, water_range = self.depth_outputs(w)
        valid = self.valid_ray & in_img & ~too_near
        out = dict(map_u=map_u, map_v=map_v, w=w, Z=Z, water_range=water_range, valid=valid,
                   background=background & valid, too_near=too_near, grid=grid)
        if edges:
            e = depth_edges_torch(inv.float()).to(self.dtype)
            out["near_edge"] = (_grid_sample(e, grid)[:, 0] > 0) & valid
        return out

    def render(self, rgb, depth, depth_type="z", invalid_value=0, edges=True):
        """Batched render. rgb: (B, Hr, Wr, C) or (Hr, Wr, C) numpy/tensor (uint8 or float),
        depth as in `trace`. Returns (image (B, H, W, C) tensor on the device, info dict)."""
        info = self.trace(depth, depth_type, edges)
        x = _upload(rgb, self.device)
        if x.ndim == 3:
            x = x[None]
        out_dtype = x.dtype
        x = x.permute(0, 3, 1, 2).to(self.dtype)                               # (B, C, Hr, Wr)
        img = _grid_sample(x, info["grid"]).permute(0, 2, 3, 1)                # (B, H, W, C)
        if out_dtype == torch.uint8:
            img = img.float().round().clamp(0, 255)
        img = torch.where(info["valid"][..., None], img, torch.full_like(img, invalid_value))
        return img.to(out_dtype), info
