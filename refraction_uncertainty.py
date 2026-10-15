"""Per-pixel covariance of refraction-corrected images (for weighted SfM / SLAM).

Usage:
    python refraction_uncertainty.py configs/dataset.yaml [--cov-dir DIR] [--device mps]
                                     [--per-image] [--set d0=0.001 tilt_deg=1 ...]

Output (interchange format): one file per image, <cov_dir>/<image_stem>.npy, float32
(H, W, 3) = (sxx, sxy, syy) in px^2 on the corrected-image grid that remove_refraction.py
writes (same canvas and crop), NaN where the pixel has no source. In fixed-depth mode
(paths.depth_dir null) all images share one map, <cov_dir>/shared.npy; --per-image also
writes it under every image stem (hard links).

Two variants are written:
    <cov_dir>/        non-absorbable covariance (recommended for BA with refined
                      intrinsics): the part of the refraction error that no change of
                      fx, fy, cx, cy (and, for the per-frame depth scale, no camera
                      translation) can explain;
    <cov_dir>_full/   full covariance (BA with fixed intrinsics, or for gating).
Neither contains keypoint detection noise: add sigma_kp^2 I in the consumer.

Priors: core.uncertainty.default_prior, overridden by the config's `uncertainty:` section
and by --set (d0, tglass in m; mu_g, mu_w; tilt_deg; fx, fy, cx, cy in px; k1..p2;
depth_global = log-sigma of the per-frame depth / z0; depth_local = relative per-pixel
depth error). Defaults for the depth terms: fixed-depth mode 0.25 / 0.10 (z0 known to
~25 %, 10 % relief), depth-map mode 0.10 / 0.03.
"""
import argparse
import glob
import os
import time

import numpy as np

from core import uncertainty as U
from core.config import load_config, camera_params, distortion_params, housing_params
from core.flatport import InvDepth
from core.rectangle import largest_valid_rectangle

IMAGE_EXTS = ("jpg", "JPG", "jpeg", "JPEG", "png", "PNG", "bmp", "BMP", "tiff", "TIFF")


def parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = float(v)
    return out


def default_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CovarianceMaps:
    """Shared set-up (zoom, canvas, reference terms) + covariance maps for a depth."""

    def __init__(self, cfg, prior, device="cpu", theta_sets=("P4",)):
        import torch
        self.cfg, self.prior, self.device = cfg, prior, device
        self.dtype = torch.float32 if device != "cpu" else torch.float64
        self.geo, self.canvas = U.setup(cfg)
        W, H, fx, fy, cx, cy = camera_params(cfg)
        self.cam = (W, H, fx, fy, cx, cy)
        xs, ys = U.source_pixels(self.geo, self.canvas, self.geo.z_ref)
        ok, ur, vr = self._valid(xs, ys)
        qx, qy = self.geo.out_grid(1)
        self.ref_ok = ok
        self.row = np.full(ok.shape, -1, np.int64)
        self.row[ok] = np.arange(ok.sum())
        self.fc = U.FastCov(self.geo, ur, vr, qx[ok], qy[ok], prior, theta_sets,
                            device=device, dtype=self.dtype)
        self._torch_inv = None

    def _valid(self, xs, ys):
        """remove_refraction's mask: a finite source whose raw (distorted) sample lies in
        the frame. Returns (mask, raw u, raw v) of the valid pixels."""
        W, H = self.cam[:2]
        ok = np.isfinite(xs) & np.isfinite(ys)
        u, v = U.raw_from_ideal(np.where(ok, xs, 0), np.where(ok, ys, 0), self.geo)
        u32, v32 = u.astype(np.float32), v.astype(np.float32)
        ok &= (u32 >= 0) & (u32 <= W - 1) & (v32 >= 0) & (v32 <= H - 1)
        return ok, u[ok], v[ok]

    def source(self, depth):
        """Ideal source pixels of the full canvas for a depth (scalar or source-grid map)."""
        if np.ndim(depth) == 0 or self.device == "cpu":
            return U.source_pixels(self.geo, self.canvas, depth)
        import torch
        from core.torch_remove import TorchInverse
        if self._torch_inv is None:
            W, H, fx, fy, cx, cy = self.cam
            self._torch_inv = TorchInverse(W, H, fx, fy, cx, cy, self.geo.housing,
                                           self.geo.zoom, self.canvas, None,
                                           device=self.device, z_ref=self.geo.z_ref)
        d = torch.as_tensor(np.asarray(depth, np.float32), device=self.device)[None]
        mx, my = self._torch_inv.remap_maps(d)
        mx, my = mx[0].cpu().numpy().astype(np.float64), my[0].cpu().numpy().astype(np.float64)
        bad = mx < 0
        mx[bad] = np.nan
        my[bad] = np.nan
        return mx, my

    def maps(self, depth):
        """(full, np, valid): float32 (H_out, W_out, 3) on the uncropped canvas, NaN where
        invalid. Pixels valid in this frame but not at z0 are computed directly."""
        import torch
        W, H, fx, fy, cx, cy = self.cam
        if np.ndim(depth) == 0 and float(depth) == self.geo.z_ref:
            ok = self.ref_ok
            w = np.full(ok.sum(), 1.0 / float(depth))
            full, npc = self.fc(w)
        else:
            xs, ys = self.source(depth)
            ok, _, _ = self._valid(xs, ys)
            if np.ndim(depth) == 0:
                w = np.full(ok.sum(), 1.0 / float(depth))
            else:
                w = InvDepth(depth, fx, fy, cx, cy).sample((xs[ok] - cx) / fx,
                                                           (ys[ok] - cy) / fy)
            rows = self.row[ok]
            have = rows >= 0
            fc = self.fc
            sel = torch.as_tensor(rows[have], device=fc.device)
            terms = ([m[sel] for m in fc.M], [m[sel] for m in fc.N], fc.Jw[sel])
            qx, qy = self.geo.out_grid(1)
            qx, qy = qx[ok], qy[ok]
            if not have.all():  # reference-invalid pixels: their own terms
                miss = ~have
                _, ur, vr = self._valid(xs, ys)
                ex = fc.terms(ur[miss], vr[miss], qx[miss], qy[miss])
                order = np.argsort(np.concatenate([np.flatnonzero(have),
                                                   np.flatnonzero(miss)]))
                o = torch.as_tensor(order, device=fc.device)
                cat = lambda x, y: torch.cat([x, y])[o]
                terms = ([cat(x, y) for x, y in zip(terms[0], ex[0])],
                         [cat(x, y) for x, y in zip(terms[1], ex[1])], cat(terms[2], ex[2]))
            full, npc = fc(w, terms=terms, qxy=(fc._t(qx), fc._t(qy)))
        out = []
        for C in (full, npc):
            M = np.full(ok.shape + (3,), np.nan, np.float32)
            M[ok] = C.cpu().numpy()
            out.append(M)
        return out[0], out[1], ok


def crop(M, box):
    if box is None:
        return M
    y0, y1, x0, x1 = box
    return np.ascontiguousarray(M[y0:y1, x0:x1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--cov-dir", default=None,
                    help="default: paths.cov_dir, else <output_dir>/../cov")
    ap.add_argument("--device", default=None, help="cpu (float64), mps, cuda (float32)")
    ap.add_argument("--per-image", action="store_true",
                    help="fixed-depth mode: also write <stem>.npy for every image")
    ap.add_argument("--absorb", default="P4", choices=("P4", "P4R3"),
                    help="absorbable set for the shared parameters (P4R3: + constant rotation)")
    ap.add_argument("--set", nargs="*", help="prior overrides, e.g. d0=0.001 tilt_deg=1")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths, corr = cfg["paths"], cfg.get("correction", {})
    depth_dir = paths.get("depth_dir")
    ov = {"depth_global": 0.25, "depth_local": 0.10} if depth_dir is None else \
        {"depth_global": 0.10, "depth_local": 0.03}
    ov.update(cfg.get("uncertainty", {}) or {})
    ov.update(parse_set(args.set))
    prior = U.default_prior(None, ov)
    cov_dir = args.cov_dir or paths.get("cov_dir") or os.path.join(
        os.path.dirname(os.path.normpath(paths["output_dir"])), "cov")
    full_dir = cov_dir.rstrip("/") + "_full"
    os.makedirs(cov_dir, exist_ok=True)
    os.makedirs(full_dir, exist_ok=True)
    device = args.device or default_device()
    sets = ("P4", "R3") if args.absorb == "P4R3" else ("P4",)

    t0 = time.time()
    cm = CovarianceMaps(cfg, prior, device=device, theta_sets=sets)
    W, H = cm.cam[:2]
    print(f"Set-up {time.time() - t0:.2f} s: zoom {cm.geo.zoom:.4f}, canvas "
          f"{cm.canvas[0]}x{cm.canvas[1]}, device {device}")
    print("Prior (1 sigma): " + ", ".join(f"{k}={v:.3g}" for k, v in prior.sigma.items())
          + f", depth_global={prior.depth_global}, depth_local={prior.depth_local}")
    crop_box = corr.get("crop_valid_bbox", False)

    stems = sorted({os.path.splitext(os.path.basename(p))[0]
                    for e in IMAGE_EXTS for p in glob.glob(f"{paths['rgb_dir']}/*.{e}")})
    stems = stems[::corr.get("step_size", 1)]

    def mask_box(ok):
        if not crop_box:
            return None
        return largest_valid_rectangle(np.where(ok, 255, 0).astype(np.uint8))

    if depth_dir is None:
        t1 = time.time()
        full, npc, ok = cm.maps(cm.geo.z_ref)
        box = mask_box(ok)
        full, npc = crop(full, box), crop(npc, box)
        np.save(os.path.join(cov_dir, "shared.npy"), npc)
        np.save(os.path.join(full_dir, "shared.npy"), full)
        s = np.sqrt(npc[..., 0] + npc[..., 2])
        sf = np.sqrt(full[..., 0] + full[..., 2])
        print(f"Shared map {npc.shape[1]}x{npc.shape[0]} in {time.time() - t1:.2f} s: "
              f"sqrt(trace) non-absorbable median {np.nanmedian(s):.2f} px "
              f"(max {np.nanmax(s):.2f}), full median {np.nanmedian(sf):.2f} px "
              f"(max {np.nanmax(sf):.2f})")
        if args.per_image:
            for d in (cov_dir, full_dir):
                src = os.path.join(d, "shared.npy")
                for st in stems:
                    dst = os.path.join(d, f"{st}.npy")
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.link(src, dst)
            print(f"Linked {len(stems)} per-image files")
    else:
        from remove_refraction import load_depth
        for st in stems:
            out = os.path.join(cov_dir, f"{st}.npy")
            if os.path.exists(out) and not args.overwrite:
                print(f"Exists: {st}.npy -> skip")
                continue
            depth = load_depth(st, depth_dir, W, H)
            if depth is None:
                print(f"No depth: {st} -> skip")
                continue
            t1 = time.time()
            full, npc, ok = cm.maps(depth)
            box = mask_box(ok)
            np.save(out, crop(npc, box))
            np.save(os.path.join(full_dir, f"{st}.npy"), crop(full, box))
            print(f"Saved: {st}.npy ({time.time() - t1:.2f} s)")
    print(f"Done in {time.time() - t0:.1f} s -> {cov_dir} (non-absorbable), {full_dir} (full)")


if __name__ == "__main__":
    main()
