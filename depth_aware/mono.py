"""Monocular relative depth (Depth Anything V2) and its robust alignment to sparse SfM depth.

Depth Anything V2 (relative) predicts affine-invariant *inverse* depth (disparity), so
the alignment is done in inverse depth: 1 / Z ~ a * disp + b, the model's native
space (fitting a disparity model in depth was ~2x worse in ORCA_splat's tests).
"""
import os

import numpy as np

MODEL_ID = os.environ.get("REFRAX_MONO_MODEL", "depth-anything/Depth-Anything-V2-Large-hf")


class MonoDepth:
    def __init__(self, model_id=MODEL_ID, device=None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        torch.set_num_threads(4)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
        self.torch, self.device = torch, device
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    def __call__(self, rgb):
        """Relative disparity (H, W) float32 for an RGB uint8 image."""
        torch = self.torch
        inputs = self.proc(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pred = self.model(**inputs).predicted_depth  # (1, h, w)
        pred = torch.nn.functional.interpolate(pred[:, None], size=rgb.shape[:2],
                                               mode="bicubic", align_corners=False)
        return pred[0, 0].float().cpu().numpy()


# ------------------------------------------------------------------ alignment
def _design(disp, xy, size, kind):
    W, H = size
    if kind == "scale":
        return disp[:, None]
    if kind == "affine":
        return np.stack([disp, np.ones_like(disp)], 1)
    if kind == "spatial":
        # gain varying bilinearly over the image, plus a constant shift
        x = xy[:, 0] / W - 0.5
        y = xy[:, 1] / H - 0.5
        return np.stack([disp, disp * x, disp * y, np.ones_like(disp)], 1)
    raise ValueError(kind)


def robust_fit(A, y, iters=10, c=1.5):
    """Huber IRLS on residuals normalised by their MAD, then one 3-sigma trim."""
    w = np.ones(len(y))
    for _ in range(iters):
        sw = np.sqrt(w)
        p, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        r = y - A @ p
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        u = np.abs(r) / (c * s)
        w = np.where(u <= 1, 1.0, 1.0 / u)
    keep = np.abs(r) < 3 * s
    p, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
    return p, keep


def fit_inverse_depth(disp_s, z_s, xy, size, kind="affine"):
    """Fit 1/z ~ A(disp) p; returns (p, inlier mask)."""
    A = _design(disp_s, xy, size, kind)
    return robust_fit(A, 1.0 / z_s)


def predict_inverse(p, disp, xy, size, kind):
    return _design(disp, xy, size, kind) @ p


def holdout_error(disp_s, z_s, xy, size, kind, k=5, seed=0):
    """Median |z_pred - z| / z on held-out sparse points (k-fold, untrimmed)."""
    idx = np.random.default_rng(seed).permutation(len(z_s))
    errs = []
    for f in range(k):
        te = idx[f::k]
        tr = np.setdiff1d(idx, te)
        p, _ = fit_inverse_depth(disp_s[tr], z_s[tr], xy[tr], size, kind)
        inv = predict_inverse(p, disp_s[te], xy[te], size, kind)
        zp = np.where(inv > 0, 1.0 / np.maximum(inv, 1e-9), np.inf)
        errs.append(np.abs(zp - z_s[te]) / z_s[te])
    return float(np.median(np.concatenate(errs)))


def dense_from_fit(p, disp, size, kind, z_s):
    """Dense depth (same unit as z_s) from the fitted model, clamped to a sane range."""
    W, H = size
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    xy = np.stack([X.ravel(), Y.ravel()], 1)
    inv = predict_inverse(p, disp.ravel(), xy, size, kind).reshape(H, W)
    lo, hi = 0.5 * np.percentile(z_s, 2), 2.0 * np.percentile(z_s, 98)
    z = np.where(inv > 1.0 / hi, 1.0 / np.maximum(inv, 1e-12), hi)
    return np.clip(z, lo, hi).astype(np.float32)
