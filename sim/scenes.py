"""Analytic scenes for validating the refraction renderer.

A scene is a list of primitives (planes, axis-aligned boxes, spheres). It can be ray-traced
exactly (a) from arbitrary ray origins/directions, i.e. directly through the flat-port
housing (ground truth), and (b) as a pinhole RGB-D render (what a simulator outputs).
"""
import numpy as np

INF = np.inf


class Plane:
    def __init__(self, point, normal):
        self.p = np.asarray(point, float)
        self.n = np.asarray(normal, float) / np.linalg.norm(normal)

    def intersect(self, O, D):
        den = D @ self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.p - O) @ self.n) / den
        return np.where(np.isfinite(t) & (t > 0), t, INF)


class Box:
    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)

    def intersect(self, O, D):
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / D
            t0, t1 = (self.lo - O) * inv, (self.hi - O) * inv
        tmin = np.nanmax(np.minimum(t0, t1), axis=-1)
        tmax = np.nanmin(np.maximum(t0, t1), axis=-1)
        hit = (tmax >= tmin) & (tmax > 0)
        t = np.where(tmin > 0, tmin, tmax)
        return np.where(hit, t, INF)


class Sphere:
    def __init__(self, centre, radius):
        self.c, self.r = np.asarray(centre, float), float(radius)

    def intersect(self, O, D):
        oc = O - self.c
        b = np.einsum('...i,...i->...', oc, D)
        c = np.einsum('...i,...i->...', oc, oc) - self.r ** 2
        disc = b * b - c
        s = np.sqrt(np.maximum(disc, 0))
        t = np.where(-b - s > 0, -b - s, -b + s)
        return np.where((disc >= 0) & (t > 0), t, INF)


PALETTE = np.array([[0.85, 0.75, 0.55], [0.35, 0.65, 0.85], [0.9, 0.45, 0.35],
                    [0.5, 0.8, 0.45], [0.75, 0.5, 0.85], [0.95, 0.85, 0.3]])


class Scene:
    def __init__(self, name, prims, background=(0.05, 0.25, 0.35), checker=0.047):
        self.name, self.prims = name, prims
        self.background = np.asarray(background, float)
        self.checker = checker

    def trace(self, O, D):
        """Nearest hit: (t, X, prim_id); t = inf / id = -1 where nothing is hit."""
        O = np.broadcast_to(O, D.shape)
        ts = np.stack([p.intersect(O, D) for p in self.prims], axis=0)
        idx = np.argmin(ts, axis=0)
        t = np.take_along_axis(ts, idx[None], 0)[0]
        idx = np.where(np.isfinite(t), idx, -1)
        X = O + np.where(np.isfinite(t), t, 0)[..., None] * D
        X = np.where(np.isfinite(t)[..., None], X, np.nan)
        return t, X, idx

    def shade(self, X, idx):
        """Procedural texture: 3D checker + sinusoid modulation, tinted per primitive."""
        s = self.checker
        Xs = np.nan_to_num(X)
        chk = (np.floor(Xs[..., 0] / s) + np.floor(Xs[..., 1] / s) + np.floor(Xs[..., 2] / s)) % 2
        mod = 0.75 + 0.25 * np.sin(2 * np.pi * Xs[..., 0] / (3.1 * s)) * np.cos(2 * np.pi * Xs[..., 1] / (2.3 * s))
        col = PALETTE[np.maximum(idx, 0) % len(PALETTE)] * (0.35 + 0.65 * chk[..., None]) * mod[..., None]
        return np.where((idx >= 0)[..., None], col, self.background).astype(np.float32)

    def render_pinhole(self, spec):
        """Pinhole RGB-D render with the camera of `spec` (core.render.PinholeSpec).

        Returns rgb (H, W, 3) float32, z-depth (H, W) in the render frame (inf = miss) and
        the world point per pixel (H, W, 3)."""
        u, v = np.meshgrid(np.arange(spec.W, dtype=float), np.arange(spec.H, dtype=float))
        D = np.stack([(u - spec.cx) / spec.fx, (v - spec.cy) / spec.fy, np.ones_like(u)], -1)
        D /= np.linalg.norm(D, axis=-1, keepdims=True)
        t, X, idx = self.trace(spec.centre, D)
        z = np.where(np.isfinite(t), X[..., 2] - spec.centre[2], np.inf)
        return self.shade(X, idx), z, X


def standard_scenes():
    """Scenes used in docs/explore_sim.md."""
    scenes = {}
    for Z in (0.3, 1.0, 3.0, 10.0):
        scenes[f"plane_{Z:g}m"] = Scene(f"fronto-parallel plane Z={Z:g} m",
                                        [Plane([0, 0, Z], [0, 0, -1])])
    th = np.radians(50)
    scenes["tilted_plane"] = Scene("plane tilted 50 deg about y, through (0, 0, 1.2)",
                                   [Plane([0, 0, 1.2], [np.sin(th), 0, -np.cos(th)])])
    scenes["steps"] = Scene("occlusion steps: boxes at 0.35 / 0.6 / 1.0 m before a wall at 2.5 m", [
        Plane([0, 0, 2.5], [0, 0, -1]),
        Box([-0.45, -0.30, 1.0], [0.05, 0.10, 1.3]),
        Box([0.02, -0.08, 0.6], [0.30, 0.35, 0.7]),
        Box([-0.12, -0.25, 0.35], [-0.04, 0.05, 0.40]),
    ])
    scenes["spheres"] = Scene("spheres before a seabed plane", [
        Plane([0, 0.6, 0], [0, -1, -0.35]),
        Sphere([0.15, 0.05, 0.55], 0.12),
        Sphere([-0.45, -0.15, 1.3], 0.35),
        Sphere([0.6, -0.3, 2.2], 0.4),
        Sphere([-0.05, -0.12, 0.32], 0.05),
    ])
    return scenes
