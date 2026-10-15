"""Per-parameter sensitivity of the corrected image, split into pinhole-absorbable and
non-absorbable parts (docs/explore_uncertainty.md, section 2).

    python uncertainty/sensitivity.py [configs/dataset.yaml] [--stride 4]

For each parameter (1-sigma prior), depth Z and projection centre (camera centre, virtual
centre), the displacement field dq(q) = J(q) * sigma over the corrected image is computed
at the scene depth Z (the correction itself made with the right depth), and fitted with
the absorbable fields:
  P4    output fx, fy, cx, cy (COLMAP/SLAM intrinsics refinement),
  P4R3  + a constant rotation (absorbed by the poses).
"3D" fits one set of coefficients (P4 + R3 + centre translation T3) jointly over all depths,
i.e. a scene spanning 0.3-5 m seen by one camera.

Depth terms: "z global" is a 25 % (1 sigma, log) error of the depth shared by all pixels
(fixed-depth mode: z0 vs the true scene distance); "z local" a 10 % per-pixel error (scene
relief around z0, or depth-map noise), which no camera parameter can absorb.

Writes docs/figures/sensitivity.json and the figures sens_*.png.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from core.config import load_config  # noqa: E402
from core import uncertainty as U  # noqa: E402

FIG = os.path.join(REPO, "docs", "figures")
DEPTHS = (0.3, 0.5, 1.0, 2.0, 5.0)


def fields_at(geo, canvas, Z, stride):
    xs, ys = U.source_pixels(geo, canvas, Z)
    xs, ys = xs[::stride, ::stride], ys[::stride, ::stride]
    qx, qy = geo.out_grid(stride)
    ok = np.isfinite(xs) & np.isfinite(ys)
    ur, vr = U.raw_from_ideal(xs[ok], ys[ok], geo)
    J, q = U.jacobian_torch(geo, ur, vr, np.full(ur.shape, 1.0 / Z))
    assert np.abs(q - np.stack([qx[ok], qy[ok]], -1)).max() < 1e-6
    return J, qx[ok], qy[ok], ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=os.path.join(REPO, "configs/dataset.yaml"))
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    prior = U.default_prior(cfg)
    sig = prior.vector()
    c_vc = U.optimal_center_z(cfg)
    print(f"virtual centre: {c_vc * 1e3:.2f} mm")
    names = list(U.PARAMS) + ["z_global", "z_local"]
    out = dict(center_vc=c_vc, prior={k: float(s) for k, s in zip(U.PARAMS, sig)},
               depth_global=prior.depth_global, depth_local=prior.depth_local, rows=[])
    keep = {}
    for centre, cz in (("cc", 0.0), ("vc", c_vc)):
        geo, canvas = U.setup(cfg, center_z=cz)
        stack = []
        for Z in DEPTHS:
            J, qx, qy, ok = fields_at(geo, canvas, Z, args.stride)
            w = 1.0 / Z
            # 1-sigma fields (N, 2, M)
            F = np.concatenate([J[:, :, :-1] * sig,
                                J[:, :, -1:] * (w * prior.depth_global),
                                J[:, :, -1:] * (w * prior.depth_local)], -1)
            B4 = U.basis(qx, qy, geo, ("P4",))
            B7 = U.basis(qx, qy, geo, ("P4", "R3"))
            r4, b4 = U.absorb(F, B4)
            r7, _ = U.absorb(F, B7)
            r4[:, :, -1] = F[:, :, -1]  # local depth error: not absorbable
            r7[:, :, -1] = F[:, :, -1]
            stack.append((F, qx, qy, w))
            for i, k in enumerate(names):
                full = U.field_stats(F[:, :, i])
                np4 = U.field_stats(r4[:, :, i])
                np7 = U.field_stats(r7[:, :, i])
                out["rows"].append(dict(
                    param=k, centre=centre, Z=Z, full_rms=full[0], full_max=full[1],
                    np_rms=np4[0], np_max=np4[1], np_rot_rms=np7[0], np_rot_max=np7[1],
                    dfx=float(b4[0, i] * geo.f_out[0]), dfy=float(b4[1, i] * geo.f_out[1]),
                    dcx=float(b4[2, i]), dcy=float(b4[3, i])))
            if Z == 1.0:
                keep[centre] = dict(F=F, r4=r4, qx=qx, qy=qy, ok=ok, geo=geo)
            print(centre, Z, "done", flush=True)
        # joint 3-D fit over all depths
        Fj = np.concatenate([s[0] for s in stack])
        Bj = np.concatenate([U.basis(s[1], s[2], geo, ("P4", "R3", "T3"), w=s[3])
                             for s in stack])
        rj, _ = U.absorb(Fj, Bj)
        rj[:, :, -1] = Fj[:, :, -1]
        for i, k in enumerate(names):
            r = U.field_stats(rj[:, :, i])
            out["rows"].append(dict(param=k, centre=centre, Z="3D", np_rms=r[0], np_max=r[1],
                                    full_rms=U.field_stats(Fj[:, :, i])[0],
                                    full_max=U.field_stats(Fj[:, :, i])[1]))
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, "sensitivity.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(REPO, "uncertainty", ".sens_1m.npz"),
                        **{f"{c}_{n}": v for c, d in keep.items() for n, v in d.items()
                           if n != "geo"},
                        shape=np.array(keep["cc"]["ok"].shape))
    print_tables(out)


def print_tables(out):
    rows = out["rows"]
    names = list(dict.fromkeys(r["param"] for r in rows))
    get = lambda p, c, Z: next(r for r in rows if r["param"] == p and r["centre"] == c
                               and r["Z"] == Z)
    print("\n1 m, camera centre: full RMS/max | np(P4) RMS/max | np(P4+R3) RMS | absorbed dfx dcx")
    for p in names:
        r = get(p, "cc", 1.0)
        print(f"{p:9s} {r['full_rms']:7.3f} {r['full_max']:7.3f} | {r['np_rms']:7.3f} "
              f"{r['np_max']:7.3f} | {r['np_rot_rms']:7.3f} | {r['dfx']:7.2f} {r['dcx']:7.2f}")
    print("\nnp(P4) RMS vs depth, cc / vc; 3D joint")
    for p in names:
        s = " ".join(f"{get(p, 'cc', Z)['np_rms']:6.3f}/{get(p, 'vc', Z)['np_rms']:6.3f}"
                     for Z in DEPTHS)
        print(f"{p:9s} {s} | 3D {get(p, 'cc', '3D')['np_rms']:6.3f}")


if __name__ == "__main__":
    main()
