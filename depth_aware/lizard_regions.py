"""Region-wise evaluation of Lizard Island reconstructions (docs/explore_lizard.md, section 9).

    python depth_aware/lizard_regions.py --work W --ref <model_txt> name=<model_txt> ... \
        [--json out.json] [--fig seg.png]

The subset is split into a shallow and a deep region by the per-image median depth of the
sparse points of the reference model (metres via GPS scale):
  * median filter over 5 images, then shallow if < --t-shallow (2.2 m), deep if
    > --t-deep (3.5 m), otherwise transition;
  * contiguity: a run of one label (consecutive images) shorter than --min-run images is
    relabelled transition, and --pad images on both sides of every transition are too.
Legs come from the GPS track: along-track coordinate s on the principal axis, legs are the
runs of constant direction of travel (turns excluded).

Per model, after ONE global similarity (Umeyama) alignment of all camera centres to GPS
(east, north, 0) (the ASV keeps the camera at constant height, so the aligned height h is
the error):
  h_mean, v_rms      mean and RMS of h over the region's images (m)
  step               h_mean(deep) - h_mean(shallow) (m); GPS: 0
  step_local         the same at each shallow/deep boundary within a leg: lines fitted to h(s)
                     on both sides (within 6 m of the transition), evaluated at its centre;
                     mean over boundaries (m)
  sag20              per region run (one leg, >= 8 m): parabola h = a + b s + c s^2,
                     sagitta c * 10^2 over a 20 m chord (mm; + = bowl), image-weighted mean
  curv               the same as curvature 2c (1/km)
  sand_curv, sand_rms  sparse points (track >= 3) seen by the deep region's images, lowest
                     layer (height within 0.6 m above the 10th percentile of the patch,
                     2-sigma clipped): quadric fit z(s, t) in leg coordinates; along-track
                     curvature (1/km) and RMS about the fitted plane / quadric (mm)
  reproj             mean reprojection error of the observations in the region's images (px)
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colmap_utils as cu  # noqa: E402
from lizard_eval import umeyama  # noqa: E402


def load_gps(path):
    out = {}
    for line in open(path):
        n, e, no = line.strip().split(",")
        out[n] = np.array([float(e), float(no), 0.0])
    return out


def read_cameras(path):
    cams = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        e = line.split()
        cams[int(e[0])] = (e[1], np.array(list(map(float, e[4:]))))
    return cams


def project(model, params, Xc):
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    if model == "PINHOLE":
        fx, fy, cx, cy = params
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2
        x, y = (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    else:
        raise ValueError(model)
    return np.stack([fx * x + cx, fy * y + cy], 1)


def legs_from_gps(names, gps):
    """leg id per image (-1 = turn) and along-track coordinate s (m)."""
    xy = np.array([gps[n][:2] for n in names])
    mu = xy.mean(0)
    u = np.linalg.svd(xy - mu, full_matrices=False)[2][0]
    s = (xy - mu) @ u
    v = np.gradient(uniform_filter1d(s, 9, mode="nearest"))
    sp = uniform_filter1d(np.linalg.norm(np.gradient(uniform_filter1d(xy, 9, axis=0,
                                                                       mode="nearest"), axis=0), axis=1),
                          9, mode="nearest")
    moving = np.abs(v) > 0.6 * np.median(sp)   # travelling along the leg axis
    sign = np.where(moving, np.sign(v), 0)
    leg = -np.ones(len(names), int)
    k = 0
    i = 0
    while i < len(names):
        j = i
        while j < len(names) and sign[j] == sign[i]:
            j += 1
        if sign[i] != 0 and j - i >= 30:
            leg[i:j] = k
            k += 1
        i = j
    return leg, s


def segment(depth, t_sh=2.2, t_dp=3.5, min_run=10, pad=2):
    d = median_filter(depth, 5, mode="nearest")
    lab = np.zeros(len(d), int)
    lab[d < t_sh] = 1
    lab[d > t_dp] = 2
    # contiguity
    i = 0
    while i < len(lab):
        j = i
        while j < len(lab) and lab[j] == lab[i]:
            j += 1
        if lab[i] and j - i < min_run:
            lab[i:j] = 0
        i = j
    from scipy.ndimage import binary_dilation
    lab[binary_dilation(lab == 0, iterations=pad)] = 0
    return lab, d


def per_image_depth(model_txt, gps):
    imgs = cu.read_images(f"{model_txt}/images.txt")
    pts = cu.read_points3d(f"{model_txt}/points3D.txt")
    names = sorted(n for n in imgs if n in gps)
    C = np.array([-imgs[n]["R"].T @ imgs[n]["t"] for n in names])
    s = umeyama(C, np.array([gps[n] for n in names]))[0]
    out = {}
    for n in names:
        d = imgs[n]
        ids = d["p3d"][d["p3d"] >= 0]
        P = np.array([pts[i][0] for i in ids if i in pts])
        if len(P) > 20:
            out[n] = float(np.median(s * (P @ d["R"].T + d["t"])[:, 2]))
    return out


def runs_of(lab, leg):
    """(label, leg, idx array) for maximal runs of a region label within a leg."""
    out = []
    i = 0
    while i < len(lab):
        j = i
        while j < len(lab) and lab[j] == lab[i] and leg[j] == leg[i]:
            j += 1
        if lab[i] > 0 and leg[i] >= 0:
            out.append((int(lab[i]), int(leg[i]), np.arange(i, j)))
        i = j
    return out


def evaluate(model_txt, names, lab, leg, s_al, gps, axis_u):
    imgs = cu.read_images(f"{model_txt}/images.txt")
    pts = cu.read_points3d(f"{model_txt}/points3D.txt")
    cams = read_cameras(f"{model_txt}/cameras.txt")
    idx = {n: i for i, n in enumerate(names)}
    reg = [n for n in names if n in imgs]
    C = np.array([-imgs[n]["R"].T @ imgs[n]["t"] for n in reg])
    s, R, t = umeyama(C, np.array([gps[n] for n in reg]))
    h = np.full(len(names), np.nan)
    for n, c in zip(reg, s * C @ R.T + t):
        h[idx[n]] = c[2]
    out = dict(scale=float(s))
    # per region height stats, reprojection
    for r, rn in ((1, "shallow"), (2, "deep")):
        m = (lab == r) & np.isfinite(h)
        if m.sum() < 10:
            out[rn] = None
            continue
        errs = []
        for n in reg:
            if lab[idx[n]] != r:
                continue
            d = imgs[n]
            sel = d["p3d"] >= 0
            ids = d["p3d"][sel]
            ok = np.array([i in pts for i in ids])
            X = np.array([pts[i][0] for i in ids[ok]])
            Xc = X @ d["R"].T + d["t"]
            mdl, prm = cams[d["cam"]]
            e = np.linalg.norm(project(mdl, prm, Xc) - d["xy"][sel][ok], axis=1)
            errs.append(e)
        errs = np.concatenate(errs)
        out[rn] = dict(n=int(m.sum()), h_mean=float(np.mean(h[m])),
                       v_rms=float(np.sqrt(np.mean(h[m] ** 2))), h_std=float(np.std(h[m])),
                       reproj=float(errs.mean()))
        # horizontal scale of this region alone vs GPS (relative to the global scale)
        rn_ = [n for n in reg if lab[idx[n]] == r]
        Cr = np.array([-imgs[n]["R"].T @ imgs[n]["t"] for n in rn_])
        out[rn]["scale_rel"] = float(umeyama(Cr, np.array([gps[n] for n in rn_]))[0] / s)
    both = out["shallow"] is not None and out["deep"] is not None
    out["step"] = out["deep"]["h_mean"] - out["shallow"]["h_mean"] if both else None
    out["v_rms"] = float(np.sqrt(np.nanmean(h ** 2)))
    # within-region bending
    rr = runs_of(lab, leg)
    for r, rn in ((1, "shallow"), (2, "deep")):
        if out[rn] is None:
            continue
        cs, ws = [], []
        for lr, lg, ii in rr:
            ii = ii[np.isfinite(h[ii])]
            if lr != r or len(ii) < 10 or np.ptp(s_al[ii]) < 8:
                continue
            c2 = np.polyfit(s_al[ii], h[ii], 2)[0]
            cs.append(c2)
            ws.append(len(ii))
        cs, ws = np.array(cs), np.array(ws)
        out[rn]["sag20_mm"] = float(1000 * np.sum(cs * ws) / ws.sum() * 100) if len(cs) else None
        out[rn]["curv_per_km"] = float(1000 * 2 * np.sum(cs * ws) / ws.sum()) if len(cs) else None
        out[rn]["sag20_runs_mm"] = [round(1000 * c * 100) for c in cs]
    # local steps at boundaries
    steps = []
    for (l1, g1, i1), (l2, g2, i2) in zip(rr[:-1], rr[1:]):
        if g1 != g2 or l1 == l2 or i2[0] - i1[-1] > 40:
            continue
        sc = 0.5 * (s_al[i1[-1]] + s_al[i2[0]])
        vals = {}
        for l_, ii in ((l1, i1), (l2, i2)):
            ii = ii[(np.abs(s_al[ii] - sc) < 6 + abs(s_al[i1[-1]] - sc)) & np.isfinite(h[ii])]
            if len(ii) < 5:
                break
            vals[l_] = np.polyval(np.polyfit(s_al[ii] - sc, h[ii], 1), 0.0)
        if len(vals) == 2:
            steps.append(vals[2] - vals[1])
    out["step_local"] = float(np.mean(steps)) if steps else None
    out["step_local_each"] = [round(x, 3) for x in steps]
    # sand flatness (deep region points)
    deep_ids = set()
    for n in reg:
        if lab[idx[n]] == 2:
            p = imgs[n]["p3d"]
            deep_ids.update(p[p >= 0].tolist())
    X = np.array([pts[i][0] for i in deep_ids if i in pts and pts[i][2] >= 3])
    if len(X) > 500:
        Xa = s * X @ R.T + t
        a1 = np.array([axis_u[0], axis_u[1], 0.0])
        a2 = np.array([-axis_u[1], axis_u[0], 0.0])
        sa, ta, za = Xa @ a1, Xa @ a2, Xa[:, 2]
        keep = za < np.percentile(za, 10) + 0.6
        for _ in range(5):
            A = np.stack([np.ones_like(sa), sa, ta, sa * sa, sa * ta, ta * ta], 1)
            c = np.linalg.lstsq(A[keep], za[keep], rcond=None)[0]
            r_ = za - A @ c
            sd = np.std(r_[keep])
            keep = keep & (np.abs(r_) < 2 * sd)
        A3 = A[keep][:, :3]
        rp = za[keep] - A3 @ np.linalg.lstsq(A3, za[keep], rcond=None)[0]
        out["sand"] = dict(n=int(keep.sum()), curv_along_per_km=float(1000 * 2 * c[3]),
                           curv_across_per_km=float(1000 * 2 * c[5]),
                           rms_plane_mm=float(1000 * np.sqrt(np.mean(rp ** 2))),
                           rms_quadric_mm=float(1000 * np.sqrt(np.mean(r_[keep] ** 2))),
                           depth_below_cam=float(-np.median(za[keep])))
    return out, h


def summary(r):
    f = lambda x, fmt: "  -  " if x is None else format(x, fmt)  # noqa: E731
    out = f"step {f(r['step'], '+.3f')} local {f(r['step_local'], '+.3f')}"
    for rn in ("shallow", "deep"):
        q = r[rn]
        if q is None:
            continue
        out += (f" | {rn[:2]}: v {q['v_rms']:.3f} sag20 {f(q['sag20_mm'], '+5.0f')} "
                f"rp {q['reproj']:.4f} sc {q['scale_rel']:.3f}")
    sd = r.get("sand")
    if sd:
        out += (f" | sand curv {sd['curv_along_per_km']:+.1f}/km rms {sd['rms_plane_mm']:.0f}/"
                f"{sd['rms_quadric_mm']:.0f} mm z {sd['depth_below_cam']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="+", help="name=<model_txt>")
    ap.add_argument("--work", default="/Users/fischert/robot/refrax-work/lizard")
    ap.add_argument("--ref", required=True, help="model whose per-image depth segments")
    ap.add_argument("--t-shallow", type=float, default=2.2)
    ap.add_argument("--t-deep", type=float, default=3.5)
    ap.add_argument("--min-run", type=int, default=10)
    ap.add_argument("--json", default=None)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()
    gps = load_gps(f"{args.work}/gps.csv")
    names = sorted(gps)
    dep = per_image_depth(args.ref, gps)
    d = np.array([dep.get(n, np.nan) for n in names])
    d = np.where(np.isfinite(d), d, np.interp(np.arange(len(d)), np.flatnonzero(np.isfinite(d)),
                                              d[np.isfinite(d)]))
    lab, dsm = segment(d, args.t_shallow, args.t_deep, args.min_run)
    leg, s_al = legs_from_gps(names, gps)
    xy = np.array([gps[n][:2] for n in names])
    axis_u = np.linalg.svd(xy - xy.mean(0), full_matrices=False)[2][0]
    seg = dict(names=names, depth=d.tolist(), label=lab.tolist(), leg=leg.tolist(),
               s=s_al.tolist(),
               levels=dict(shallow=float(np.median(d[lab == 1])), deep=float(np.median(d[lab == 2]))),
               counts=dict(shallow=int((lab == 1).sum()), deep=int((lab == 2).sum()),
                           transition=int((lab == 0).sum())))
    print(f"segmentation: {seg['counts']} levels {seg['levels']}", flush=True)
    res, heights = {}, {}
    for m in args.models:
        name, path = m.split("=", 1)
        res[name], heights[name] = evaluate(path, names, lab, leg, s_al, gps, axis_u)
        r = res[name]
        sd = r.get("sand", {})
        print(f"{name:16s}", summary(r), flush=True)
    if args.json:
        json.dump(dict(segmentation=seg, results=res,
                       heights={k: [None if not np.isfinite(x) else round(float(x), 4) for x in v]
                                for k, v in heights.items()}),
                  open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
