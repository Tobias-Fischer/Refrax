"""Synthetic evaluation of the tilt-estimating refractive BA (docs/explore_tilt.md, §4a).

    python tilt/synthetic.py [--trials 6] [--tilts 0 0.5 1 2] [--scenes tank reef]

Per trial: true housing / lens parameters drawn from the uncertainty prior
(core.uncertainty.default_prior), except the port tilt, which is set to the requested
magnitude in a random direction. Raw keypoints are simulated exactly through the true
model with SIG_RAW px noise. Methods (all start from the same perturbed poses / points):

  refrax+pinhole     Refrax correction with the nominal parameters (port on axis, fixed
                     depth z0), pinhole BA with fx, fy, cx, cy refined (= today's pipeline)
  refrac, no tilt    refractive BA on the raw keypoints (exact depth from the points),
                     in-air fx, fy, cx, cy refined, tilt fixed at 0
  refrac + tilt      ... + tilt_x, tilt_y free                      (the proposal)
  refrac + tilt + d0 ... + d0 free
  refrax(tilt)+pin   Refrax correction re-done with the *estimated* tilt, pinhole BA
                     (the "feed it back into Refrax" route)
  oracle             refractive BA with all parameters at their true values (noise floor)

Writes docs/figures/tilt_synthetic.json and prints medians.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "uncertainty"))
from core.config import load_config  # noqa: E402
from core import uncertainty as U  # noqa: E402
from tilt.model import Decl, IDX  # noqa: E402
from tilt.ba import RefracBA  # noqa: E402
import synthetic_ba as S  # noqa: E402

SIG_RAW = 0.5


def align_eval(R, t, X, cams_gt, X_gt, planar):
    out = {}
    C = np.einsum("nji,nj->ni", R, -t)
    Cg = np.array([-Rg.T @ tg for Rg, tg in cams_gt])
    s, Ra, ta = S.umeyama(C, Cg)
    Ca = s * C @ Ra.T + ta
    Xa = s * X @ Ra.T + ta
    Errs = np.stack([Rg @ (Re @ Ra.T).T for (Rg, _), Re in zip(cams_gt, R)])
    rot = [np.degrees(np.linalg.norm(Rotation.from_matrix(E).as_rotvec())) for E in Errs]
    Uq, _, Vq = np.linalg.svd(Errs.sum(0))
    Q = Uq @ np.diag([1, 1, np.linalg.det(Uq @ Vq)]) @ Vq
    rot_rel = [np.degrees(np.linalg.norm(Rotation.from_matrix(E @ Q.T).as_rotvec()))
               for E in Errs]
    out = dict(pos_mm=1e3 * float(np.sqrt(((Ca - Cg) ** 2).sum(1).mean())),
               rot_deg=float(np.sqrt(np.mean(np.square(rot)))),
               rot_rel_deg=float(np.sqrt(np.mean(np.square(rot_rel)))),
               pts_mm=1e3 * float(np.sqrt(((Xa - X_gt) ** 2).sum(1).mean())))
    if planar:
        r = S.pl.fit_quadric(S.pl.analyze(Xa))
        out["plan_rel_rmse_pct"] = float(np.sqrt(np.mean(r["dist"] ** 2)) / r["D"] * 100)
        out["kappa"] = float(r["c3"] * r["D"])
    return out


def pinhole_ba(cams_i, X_i, cam, ptu, q, intr0, baseline):
    ba = S.BA(cams_i, X_i, cam, ptu, q, intr0, cov=None, refine_intr=True, baseline=baseline)
    Rs, ts, X, intr, _ = ba.solve()
    return np.asarray(Rs), np.asarray(ts), X, intr


def run_trial(kind, X, cams, geo, z0, th_true, rng, planar, baseline):
    cam, pt, u_obs, w_corr = S.simulate(X, cams, th_true, geo, rng, False, z0)
    # S.simulate adds SIG_RAW of synthetic_ba (0.35 px); add the rest to reach SIG_RAW
    extra = np.sqrt(max(SIG_RAW ** 2 - S.SIG_RAW ** 2, 0.0))
    u_obs = u_obs + rng.normal(0, extra, u_obs.shape)
    used = np.unique(pt)
    remap = -np.ones(len(X), int)
    remap[used] = np.arange(len(used))
    Xu, ptu = X[used], remap[pt]
    cams_i = [(Rotation.from_rotvec(rng.normal(0, 0.005, 3)).as_matrix() @ R,
               t + rng.normal(0, 0.005, 3)) if i else (R, t) for i, (R, t) in enumerate(cams)]
    X_i = Xu + rng.normal(0, 0.005, Xu.shape)
    R_i = np.stack([R for R, _ in cams_i])
    t_i = np.stack([t for _, t in cams_i])
    decl = Decl(geo.f_out, geo.c_out, geo.R0, geo.mu_a)
    row = dict(theta_true=th_true.tolist(), n_obs=int(len(cam)),
               tilt_true_deg=[float(np.degrees(th_true[4])), float(np.degrees(th_true[5]))])

    # today's pipeline
    q_obs = S.corrected(geo.theta0, u_obs, w_corr, geo)
    intr0 = [geo.f_out[0], geo.f_out[1], geo.c_out[0], geo.c_out[1]]
    Rs, ts, Xe, intr = pinhole_ba(cams_i, X_i, cam, ptu, q_obs, intr0, baseline)
    row["refrax+pinhole"] = align_eval(Rs, ts, Xe, cams, Xu, planar)
    row["refrax+pinhole"]["intr"] = [float(v) for v in intr]

    variants = {
        "refrac, no tilt": ("fx", "fy", "cx", "cy"),
        "refrac + tilt": ("tilt_x", "tilt_y", "fx", "fy", "cx", "cy"),
        "refrac + tilt + d0": ("tilt_x", "tilt_y", "d0", "fx", "fy", "cx", "cy"),
        "oracle": (),
    }
    th_hat = None
    for name, free in variants.items():
        th0 = th_true.copy() if name == "oracle" else geo.theta0.copy()
        ba = RefracBA(R_i, t_i, X_i, cam, ptu, u_obs, th0, decl, free=free, huber=None,
                      sigma_px=SIG_RAW, verbose=False)
        ba.solve(max_iter=40)
        res = align_eval(ba.R, ba.t, ba.X, cams, Xu, planar)
        if free:
            sm = ba.summary()
            res.update(params=sm["params"], corr=sm["corr"], rms_px=sm["rms_px"])
        row[name] = res
        if name == "refrac + tilt":
            th_hat = ba.theta.copy()
    # feed the estimated tilt back into the Refrax correction (fixed depth), pinhole BA
    th_fb = geo.theta0.copy()
    th_fb[4:6] = th_hat[4:6]
    q_fb = S.corrected(th_fb, u_obs, w_corr, geo)
    Rs, ts, Xe, intr = pinhole_ba(cams_i, X_i, cam, ptu, q_fb, intr0, baseline)
    row["refrax(tilt)+pinhole"] = align_eval(Rs, ts, Xe, cams, Xu, planar)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--tilts", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--scenes", nargs="*", default=["tank", "reef"])
    ap.add_argument("--out", default=os.path.join(REPO, "docs/figures/tilt_synthetic.json"))
    args = ap.parse_args()
    cfg = load_config(os.path.join(REPO, "configs/dataset.yaml"))
    results = {}
    for kind in args.scenes:
        rng = np.random.default_rng(20)
        X, cams = S.make_scene(kind, rng)
        Zs = np.concatenate([(X @ R.T + t)[:, 2] for R, t in cams])
        z0 = float(np.median(Zs[Zs > 0.2]))
        geo, _ = U.setup(dict(cfg, correction=dict(cfg["correction"], z0_fixed=z0)))
        prior = U.default_prior()
        sig = prior.vector()
        Cg = np.array([-R.T @ t for R, t in cams])
        baseline = float(np.linalg.norm(Cg[1] - Cg[0]))
        rows = []
        for tilt in args.tilts:
            for tr in range(args.trials):
                t0 = time.time()
                th = geo.theta0 + sig * rng.standard_normal(len(sig))
                phi = rng.uniform(0, 2 * np.pi)
                th[4], th[5] = np.radians(tilt) * np.cos(phi), np.radians(tilt) * np.sin(phi)
                row = run_trial(kind, X, cams, geo, z0, th, rng, kind == "tank", baseline)
                row.update(tilt=tilt, trial=tr)
                rows.append(row)
                p = row["refrac + tilt"]["params"]
                print(f"{kind} tilt {tilt} tr {tr} ({time.time() - t0:.0f}s) true "
                      f"({row['tilt_true_deg'][0]:+.3f}, {row['tilt_true_deg'][1]:+.3f}) est "
                      f"({p['tilt_x'][0]:+.3f}±{p['tilt_x'][1]:.3f}, "
                      f"{p['tilt_y'][0]:+.3f}±{p['tilt_y'][1]:.3f}) pts mm: " +
                      ", ".join(f"{k}={v['pts_mm']:.2f}" for k, v in row.items()
                                if isinstance(v, dict) and "pts_mm" in v), flush=True)
                results[kind] = dict(z0=z0, rows=rows)
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=1)
    summarize(results)


def summarize(results):
    meths = ["refrax+pinhole", "refrac, no tilt", "refrac + tilt", "refrac + tilt + d0",
             "refrax(tilt)+pinhole", "oracle"]
    for kind, d in results.items():
        print(f"\n{kind} (z0 {d['z0']:.2f} m): median over trials")
        tilts = sorted({r["tilt"] for r in d["rows"]})
        for tilt in tilts:
            rows = [r for r in d["rows"] if r["tilt"] == tilt]
            err, sd, nees = [], [], []
            for r in rows:
                p = r["refrac + tilt"]["params"]
                e = np.array([p["tilt_x"][0], p["tilt_y"][0]]) - r["tilt_true_deg"]
                s = np.array([p["tilt_x"][1], p["tilt_y"][1]])
                err.append(e)
                sd.append(s)
                nees.append(np.sum((e / s) ** 2))
            err, sd = np.array(err), np.array(sd)
            print(f" tilt {tilt}°: tilt error RMS x/y {np.sqrt((err ** 2).mean(0)).round(3)}°, "
                  f"posterior sd {sd.mean(0).round(3)}°, NEES {np.mean(nees):.2f} (2 expected)")
            keys = ["pos_mm", "rot_rel_deg", "pts_mm"] + (["plan_rel_rmse_pct", "kappa"]
                                                          if kind == "tank" else [])
            print(f"  {'method':22s}" + "".join(f"{k:>18s}" for k in keys))
            for m in meths:
                vals = [np.median([abs(r[m][k]) if k == "kappa" else r[m][k] for r in rows])
                        for k in keys]
                print(f"  {m:22s}" + "".join(f"{v:18.4f}" for v in vals))


if __name__ == "__main__":
    main()
