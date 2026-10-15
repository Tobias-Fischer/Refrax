"""SfM spot checks of the sounder depth proxy (lizard_screen.py) on every sequence.

    python depth_aware/lizard_windows.py [--per-seq 8] [--fetch] [--sfm]

For each sequence, --per-seq windows are placed evenly over the survey, half in long
shallow runs and half in long deep runs of the sounder labels. A window is every 2nd of 20
consecutive images (10 images, 0.34 m apart, 3.4 m along-track). Per window:
  * --fetch: rsync the 10 JPGs from the HPC data-transfer node (copy only) to
    /Users/fischert/robot/refrax-data/lizard_windows/<seq>/ and downscale 6x (area) locally
  * --sfm: COLMAP (exhaustive matching, 4 threads) with the GP1 underwater
    self-calibration held fixed (OPENCV, lizard calib.json; for the screening the GP2
    camera is assumed similar, the depth error of a wrong focal length is ~1:1 in scale)
  * metric scale: the window's camera centres are aligned (similarity) to the GPS track
    smoothed over +-25 images, which gives the along-track distance to a few %
  * depth: median camera-to-point depth over the window's images (m)
Output: <out>/windows.json with sfm depth vs sounder depth per window.
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colmap_utils as cu  # noqa: E402
from lizard_eval import umeyama  # noqa: E402
from lizard_screen import OUT, SEQS, counted  # noqa: E402

DATA = "/Users/fischert/robot/refrax-data/lizard_windows"
META = "/Users/fischert/robot/refrax-data/lizard_meta"
HOST = "eresdtn01.qut.edu.au"
ROOT = "/work/qvpr/data/raw/Lizard_Island/Serena_Mou_Data"
SEQ_DIR = {"feb_gp1": "LIRS_Feb_24/South_Palfrey_1/GoPro1",
           "mar_gp1": "LIRS_Mar_24/South_Palfrey_1/GoPro1",
           "mar_gp2": "LIRS_Mar_24/South_Palfrey_1/GoPro2",
           "sep_gp1": "LIRS_Sep_25/GP1/images", "sep_gp2": "LIRS_Sep_25/GP2/images"}
# GP1 underwater self-calibration at 928x812 (docs/explore_lizard.md, section 3)
CAM = (928, 812, 589.89, 590.44, 466.05, 404.77, 0.315, 0.402, 0.0, 0.0)


def hpc_paths():
    """name -> path relative to ROOT, from the listing of lizard_meta/listing.txt."""
    out = {}
    for line in open(f"{META}/listing.txt"):
        p = line.split()[0]
        for seq, d in SEQ_DIR.items():
            if p.startswith(d + "/"):
                out[(seq, os.path.basename(p)[:-4])] = p
    return out


def pick_windows(seq, per_seq, span=20):
    z = np.load(f"{OUT}/screen_{seq}.npz")
    lab, n = z["lab"], len(z["lab"])
    win = []
    for v in (1, 2):
        c = np.flatnonzero(counted(lab, v))
        c = c[(c > span) & (c < n - span)]
        for q in np.linspace(0.05, 0.95, per_seq // 2):
            win.append(int(c[int(q * (len(c) - 1))]))
    return sorted(win)


def fetch(files, dest):
    os.makedirs(dest, exist_ok=True)
    need = [f for f in files if not os.path.exists(f"{dest}/{os.path.basename(f)}")]
    if need:
        lst = f"{dest}/.files"
        open(lst, "w").write("\n".join(need) + "\n")
        subprocess.run(["rsync", "-a", "--no-relative", "-e", "ssh -o BatchMode=yes",
                        f"--files-from={lst}", f"{HOST}:{ROOT}/", dest], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def downscale(src, dst, f=6):
    if os.path.exists(dst):
        return
    img = cv2.imread(src)
    h, w = img.shape[:2]
    cv2.imwrite(dst, cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-seq", type=int, default=8)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--sfm", action="store_true")
    ap.add_argument("--seqs", nargs="*", default=list(SEQS))
    args = ap.parse_args()
    paths = hpc_paths()
    res = json.load(open(f"{OUT}/windows.json")) if os.path.exists(f"{OUT}/windows.json") else {}
    for seq in args.seqs:
        z = np.load(f"{OUT}/screen_{seq}.npz")
        names, xy, dep = list(z["names"]), z["xy"], z["dep"]
        xs = uniform_filter1d(xy, 51, axis=0, mode="nearest")
        for c in pick_windows(seq, args.per_seq):
            idx = list(range(c - 10, c + 10, 2))
            key = f"{seq}/{names[c]}"
            files = [paths[(seq, names[i])] for i in idx if (seq, names[i]) in paths]
            if len(files) < len(idx):
                print(key, "missing files", flush=True)
                continue
            wdir = f"{DATA}/{seq}/{names[c]}"
            if args.fetch:
                fetch(files, f"{wdir}/jpg")
                os.makedirs(f"{wdir}/img", exist_ok=True)
                for f in files:
                    b = os.path.basename(f)[:-4]
                    downscale(f"{wdir}/jpg/{b}.JPG", f"{wdir}/img/{b}.png")
            r = dict(seq=seq, centre=names[c], index=c, sounder=float(np.nanmedian(dep[idx])))
            if args.sfm and os.path.isdir(f"{wdir}/img"):
                work = f"{OUT}/windows/{seq}_{names[c]}"
                try:
                    m = cu.sfm(f"{wdir}/img", work, CAM, camera_model="OPENCV")
                    imgs = cu.read_images(f"{m}/images.txt")
                    pts = cu.read_points3d(f"{m}/points3D.txt")
                    nm = sorted(imgs)
                    C = np.array([-imgs[n]["R"].T @ imgs[n]["t"] for n in nm])
                    G = np.array([[*xs[names.index(n[:-4])], 0.0] for n in nm])
                    s = umeyama(C, G)[0]
                    meds = []
                    for n in nm:
                        d = imgs[n]
                        ids = d["p3d"][d["p3d"] >= 0]
                        P = np.array([pts[i][0] for i in ids if i in pts])
                        if len(P) > 50:
                            meds.append(np.median(s * (P @ d["R"].T + d["t"])[:, 2]))
                    r.update(registered=len(nm), sfm=float(np.median(meds)),
                             sfm_spread=float(np.ptp(meds)) if meds else None,
                             baseline=float(np.ptp(G[:, :2], 0).max()))
                except Exception as e:  # noqa: BLE001
                    r.update(error=str(e)[:200])
            res[key] = r
            print(key, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()},
                  flush=True)
            json.dump(res, open(f"{OUT}/windows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
