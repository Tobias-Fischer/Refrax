"""Running COLMAP on corrected images and reading its text models."""
import os
import shutil
import subprocess

import numpy as np

THREADS = int(os.environ.get("REFRAX_THREADS", "4"))


def run(cmd, log):
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def sfm(image_dir, work, calib, mask_dir=None, matcher="exhaustive", overwrite=False,
        camera_model="PINHOLE", refine=False, pairs=None, seed=0):
    """Feature extraction, matching and mapping (single camera).

    calib = (W, H, fx, fy, cx, cy) for PINHOLE, or (W, H, *params) for other models; the
    intrinsics are kept fixed unless refine. pairs: a COLMAP pair list (one "a b" per line)
    to match instead of running `matcher`. Returns the path of the largest model, in TXT form.
    """
    db, sparse, log = f"{work}/database.db", f"{work}/sparse", f"{work}/colmap.log"
    if overwrite and os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(sparse, exist_ok=True)
    params = ",".join(f"{float(p):.12g}" for p in calib[2:])
    if not os.path.exists(f"{work}/.features"):
        if os.path.exists(db):
            os.remove(db)
        cmd = ["colmap", "feature_extractor", "--database_path", db, "--image_path", image_dir,
               "--ImageReader.camera_model", camera_model, "--ImageReader.single_camera", "1",
               "--ImageReader.camera_params", params,
               "--SiftExtraction.num_threads", str(THREADS),
               "--SiftExtraction.use_gpu", "0", "--SiftExtraction.max_num_features", "8192"]
        if mask_dir:
            cmd += ["--ImageReader.mask_path", mask_dir]
        run(cmd, log)
        open(f"{work}/.features", "w").close()
    if not os.path.exists(f"{work}/.matches"):
        if pairs:
            run(["colmap", "matches_importer", "--database_path", db, "--match_list_path", pairs,
                 "--match_type", "pairs", "--SiftMatching.num_threads", str(THREADS),
                 "--SiftMatching.use_gpu", "0"], log)
        else:
            run(["colmap", f"{matcher}_matcher", "--database_path", db,
                 "--SiftMatching.num_threads", str(THREADS), "--SiftMatching.use_gpu", "0"], log)
        open(f"{work}/.matches", "w").close()
    if not os.path.exists(f"{work}/.mapped"):
        r = "1" if refine else "0"
        run(["colmap", "mapper", "--database_path", db, "--image_path", image_dir,
             "--output_path", sparse, "--Mapper.num_threads", str(THREADS),
             "--random_seed", str(seed),
             "--Mapper.ba_refine_focal_length", r, "--Mapper.ba_refine_principal_point", r,
             "--Mapper.ba_refine_extra_params", r], log)
        open(f"{work}/.mapped", "w").close()
    return largest_model_txt(sparse, log)


def remap(work, image_dir, seed, log_name="remap"):
    """Re-run only the mapper on an existing database with another random seed."""
    out = f"{work}/sparse_seed{seed}"
    log = f"{work}/{log_name}.log"
    if not os.path.exists(f"{out}/.mapped"):
        os.makedirs(out, exist_ok=True)
        run(["colmap", "mapper", "--database_path", f"{work}/database.db",
             "--image_path", image_dir, "--output_path", out, "--Mapper.num_threads",
             str(THREADS), "--random_seed", str(seed),
             "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
             "--Mapper.ba_refine_extra_params", "0"], log)
        open(f"{out}/.mapped", "w").close()
    return largest_model_txt(out, log)


def largest_model_txt(sparse, log):
    best, best_n = None, -1
    for m in sorted(os.listdir(sparse)):
        p = f"{sparse}/{m}"
        if not os.path.isdir(p) or m.endswith("_txt"):
            continue
        txt = p + "_txt"
        if not os.path.exists(f"{txt}/images.txt"):
            os.makedirs(txt, exist_ok=True)
            run(["colmap", "model_converter", "--input_path", p, "--output_path", txt,
                 "--output_type", "TXT"], log)
        n = len(read_images(f"{txt}/images.txt"))
        if n > best_n:
            best, best_n = txt, n
    return best


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_images(path):
    """{name: dict(id, R, t, xy (N,2), p3d (N,) int)}."""
    out = {}
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    for i in range(0, len(lines) - 1, 2):
        e = lines[i].split()
        if not e:
            continue
        q = np.array(list(map(float, e[1:5])))
        t = np.array(list(map(float, e[5:8])))
        pts = np.array(lines[i + 1].split(), dtype=float).reshape(-1, 3)
        out[e[9]] = dict(id=int(e[0]), R=qvec2rotmat(q), t=t, q=q, cam=int(e[8]),
                         xy=pts[:, :2], p3d=pts[:, 2].astype(np.int64))
    return out


def read_points3d(path):
    """{id: (xyz, err, track_len)}."""
    out = {}
    with open(path) as f:
        for l in f:
            if l.startswith("#"):
                continue
            e = l.split(maxsplit=8)
            n = (len(e[8].split()) // 2) if len(e) > 8 else 0
            out[int(e[0])] = (np.array([float(e[1]), float(e[2]), float(e[3])]), float(e[7]), n)
    return out


def model_stats(txt, n_images):
    imgs = read_images(f"{txt}/images.txt")
    pts = read_points3d(f"{txt}/points3D.txt")
    err = np.array([p[1] for p in pts.values()])
    tl = np.array([p[2] for p in pts.values()])
    return dict(registered=len(imgs), reg_ratio=len(imgs) / n_images, points=len(pts),
                track_len=float(tl.mean()), reproj=float(err.mean()))


def camera_centers(imgs):
    return {n: -d["R"].T @ d["t"] for n, d in imgs.items()}
