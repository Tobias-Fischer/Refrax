"""Region-only reconstructions: re-run the mapper of existing runs on one depth region.

    python depth_aware/lizard_submaps.py --regions <regions.json> --out DIR name=<run_dir> ...

In a joint reconstruction the two regions are coupled (overlap across legs, the turns), so
a region's bending can be driven by the other region's z0 error. Here the database of each
run (copied, read-only use of the source) is mapped again with --image_list_path restricted
to the images of one region (labels of lizard_regions.py), same mapper settings as
colmap_utils.sfm (fixed intrinsics, seed 0, REFRAX_THREADS threads). Models go to
<out>/<name>_<region>/sparse; evaluate them with lizard_regions.py.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colmap_utils as cu  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="name=<run_dir with colmap/ and images/>")
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seg = json.load(open(args.regions))["segmentation"]
    for spec in args.runs:
        name, run = spec.split("=", 1)
        for r, rn in ((1, "shallow"), (2, "deep")):
            work = f"{args.out}/{name}_{rn}"
            os.makedirs(f"{work}/sparse", exist_ok=True)
            if os.path.exists(f"{work}/.mapped"):
                continue
            lst = f"{work}/images.txt"
            with open(lst, "w") as f:
                for n, lab in zip(seg["names"], seg["label"]):
                    if lab == r:
                        f.write(n + "\n")
            db = f"{work}/database.db"
            shutil.copy(f"{run}/colmap/database.db", db)
            cu.run(["colmap", "mapper", "--database_path", db, "--image_path", f"{run}/images",
                    "--image_list_path", lst, "--output_path", f"{work}/sparse",
                    "--Mapper.num_threads", str(cu.THREADS), "--random_seed", "0",
                    "--Mapper.ba_refine_focal_length", "0",
                    "--Mapper.ba_refine_principal_point", "0",
                    "--Mapper.ba_refine_extra_params", "0"], f"{work}/colmap.log")
            os.remove(db)
            open(f"{work}/.mapped", "w").close()
            print(name, rn, cu.largest_model_txt(f"{work}/sparse", f"{work}/colmap.log"),
                  flush=True)


if __name__ == "__main__":
    main()
