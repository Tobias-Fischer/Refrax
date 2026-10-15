#!/bin/bash
# All Lizard Island runs of docs/explore_lizard.md, one COLMAP job at a time (4 threads).
#   pixi run -e depth bash depth_aware/lizard_runs.sh
# Assumes depth_aware/lizard_prep.py has been run (images, gps.csv, pairs.txt). ~45 min per
# COLMAP run on an M-series Mac (14.6k pairs, 476 images of ~790x690).
set -euo pipefail
cd "$(dirname "$0")"
W=${REFRAX_LIZARD_WORK:-/Users/fischert/robot/refrax-work/lizard}
export REFRAX_THREADS=${REFRAX_THREADS:-4}
P=$W/lizard_gp11_paper.yaml   # paper's rule, z0 from the data (written by lizard_calib.py)
S=$W/lizard_gp11_snell.yaml   # Snell conversion of the underwater self-calibration

# 1. raw images, COLMAP self-calibration (OPENCV, refined): baseline + calibration source
RAW=$(python - <<EOF
import colmap_utils as cu
print(cu.sfm("$W/raw928", "$W/runs/raw_opencv", (928, 812, 600, 600, 463.5, 405.5, 0, 0, 0, 0),
             camera_model="OPENCV", refine=True, pairs="$W/pairs.txt"))
EOF
)
python lizard_eval.py raw=$RAW --json $W/runs/raw_opencv/eval.json

# 2. self-calibration -> configs; z0 = median metric depth of the raw model (0.1 m)
Z0=$(python -c "import json; print(round(json.load(open('$W/runs/raw_opencv/eval.json'))['raw']['depth_med'], 1))")
python lizard_calib.py $RAW --json $W/calib.json --write-config $S $P --z0 $Z0 \
  --z0-note "median of the raw model"

# 3. fixed depth (paper method): z0 from the data and a second z0 for the flatness sweep;
#    z0* = zero crossing of the sag in 1/z0 (1.72 m here), checked with a run at 1.7
python lizard.py --config $P --calib-tag paper_ --mode fixed --z0 $Z0
python lizard.py --config $P --calib-tag paper_ --mode fixed --z0 2.0
python lizard.py --config $P --calib-tag paper_ --mode fixed --z0 1.7
# 4. depth-aware, depths from the z0 = 4.7 fixed run (GPS scale); nominal housing
python lizard.py --config $P --calib-tag paper_ --mode dense --z0 $Z0
python lizard.py --config $P --calib-tag paper_ --mode median --z0 $Z0
python lizard.py --config $P --calib-tag paper_ --mode dense --z0 $Z0 --center-z auto --source-c 0
# 5. mapper seed reruns (variance)
python - <<EOF
import colmap_utils as cu
for r in ["paper_z4.7_c0.00mm_fixed", "paper_z4.7_c0.00mm_dense", "paper_z1.7_c0.00mm_fixed"]:
    for seed in (1, 2):
        cu.remap(f"$W/runs/{r}/colmap", f"$W/runs/{r}/images", seed)
EOF
