#!/usr/bin/env bash
# Incremental mapping of gopro10 with the covariance-aware COLMAP (colmap-uncertainty):
# same database (features + matches of explore/depth's fixed-depth z0 = 1 m run), mapped
#   plain   no covariances (identical to stock COLMAP)
#   np      Sigma_np + (0.5 px)^2 I imported from <prep>/cov/shared.npy
# with PINHOLE intrinsics fixed (as explore/depth did) and refined.
#
#   bash uncertainty/real_mapper.sh RUN_DIR PREP_DIR
# Outputs PREP_DIR/mapper/<variant>/{0,0_txt}; 4 threads, one job at a time.
set -euo pipefail
RUN="$1"; PREP="$2"
CU=/Users/fischert/robot/colmap-uncertainty
COLMAP="$CU/install/bin/colmap"
PY="$CU/.pixi/envs/default/bin/python"
OUT="$PREP/mapper"; mkdir -p "$OUT"
for v in plain np; do
  [[ -f "$OUT/db_$v.db" ]] || sqlite3 "$RUN/colmap/database.db" ".backup '$OUT/db_$v.db'"
done
"$PY" "$CU/tools/import_covariance_maps.py" --database_path "$OUT/db_np.db" \
  --cov_dir "$PREP/cov" --detector-sigma 0.5 --clear > "$OUT/import.log" 2>&1
for v in plain np; do
  for refine in 0 1; do
    name="${v}_refine${refine}"
    [[ -d "$OUT/$name/0_txt" ]] && continue
    mkdir -p "$OUT/$name"
    /usr/bin/time -p "$COLMAP" mapper --database_path "$OUT/db_$v.db" --image_path "$RUN/images" \
      --output_path "$OUT/$name" --Mapper.num_threads 4 \
      --Mapper.ba_refine_focal_length $refine --Mapper.ba_refine_principal_point $refine \
      --Mapper.ba_refine_extra_params 0 > "$OUT/$name.log" 2>&1
    best=$(ls -d "$OUT/$name"/[0-9]* | head -1)
    mkdir -p "$best"_txt
    "$COLMAP" model_converter --input_path "$best" --output_path "$best"_txt --output_type TXT
    ln -sfn "$best"_txt "$PREP/ba/mapper_$name"
    echo "$name done"
  done
done
