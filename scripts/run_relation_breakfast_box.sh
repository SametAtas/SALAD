#!/usr/bin/env bash
set -euo pipefail

CATEGORY="${CATEGORY:-breakfast_box}"
SEG_ROOT="${SEG_ROOT:-data/mvtec_loco_composition_maps}"
MODEL="${MODEL:-relation_model_${CATEGORY}.npz}"
REL_TEST="${REL_TEST:-relation_scores_${CATEGORY}_test.csv}"
PSEUDO_VAL="${PSEUDO_VAL:-relation_pseudo_val_${CATEGORY}.csv}"
BRANCH_SCORES="${BRANCH_SCORES:-results/${CATEGORY}/branch_scores_${CATEGORY}_final.csv}"

python relation_features.py fit \
  --category "$CATEGORY" \
  --seg_root "$SEG_ROOT" \
  --output "$MODEL"

python relation_features.py score \
  --category "$CATEGORY" \
  --seg_root "$SEG_ROOT" \
  --split test \
  --model "$MODEL" \
  --output "$REL_TEST"

python pseudo_relation_validation.py \
  --category "$CATEGORY" \
  --seg_root "$SEG_ROOT" \
  --model "$MODEL" \
  --output "$PSEUDO_VAL"

if [[ -f "$BRANCH_SCORES" ]]; then
  python relation_fusion.py \
    --branch_scores "$BRANCH_SCORES" \
    --relation_scores "$REL_TEST" \
    --weights 1,1,1,0.5 \
    --output "relation_fusion_${CATEGORY}_test.csv" \
    --summary_json "relation_fusion_${CATEGORY}_summary.json"
else
  echo "Skipping fusion: branch-score CSV not found at $BRANCH_SCORES"
  echo "Set BRANCH_SCORES=/path/to/branch_scores_${CATEGORY}_*.csv and rerun this script."
fi
