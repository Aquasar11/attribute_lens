#!/usr/bin/env bash
# Run layer-weight notebook then all 9 experiment configs sequentially.
# Usage (from project root):
#   source venv/bin/activate
#   bash run_experiments.sh 2>&1 | tee run_experiments.log

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

NOTEBOOK="notebooks/fg_bg_layer_weights.ipynb"
WEIGHTS_PT="/home/dev/attribute_lens/tmp/layer_avg_weights_patch_map_full.pt"

CONFIGS=(
    "configs/experiments/exp_no_nb_no_la.yaml"
    "configs/experiments/exp_no_nb_la_uniform.yaml"
    "configs/experiments/exp_no_nb_la_weighted.yaml"
    "configs/experiments/exp_nb_emb_no_la.yaml"
    "configs/experiments/exp_nb_emb_la_uniform.yaml"
    "configs/experiments/exp_nb_emb_la_weighted.yaml"
    "configs/experiments/exp_nb_score_no_la.yaml"
    "configs/experiments/exp_nb_score_la_uniform.yaml"
    "configs/experiments/exp_nb_score_la_weighted.yaml"
)

# ── Step 1: compute layer weights (one-time, full val set) ───────────────────
echo "============================================================"
echo "Step 1: Computing FG/BG layer weights from full val set"
echo "============================================================"

mkdir -p /home/dev/attribute_lens/tmp

jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=7200 \
    --output "${NOTEBOOK}" \
    "$NOTEBOOK"

if [[ ! -f "$WEIGHTS_PT" ]]; then
    echo "ERROR: weights file not found after notebook run: $WEIGHTS_PT"
    exit 1
fi
echo "Weights saved: $WEIGHTS_PT"

# ── Step 2: run 9 experiments ────────────────────────────────────────────────
TOTAL=${#CONFIGS[@]}
for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name="$(basename "$cfg" .yaml)"
    echo ""
    echo "============================================================"
    echo "Experiment $((i+1))/$TOTAL: $name"
    echo "============================================================"
    python -m attribute_lens.evaluate --config "$cfg"
done

echo ""
echo "============================================================"
echo "All $TOTAL experiments complete."
echo "Results in /home/dev/attribute_lens/outputs/experiments/"
echo "============================================================"
