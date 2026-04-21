#!/usr/bin/env bash
# Run all 9 experiment configs sequentially.
# Usage (from project root):
#   source venv/bin/activate
#   bash run_experiments.sh 2>&1 | tee run_experiments.log

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

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
