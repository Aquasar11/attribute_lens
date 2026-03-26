#!/usr/bin/env bash
# Run all lens × loss configuration combinations sequentially.
# Logs for each run are saved to outputs/<name>/train.log
# Usage: bash scripts/run_all_configs.sh

set -euo pipefail

# Use cached HuggingFace models — avoids any network requests to HF Hub.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Force Python stdout/stderr to be unbuffered so tee receives output line-by-line.
export PYTHONUNBUFFERED=1

PYTHON="venv/bin/python"
CONFIGS=(
    "configs/affine_kld.yaml"
    "configs/affine_ce.yaml"
    "configs/affine_combined.yaml"
    "configs/mlp_kld.yaml"
    "configs/mlp_ce.yaml"
    "configs/mlp_combined.yaml"
)

TOTAL=${#CONFIGS[@]}
FAILED=()

echo "======================================================"
echo " Tuned lens sweep: ${TOTAL} configs"
echo " Started: $(date)"
echo "======================================================"

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    NAME=$(basename "${CONFIG}" .yaml)
    OUTPUT_DIR="outputs/${NAME}"
    LOG_FILE="${OUTPUT_DIR}/train.log"
    RUN_NUM=$((i + 1))

    echo ""
    echo "[${RUN_NUM}/${TOTAL}] ----------------------------------------"
    echo "  Config : ${CONFIG}"
    echo "  Output : ${OUTPUT_DIR}"
    echo "  Log    : ${LOG_FILE}"
    echo "  Started: $(date)"

    mkdir -p "${OUTPUT_DIR}"

    if ${PYTHON} -m tuned_lens.scripts.train \
        --config "${CONFIG}" \
        2>&1 | tee "${LOG_FILE}"; then
        echo "  Status : DONE ($(date))"
    else
        echo "  Status : FAILED ($(date))"
        FAILED+=("${NAME}")
    fi
done

echo ""
echo "======================================================"
echo " Finished: $(date)"
echo " Completed: $((TOTAL - ${#FAILED[@]})) / ${TOTAL}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo " Failed runs:"
    for name in "${FAILED[@]}"; do
        echo "   - ${name}  (log: outputs/${name}/train.log)"
    done
    exit 1
fi

echo " All runs completed successfully."
echo "======================================================"
