#!/usr/bin/env bash
# Run all lens × loss configuration combinations sequentially.
# Logs for each run are saved to outputs/<name>/train.log
# Usage: bash scripts/run_all_configs.sh

set -euo pipefail

# Use cached HuggingFace models — avoids any network requests to HF Hub.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Force Python stdout/stderr to be unbuffered so output is written immediately.
export PYTHONUNBUFFERED=1

# Use 'script' to allocate a pseudo-TTY so Lightning's progress bar and ETA
# render exactly as in an interactive single run. Falls back to plain tee if
# 'script' is not available.
_run_with_log() {
    local cmd="$1"
    local log="$2"
    if command -v script &>/dev/null; then
        # -q: suppress "Script started/ended" messages
        # -e: exit with the command's exit code
        # -c: command to run
        script -q -e -c "${cmd}" "${log}"
    else
        eval "${cmd}" 2>&1 | tee "${log}"
    fi
}

PYTHON="venv/bin/python"
CONFIGS=(
    "configs/patch_affine_kld.yaml"
    "configs/patch_mlp_kld.yaml"
)

TOTAL=${#CONFIGS[@]}
FAILED=()

echo "======================================================"
echo " Patch lens training: ${TOTAL} configs"
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

    if _run_with_log \
        "${PYTHON} -m tuned_lens.scripts.train --config ${CONFIG}" \
        "${LOG_FILE}"; then
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
