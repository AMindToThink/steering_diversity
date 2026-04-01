#!/usr/bin/env bash
set -euo pipefail

# QwQ-32B pass@k scaleup: add 100 more BigCodeBench problems (50-150)
# to existing 50-problem dataset, for both unsteered + steered conditions.
#
# Prerequisites:
#   - Steered model at /tmp/qwq-32b-steered-unaware-0.25/
#   - Existing results in outputs/passk_qwq_v1/code/bigcodebench/
#   - No vLLM server running on port 8017
#
# Usage: nohup bash scripts/eval/run_qwq_scaleup.sh > /tmp/qwq_scaleup.log 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PORT=8017
UNSTEERED_MODEL="Qwen/QwQ-32B"
STEERED_MODEL="/tmp/qwq-32b-steered-unaware-0.25"

# BigCodeBench parameters
SPLIT="instruct"
SUBSET="full"
N_SAMPLES=5
TEMPERATURE=0.8
MAX_TOKENS=4096
BATCH_SIZE=10
ID_RANGE="50-150"

# New batch goes into batch2 subdirectories
UNSTEERED_BATCH2="outputs/passk_qwq_v1/code/bigcodebench/scale_0.0_temp_0.8/batch2"
STEERED_BATCH2="outputs/passk_qwq_v1/code/bigcodebench/scale_0.25_temp_0.8/batch2"

# Existing batch1 directories (for merge)
UNSTEERED_OUT="outputs/passk_qwq_v1/code/bigcodebench/scale_0.0_temp_0.8"
STEERED_OUT="outputs/passk_qwq_v1/code/bigcodebench/scale_0.25_temp_0.8"

echo "============================================"
echo "QwQ-32B Scaleup: +100 problems (50-150)"
echo "Started: $(date)"
echo "Problems: ${ID_RANGE}, n=${N_SAMPLES}, max_tokens=${MAX_TOKENS}"
echo "============================================"

# --- Precondition checks ---
echo ""
echo "[CHECK] Verifying prerequisites..."

if [[ ! -f "${STEERED_MODEL}/config.json" ]]; then
    echo "ERROR: Steered model not found at ${STEERED_MODEL}" >&2
    exit 1
fi
echo "  Steered model: OK"

AVAIL_GB=$(df --output=avail / | tail -1 | awk '{print int($1/1024/1024)}')
if (( AVAIL_GB < 5 )); then
    echo "ERROR: Only ${AVAIL_GB}GB free, need at least 5GB" >&2
    exit 1
fi
echo "  Disk space: ${AVAIL_GB}GB free"

uv run python -c "import bigcodebench" 2>/dev/null || {
    echo "ERROR: bigcodebench not installed in project venv" >&2
    exit 1
}
echo "  bigcodebench: OK"

# Check existing batch1 results exist
EXISTING_UNSTEERED=$(ls -t "${UNSTEERED_OUT}"/*.jsonl 2>/dev/null | head -1)
EXISTING_STEERED=$(ls -t "${STEERED_OUT}"/*.jsonl 2>/dev/null | head -1)
if [[ -z "${EXISTING_UNSTEERED}" ]]; then
    echo "ERROR: No existing unsteered samples in ${UNSTEERED_OUT}" >&2
    exit 1
fi
if [[ -z "${EXISTING_STEERED}" ]]; then
    echo "ERROR: No existing steered samples in ${STEERED_OUT}" >&2
    exit 1
fi
echo "  Existing unsteered: ${EXISTING_UNSTEERED}"
echo "  Existing steered:   ${EXISTING_STEERED}"

# --- Helper: start vLLM server and wait for ready ---
start_server() {
    local MODEL_PATH="$1"
    local MODEL_NAME="$2"
    local LOG_FILE="$3"

    echo "Starting vLLM server: ${MODEL_NAME}..."
    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${MODEL_NAME}" \
        --port "${PORT}" \
        --tensor-parallel-size 2 \
        --trust-remote-code \
        --dtype float16 \
        --max-model-len 32768 \
        > "${LOG_FILE}" 2>&1 &
    local PID=$!
    echo "  PID: ${PID}"

    echo "  Waiting for server..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
            echo "  Server ready after $((i*5))s"
            return 0
        fi
        if ! kill -0 "${PID}" 2>/dev/null; then
            echo "ERROR: Server died during startup. Log:" >&2
            tail -20 "${LOG_FILE}" >&2
            exit 1
        fi
        sleep 5
    done
    echo "ERROR: Server not ready after 10 minutes" >&2
    tail -20 "${LOG_FILE}" >&2
    exit 1
}

# --- Helper: kill vLLM server ---
kill_server() {
    echo "Killing vLLM server..."
    pkill -f "vllm.entrypoints.*${PORT}" || true
    sleep 5
    pkill -9 -f "vllm.entrypoints.*${PORT}" 2>/dev/null || true
    sleep 5
    for i in $(seq 1 12); do
        GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        if (( GPU_PROCS == 0 )); then
            echo "  GPUs clear"
            return 0
        fi
        echo "  Waiting for ${GPU_PROCS} GPU processes to exit..."
        sleep 5
    done
    echo "WARNING: GPU processes still running after 60s"
}

# --- Helper: run codegen ---
run_codegen() {
    local MODEL_NAME="$1"
    local OUT_DIR="$2"

    mkdir -p "${OUT_DIR}/logs"
    uv run python -m bigcodebench.generate \
        --model "${MODEL_NAME}" \
        --split "${SPLIT}" \
        --subset "${SUBSET}" \
        --backend openai \
        --base_url "http://localhost:${PORT}/v1" \
        --n_samples "${N_SAMPLES}" \
        --temperature "${TEMPERATURE}" \
        --max_new_tokens "${MAX_TOKENS}" \
        --bs "${BATCH_SIZE}" \
        --id_range "${ID_RANGE}" \
        --root "${OUT_DIR}" \
        --resume True \
        2>&1 | tee "${OUT_DIR}/logs/generate.log"
}

# ============================================
# PHASE 1: Unsteered codegen (problems 50-150)
# ============================================
echo ""
echo "============================================"
echo "PHASE 1: Unsteered codegen (problems 50-150)"
echo "Started: $(date)"
echo "============================================"

kill_server 2>/dev/null || true
start_server "${UNSTEERED_MODEL}" "qwq-32b" "/tmp/vllm_qwq_unsteered_scaleup.log"

PHASE1_START=$(date +%s)
run_codegen "qwq-32b" "${UNSTEERED_BATCH2}"
PHASE1_END=$(date +%s)
echo "Phase 1 completed in $(( (PHASE1_END - PHASE1_START) / 60 )) minutes"

# ============================================
# PHASE 2: Steered codegen (GPU) + unsteered merge+eval (CPU) in parallel
# ============================================
echo ""
echo "============================================"
echo "PHASE 2: Steered codegen (GPU) || unsteered eval (CPU)"
echo "Started: $(date)"
echo "============================================"

kill_server
start_server "${STEERED_MODEL}" "qwq-32b-steered" "/tmp/vllm_qwq_steered_scaleup.log"

# --- Background: merge + evaluate unsteered (CPU-only) ---
(
    echo "[BG] Merging unsteered batch1 + batch2..."
    BATCH2_UNSTEERED=$(ls -t "${UNSTEERED_BATCH2}"/*.jsonl 2>/dev/null | head -1)
    if [[ -z "${BATCH2_UNSTEERED}" ]]; then
        echo "[BG] ERROR: No unsteered batch2 samples in ${UNSTEERED_BATCH2}" >&2
        exit 1
    fi
    MERGED_UNSTEERED="${UNSTEERED_OUT}/merged_0-150.jsonl"
    cat "${EXISTING_UNSTEERED}" "${BATCH2_UNSTEERED}" > "${MERGED_UNSTEERED}"
    echo "[BG] Merged unsteered: ${MERGED_UNSTEERED} ($(wc -l < "${MERGED_UNSTEERED}") lines)"

    TASK_IDS=$(python3 -c "print(','.join(f'BigCodeBench/{i}' for i in range(150)))")
    echo "[BG] Evaluating merged unsteered..."
    uv run python -m bigcodebench.evaluate \
        --split "${SPLIT}" \
        --subset "${SUBSET}" \
        --samples "${MERGED_UNSTEERED}" \
        --execution local \
        --pass_k "1,2,3,4,5" \
        --no_gt \
        --selective_evaluate "${TASK_IDS}" \
        2>&1 | tee "${UNSTEERED_OUT}/logs/evaluate_merged.log"
    echo "[BG] Unsteered eval done at $(date)"
) &
EVAL_BG_PID=$!
echo "  Background unsteered eval PID: ${EVAL_BG_PID}"

# --- Foreground: steered codegen (GPU) ---
PHASE2_START=$(date +%s)
run_codegen "qwq-32b-steered" "${STEERED_BATCH2}"
PHASE2_END=$(date +%s)
echo "Phase 2 codegen completed in $(( (PHASE2_END - PHASE2_START) / 60 )) minutes"

kill_server

# Wait for background unsteered eval to finish
echo "Waiting for background unsteered eval (PID ${EVAL_BG_PID})..."
wait ${EVAL_BG_PID}
EVAL_BG_EXIT=$?
if (( EVAL_BG_EXIT != 0 )); then
    echo "ERROR: Background unsteered eval failed (exit ${EVAL_BG_EXIT})" >&2
    exit 1
fi
echo "Background unsteered eval completed successfully"

# ============================================
# PHASE 3: Merge + evaluate steered (sequential, since codegen just finished)
# ============================================
echo ""
echo "============================================"
echo "PHASE 3: Merge + evaluate steered"
echo "Started: $(date)"
echo "============================================"

BATCH2_STEERED=$(ls -t "${STEERED_BATCH2}"/*.jsonl 2>/dev/null | head -1)
if [[ -z "${BATCH2_STEERED}" ]]; then
    echo "ERROR: No steered batch2 samples in ${STEERED_BATCH2}" >&2
    exit 1
fi

MERGED_UNSTEERED="${UNSTEERED_OUT}/merged_0-150.jsonl"
MERGED_STEERED="${STEERED_OUT}/merged_0-150.jsonl"

cat "${EXISTING_STEERED}" "${BATCH2_STEERED}" > "${MERGED_STEERED}"
echo "  Merged steered: ${MERGED_STEERED} ($(wc -l < "${MERGED_STEERED}") lines)"

TASK_IDS=$(python3 -c "print(','.join(f'BigCodeBench/{i}' for i in range(150)))")
echo "Evaluating merged steered..."
uv run python -m bigcodebench.evaluate \
    --split "${SPLIT}" \
    --subset "${SUBSET}" \
    --samples "${MERGED_STEERED}" \
    --execution local \
    --pass_k "1,2,3,4,5" \
    --no_gt \
    --selective_evaluate "${TASK_IDS}" \
    2>&1 | tee "${STEERED_OUT}/logs/evaluate_merged.log"

# ============================================
# PHASE 4: Compute pass@k, test, plot
# ============================================
echo ""
echo "============================================"
echo "PHASE 4: Compute pass@k + coverage gain test"
echo "Started: $(date)"
echo "============================================"

# Find eval_results files
EVAL_UNSTEERED="${MERGED_UNSTEERED%.jsonl}_eval_results.json"
EVAL_STEERED="${MERGED_STEERED%.jsonl}_eval_results.json"

if [[ ! -f "${EVAL_UNSTEERED}" ]]; then
    echo "ERROR: No eval results at ${EVAL_UNSTEERED}" >&2
    exit 1
fi
if [[ ! -f "${EVAL_STEERED}" ]]; then
    echo "ERROR: No eval results at ${EVAL_STEERED}" >&2
    exit 1
fi

# Compute pass@k for each condition
PASSK_UNSTEERED="${UNSTEERED_OUT}/pass_at_k_merged.json"
PASSK_STEERED="${STEERED_OUT}/pass_at_k_merged.json"

uv run python scripts/eval/compute_passk_from_eval.py compute-bcb \
    --eval-results "${EVAL_UNSTEERED}" \
    --scale 0.0 \
    --temperature "${TEMPERATURE}" \
    --dataset bigcodebench \
    --k-values 1 2 3 4 5 \
    --output "${PASSK_UNSTEERED}"

uv run python scripts/eval/compute_passk_from_eval.py compute-bcb \
    --eval-results "${EVAL_STEERED}" \
    --scale 0.25 \
    --temperature "${TEMPERATURE}" \
    --dataset bigcodebench \
    --k-values 1 2 3 4 5 \
    --steering-label "Neg. test-awareness steered" \
    --output "${PASSK_STEERED}"

# Combine into curves file
CURVES="${UNSTEERED_OUT}/../pass_at_k_curves_merged.json"
uv run python scripts/eval/compute_passk_from_eval.py combine \
    "${PASSK_UNSTEERED}" "${PASSK_STEERED}" \
    --output "${CURVES}"

# Coverage gain test
uv run python scripts/eval/compute_passk_from_eval.py test \
    --baseline "${PASSK_UNSTEERED}" \
    --steered "${PASSK_STEERED}" \
    --metric plus \
    --output "${UNSTEERED_OUT}/../coverage_gain_test_merged.json"

# Plot
uv run python scripts/eval/plot_pass_at_k.py \
    --input "${CURVES}" \
    --output "${UNSTEERED_OUT}/../plots_merged"

# ============================================
# DONE
# ============================================
echo ""
echo "============================================"
echo "DONE"
echo "Finished: $(date)"
TOTAL_END=$(date +%s)
echo "Total runtime: $(( (TOTAL_END - PHASE1_START) / 3600 ))h $(( ((TOTAL_END - PHASE1_START) % 3600) / 60 ))m"
echo ""
echo "Results:"
echo "  Merged unsteered:  ${MERGED_UNSTEERED}"
echo "  Merged steered:    ${MERGED_STEERED}"
echo "  Curves:            ${CURVES}"
echo "  Coverage gain test: ${UNSTEERED_OUT}/../coverage_gain_test_merged.json"
echo "  Plots:             ${UNSTEERED_OUT}/../plots_merged/"
echo "============================================"
