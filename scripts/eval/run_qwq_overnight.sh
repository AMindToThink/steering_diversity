#!/usr/bin/env bash
set -euo pipefail

# Overnight QwQ-32B pass@k evaluation: unsteered + steered (α=-0.25)
# Starts servers with thinking disabled, runs codegen, evaluates.
#
# Prerequisites:
#   - Steered model saved at /tmp/qwq-32b-steered-unaware-0.25/
#   - No vLLM server running on port 8017
#
# Usage: nohup bash scripts/eval/run_qwq_overnight.sh > /tmp/qwq_overnight.log 2>&1 &

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
ID_RANGE="0-50"

UNSTEERED_OUT="outputs/passk_qwq_v1/code/bigcodebench/scale_0.0_temp_0.8"
STEERED_OUT="outputs/passk_qwq_v1/code/bigcodebench/scale_0.25_temp_0.8"

echo "============================================"
echo "QwQ-32B Overnight Eval (thinking disabled)"
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
    # Wait for GPUs to clear
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
# PHASE 1: Unsteered
# ============================================
echo ""
echo "============================================"
echo "PHASE 1: Unsteered codegen"
echo "Started: $(date)"
echo "============================================"

# Kill any existing server first
kill_server 2>/dev/null || true

start_server "${UNSTEERED_MODEL}" "qwq-32b" "/tmp/vllm_qwq_unsteered_overnight.log"

PHASE1_START=$(date +%s)
run_codegen "qwq-32b" "${UNSTEERED_OUT}"
PHASE1_END=$(date +%s)
echo "Phase 1 completed in $(( (PHASE1_END - PHASE1_START) / 60 )) minutes"

# ============================================
# PHASE 2: Steered
# ============================================
echo ""
echo "============================================"
echo "PHASE 2: Steered codegen"
echo "Started: $(date)"
echo "============================================"

kill_server
start_server "${STEERED_MODEL}" "qwq-32b-steered" "/tmp/vllm_qwq_steered_overnight.log"

PHASE2_START=$(date +%s)
run_codegen "qwq-32b-steered" "${STEERED_OUT}"
PHASE2_END=$(date +%s)
echo "Phase 2 completed in $(( (PHASE2_END - PHASE2_START) / 60 )) minutes"

# Kill server — don't need it for evaluation
kill_server

# ============================================
# PHASE 3: Evaluate
# ============================================
echo ""
echo "============================================"
echo "PHASE 3: Evaluate both conditions"
echo "Started: $(date)"
echo "============================================"

UNSTEERED_SAMPLES=$(ls -t "${UNSTEERED_OUT}"/*.jsonl 2>/dev/null | head -1)
STEERED_SAMPLES=$(ls -t "${STEERED_OUT}"/*.jsonl 2>/dev/null | head -1)

if [[ -z "${UNSTEERED_SAMPLES}" ]]; then
    echo "ERROR: No unsteered samples found in ${UNSTEERED_OUT}" >&2
    exit 1
fi
if [[ -z "${STEERED_SAMPLES}" ]]; then
    echo "ERROR: No steered samples found in ${STEERED_OUT}" >&2
    exit 1
fi

echo "Evaluating unsteered: ${UNSTEERED_SAMPLES}"
uv run python -m bigcodebench.evaluate \
    --split "${SPLIT}" \
    --subset "${SUBSET}" \
    --samples "${UNSTEERED_SAMPLES}" \
    --execution local \
    --pass_k "1,2,3,4,5" \
    --no_gt \
    2>&1 | tee "${UNSTEERED_OUT}/logs/evaluate.log"

echo ""
echo "Evaluating steered: ${STEERED_SAMPLES}"
uv run python -m bigcodebench.evaluate \
    --split "${SPLIT}" \
    --subset "${SUBSET}" \
    --samples "${STEERED_SAMPLES}" \
    --execution local \
    --pass_k "1,2,3,4,5" \
    --no_gt \
    2>&1 | tee "${STEERED_OUT}/logs/evaluate.log"

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
echo "  Unsteered: ${UNSTEERED_OUT}/"
echo "  Steered:   ${STEERED_OUT}/"
echo "============================================"
