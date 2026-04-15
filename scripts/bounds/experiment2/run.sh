#!/usr/bin/env bash
# Master orchestration script for experiment 2.
#
# Runs the whole pipeline end-to-end:
#   Phase 1: per-layer diagnostic on both models (GPU 0 for Qwen, GPU 1 for Llama, in parallel)
#   Phase 2: pick 3 layers per model (edge/middle/edge of target range)
#   Phase 3: generate all 14 config YAMLs
#   Phase 4: verify all 14 configs (parallel on both GPUs)
#   Phase 5: record all 14 configs (parallel on both GPUs, sequential within GPU)
#   Phase 6: compute + visualize all 14 runs (CPU)
#   Phase 7: summary table
#
# IMPORTANT: checks nvidia-smi before every GPU-using phase and aborts if
# either GPU is >1 GiB in use. Can be re-entered at a specific phase via
# ``--from-phase N``.
#
# Usage:
#   scripts/bounds/experiment2/run.sh              # run all phases
#   scripts/bounds/experiment2/run.sh --from-phase 4  # skip diagnostic, start at verify

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DIAGNOSTIC_DIR="outputs/bounds/experiment2"
CONFIGS_DIR="configs/bounds/experiment2"
QWEN_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
LLAMA_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
QWEN_VECTOR="EasySteer/vectors/happy_diffmean.gguf"
LLAMA_VECTOR="EasySteer/replications/creative_writing/create.gguf"
QWEN_TARGET_LAYERS=(10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
LLAMA_TARGET_LAYERS=(16 17 18 19 20 21 22 23 24 25 26 27 28 29)

FROM_PHASE=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --from-phase)
            FROM_PHASE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

mkdir -p "$DIAGNOSTIC_DIR" "$CONFIGS_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

check_gpus_idle() {
    # Fail if either GPU has > 1 GiB used (another process probably owns it).
    echo "[gpu-check] $(date +%T) — checking nvidia-smi ..."
    local out
    out=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    echo "$out" | awk -F, '
        { gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2+0 > 1024) {
            printf("[gpu-check] GPU %s has %s MiB in use — aborting phase (another process is using it)\n", $1, $2) > "/dev/stderr";
            exit 1
        }}
    '
    if [[ "${PIPESTATUS[0]}" != "0" ]]; then
        echo "[gpu-check] nvidia-smi failed" >&2
        return 1
    fi
    echo "$out" | awk -F, '{ gsub(/ /, "", $1); gsub(/ /, "", $2); printf("[gpu-check]   GPU %s: %s MiB used (OK)\n", $1, $2) }'
}

phase_header() {
    echo
    echo "=============================================================="
    echo "  Phase $1: $2"
    echo "  started at $(date)"
    echo "=============================================================="
}

# ---------------------------------------------------------------------------
# Phase 1: per-layer diagnostic
# ---------------------------------------------------------------------------

if [[ "$FROM_PHASE" -le 1 ]]; then
    phase_header 1 "per-layer diagnostic (parallel on both GPUs)"
    check_gpus_idle

    # Qwen on GPU 0, Llama on GPU 1, in parallel.
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/bounds/experiment2/diagnose_per_layer_norms.py \
        --model "$QWEN_MODEL" \
        --vector "$QWEN_VECTOR" \
        --out "$DIAGNOSTIC_DIR/diagnostic_qwen.json" \
        --num-prompts 100 \
        --batch-size 8 > "$DIAGNOSTIC_DIR/diagnostic_qwen.log" 2>&1 &
    QWEN_PID=$!

    # Llama's diagnostic saves all 32 layer outputs inside one trace,
    # which is much heavier than the main recording path (2 tensors).
    # Use batch size 1 to stay under the 44 GB VRAM budget even with
    # inference_mode fully disabling the grad graph.
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/bounds/experiment2/diagnose_per_layer_norms.py \
        --model "$LLAMA_MODEL" \
        --vector "$LLAMA_VECTOR" \
        --out "$DIAGNOSTIC_DIR/diagnostic_llama.json" \
        --num-prompts 100 \
        --batch-size 1 > "$DIAGNOSTIC_DIR/diagnostic_llama.log" 2>&1 &
    LLAMA_PID=$!

    echo "[phase1] launched qwen (PID $QWEN_PID) + llama (PID $LLAMA_PID)"
    wait "$QWEN_PID" || { echo "[phase1] qwen diagnostic failed"; tail -20 "$DIAGNOSTIC_DIR/diagnostic_qwen.log"; exit 1; }
    wait "$LLAMA_PID" || { echo "[phase1] llama diagnostic failed"; tail -20 "$DIAGNOSTIC_DIR/diagnostic_llama.log"; exit 1; }
    echo "[phase1] both diagnostics done"
fi

# ---------------------------------------------------------------------------
# Phase 2: pick layers
# ---------------------------------------------------------------------------

if [[ "$FROM_PHASE" -le 2 ]]; then
    phase_header 2 "pick layers (edge/middle/edge of target range)"

    uv run python scripts/bounds/experiment2/pick_layers.py \
        --qwen-diagnostic "$DIAGNOSTIC_DIR/diagnostic_qwen.json" \
        --qwen-target-layers "${QWEN_TARGET_LAYERS[@]}" \
        --llama-diagnostic "$DIAGNOSTIC_DIR/diagnostic_llama.json" \
        --llama-target-layers "${LLAMA_TARGET_LAYERS[@]}" \
        --out "$DIAGNOSTIC_DIR/layer_picks.json"
fi

# ---------------------------------------------------------------------------
# Phase 3: generate configs
# ---------------------------------------------------------------------------

if [[ "$FROM_PHASE" -le 3 ]]; then
    phase_header 3 "generate config YAMLs"

    uv run python scripts/bounds/experiment2/make_configs.py \
        --picks "$DIAGNOSTIC_DIR/layer_picks.json" \
        --out-dir "$CONFIGS_DIR"
fi

# ---------------------------------------------------------------------------
# Phase 4: verify all 14 configs
# ---------------------------------------------------------------------------

QWEN_CONFIGS=(
    "$CONFIGS_DIR/qwen_random_agg_matched.yaml"
)
LLAMA_CONFIGS=(
    "$CONFIGS_DIR/llama_random_agg_matched.yaml"
)
# Single-layer configs picked up dynamically (depends on layer picks in phase 2)
while IFS= read -r f; do QWEN_CONFIGS+=("$f"); done < <(ls -1 "$CONFIGS_DIR"/qwen_*_single_L*.yaml 2>/dev/null || true)
while IFS= read -r f; do LLAMA_CONFIGS+=("$f"); done < <(ls -1 "$CONFIGS_DIR"/llama_*_single_L*.yaml 2>/dev/null || true)

if [[ "$FROM_PHASE" -le 4 ]]; then
    phase_header 4 "verify all configs"
    check_gpus_idle

    verify_chain() {
        local gpu=$1; shift
        local configs=("$@")
        for cfg in "${configs[@]}"; do
            echo "[verify GPU$gpu] $(basename $cfg)"
            CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/bounds/01_verify_steering.py \
                --config "$cfg" --auto-escalate \
                >> "$DIAGNOSTIC_DIR/verify_gpu${gpu}.log" 2>&1 \
                || { echo "[verify GPU$gpu] FAILED on $(basename $cfg) — see $DIAGNOSTIC_DIR/verify_gpu${gpu}.log"; return 1; }
        done
    }

    : > "$DIAGNOSTIC_DIR/verify_gpu0.log"
    : > "$DIAGNOSTIC_DIR/verify_gpu1.log"

    verify_chain 0 "${QWEN_CONFIGS[@]}" &
    VERIFY_QWEN_PID=$!
    verify_chain 1 "${LLAMA_CONFIGS[@]}" &
    VERIFY_LLAMA_PID=$!

    wait "$VERIFY_QWEN_PID" || { echo "[phase4] qwen verification chain failed"; exit 1; }
    wait "$VERIFY_LLAMA_PID" || { echo "[phase4] llama verification chain failed"; exit 1; }
    echo "[phase4] all verifications passed"
fi

# ---------------------------------------------------------------------------
# Phase 5: record all 14 configs
# ---------------------------------------------------------------------------

if [[ "$FROM_PHASE" -le 5 ]]; then
    phase_header 5 "record all configs"
    check_gpus_idle

    record_chain() {
        local gpu=$1; shift
        local configs=("$@")
        for cfg in "${configs[@]}"; do
            local start_time=$(date +%s)
            echo "[record GPU$gpu] $(date +%T) $(basename $cfg)"
            CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/bounds/02_record_stats.py \
                --config "$cfg" \
                >> "$DIAGNOSTIC_DIR/record_gpu${gpu}.log" 2>&1 \
                || { echo "[record GPU$gpu] FAILED on $(basename $cfg) — see $DIAGNOSTIC_DIR/record_gpu${gpu}.log"; return 1; }
            local dt=$(( $(date +%s) - start_time ))
            echo "[record GPU$gpu] $(basename $cfg) done in ${dt}s"
        done
    }

    : > "$DIAGNOSTIC_DIR/record_gpu0.log"
    : > "$DIAGNOSTIC_DIR/record_gpu1.log"

    record_chain 0 "${QWEN_CONFIGS[@]}" &
    RECORD_QWEN_PID=$!
    record_chain 1 "${LLAMA_CONFIGS[@]}" &
    RECORD_LLAMA_PID=$!

    wait "$RECORD_QWEN_PID" || { echo "[phase5] qwen record chain failed"; exit 1; }
    wait "$RECORD_LLAMA_PID" || { echo "[phase5] llama record chain failed"; exit 1; }
    echo "[phase5] all records complete"
fi

# ---------------------------------------------------------------------------
# Phase 6: compute + visualize (CPU)
# ---------------------------------------------------------------------------

if [[ "$FROM_PHASE" -le 6 ]]; then
    phase_header 6 "compute bounds + render plots (CPU)"

    ALL_CONFIGS=("${QWEN_CONFIGS[@]}" "${LLAMA_CONFIGS[@]}")
    for cfg in "${ALL_CONFIGS[@]}"; do
        # Derive run_name from the YAML (first line usually "run_name: <name>").
        run_name=$(grep -E '^run_name:' "$cfg" | head -1 | awk '{print $2}')
        stats_path="outputs/bounds/${run_name}/stats.pt"
        metrics_path="outputs/bounds/${run_name}/bounds_metrics.json"
        echo "[compute] $run_name"
        uv run python scripts/bounds/03_compute.py --stats "$stats_path" --config "$cfg" > /dev/null \
            || { echo "[compute] FAILED for $run_name"; exit 1; }
        uv run python scripts/bounds/04_visualize.py --metrics "$metrics_path" > /dev/null \
            || { echo "[viz] FAILED for $run_name"; exit 1; }
    done
    echo "[phase6] all compute+viz done"
fi

# ---------------------------------------------------------------------------
# Phase 7: summary table
# ---------------------------------------------------------------------------

if [[ "$FROM_PHASE" -le 7 ]]; then
    phase_header 7 "summary table"

    uv run python scripts/bounds/experiment2/summarize.py \
        --configs-dir "$CONFIGS_DIR" \
        --out "$DIAGNOSTIC_DIR/summary.md" \
        || { echo "[phase7] summarize failed"; exit 1; }

    cat "$DIAGNOSTIC_DIR/summary.md"
fi

echo
echo "=============================================================="
echo "Experiment 2 complete at $(date)"
echo "=============================================================="
