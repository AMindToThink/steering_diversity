#!/usr/bin/env bash
set -euo pipefail

# Start a QwQ-32B vLLM server with thinking disabled
# Usage: .claude-tools/start-qwq-server.sh <model_path_or_name> <served_name> [log_file] [port]

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model_path_or_name> <served_name> [log_file=/tmp/vllm_qwq.log] [port=8017]" >&2
    exit 1
fi

MODEL="$1"
SERVED_NAME="$2"
LOG_FILE="${3:-/tmp/vllm_qwq.log}"
PORT="${4:-8017}"

# Check no existing server on this port
if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "200"; then
    echo "ERROR: Server already running on port ${PORT}" >&2
    exit 1
fi

# Check GPUs are available
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if (( GPU_PROCS > 0 )); then
    echo "ERROR: ${GPU_PROCS} GPU processes already running" >&2
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null
    exit 1
fi

echo "Starting vLLM: model=${MODEL}, name=${SERVED_NAME}, port=${PORT}"
echo "  Thinking: disabled (enable_thinking=false)"
echo "  Log: ${LOG_FILE}"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --dtype float16 \
    --max-model-len 32768 \
    --chat-template "$(cd "$(dirname "$0")/.." && pwd)/configs/qwq_no_thinking.jinja" \
    > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "  PID: ${SERVER_PID}"

# Wait for server to be ready
echo "  Waiting for server..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
        echo "  Server ready after $((i*5))s"
        exit 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: Server died during startup. Last 20 lines:" >&2
        tail -20 "${LOG_FILE}" >&2
        exit 1
    fi
    sleep 5
done
echo "ERROR: Server not ready after 10 minutes" >&2
tail -20 "${LOG_FILE}" >&2
exit 1
