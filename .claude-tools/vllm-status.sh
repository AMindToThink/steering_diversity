#!/usr/bin/env bash
set -euo pipefail

# Check vLLM server status: process alive, GPU usage, API health, last log lines
# Usage: .claude-tools/vllm-status.sh [log-file] [port]

LOG_FILE="${1:-/tmp/vllm_qwq_unsteered.log}"
PORT="${2:-8017}"

echo "=== vLLM Process ==="
if pgrep -f "vllm.entrypoints" > /dev/null 2>&1; then
    pgrep -af "vllm.entrypoints" | head -3
else
    echo "No vLLM process found"
fi

echo ""
echo "=== GPU Memory ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null || echo "No GPU processes"
    echo "---"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null
else
    echo "nvidia-smi not available"
fi

echo ""
echo "=== API Health (port $PORT) ==="
HTTP_CODE=$(curl -s -o /tmp/.vllm_status_resp -w "%{http_code}" "http://localhost:${PORT}/v1/models" 2>/dev/null || echo "000")
if [[ "$HTTP_CODE" == "200" ]]; then
    echo "SERVING (HTTP 200)"
    python3 -c "import json; d=json.load(open('/tmp/.vllm_status_resp')); [print(f'  Model: {m[\"id\"]}') for m in d.get('data',[])]"
elif [[ "$HTTP_CODE" == "000" ]]; then
    echo "NOT RESPONDING (connection refused or timeout)"
else
    echo "HTTP $HTTP_CODE"
fi

echo ""
echo "=== Last 5 Log Lines ==="
if [[ -f "$LOG_FILE" ]]; then
    tail -5 "$LOG_FILE" | sed 's/\x1b\[[0-9;]*m//g'
else
    echo "Log file not found: $LOG_FILE"
fi

echo ""
echo "=== Disk Space ==="
df -h / | tail -1
