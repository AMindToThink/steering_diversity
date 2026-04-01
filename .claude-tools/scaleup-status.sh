#!/usr/bin/env bash
set -euo pipefail

# Check status of the QwQ scaleup run: log tail, process, GPU usage.
# Usage: .claude-tools/scaleup-status.sh [log-file] [pid]
#   log-file: path to the scaleup log (default: /tmp/qwq_scaleup.log)
#   pid: PID of the main script (default: auto-detect from pgrep)

LOG_FILE="${1:-/tmp/qwq_scaleup.log}"
SCRIPT_PID="${2:-}"

# --- Validate inputs ---
if [[ ! -f "${LOG_FILE}" ]]; then
    echo "Error: log file not found: ${LOG_FILE}" >&2
    exit 1
fi

# Auto-detect PID if not provided
if [[ -z "${SCRIPT_PID}" ]]; then
    SCRIPT_PID=$(pgrep -f "run_qwq_scaleup.sh" 2>/dev/null | head -1 || true)
fi

echo "=== Scaleup Status ==="
echo ""

# Process status
echo "--- Process ---"
if [[ -n "${SCRIPT_PID}" ]]; then
    if ps -p "${SCRIPT_PID}" -o pid,etime,args --no-headers 2>/dev/null; then
        true
    else
        echo "PID ${SCRIPT_PID} not running (script may have finished)"
    fi
else
    echo "No running scaleup process found"
fi
echo ""

# GPU status
echo "--- GPU ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || echo "No GPU processes"
echo ""

# Log tail
echo "--- Log (last 10 lines) ---"
tail -10 "${LOG_FILE}"
echo ""

# Disk space
echo "--- Disk ---"
df -h / | tail -1 | awk '{print "Free: " $4 " (" $5 " used)"}'
