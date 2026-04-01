#!/usr/bin/env bash
set -euo pipefail

# Send a test prompt to vLLM and show response with token stats and optional logprobs
# Usage: .claude-tools/vllm-test-prompt.sh <prompt> [model] [max_tokens] [temperature] [logprobs]

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <prompt> [model=qwq-32b] [max_tokens=500] [temperature=0] [logprobs=false]" >&2
    exit 1
fi

PROMPT="$1"
MODEL="${2:-qwq-32b}"
MAX_TOKENS="${3:-500}"
TEMPERATURE="${4:-0}"
LOGPROBS="${5:-false}"
PORT="${VLLM_PORT:-8017}"

# Check server is responding
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/v1/models" 2>/dev/null || echo "000")
if [[ "$HTTP_CODE" != "200" ]]; then
    echo "Error: vLLM server not responding on port $PORT (HTTP $HTTP_CODE)" >&2
    exit 1
fi

# Build logprobs JSON fragment
if [[ "$LOGPROBS" == "true" ]]; then
    LP_JSON='"logprobs": true, "top_logprobs": 5,'
else
    LP_JSON=""
fi

# Send request
RESPONSE=$(curl -s "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": $(python3 -c "import json; print(json.dumps('${PROMPT//\'/\\\'}'))")}],
        \"max_tokens\": ${MAX_TOKENS},
        \"temperature\": ${TEMPERATURE},
        ${LP_JSON}
        \"stream\": false
    }" 2>/dev/null)

# Write response to temp file for Python to read
TMPFILE=$(mktemp /tmp/.vllm_test_XXXXXX.json)
trap 'rm -f "$TMPFILE"' EXIT
echo "$RESPONSE" > "$TMPFILE"

# Parse and display
export _VLLM_RESP="$TMPFILE"
export _SHOW_LOGPROBS="$LOGPROBS"
python3 << 'PYTHON_END'
import json, os

with open(os.environ["_VLLM_RESP"]) as f:
    resp = json.load(f)
show_logprobs = os.environ.get("_SHOW_LOGPROBS") == "true"

if "error" in resp:
    print(f"ERROR: {resp['error']}", file=sys.stderr)
    sys.exit(1)

c = resp["choices"][0]
text = c["message"]["content"]
usage = resp["usage"]

print(f"Model: {resp.get('model', 'unknown')}")
print(f"Finish: {c['finish_reason']}")
print(f"Prompt tokens: {usage['prompt_tokens']}, Completion tokens: {usage['completion_tokens']}")
print(f"Response ({len(text)} chars):")
print("---")
print(text[:2000])
if len(text) > 2000:
    print(f"\n... ({len(text) - 2000} chars truncated)")

if show_logprobs:
    lps = c.get("logprobs", {}).get("content", [])
    if lps:
        print("\n=== Logprobs (first 20 tokens) ===")
        for i, tok in enumerate(lps[:20]):
            top = tok.get("top_logprobs", [])
            top_str = ", ".join(f"{t['token']}:{t['logprob']:.4f}" for t in top[:3])
            print(f"  [{i}] {tok['token']!r} lp={tok['logprob']:.4f}  top3: {top_str}")
PYTHON_END
