#!/usr/bin/env bash
# Poll Ollama /api/ps for VRAM usage when nvidia-smi is not reachable locally.
# Usage: ./ollama_ps_poll.sh ../raw/ps_poll.csv 500 &
set -euo pipefail
HOST="${OLLAMA_HOST:-${HOST:-localhost:11434}}"
out="${1:-../raw/ps_poll_$(date +%Y%m%d-%H%M%S).csv}"
ms="${2:-500}"
interval="$(awk "BEGIN{print $ms/1000}")"
echo "timestamp,size_vram_bytes,size_total_bytes,context_length,expires_at" > "$out"
while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  curl -s "http://${HOST}/api/ps" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for m in d.get('models', []):
    print(f\"{sys.argv[1]},{m.get('size_vram',0)},{m.get('size',0)},{m.get('context_length',0)},{m.get('expires_at','')}\")
" "$ts" >> "$out" 2>/dev/null || true
  sleep "$interval"
done
