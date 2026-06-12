#!/usr/bin/env bash
# Full sweep. KV cache type is a SERVER setting, not per-request: restart Ollama with
# the desired env before each KV-mode pass, e.g.
#   OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
# then run this with KV=q8_0 so artifacts are labeled correctly.
set -euo pipefail

HOST="${OLLAMA_HOST:-${HOST:-localhost:11434}}"
MODEL="${MODEL:-hf.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL}"
KV="${OLLAMA_KV_CACHE_TYPE:-f16}"
PROMPT_FILE="${PROMPT_FILE:-../prompts/bench_prompt.txt}"
CONTEXTS="${CONTEXTS:-65536 131072 262144}"

mkdir -p ../raw

for ctx in $CONTEXTS; do
  echo "=== context $ctx (kv=$KV) ==="
  gpu_csv="../raw/gpu_ctx${ctx}_${KV}.csv"
  ./gpu_poll.sh "$gpu_csv" 200 & GP=$!
  trap 'kill $GP 2>/dev/null || true' EXIT

  python3 bench.py --host "$HOST" --model "$MODEL" --prompt-file "$PROMPT_FILE" \
    --num-ctx "$ctx" --kv-cache "$KV" --label sweep || echo "bench failed at $ctx (likely VRAM)"

  python3 niah.py --host "$HOST" --model "$MODEL" --num-ctx "$ctx" --mode single \
    || echo "niah single failed at $ctx"
  python3 niah.py --host "$HOST" --model "$MODEL" --num-ctx "$ctx" --mode multi \
    || echo "niah multi failed at $ctx"

  kill $GP 2>/dev/null || true
  trap - EXIT
  peak=$(awk -F, 'NR>1{if($2>m)m=$2}END{print m}' "$gpu_csv")
  echo "ctx $ctx peak VRAM: ${peak} MiB"
done

echo "=== quality (ctx 8192) ==="
python3 run_quality.py --host "$HOST" --model "$MODEL" --task all \
  ${GSM8K_PATH:+--gsm8k-path "$GSM8K_PATH"} \
  ${HUMANEVAL_PATH:+--humaneval-path "$HUMANEVAL_PATH"} \
  || echo "quality run had errors"

echo "done. raw artifacts in ../raw"
