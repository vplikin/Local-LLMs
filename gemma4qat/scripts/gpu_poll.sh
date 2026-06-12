#!/usr/bin/env bash
# Run before a bench, kill after (saved PID or Ctrl-C). Peak VRAM = max(mem_used).
#   ./gpu_poll.sh ../raw/gpu_run.csv 200 &  ; GP=$! ; ... ; kill $GP
#   peak: awk -F, 'NR>1{if($2>m)m=$2}END{print m" MiB"}' ../raw/gpu_run.csv
set -euo pipefail
out="${1:-../raw/gpu_$(date +%Y%m%d-%H%M%S).csv}"
ms="${2:-200}"
interval="$(awk "BEGIN{print $ms/1000}")"
echo "timestamp,mem_used_mib,mem_total_mib,power_w,temp_c,util_pct" > "$out"
while true; do
  nvidia-smi \
    --query-gpu=timestamp,memory.used,memory.total,power.draw,temperature.gpu,utilization.gpu \
    --format=csv,noheader,nounits >> "$out"
  sleep "$interval"
done
