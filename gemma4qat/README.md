# Gemma 4 26B-A4B QAT — local benchmark (16 GB VRAM)

Reproducible harness for `hf.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL` on Ollama.

## Layout

```
gemma4qat/
├── posts/         # LinkedIn post drafts (EN + RU)
├── raw/           # JSON/CSV measurements (machine-readable)
├── prompts/       # verbatim test prompts
├── data/          # GSM8K + HumanEval subsets (seed=42)
├── scripts/       # runners and graders
├── report.md      # full engineering report
└── README.md
```

## Prerequisites

```bash
pip install -r scripts/requirements.txt
# Ollama reachable (default localhost:11434; set OLLAMA_HOST for remote)
export OLLAMA_HOST="${OLLAMA_HOST:-localhost:11434}"
ollama show hf.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL
```

## KV cache mode (server-side)

`num_ctx` is per-request; KV cache quantization is set when **Ollama starts**:

```bash
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

Restart the server for each mode you want to compare (`f16`, `q8_0`, `q4_0`). Label runs with `--kv-cache q8_0` so artifacts match the server config.

## One-shot reproduction

```bash
cd scripts

# 1) Environment snapshot
python3 capture_env.py

# 2) Datasets (GSM8K n=30, HumanEval n=20, seed=42)
python3 prepare_data.py

# 3) Throughput at 8K / 64K / 128K / 256K
python3 bench.py --prompt-file ../prompts/bench_prompt.txt --num-ctx 8192 --label baseline8192
python3 make_long_prompt.py --num-ctx 65536 --out ../prompts/bench_prompt_65536.txt
python3 bench.py --prompt-file ../prompts/bench_prompt_65536.txt --num-ctx 65536 --label sweep_long

# 4) NIAH long-context retrieval
python3 niah.py --num-ctx 65536 --mode single
python3 niah.py --num-ctx 65536 --mode multi

# 5) Quality (num_ctx=8192, temperature=0, seed=0, 3 reps)
python3 run_quality.py --task all \
  --gsm8k-path ../data/gsm8k.jsonl \
  --humaneval-path ../data/humaneval.jsonl

# Full sweep (after setting OLLAMA_KV_CACHE_TYPE on the server)
./run_all.sh
```

## VRAM polling

If `nvidia-smi` is not available on the client, poll Ollama instead:

```bash
HOST=localhost:11434 ./ollama_ps_poll.sh ../raw/ps_run.csv 500 &
# run benchmark…
kill %1
awk -F, 'NR>1{if($2>m)m=$2}END{print m/1024/1024" GiB VRAM"}' ../raw/ps_run.csv
```

## HumanEval warning

`run_quality.py` executes model-generated Python. Use an isolated environment only.

## Determinism

Quality runs: `temperature=0`, `seed=0`, fixed `num_predict`. Gemma 4 emits `<|channel>thought` template tokens; graders strip them before scoring (see `graders/strip.py`).
