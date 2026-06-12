# Local-LLMs

Reproducible benchmarks of open-weight language models on **consumer hardware** — honest numbers, raw JSON, one-command reruns.

Principles: measured metrics only, small fixed evaluation sets, warmup discarded, threats to validity documented.

## Benchmarks

| Model | Quant | VRAM | Report | Key result |
|-------|-------|------|--------|------------|
| [Gemma 4 26B-A4B IT QAT](gemma4qat/) | Unsloth UD-Q4_K_XL | 16 GB (RTX 4080 Super) | [report.md](gemma4qat/report.md) | **125 tok/s** decode @ 8K → **44 tok/s** @ 256K; CPU offload from 64K |

## Repository layout

```
Local-LLMs/
├── README.md                 ← you are here
└── gemma4qat/
    ├── report.md             ← full engineering report
    ├── posts/linkedin.md     ← EN + RU post drafts
    ├── raw/                  ← JSON/CSV measurements
    ├── prompts/              ← verbatim test prompts
    ├── data/                 ← GSM8K + HumanEval subsets (seed=42)
    └── scripts/              ← runners + graders
```

## Headline numbers (Gemma 4 QAT, own measurements)

See [`gemma4qat/raw/results_summary.json`](gemma4qat/raw/results_summary.json) for artifact pointers.

| Context | Decode tok/s (med) | Notes |
|--------:|-------------------:|-------|
| 8K | 125 | ~14.6 GiB VRAM, short prompt |
| 64K | 80 | ~2 GiB CPU offload, long prompt |
| 128K | 63 | TTFT ~78 s |
| 256K | 44 | TTFT ~132 s |

Quality @ 8K (`t=0`, seed=0): GSM8K **96.7%** · HumanEval **90%** · IFEval **80%** · Multilingual **100%** (small samples).
