# Gemma 4 26B-A4B QAT on 16 GB VRAM — benchmark report

**Model:** `hf.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL`  
**Digest:** `ec94fe5da7eb9c046c7b28fb2e1b132a0ebb602d5ab4ebbb6173305304c3ab3e`  
**Host:** Ollama `localhost:11434` · RTX 4080 Super 16 GB VRAM, i5 12400F in pair with DDR5 5600 MHz for offloading
**Date:** 2026-06-12  
**Artifacts:** `raw/` (see `results_summary.json`), `prompts/`, `scripts/`  
**Repository:** https://github.com/vplikin/Local-LLMs

---

## 1. Methodology

| Parameter | Value |
|-----------|-------|
| Inference API | Ollama `/api/generate`, streaming |
| Determinism (quality) | `temperature=0`, `top_p=1`, `seed=0` |
| Throughput reps | N≥5 (or 4 for long-context), **rep 0 = warmup, discarded** |
| Reported stats | median ± population stddev (min/max in JSON) |
| Metrics | `prefill tok/s = prompt_eval_count / prompt_eval_duration × 1e9`; same for decode; TTFT = wall-clock to first token |
| Quality reps | 3 per item; `unstable` = pass rate strictly between 0 and 1 |
| Grading | Programmatic (`graders/`); Gemma 4 `<\|channel>thought` tokens stripped before scoring |

**KV cache:** `OLLAMA_KV_CACHE_TYPE` is a **server** env var (`f16` \| `q8_0` \| `q4_0`, requires `OLLAMA_FLASH_ATTENTION=1`). This run used the server's default configuration — **mode not confirmed remotely** (labeled `unknown` in CSV). Comparison of `f16` vs `q8_0` was **not measured** in this session; restart Ollama per mode and re-run `run_all.sh`.

**GPU telemetry:** SSH to the inference host was unavailable. VRAM peaks come from polling `GET /api/ps` (`size_vram` vs `size`). Power/temperature **not measured**.

---

## 2. Environment

| Component | Value | Source |
|-----------|-------|--------|
| Ollama | **0.30.7** | `/api/version` |
| llama.cpp build | not exposed via API | — |
| NVIDIA driver / CUDA / OS | not measured (no SSH) | operator: Windows host, RTX 4080 Super 16 GB |
| Model on disk | 15.44 GB | `/api/tags` |
| GGUF quant label | `Q4_0` (Unsloth **UD-Q4_K_XL** dynamic) | `/api/show` → `details.quantization_level` |
| Architecture | `gemma4`, 30 layers, **8 active / 128 total experts**, sliding window **1024**, context **262144** | `/api/show` → `model_info` |
| Capabilities (Ollama) | `completion`, `vision` (mmproj present; text-only tests here) | `/api/tags` |

### Architecture cross-check (first sources)

| Claim | Official source | Ollama `/api/show` |
|-------|-----------------|---------------------|
| 26B total / ~4B active MoE | [Model card](https://ai.google.dev/gemma/docs/core/model_card_4): 25.2B total, 3.8B active, 8/128 experts | 25.2B params, 8/128 experts ✓ |
| Hybrid attention (sliding + global, last layer global) | [Model card § Architecture](https://ai.google.dev/gemma/docs/core/model_card_4) | `sliding_window=1024`; global layers not individually listed |
| Unified KV + p-RoPE on global layers | [Model card](https://ai.google.dev/gemma/docs/core/model_card_4) | not separately exposed in GGUF metadata |
| Context up to 256K | [Model card](https://ai.google.dev/gemma/docs/core/model_card_4) | `gemma4.context_length=262144` ✓ |
| 140+ languages | [Model card](https://ai.google.dev/gemma/docs/core/model_card_4) | not runtime-verifiable |
| QAT Q4_0 + mobile schema | [Google QAT blog](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/) | QAT GGUF loaded; `finetune=qat-it` |
| UD-Q4_K_XL beats naive q4_0 (Unsloth) | [Unsloth QAT docs](https://unsloth.ai/docs/models/gemma-4/qat): MMLU-style delta q4_0 70.2% vs UD-Q4_K_XL 85.6% on 26B | **not re-benchmarked here** |
| draft-MTP speculative decoding | [MTP overview](https://ai.google.dev/gemma/docs/mtp/overview), [Ollama PR #15980](https://github.com/ollama/ollama/pull/15980) | **N/A** — see §6 |

---

## 3. Throughput & memory vs context length

### 3.1 Short prompt (69 tokens) — context slot sizing

| num_ctx | prompt tok | prefill tok/s (med) | decode tok/s (med) | TTFT (med) | load (med) | peak VRAM / total* |
|---------|------------|---------------------|--------------------|-----------:|-----------:|-------------------|
| 8 192 | 69 | **1258 ± 44** | **125 ± 0.4** | 0.65 s | 0.52 s | 13.6 / 14.5 GiB |
| 65 536 | 69 | 254 ± 5 | **92 ± 0.5** | 25.7 s | 25.3 s | 12.4 / 14.5 GiB |

\*Peak from `/api/ps` polling (`size_vram` / `size`). At 64K slot, **~2.1 GiB of model+KV state sits off-GPU** (total > VRAM).

**Insight:** Raising `num_ctx` from 8K→64K with a tiny prompt cuts decode **125 → 92 tok/s** and adds **~25 s** load/TTFT — mostly KV-cache allocation and partial CPU offload, not prompt work.

### 3.2 Long prompt (~90% fill) — realistic prefill + decode

| num_ctx | prompt tok (actual) | prefill tok/s (med) | decode tok/s (med) | TTFT (med) | peak VRAM / total |
|---------|---------------------|---------------------|--------------------|-----------:|-------------------|
| 65 536 | 37 369 | ~**3400**† | **80 ± 2** | 11.8 s‡ | 12.4 / 14.6 GiB |
| 131 072 | 74 764 | **2412 ± 3** | **63 ± 0.1** | 78.3 s | 12.4 / 14.6 GiB |
| 262 144 | 149 509 | **1524 ± 8** | **44 ± 0.4** | 131.8 s | 12.4 / 14.9 GiB |

†Reps 1 and 3 at 64K showed **>400k tok/s** prefill — consistent with Ollama **prompt KV reuse** after identical prompts; credible cold prefill ≈ **3.4k tok/s** (reps 0, 2).  
‡TTFT for cached reps; cold prefill TTFT ≈ **65 s** (rep 0).

**Scaling:** Decode falls **125 → 44 tok/s** from 8K-short to 256K-long. Prefill at 150K tokens takes **~2.2 min** before the first decode token.

### 3.3 16 GB ceiling (key result)

1. **Weights alone ~14.6 GiB VRAM** at 8K — leaves ~1.4 GiB headroom on a 16 GB card.
2. From **64K context upward**, Ollama reports **`size` > `size_vram`** (~14.6 GiB total vs ~12.4 GiB VRAM) → **mandatory CPU/RAM offload** for KV + possibly layers.
3. **256K is achievable** but impractical for interactive use: TTFT **>2 min**, decode **~44 tok/s**.
4. **Practical home ceiling:** **~128K tokens** at ~63 tok/s decode with ~75 s prefill for ~75K-token prompts; beyond that, latency dominates.

KV cache quantization (`q8_0` / `q4_0`) was **not swept** — likely required to push usable context on 16 GB without offload; not measured here.

---

## 4. Needle-in-a-haystack (NIAH)

Haystack: neutral maritime filler (`niah.py`). Needle: `vault access code = ratatoskr-7741`. Depths: 0/10/25/50/75/90/100%.  
Multi-needle: 3 facts at 10/50/90% depths.

### Single-needle (exact match)

| depth % | 64K (37k tok) | 128K (75k tok) | 256K (150k tok) |
|--------:|:-------------:|:--------------:|:---------------:|
| 0 | ✓ | ✓ | ✓ |
| 10 | ✓ | ✓ | ✓ |
| 25 | ✓ | ✓ | ✓ |
| 50 | ✓ | ✓ | ✓ |
| 75 | ✓ | ✓ | ✓ |
| 90 | ✓ | ✓ | ✓ |
| 100 | ✓ | ✓ | ✓ |
| **Score** | **7/7** | **7/7** | **7/7** |

### Multi-needle (values found / 3)

| Context | Score |
|--------:|------:|
| 64K | 3/3 |
| 128K | 3/3 |
| 256K | **2/3** |

NIAH ≠ real-world long-document QA; synthetic exact-match only.

---

## 5. Base quality (num_ctx=8192, t=0, seed=0, 3 reps)

| Task | n | Metric | Score | Unstable |
|------|--:|--------|------:|---------:|
| GSM8K | 30 | accuracy | **96.7%** | 0 |
| HumanEval | 20 | pass@1 | **90.0%** | 0 |
| IFEval-style | 15 | constraint accuracy | **80.0%** | 0 |
| Multilingual | 10 | keyword/rubric | **100%** | 0 |

Prompts: `data/` (GSM8K/HumanEval), `graders/ifeval.py` (IFEval), `graders/multilingual.py`.

No prior models in this repo for apples-to-apples comparison (§4 skipped).

---

## 6. Speculative decoding (draft-MTP)

**Status: N/A**

- Loaded artifact is a **single GGUF** without `DRAFT` / MTP head.
- Ollama MTP requires a Modelfile pairing target + draft safetensors ([Ollama PR #15980](https://github.com/ollama/ollama/pull/15980)); server tags list no Gemma 4 MTP variant for this QAT quant.
- Unsloth ships separate `*MTP*` files on [HF GGUF repo](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) — not pulled into this Ollama instance.

Optional llama.cpp path (`--spec-type draft-mtp`) not exercised — out of scope for this Ollama-focused run.

---

## 7. Conclusions (consumer 16 GB)

| Question | Answer |
|----------|--------|
| Does UD-Q4_K_XL QAT fit on 16 GB? | **Yes** for inference at ≤8K context (~14.6 GiB VRAM). |
| Usable long context at home? | **~64–128K** with CPU offload; **256K runs** but **44 tok/s** decode and **multi-minute** prefills. |
| Retrieval quality (NIAH)? | **Perfect single-needle** through 150K prompt tokens; **one multi-needle miss** at 256K. |
| Quality vs size? | Strong on small samples (GSM8K 97%, HumanEval 90%) — not comparable to Google bench tables. |
| Biggest home-GPU bottleneck? | **KV cache + full 26B weight residency**, not active 4B compute — matches [Google docs](https://ai.google.dev/gemma/docs/core): all 26B must reside in memory. |

---

## 8. Threats to validity

1. **Small samples** — 30 GSM8K / 20 HumanEval / 15 IFEval / 10 multilingual; wide confidence intervals.
2. **`t=0` only** — production uses `temperature=1, top_p=0.95, top_k=64` per [model card](https://ai.google.dev/gemma/docs/core/model_card_4).
3. **NIAH ≠ comprehension** — single-fact extraction in synthetic prose.
4. **KV cache mode unknown** — no `f16` vs `q8_0` sweep; offload confounds throughput.
5. **Prompt caching** — duplicate long prompts inflate prefill tok/s (observed at 64K).
6. **Ollama vs raw llama.cpp** — different schedulers, offload heuristics.
7. **MTP / multimodal untested** — vision mmproj present but not evaluated.
8. **No GPU thermals/power** — sustained loads may throttle; not measured.
9. **Gemma 4 template overhead** — empty `<|channel>thought` block on every reply; graders strip it but latency impact remains.
10. **Remote host metadata** — driver/CUDA/OS unverified from benchmark client.

---

## 9. References

- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 overview / MoE memory note](https://ai.google.dev/gemma/docs/core)
- [QAT release blog](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)
- [MTP / speculative decoding](https://ai.google.dev/gemma/docs/mtp/overview)
- [google/gemma-4-26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B)
- [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF)
- [Unsloth UD-Q4_K_XL vs q4_0 claims](https://unsloth.ai/docs/models/gemma-4/qat)
- [Ollama Gemma4 MTP PR #15980](https://github.com/ollama/ollama/pull/15980)
