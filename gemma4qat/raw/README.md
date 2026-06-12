# Raw benchmark artifacts

Machine-readable outputs from the Gemma 4 QAT run (2026-06-12).

| File | Description |
|------|-------------|
| `results_summary.json` | Curated headline metrics (start here) |
| `environment.json` | Ollama + model metadata (sanitized for publication) |
| `bench_summary.csv` | Throughput runs aggregated |
| `quality_summary.csv` | Quality task scores |
| `niah_matrix.csv` | Needle-in-a-haystack depth × context |
| `*_ctx*.json` | Per-run JSON with all reps (warmup flagged) |
| `ps_ctx*.csv` | VRAM polling via `GET /api/ps` |

Regenerate with scripts in `../scripts/`. Set `OLLAMA_HOST=host:11434` if Ollama is remote.

Superseded artifacts (broken Gemma 4 grader before `strip.py`) were removed before publication.
