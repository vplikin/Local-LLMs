import argparse
import csv
import json
import os
import statistics
import time

from ollama_client import generate, metrics


def agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {}
    return {
        "median": statistics.median(vals),
        "mean": statistics.fmean(vals),
        "stddev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "localhost:11434"))
    ap.add_argument("--model",
                    default="hf.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--num-ctx", type=int, required=True)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--reps", type=int, default=6)  # rep 0 is warmup, discarded
    ap.add_argument("--seed", type=int, default=0)
    # KV cache type is set on the server (OLLAMA_KV_CACHE_TYPE); recorded here only.
    ap.add_argument("--kv-cache", default=os.environ.get("OLLAMA_KV_CACHE_TYPE", "f16"))
    ap.add_argument("--label", default="run")
    ap.add_argument("--out-dir", default="../raw")
    args = ap.parse_args()

    prompt = open(args.prompt_file, encoding="utf-8").read()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = []
    for i in range(args.reps):
        d = generate(args.host, args.model, prompt, args.num_ctx,
                     num_predict=args.num_predict, seed=args.seed)
        m = metrics(d)
        m["rep"] = i
        m["warmup"] = (i == 0)
        runs.append(m)
        pf = m["prefill_tok_s"] or 0
        dc = m["decode_tok_s"] or 0
        tt = m["ttft_s"] or 0
        tag = " (warmup)" if i == 0 else ""
        print(f"rep {i}{tag}: prefill={pf:.1f} decode={dc:.1f} ttft={tt:.3f}s "
              f"prompt_tok={m['prompt_tokens']}")

    measured = [r for r in runs if not r["warmup"]]
    summary = {
        "label": args.label,
        "model": args.model,
        "num_ctx": args.num_ctx,
        "kv_cache": args.kv_cache,
        "num_predict": args.num_predict,
        "reps_measured": len(measured),
        "prompt_tokens": measured[0]["prompt_tokens"] if measured else None,
        "prefill_tok_s": agg([r["prefill_tok_s"] for r in measured]),
        "decode_tok_s": agg([r["decode_tok_s"] for r in measured]),
        "ttft_s": agg([r["ttft_s"] for r in measured]),
        "load_s": agg([r["load_s"] for r in measured]),
        "total_s": agg([r["total_s"] for r in measured]),
    }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{args.out_dir}/{args.label}_ctx{args.num_ctx}_{args.kv_cache}_{stamp}"
    with open(base + ".json", "w") as f:
        json.dump({"summary": summary, "runs": runs}, f, indent=2)

    csv_path = f"{args.out_dir}/bench_summary.csv"
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "num_ctx", "kv_cache", "num_predict", "prompt_tokens",
                        "prefill_med", "prefill_std", "decode_med", "decode_std",
                        "ttft_med", "ttft_std", "load_med", "n"])

        def g(k, s):
            return summary[k].get(s)

        w.writerow([args.label, args.num_ctx, args.kv_cache, args.num_predict,
                    summary["prompt_tokens"],
                    g("prefill_tok_s", "median"), g("prefill_tok_s", "stddev"),
                    g("decode_tok_s", "median"), g("decode_tok_s", "stddev"),
                    g("ttft_s", "median"), g("ttft_s", "stddev"),
                    g("load_s", "median"), len(measured)])
    print("wrote", base + ".json")


if __name__ == "__main__":
    main()
