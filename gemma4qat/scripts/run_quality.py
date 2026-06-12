import argparse
import csv
import json
import os
import statistics
import time

from ollama_client import generate
from graders import gsm8k, humaneval, ifeval, multilingual


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def call(host, model, prompt, num_predict, seed):
    out = generate(host, model, prompt, num_ctx=8192,
                   num_predict=num_predict, seed=seed)
    return out.get("response", "")


def run_gsm8k(host, model, path, n, reps, seed):
    items = load_jsonl(path)[:n]
    per_item = []
    for it in items:
        prompt = (f"{it['question']}\n\n"
                  "Solve step by step, then give the final numeric answer "
                  "on the last line as: #### <number>")
        oks = [gsm8k.grade(call(host, model, prompt, 512, seed + r), it["answer"])
               for r in range(reps)]
        per_item.append(sum(oks) / len(oks))
    return {"task": "gsm8k", "n": len(items), "reps": reps,
            "accuracy": statistics.fmean(per_item) if per_item else None,
            "unstable": sum(1 for x in per_item if 0 < x < 1)}


def run_humaneval(host, model, path, n, reps, seed):
    items = load_jsonl(path)[:n]
    per_item = []
    for it in items:
        prompt = (f"Complete this Python function. Return only the full function "
                  f"in a python code block.\n\n{it['prompt']}")
        oks = []
        for r in range(reps):
            code = humaneval.extract_code(call(host, model, prompt, 768, seed + r))
            oks.append(humaneval.check(code, it["test"], it["entry_point"]))
        per_item.append(sum(oks) / len(oks))
    return {"task": "humaneval", "n": len(items), "reps": reps,
            "pass@1": statistics.fmean(per_item) if per_item else None,
            "unstable": sum(1 for x in per_item if 0 < x < 1)}


def run_multilingual(host, model, reps, seed):
    per_case = []
    for case in multilingual.ALL:
        oks = [multilingual.grade_case(
            case["id"], call(host, model, case["prompt"], 128, seed + r))
               for r in range(reps)]
        per_case.append(sum(oks) / len(oks))
    return {"task": "multilingual", "n": len(multilingual.ALL), "reps": reps,
            "accuracy": statistics.fmean(per_case) if per_case else None,
            "unstable": sum(1 for x in per_case if 0 < x < 1)}


def run_ifeval(host, model, reps, seed):
    per_case = []
    for case in ifeval.CASES:
        oks = [ifeval.grade(case["id"], call(host, model, case["prompt"], 256, seed + r))
               for r in range(reps)]
        per_case.append(sum(oks) / len(oks))
    return {"task": "ifeval", "n": len(ifeval.CASES), "reps": reps,
            "accuracy": statistics.fmean(per_case) if per_case else None,
            "unstable": sum(1 for x in per_case if 0 < x < 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "localhost:11434"))
    ap.add_argument("--model",
                    default="hf.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL")
    ap.add_argument("--task",
                    choices=["gsm8k", "humaneval", "ifeval", "multilingual", "all"],
                    default="all")
    ap.add_argument("--gsm8k-path", help="JSONL with 'question' and 'answer' fields")
    ap.add_argument("--humaneval-path",
                    help="JSONL with 'prompt','test','entry_point' fields")
    ap.add_argument("--gsm8k-n", type=int, default=30)
    ap.add_argument("--humaneval-n", type=int, default=20)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="../raw")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    if args.task in ("gsm8k", "all"):
        if not args.gsm8k_path:
            print("skip gsm8k: --gsm8k-path not provided")
        else:
            results.append(run_gsm8k(args.host, args.model, args.gsm8k_path,
                                     args.gsm8k_n, args.reps, args.seed))
    if args.task in ("humaneval", "all"):
        if not args.humaneval_path:
            print("skip humaneval: --humaneval-path not provided")
        else:
            results.append(run_humaneval(args.host, args.model, args.humaneval_path,
                                         args.humaneval_n, args.reps, args.seed))
    if args.task in ("ifeval", "all"):
        results.append(run_ifeval(args.host, args.model, args.reps, args.seed))
    if args.task in ("multilingual", "all"):
        results.append(run_multilingual(args.host, args.model, args.reps, args.seed))

    for r in results:
        print(r)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    with open(f"{args.out_dir}/quality_{stamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    csv_path = f"{args.out_dir}/quality_summary.csv"
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["task", "n", "reps", "score", "metric", "unstable"])
        for r in results:
            metric = ("pass@1" if "pass@1" in r else "accuracy")
            w.writerow([r["task"], r["n"], r["reps"], r.get(metric), metric,
                        r.get("unstable")])
    print(f"wrote {args.out_dir}/quality_{stamp}.json")


if __name__ == "__main__":
    main()
