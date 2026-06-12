#!/usr/bin/env python3
"""Download fixed GSM8K + HumanEval subsets for reproducible quality runs."""
import argparse
import json
import random
import urllib.request


def fetch_gsm8k(n, seed, out):
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    lines = urllib.request.urlopen(url, timeout=60).read().decode().strip().splitlines()
    items = [json.loads(l) for l in lines]
    rng = random.Random(seed)
    rng.shuffle(items)
    subset = items[:n]
    with open(out, "w") as f:
        for it in subset:
            f.write(json.dumps({"question": it["question"], "answer": it["answer"]}) + "\n")
    print(f"gsm8k: {len(subset)} items -> {out}")


def fetch_humaneval(n, seed, out):
    url = ("https://raw.githubusercontent.com/openai/human-eval/master/"
           "data/HumanEval.jsonl.gz")
    import gzip, io
    raw = gzip.decompress(urllib.request.urlopen(url, timeout=60).read())
    items = [json.loads(l) for l in raw.decode().strip().splitlines()]
    rng = random.Random(seed)
    rng.shuffle(items)
    subset = items[:n]
    with open(out, "w") as f:
        for it in subset:
            f.write(json.dumps({
                "prompt": it["prompt"],
                "test": it["test"],
                "entry_point": it["entry_point"],
            }) + "\n")
    print(f"humaneval: {len(subset)} items -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../data")
    ap.add_argument("--gsm8k-n", type=int, default=30)
    ap.add_argument("--humaneval-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    fetch_gsm8k(args.gsm8k_n, args.seed, f"{args.out_dir}/gsm8k.jsonl")
    fetch_humaneval(args.humaneval_n, args.seed, f"{args.out_dir}/humaneval.jsonl")


if __name__ == "__main__":
    main()
