import argparse
import csv
import json
import os
import time

from ollama_client import generate
from graders.strip import strip

# Neutral, answer-free filler. Must not contain digits or words that could leak a needle.
FILLER = (
    "The history of maritime navigation spans many generations of incremental "
    "engineering work. Crews refined hull forms, rigging arrangements, and "
    "instruments through trial and error long before the underlying theory was "
    "written down. Ports grew around these practices and trade followed. "
)


def approx_tokens(text):
    return max(1, len(text) // 4)  # ~4 chars/token; calibrate against prompt_eval_count


def build_haystack(target_tokens):
    n = max(1, target_tokens // approx_tokens(FILLER))
    return FILLER * n


def insert_at_depth(haystack, sentence, depth_pct):
    words = haystack.split(" ")
    pos = min(len(words), int(len(words) * depth_pct / 100))
    words[pos:pos] = sentence.split(" ")
    return " ".join(words)


def run_single(host, model, num_ctx, depths, key, value):
    target = int(num_ctx * 0.9)
    needle = f"Important fact: the {key} is {value}."
    rows = []
    for d in depths:
        ctx = insert_at_depth(build_haystack(target), needle, d)
        prompt = (f"{ctx}\n\nQuestion: What is the {key}? "
                  f"Reply with the value only, nothing else.")
        out = generate(host, model, prompt, num_ctx, num_predict=32, seed=0)
        ans = strip(out.get("response", "")).strip()
        ok = value.lower() in ans.lower()
        rows.append({
            "mode": "single", "num_ctx": num_ctx, "depth_pct": d,
            "found": ok, "answer": ans[:160],
            "prompt_tokens": out.get("prompt_eval_count"),
        })
        print(f"  ctx={num_ctx} depth={d:>3}%  {'OK' if ok else 'MISS'}  "
              f"({out.get('prompt_eval_count')} tok)")
    return rows


def run_multi(host, model, num_ctx, depths, facts):
    # facts: list of (key, value). Each placed at a separate depth, all asked at once.
    target = int(num_ctx * 0.9)
    hs = build_haystack(target)
    pairs = list(zip(facts, depths))
    for (key, value), d in pairs:
        hs = insert_at_depth(hs, f"Important fact: the {key} is {value}.", d)
    q = " ".join(f"What is the {k}?" for k, _ in facts)
    prompt = (f"{hs}\n\nQuestions: {q} "
              f"Answer each with 'key=value' on its own line.")
    out = generate(host, model, prompt, num_ctx, num_predict=128, seed=0)
    ans = strip(out.get("response", ""))
    found = sum(1 for _, v in facts if v.lower() in ans.lower())
    print(f"  ctx={num_ctx} multi: {found}/{len(facts)} found "
          f"({out.get('prompt_eval_count')} tok)")
    return [{
        "mode": "multi", "num_ctx": num_ctx, "depth_pct": None,
        "found": found, "total": len(facts), "answer": ans[:300],
        "prompt_tokens": out.get("prompt_eval_count"),
    }]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "localhost:11434"))
    ap.add_argument("--model",
                    default="hf.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL")
    ap.add_argument("--num-ctx", type=int, required=True)
    ap.add_argument("--mode", choices=["single", "multi"], default="single")
    ap.add_argument("--depths", default="0,10,25,50,75,90,100")
    ap.add_argument("--out-dir", default="../raw")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    depths = [int(x) for x in args.depths.split(",")]

    if args.mode == "single":
        rows = run_single(args.host, args.model, args.num_ctx, depths,
                          key="vault access code", value="ratatoskr-7741")
    else:
        facts = [("vault access code", "ratatoskr-7741"),
                 ("relay callsign", "anatr0p-zulu"),
                 ("checkpoint phrase", "lord-yorkshire-9")]
        rows = run_multi(args.host, args.model, args.num_ctx,
                         depths[:len(facts)] or [10, 50, 90], facts)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{args.out_dir}/niah_{args.mode}_ctx{args.num_ctx}_{stamp}"
    with open(base + ".json", "w") as f:
        json.dump(rows, f, indent=2)

    csv_path = f"{args.out_dir}/niah_matrix.csv"
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["mode", "num_ctx", "depth_pct", "found", "prompt_tokens"])
        for r in rows:
            w.writerow([r["mode"], r["num_ctx"], r.get("depth_pct"),
                        r["found"], r["prompt_tokens"]])
    print("wrote", base + ".json")


if __name__ == "__main__":
    main()
