#!/usr/bin/env python3
"""Generate a filler prompt targeting ~90% of num_ctx tokens."""
import argparse

FILLER = (
    "The history of maritime navigation spans many generations of incremental "
    "engineering work. Crews refined hull forms, rigging arrangements, and "
    "instruments through trial and error long before the underlying theory was "
    "written down. Ports grew around these practices and trade followed. "
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-ctx", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fraction", type=float, default=0.9)
    args = ap.parse_args()
    target = int(args.num_ctx * args.fraction)
    n = max(1, target // max(1, len(FILLER) // 4))
    text = (FILLER * n) + "\n\nSummarize the above in one sentence."
    with open(args.out, "w") as f:
        f.write(text)
    print(f"wrote {args.out} (~{len(text)//4} est tokens, target {target})")


if __name__ == "__main__":
    main()
