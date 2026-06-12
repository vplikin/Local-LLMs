#!/usr/bin/env python3
"""Capture Ollama server + model metadata for reproducibility."""
import argparse
import json
import os
import time

import requests


def get(host, path):
    r = requests.get(f"http://{host}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def post(host, path, body):
    r = requests.post(f"http://{host}{path}", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def slim_env(host, model, version, tags, show, ps):
    mi = show.get("model_info", {})
    tag = next((m for m in tags.get("models", []) if m["name"] == model), {})
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": host,
        "model": model,
        "digest": tag.get("digest"),
        "ollama_version": version.get("version"),
        "model_details": {
            "format": show.get("details", {}).get("format"),
            "family": show.get("details", {}).get("family"),
            "parameter_size": show.get("details", {}).get("parameter_size"),
            "quantization_level": show.get("details", {}).get("quantization_level"),
            "context_length": mi.get("gemma4.context_length"),
            "embedding_length": mi.get("gemma4.embedding_length"),
            "layers": mi.get("gemma4.block_count"),
            "sliding_window": mi.get("gemma4.attention.sliding_window"),
            "expert_used_count": mi.get("gemma4.expert_used_count"),
            "expert_count": mi.get("gemma4.expert_count"),
            "size_bytes": tag.get("size"),
            "capabilities": tag.get("capabilities"),
        },
        "kv_cache_mode_note": (
            "OLLAMA_KV_CACHE_TYPE is server-side; set manually when publishing results."
        ),
        "ps_snapshot": ps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "localhost:11434"))
    ap.add_argument("--model",
                    default="hf.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL")
    ap.add_argument("--out-dir", default="../raw")
    ap.add_argument("--full", action="store_true",
                    help="Also write full /api/show blob (large; omit for publication)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    version = get(args.host, "/api/version")
    tags = get(args.host, "/api/tags")
    show = post(args.host, "/api/show", {"name": args.model})
    ps = get(args.host, "/api/ps")

    slim = slim_env(args.host, args.model, version, tags, show, ps)
    slim_path = f"{args.out_dir}/environment.json"
    with open(slim_path, "w") as f:
        json.dump(slim, f, indent=2)
    print("wrote", slim_path)

    if args.full:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        full_path = f"{args.out_dir}/environment_{stamp}_full.json"
        with open(full_path, "w") as f:
            json.dump({"version": version, "tags": tags, "show": show, "ps": ps},
                      f, indent=2)
        print("wrote", full_path)

    mi = show.get("model_info", {})
    print(f"digest={slim.get('digest')}")
    print(f"quant={slim['model_details'].get('quantization_level')}")
    print(f"context_length={mi.get('gemma4.context_length')}")


if __name__ == "__main__":
    main()
