import json
import time

import requests


def generate(host, model, prompt, num_ctx, num_predict=256, seed=0,
             temperature=0.0, top_p=1.0, timeout=1800):
    url = f"http://{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    t0 = time.perf_counter()
    ttft = None
    text = []
    final = {}
    with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            piece = chunk.get("response", "")
            if piece and ttft is None:
                ttft = time.perf_counter() - t0  # wall-clock to first token
            text.append(piece)
            if chunk.get("done"):
                final = chunk
    final["response"] = "".join(text)
    final["_ttft_s"] = ttft
    final["_wall_s"] = time.perf_counter() - t0
    return final


def metrics(d):
    def s(ns):
        return (ns or 0) / 1e9

    pe_c = d.get("prompt_eval_count") or 0
    pe_d = s(d.get("prompt_eval_duration"))
    e_c = d.get("eval_count") or 0
    e_d = s(d.get("eval_duration"))
    return {
        "prompt_tokens": pe_c,
        "gen_tokens": e_c,
        "prefill_tok_s": pe_c / pe_d if pe_d else None,
        "decode_tok_s": e_c / e_d if e_d else None,
        "ttft_s": d.get("_ttft_s"),
        "load_s": s(d.get("load_duration")),
        "total_s": s(d.get("total_duration")),
        "wall_s": d.get("_wall_s"),
    }
