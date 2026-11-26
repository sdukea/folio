"""Turn the raw downloads in data/raw/ into small, benchmark-ready files in data/processed/.

Run after scripts/download_data.sh:
    .venv/bin/python scripts/preprocess_data.py

Outputs (all gitignored):
    sharegpt.jsonl                    one {id, prompt, prompt_len, output_len} per usable conversation
    sharegpt_sample_1000_seed0.jsonl  fixed-seed sample used by the load generator
    burstgpt.parquet                  arrivals: t (s from first request), request/response tokens, model
    azure_conv.parquet, azure_code.parquet   same shape for the Azure 2023 traces
    wikitext2_test.txt                text for the perplexity harness ("\n\n".join of the test split)
    stats.json                        summary numbers for every dataset

Token lengths use the real Qwen2.5 tokenizer on the *raw* prompt text (no chat template;
the template adds roughly 20-30 tokens per request).
"""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
TOKENIZER = ROOT / "models" / "Qwen2.5-0.5B-Instruct" / "tokenizer.json"

# Same length filter as vLLM's ShareGPT benchmark: keeps requests that fit a small edge model.
MIN_LEN, MAX_PROMPT, MAX_TOTAL = 4, 1024, 2048
SEED, SAMPLE_SIZE = 0, 1000


def summarize(xs) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 1),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def sharegpt(tok: Tokenizer) -> dict:
    data = json.loads((RAW / "sharegpt" / "ShareGPT_V3_unfiltered_cleaned_split.json").read_text())
    pairs = []
    for conv in data:
        turns = conv.get("conversations") or []
        if len(turns) >= 2 and turns[0]["from"] == "human" and turns[1]["from"] == "gpt":
            pairs.append((conv["id"], turns[0]["value"], turns[1]["value"]))

    p_lens = [len(e.ids) for e in tok.encode_batch([p for _, p, _ in pairs], add_special_tokens=False)]
    o_lens = [len(e.ids) for e in tok.encode_batch([o for _, _, o in pairs], add_special_tokens=False)]

    rows = [
        {"id": cid, "prompt": p, "prompt_len": pl, "output_len": ol}
        for (cid, p, _), pl, ol in zip(pairs, p_lens, o_lens)
        if pl >= MIN_LEN and ol >= MIN_LEN and pl <= MAX_PROMPT and pl + ol <= MAX_TOTAL
    ]
    with open(OUT / "sharegpt.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sample = random.Random(SEED).sample(rows, SAMPLE_SIZE)
    with open(OUT / f"sharegpt_sample_{SAMPLE_SIZE}_seed{SEED}.jsonl", "w") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "conversations": len(data),
        "first_turn_pairs": len(pairs),
        "kept_after_filter": len(rows),
        "prompt_len": summarize([r["prompt_len"] for r in rows]),
        "output_len": summarize([r["output_len"] for r in rows]),
        "sample_prompt_len": summarize([r["prompt_len"] for r in sample]),
        "sample_output_len": summarize([r["output_len"] for r in sample]),
    }


def arrivals_stats(t: np.ndarray) -> dict:
    gaps = np.diff(t)
    span = float(t[-1] - t[0])
    return {
        "requests": int(t.size),
        "span_hours": round(span / 3600, 2),
        "mean_rate_per_s": round(t.size / span, 4) if span else None,
        # CV = 1 for a Poisson process; > 1 means burstier than Poisson.
        "interarrival_cv": round(float(gaps.std() / gaps.mean()), 2) if gaps.size else None,
    }


def burstgpt() -> dict:
    ts, req, resp, model = [], [], [], []
    with open(RAW / "burstgpt" / "BurstGPT_without_fails_2.csv") as f:
        for row in csv.DictReader(f):
            ts.append(float(row["Timestamp"]))
            req.append(int(row["Request tokens"]))
            resp.append(int(row["Response tokens"]))
            model.append(row["Model"])
    order = np.argsort(ts, kind="stable")
    t = np.asarray(ts)[order]
    t = t - t[0]
    table = pa.table({
        "t": t,
        "request_tokens": np.asarray(req)[order],
        "response_tokens": np.asarray(resp)[order],
        "model": [model[i] for i in order],
    })
    pq.write_table(table, OUT / "burstgpt.parquet")
    return {**arrivals_stats(t), "request_tokens": summarize(req), "response_tokens": summarize(resp)}


def azure(kind: str) -> dict:
    ts, ctx, gen = [], [], []
    with open(RAW / "azure" / f"AzureLLMInferenceTrace_{kind}.csv") as f:
        for row in csv.DictReader(f):
            # Timestamps carry 7 fractional digits; Python parses at most 6.
            stamp = row["TIMESTAMP"][:26]
            ts.append(datetime.fromisoformat(stamp).timestamp())
            ctx.append(int(row["ContextTokens"]))
            gen.append(int(row["GeneratedTokens"]))
    t = np.sort(np.asarray(ts))
    t = t - t[0]
    pq.write_table(pa.table({"t": t, "context_tokens": ctx, "generated_tokens": gen}), OUT / f"azure_{kind}.parquet")
    return {**arrivals_stats(t), "context_tokens": summarize(ctx), "generated_tokens": summarize(gen)}


def wikitext2(tok: Tokenizer) -> dict:
    lines = pq.read_table(RAW / "wikitext2" / "test.parquet").column("text").to_pylist()
    text = "\n\n".join(lines)  # same join as the HF perplexity guide
    (OUT / "wikitext2_test.txt").write_text(text)
    return {"lines": len(lines), "chars": len(text), "qwen_tokens": len(tok.encode(text, add_special_tokens=False).ids)}


# Expected byte sizes (same as scripts/download_data.sh) so partial downloads are skipped.
RAW_FILES = {
    "wikitext2_test": ("wikitext2/test.parquet", 732610),
    "burstgpt": ("burstgpt/BurstGPT_without_fails_2.csv", 142376815),
    "azure_conv": ("azure/AzureLLMInferenceTrace_conv.csv", 719188),
    "azure_code": ("azure/AzureLLMInferenceTrace_code.csv", 320117),
    "sharegpt": ("sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json", 672837942),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(TOKENIZER))
    stats_path = OUT / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for name, fn in [
        ("wikitext2_test", lambda: wikitext2(tok)),
        ("burstgpt", burstgpt),
        ("azure_conv", lambda: azure("conv")),
        ("azure_code", lambda: azure("code")),
        ("sharegpt", lambda: sharegpt(tok)),
    ]:
        rel, size = RAW_FILES[name]
        path = RAW / rel
        if not path.exists() or path.stat().st_size != size:
            print(f"⨯ skipping {name}: {rel} missing or incomplete (re-run scripts/download_data.sh)")
            continue
        print(f"→ {name}", flush=True)
        stats[name] = fn()
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
