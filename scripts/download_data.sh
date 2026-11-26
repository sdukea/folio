#!/usr/bin/env bash
# Fetch every dataset Folio uses into data/raw/ (gitignored). Safe to re-run:
# files that already exist with the expected size are skipped.
#
# Usage:
#   bash scripts/download_data.sh            # core datasets + model tokenizer/config
#   bash scripts/download_data.sh --model    # also the full Qwen2.5-0.5B-Instruct weights (~988 MB)
#
# LMSYS-Chat-1M is gated: accept its terms on Hugging Face, run `hf auth login`,
# then fetch it separately (Track C, Week 9 — Stretch).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
MODEL_DIR="$ROOT/models/Qwen2.5-0.5B-Instruct"
WITH_MODEL=0
[[ "${1:-}" == "--model" ]] && WITH_MODEL=1

mkdir -p "$RAW/sharegpt" "$RAW/burstgpt" "$RAW/wikitext2" "$RAW/azure" "$MODEL_DIR"

fetch() {  # fetch <url> <dest> <expected_bytes>
  local url="$1" dest="$2" size="$3"
  if [[ -f "$dest" && "$(wc -c < "$dest" | tr -d ' ')" == "$size" ]]; then
    echo "✓ already have $(basename "$dest")"; return
  fi
  echo "↓ $(basename "$dest")"
  curl -L --fail --retry 3 --retry-delay 2 -C - -o "$dest" "$url"
  local got; got="$(wc -c < "$dest" | tr -d ' ')"
  if [[ "$got" != "$size" ]]; then
    echo "✗ $(basename "$dest"): expected $size bytes, got $got" >&2; exit 1
  fi
}

HF=https://huggingface.co

# Main workload: ShareGPT V3 (~642 MB)
fetch "$HF/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json" \
      "$RAW/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json" 672837942

# Bursty arrivals: BurstGPT v1.1, failures removed (~136 MB)
fetch "https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv" \
      "$RAW/burstgpt/BurstGPT_without_fails_2.csv" 142376815

# Quality: WikiText-2 raw (parquet from the HF dataset repo)
for split in test:732610 validation:657209 train:6357543; do
  name="${split%%:*}"; bytes="${split##*:}"
  fetch "$HF/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/${name}-00000-of-00001.parquet" \
        "$RAW/wikitext2/${name}.parquet" "$bytes"
done

# Production arrivals: Azure LLM Inference Trace 2023 (small; Stretch workload)
AZ=https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data
fetch "$AZ/AzureLLMInferenceTrace_conv.csv" "$RAW/azure/AzureLLMInferenceTrace_conv.csv" 719188
fetch "$AZ/AzureLLMInferenceTrace_code.csv" "$RAW/azure/AzureLLMInferenceTrace_code.csv" 320117

# Model: tokenizer + config always (needed for preprocessing); weights with --model
QWEN=$HF/Qwen/Qwen2.5-0.5B-Instruct/resolve/main
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  [[ -f "$MODEL_DIR/$f" ]] || { echo "↓ $f"; curl -sL --fail -o "$MODEL_DIR/$f" "$QWEN/$f"; }
done
if [[ $WITH_MODEL == 1 ]]; then
  [[ -f "$MODEL_DIR/model.safetensors" ]] || { echo "↓ model.safetensors (~988 MB)"; \
    curl -L --fail --retry 3 -C - -o "$MODEL_DIR/model.safetensors" "$QWEN/model.safetensors"; }
fi

echo "Done. Next: python scripts/preprocess_data.py"
