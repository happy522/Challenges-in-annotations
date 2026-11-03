#!/usr/bin/env bash
# Held-out test (binary V3 -> appraisal V2) for the chosen models through Ollama, then compare.
# Start the Ollama server first and export GOLD_CSV.
# Pick models with RUN_MODELS, e.g. RUN_MODELS="qwen3-30b-a3b-instruct-2507 llama-3.3-70b-instruct"
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-4}"
RUN_MODELS="${RUN_MODELS:-qwen3-30b-a3b-instruct-2507 llama-3.3-70b-instruct mistral-large-instruct}"

for MODEL in $RUN_MODELS; do
    python heldout_pipeline.py run --backend ollama --model "$MODEL" \
        --workers "$WORKERS" ${EXTRA_ARGS:-}
done

python heldout_pipeline.py evaluate
