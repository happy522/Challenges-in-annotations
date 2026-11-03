#!/usr/bin/env bash
# Run all 3 appraisal prompts on the chosen models through Ollama, then compare.
# Start the Ollama server first and export GOLD_CSV.
# Pick models with RUN_MODELS, e.g. RUN_MODELS="qwen3-30b-a3b-instruct-2507 llama-3.3-70b-instruct"
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-4}"
MAX_LEN="${MAX_LEN:-4096}"
RUN_MODELS="${RUN_MODELS:-qwen3-30b-a3b-instruct-2507 llama-3.3-70b-instruct mistral-large-instruct}"

for MODEL in $RUN_MODELS; do
    python appraisal_prompt_selection.py run --backend ollama --model "$MODEL" \
        --workers "$WORKERS" --max-model-len "$MAX_LEN" ${EXTRA_ARGS:-}
done

python appraisal_prompt_selection.py evaluate
