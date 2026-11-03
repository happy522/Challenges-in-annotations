#!/usr/bin/env bash
# Run all 3 binary prompts on the chosen models through Ollama, then compare.
# Start the Ollama server first, e.g.:
#   OLLAMA_NUM_PARALLEL=4 nohup ollama serve > ollama.log 2>&1 &
# Pick models with RUN_MODELS, e.g. RUN_MODELS="qwen3-30b-a3b-instruct-2507 llama-3.3-70b-instruct"
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-4}"
MAX_LEN="${MAX_LEN:-8192}"
RUN_MODELS="${RUN_MODELS:-qwen3-30b-a3b-instruct-2507 llama-3.3-70b-instruct mistral-large-instruct}"

for MODEL in $RUN_MODELS; do
    python binary_prompt_selection.py run --backend ollama --model "$MODEL" \
        --workers "$WORKERS" --max-model-len "$MAX_LEN"
done

python binary_prompt_selection.py evaluate
