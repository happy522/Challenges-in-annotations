#!/usr/bin/env bash
# Run all 3 binary prompts on all 3 models, then compare.
# Each model runs in its own process so GPU memory is released in between.
# Adjust TP sizes to your GPUs (values below assume 80 GB cards, bf16).
set -euo pipefail
cd "$(dirname "$0")"

python binary_prompt_selection.py run --model qwen3-30b-a3b-instruct-2507 --tp "${TP_QWEN:-2}"
python binary_prompt_selection.py run --model llama-3.3-70b-instruct      --tp "${TP_LLAMA:-4}"
python binary_prompt_selection.py run --model mistral-large-instruct      --tp "${TP_MISTRAL:-8}"

python binary_prompt_selection.py evaluate
