#!/usr/bin/env bash
# Split, fine-tune both tasks, test fine-tuned and zero-shot Llama on the held-out 30%, compare.
# Export GOLD_CSV first; pick GPUs with CUDA_VISIBLE_DEVICES. Extra model flags via EXTRA_ARGS,
# e.g. EXTRA_ARGS="--base unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
set -euo pipefail
cd "$(dirname "$0")"

python finetune_llama.py split
python finetune_llama.py train --task binary ${EXTRA_ARGS:-}
python finetune_llama.py train --task appraisal ${EXTRA_ARGS:-}
python finetune_llama.py test ${EXTRA_ARGS:-}
python finetune_llama.py test --zero-shot ${EXTRA_ARGS:-}
python finetune_llama.py compare
