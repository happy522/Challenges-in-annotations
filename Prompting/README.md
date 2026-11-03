# Prompting (zero-shot LLMs)

Three prompt versions are tested for each step on three models, without any training:

| Prompt | Strategy |
|---|---|
| V1 | Few-shot with example sentences, chain-of-thought ("Let's think step-by-step"), detailed class definitions |
| V2 | Zero-shot, short class definitions |
| V3 | Few-shot with indicative adjectives and n-grams for each class instead of example sentences |

Each prompt gets the TARGET sentence plus the PREVIOUS and NEXT sentence as context, and the model
answers in JSON. Models: `qwen3-30b-a3b-instruct-2507`, `llama-3.3-70b-instruct`, `mistral-large-instruct`.

## Steps (run in this order)

| Folder | What it does | Data |
|---|---|---|
| [Binary_prompt_selection/](Binary_prompt_selection/) | Makes the 70 / 30 split; runs binary prompts V1–V3 x 3 models | 308 dev sentences (118 non-eval / 190 eval) |
| [Appraisal_prompt_selection/](Appraisal_prompt_selection/) | Runs appraisal prompts V1–V3 x 3 models | 190 evaluative dev sentences (32 Affect / 76 Judgement / 62 Appreciation / 20 ambiguous) |
| [Prompt_selection/](Prompt_selection/) | Picks the best prompt and model on class-balanced subsets (3 runs) with significance tests; no model is run | binary 118 + 118 per run; appraisal 32 + 32 + 32 per run |
| [Heldout_pipeline/](Heldout_pipeline/) | Final test: binary V3 → appraisal V2 on the held-out 30% | 125 sentences (54 non-eval / 71 eval: 9 / 33 / 29) |
| [Figures/](Figures/) | Figures and tables made by `../make_result_figures.py` | – |

## Result

- **Selected prompts**: binary **V3**, appraisal **V2** (highest mean macro-F1 over models and runs).
- Prompt selection (balanced sets, mean of 3 runs), macro-F1:

| Model | Binary V1 | Binary V2 | Binary V3 | Appraisal V1 | Appraisal V2 | Appraisal V3 |
|---|---|---|---|---|---|---|
| Qwen3-30B | 0.714 | 0.637 | 0.740 | 0.566 | **0.594** | 0.381 |
| Llama-3.3-70B | 0.691 | 0.708 | 0.743 | 0.571 | 0.583 | 0.520 |
| Mistral-Large | 0.688 | 0.709 | **0.764** | 0.582 | 0.574 | 0.552 |

  With the selected prompts, none of the model differences is significant.

- Held-out test (125 sentences), macro-F1:

| Model | Binary | Appraisal (on what step 1 called evaluative) | Combined 4-class |
|---|---|---|---|
| Qwen3-30B | 0.780 | 0.490 (n = 70) | 0.519 |
| Llama-3.3-70B | 0.799 | 0.487 (n = 62) | 0.521 |
| **Mistral-Large** | **0.814** | **0.671** (n = 68) | **0.655** |
| Mistral binary + Qwen appraisal | 0.814 | 0.480 (n = 68) | 0.527 |
| Llama-3.3-70B fine-tuned (for comparison) | 0.833 | 0.558 (n = 79) | 0.602 |

  Mistral is significantly better than Llama and Qwen on the combined task (randomization test,
  Holm p = 0.037). The models do not differ significantly on the binary step.

## Running

Each folder has `run_all_ollama.sh` (Ollama backend) and the scripts also support vLLM
(`--backend vllm`). By default they read `../../Data/gold_standard_with_context.csv`; set `GOLD_CSV` to use another
path. Re-scoring saved outputs (`evaluate`) needs no GPU.
