# Appraisal prompt selection (Affect / Judgement / Appreciation)

Runs the 3 appraisal prompts on the **evaluative** sentences of the 70% development split, with all 3 models.
The split comes from `../Binary_prompt_selection/outputs/gold_split.csv`, so the held-out 30% is never used here either.

## Data

Saved in `outputs/appraisal_split.csv` (all 268 evaluative gold sentences, with their split).

| Split | Evaluative sentences | Affect | Judgement | Appreciation | Ambiguous (2 categories) |
|---|---|---|---|---|---|
| `prompt_dev` (used here) | **190** | 32 | 76 | 62 | 20 |
| `held_out` (never used here) | 78 | 9 | 33 | 29 | 7 |

- The model may give one or more categories (multi-label), so the 20 ambiguous sentences are scored too.
  `n_single` / `single_*` in the metrics = the 170 sentences with one gold category.
- Runs: 3 prompts x 3 models x 190 sentences = 1,710 model answers.

## Files

| File | Content |
|---|---|
| `appraisal_prompt_selection.py` | Inference (vLLM or Ollama), parsing, scoring |
| `prompts/V1.txt`, `V2.txt`, `V3.txt` | The three prompts (read by the script) |
| `run_all_ollama.sh` | All models with Ollama, then `evaluate` |
| `outputs/appraisal_split.csv` | The evaluative sentences with their split |
| `outputs/appraisal_metrics_summary.csv` | Metrics per model x prompt (`evaluate` writes this as `metrics_summary.csv`) |

In `Final_extra/` (same path): `outputs/pred_<model>_<prompt>.csv` (the raw and parsed answer for each sentence),
`outputs/metrics_by_domain.csv` and `logs/`.

## Results on all 190 evaluative dev sentences (multi-label macro-F1)

| Model | V1 | V2 | V3 |
|---|---|---|---|
| Qwen3-30B | 0.550 | 0.575 | 0.397 |
| Llama-3.3-70B | 0.574 | **0.598** | 0.537 |
| Mistral-Large | 0.584 | 0.580 | 0.556 |

The final choice is made on class-balanced subsets in `../Prompt_selection` (winner: **V2**).

## Usage

```bash
python appraisal_prompt_selection.py run --backend ollama --model qwen3-30b-a3b-instruct-2507
python appraisal_prompt_selection.py evaluate
```
