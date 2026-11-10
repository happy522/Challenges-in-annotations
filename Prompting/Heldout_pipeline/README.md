# Held-out test: two-step pipeline with the winning prompts

The final zero-shot test on the **held-out 30%**, which no earlier step has seen:

1. **Step 1, binary**, prompt **V3**, on every held-out sentence.
2. **Step 2, appraisal**, prompt **V2**, on the sentences step 1 called evaluative.
   (Step 2 also runs on the gold-evaluative sentences that step 1 missed, so it can be scored on its own; these per-sentence predictions are not included in this repository.)

Runs for each of the 3 models, plus one combination: Mistral for step 1 + Qwen for step 2
(the best model of each step in `../Prompt_selection`).

## Data

| | Sentences | Non-evaluative | Evaluative | Affect | Judgement | Appreciation |
|---|---|---|---|---|---|---|
| Held-out split | 132 | 54 | 78 | 9 | 33 | 29 (+ 7 ambiguous) |
| **Scored** (ambiguous removed, same as `Finetuning`) | **125** | **54** | **71** | **9** | **33** | **29** |

Scored steps:
- **1 binary**: all 125 sentences.
- **2 appraisal**: the sentences each model's step 1 called evaluative. The count differs per model:
  Llama 62, Mistral 68, Qwen 70, fine-tuned Llama 79. Gold non-evaluative sentences among them always count as wrong.
- **Combined**: all 125 sentences with 4 labels (none / affect / judgment / appreciation).

## Results

| Model | Binary F1 | Binary κ | Appraisal F1 | Combined F1 | Combined κ |
|---|---|---|---|---|---|
| Qwen3-30B | 0.780 | 0.561 | 0.490 | 0.519 | 0.399 |
| Llama-3.3-70B | 0.799 | 0.600 | 0.487 | 0.521 | 0.384 |
| **Mistral-Large** | **0.814** | **0.628** | **0.671** | **0.655** | **0.536** |
| Mistral (binary) + Qwen (appraisal) | 0.814 | 0.628 | 0.480 | 0.527 | 0.407 |
| Llama-3.3-70B fine-tuned (reference, from `Finetuning`) | 0.833 | 0.668 | 0.558 | 0.602 | 0.516 |

All F1 values are macro-F1. Answers that can't be read or are "ambiguous" count as wrong
(Mistral: 7 such answers in step 2).

Significance (paired randomization and McNemar tests, Holm-corrected):
- Binary: no significant difference between any of the models.
- Combined: Mistral beats Llama (+0.134) and Qwen (+0.136), randomization p = 0.037 for both;
  McNemar p = 0.087 for both.

### Agreement with the gold standard (heatmap)

Measured as in the human annotation study: **Cohen's κ** for Evaluative (all 125 sentences) and **Krippendorff's α**
(nominal) for each subclass, computed on the sentences that **both** the gold standard and the model call evaluative.
Each subclass is scored as a yes / no label.

| Model | Both evaluative (n) | Evaluative κ | Affect α | Judgement α | Appreciation α |
|---|---|---|---|---|---|
| Qwen3-30B | 57 | 0.561 | 0.423 | 0.207 | 0.304 |
| Llama-3.3-70B | 54 | 0.600 | 0.463 | 0.151 | 0.149 |
| **Mistral-Large** | 58 | **0.628** | **0.571** | **0.582** | **0.577** |
| Mistral (binary) + Qwen (appraisal) | 58 | 0.628 | 0.425 | 0.129 | 0.244 |
| Llama-3.3-70B fine-tuned | 65 | 0.668 | 0.457 | 0.473 | 0.392 |

These are the `cohen_kappa` (step 1) and `alpha_*` / `n_both_eval` (step 2) columns of `heldout_metrics.csv`. The `kappa_*`
columns there are one-vs-rest Cohen's κ on every sentence the model called evaluative, including gold non-evaluative ones.

## Files

| File | Content |
|---|---|
| `heldout_pipeline.py` | Runs the pipeline (imports prompts and parsers from the two prompt-selection folders) and evaluates it |
| `run_all_ollama.sh` | All 3 models + the combination with Ollama, then `evaluate` |
| `outputs/results/heldout_metrics.csv` | All metrics per step and model, including per-class precision / recall / F1 / κ |
| `outputs/results/heldout_significance.csv` | Pairwise tests |

Figures and a summary table (`heldout_test_results.csv`) are in `../Figures/` (files starting with "Held-out" or "Heatmap").

Not included in this repository: `outputs/pred_<model>.csv` (raw answers for both steps, all 132 held-out sentences),
`outputs/results/heldout_predictions.csv` (gold and predicted labels for each sentence, all models), `heldout_results.txt` and `logs/`.

```bash
python heldout_pipeline.py run --backend ollama --model mistral-large-instruct
python heldout_pipeline.py evaluate      # needs pred_*.csv; adds the fine-tuned row if ../../Finetuning/outputs exists
```
