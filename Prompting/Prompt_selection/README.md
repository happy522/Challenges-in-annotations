# Final prompt + model selection

Chooses the best prompt and model per step from the answers already saved in
`../Binary_prompt_selection/outputs` and `../Appraisal_prompt_selection/outputs`. **No model is run.**
The dev data is unbalanced (more evaluative sentences, few Affect sentences), so each comparison is made on class-balanced subsets.

## Data (per run; 3 runs with seeds 42, 43, 44)

| Step | Taken from | Balanced set per run |
|---|---|---|
| Binary | 308 dev sentences (118 non-eval / 190 eval) | **236** = 118 non-evaluative (all) + 118 evaluative (sampled with the same domain mix) |
| Appraisal | 190 evaluative dev sentences, 20 ambiguous removed → 170 (32 Affect / 76 Judgement / 62 Appreciation) | **96** = 32 Affect (all) + 32 Judgement + 32 Appreciation (sampled) |

Over the 3 binary runs, 298 different sentences are used (164 of them in every run).
The sets are listed in `Final_extra/.../selection_results/*_selection_sets.csv`.

## Method

- Metrics: macro-F1, accuracy, Cohen's κ, Krippendorff's α (vs gold). In appraisal, an answer with several categories counts as wrong.
- Best prompt = highest macro-F1 averaged over models and runs; best model = highest mean macro-F1 with that prompt.
- Significance (every model pair, same prompt, same sentences): paired approximate randomization test on
  macro-F1 (10,000 permutations) and exact McNemar test, both Holm-corrected over the 3 pairs.

## Result (mean ± sd over 3 runs, macro-F1)

| Model | Binary V1 | Binary V2 | Binary V3 | Appraisal V1 | Appraisal V2 | Appraisal V3 |
|---|---|---|---|---|---|---|
| Qwen3-30B | 0.714 ± 0.021 | 0.637 ± 0.002 | 0.740 ± 0.010 | 0.566 ± 0.024 | **0.594 ± 0.017** | 0.381 ± 0.005 |
| Llama-3.3-70B | 0.691 ± 0.000 | 0.708 ± 0.012 | 0.743 ± 0.016 | 0.571 ± 0.022 | 0.583 ± 0.024 | 0.520 ± 0.031 |
| Mistral-Large | 0.688 ± 0.010 | 0.709 ± 0.012 | **0.764 ± 0.012** | 0.582 ± 0.045 | 0.574 ± 0.025 | 0.552 ± 0.013 |

- **Chosen: binary V3** (best model Mistral-Large) and **appraisal V2** (best model Qwen3-30B).
- With the chosen prompts, no model is significantly better than the others in any run.
  The only significant gap is Qwen with appraisal V3, which is clearly worse (3/3 runs).

## Files

| File | Content |
|---|---|
| `prompt_selection.py` | The selection and the tests |
| `selection_results/final_results.csv` | Every model x prompt: mean ± sd, rank, p-values against the top model, chosen setup |
| `selection_results/*_metrics_mean_sd.csv` | All metrics, mean and sd over the 3 runs |

In `Final_extra/` (same path): `*_metrics_per_run.csv`, `*_significance.csv` (all pairwise tests),
`*_selection_sets.csv`, `final_results.txt` and `selection_summary.txt`.

```bash
python prompt_selection.py      # needs the pred_*.csv files from Final_extra
```
