# Final prompt + model selection

Chooses the best prompt and model per step from the answers already saved in
`../Binary_prompt_selection/outputs` and `../Appraisal_prompt_selection/outputs`. **No model is run.**
The dev data is unbalanced (more evaluative sentences, few Affect sentences), so each comparison is made on class-balanced subsets.

## Data: 3 runs (seeds 42, 43, 44)

| Step | Taken from | Balanced set per run |
|---|---|---|
| Binary | 308 dev sentences (118 non-eval / 190 eval) | **236** = 118 non-evaluative (all) + 118 evaluative (sampled with the same domain mix) |
| Appraisal | 190 evaluative dev sentences, 20 ambiguous removed → 170 (32 Affect / 76 Judgement / 62 Appreciation) | **96** = 32 Affect (all) + 32 Judgement + 32 Appreciation (sampled) |

**What a "run" is:** each run draws a different balanced subset from the same saved model answers. The models
answered every sentence **once** (temperature 0) and were not run again. Repeating the sampling 3 times shows
whether the ranking of prompts and models depends on which sentences happen to be picked. It also lets the
selection use more of the data: over the 3 binary runs, 298 of the 308 dev sentences are used (164 of them in every run).
The sentences of each run are listed in `selection_results/*_selection_sets.csv`.

## Method

- Metrics: macro-F1, accuracy, Cohen's κ, Krippendorff's α (vs gold). In appraisal, an answer with several categories counts as wrong.
- Best prompt = highest macro-F1 averaged over models and runs; best model = highest mean macro-F1 with that prompt.
- Significance, in **each run separately**, for every model pair with the same prompt on the same sentences:
  - paired approximate randomization test on macro-F1 (10,000 permutations; Noreen, 1989; Yeh, 2000)
  - exact McNemar test on per-sentence correctness (McNemar, 1947)
  - both Holm-corrected over the 3 model pairs of that prompt and run (Holm, 1979); significant = p < 0.05

## Result (mean ± sd over 3 runs, macro-F1)

| Model | Binary V1 | Binary V2 | Binary V3 | Appraisal V1 | Appraisal V2 | Appraisal V3 |
|---|---|---|---|---|---|---|
| Qwen3-30B | 0.714 ± 0.021 | 0.637 ± 0.002 | 0.740 ± 0.010 | 0.566 ± 0.024 | **0.594 ± 0.017** | 0.381 ± 0.005 |
| Llama-3.3-70B | 0.691 ± 0.000 | 0.708 ± 0.012 | 0.743 ± 0.016 | 0.571 ± 0.022 | 0.583 ± 0.024 | 0.520 ± 0.031 |
| Mistral-Large | 0.688 ± 0.010 | 0.709 ± 0.012 | **0.764 ± 0.012** | 0.582 ± 0.045 | 0.574 ± 0.025 | 0.552 ± 0.013 |

Macro-F1 per run (seed 42 / 43 / 44):

| Model | Binary V1 | Binary V2 | Binary V3 | Appraisal V1 | Appraisal V2 | Appraisal V3 |
|---|---|---|---|---|---|---|
| Qwen3-30B | 0.725 / 0.729 / 0.690 | 0.636 / 0.640 / 0.636 | 0.741 / 0.749 / 0.729 | 0.545 / 0.561 / 0.593 | 0.575 / 0.605 / 0.603 | 0.385 / 0.383 / 0.376 |
| Llama-3.3-70B | 0.691 / 0.691 / 0.691 | 0.717 / 0.712 / 0.694 | 0.754 / 0.750 / 0.724 | 0.546 / 0.586 / 0.580 | 0.558 / 0.604 / 0.588 | 0.533 / 0.485 / 0.543 |
| Mistral-Large | 0.678 / 0.698 / 0.688 | 0.722 / 0.698 / 0.707 | 0.771 / 0.771 / 0.750 | 0.531 / 0.598 / 0.617 | 0.545 / 0.586 / 0.591 | 0.539 / 0.553 / 0.564 |

- **Chosen: binary V3** (best model Mistral-Large) and **appraisal V2** (best model Qwen3-30B).
  V3 is the best binary prompt for every model in every run.
- With the chosen prompts, no model is significantly better than another in any run (table below).
- The only significant gaps: with appraisal V3, Qwen is worse than Mistral (3/3 runs) and than Llama (2/3 runs),
  with the randomization test only.

## Significance per run

Holm-corrected p-values per run (seed 42 / 43 / 44); **bold** = p < 0.05. "Only A / only B" = sentences only model A
(or only B) labelled correctly, which is what McNemar tests. Bold prompt = chosen prompt.

### Binary (236 sentences per run)

| Prompt | Pair (A vs B) | Δ macro-F1 (A − B) | p randomization | p McNemar | Only A – only B correct |
|---|---|---|---|---|---|
| V1 | Llama vs Mistral | +0.013 / −0.008 / +0.002 | 0.624 / 0.754 / 1.000 | 1.000 / 1.000 / 1.000 | 15–14 / 12–15 / 13–14 |
| V1 | Llama vs Qwen | −0.034 / −0.038 / +0.001 | 0.525 / 0.639 / 1.000 | 1.000 / 1.000 / 1.000 | 20–26 / 19–26 / 21–19 |
| V1 | Mistral vs Qwen | −0.046 / −0.030 / −0.002 | 0.488 / 0.660 / 1.000 | 1.000 / 1.000 / 1.000 | 24–31 / 24–28 / 25–22 |
| V2 | Llama vs Mistral | −0.005 / +0.015 / −0.013 | 0.863 / 0.573 / 0.567 | 0.864 / 0.856 / 0.557 | 16–18 / 16–14 / 11–15 |
| V2 | Llama vs Qwen | +0.081 / +0.073 / +0.058 | 0.052 / 0.099 / 0.244 | 0.066 / 0.181 / 0.250 | 45–26 / 45–28 / 43–29 |
| V2 | Mistral vs Qwen | +0.086 / +0.058 / +0.072 | 0.052 / 0.249 / 0.175 | 0.066 / 0.248 / 0.161 | 49–28 / 49–34 / 48–30 |
| **V3** | Llama vs Mistral | −0.017 / −0.021 / −0.026 | 0.996 / 0.835 / 0.938 | 1.000 / 1.000 / 1.000 | 18–22 / 19–24 / 17–23 |
| **V3** | Llama vs Qwen | +0.013 / +0.000 / −0.005 | 0.996 / 0.881 / 0.938 | 1.000 / 1.000 / 1.000 | 20–17 / 19–19 / 19–20 |
| **V3** | Mistral vs Qwen | +0.030 / +0.022 / +0.021 | 0.437 / 0.814 / 0.938 | 0.795 / 1.000 / 1.000 | 18–11 / 17–12 / 17–12 |

### Appraisal (96 sentences per run)

| Prompt | Pair (A vs B) | Δ macro-F1 (A − B) | p randomization | p McNemar | Only A – only B correct |
|---|---|---|---|---|---|
| V1 | Llama vs Mistral | +0.015 / −0.012 / −0.037 | 1.000 / 1.000 / 0.667 | 1.000 / 0.719 / 0.790 | 8–4 / 11–8 / 4–8 |
| V1 | Llama vs Qwen | +0.001 / +0.025 / −0.013 | 1.000 / 1.000 / 1.000 | 1.000 / 0.346 / 0.832 | 11–7 / 14–6 / 12–10 |
| V1 | Mistral vs Qwen | −0.014 / +0.037 / +0.024 | 1.000 / 1.000 / 1.000 | 1.000 / 0.719 / 0.790 | 8–8 / 12–7 / 13–7 |
| **V2** | Llama vs Mistral | +0.013 / +0.018 / −0.003 | 1.000 / 1.000 / 1.000 | 1.000 / 0.790 / 1.000 | 8–6 / 13–7 / 7–6 |
| **V2** | Llama vs Qwen | −0.018 / −0.001 / −0.015 | 1.000 / 1.000 / 1.000 | 1.000 / 0.848 / 1.000 | 7–9 / 8–6 / 8–8 |
| **V2** | Mistral vs Qwen | −0.031 / −0.020 / −0.012 | 1.000 / 1.000 / 1.000 | 1.000 / 0.848 / 1.000 | 5–9 / 5–9 / 6–7 |
| V3 | Llama vs Mistral | −0.006 / −0.068 / −0.021 | 0.902 / 0.231 / 0.660 | 0.860 / 0.856 / 0.711 | 15–17 / 14–16 / 13–16 |
| V3 | Llama vs Qwen | +0.148 / +0.101 / +0.168 | **0.043** / 0.231 / **0.024** | 0.082 / 0.288 / 0.064 | 24–11 / 23–13 / 25–11 |
| V3 | Mistral vs Qwen | +0.155 / +0.170 / +0.189 | **0.043** / **0.035** / **0.024** | 0.071 / 0.288 / 0.064 | 27–12 / 28–16 / 33–16 |

## Files

| File | Content |
|---|---|
| `prompt_selection.py` | The selection and the tests |
| `selection_results/final_results.csv` | Every model x prompt: mean ± sd, rank, p-values against the top model, chosen setup |
| `selection_results/*_metrics_mean_sd.csv` | All metrics, mean and sd over the 3 runs |
| `selection_results/*_metrics_per_run.csv` | All metrics for each run |
| `selection_results/*_significance.csv` | Every pairwise test in every run (raw and Holm-corrected p-values) |
| `selection_results/*_selection_sets.csv` | The sentences of each run |

```bash
python prompt_selection.py      # needs the per-sentence answers (pred_*.csv), which are not in this repository
```
