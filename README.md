# Appraisal annotation with LLMs: prompting and fine-tuning

Code, model outputs and results for the LLM experiments of
**"Challenges in annotations by humans and LLMs: A case study of evaluative language"**
(M. Imamovic, A. Knierim, K. Pitroda, E. Lapshinova-Koltunski).

The task: label every sentence of English TED talk transcripts (EmotionalizTED corpus) with
Appraisal Theory (Martin & White, 2005), in two steps:

1. **Binary**: is the sentence evaluative (1) or not (0)?
2. **Appraisal**: if it is evaluative, is it **Affect** (emotion), **Judgement** (people / behaviour)
   or **Appreciation** (things / quality / aesthetics)?

Models: Qwen3-30B-A3B-Instruct-2507, Llama-3.3-70B-Instruct and Mistral-Large-Instruct-2411 (zero-shot),
and Llama-3.3-70B-Instruct fine-tuned with QLoRA.

## Folders

| Folder | What it contains |
|---|---|
| [Data/](Data/) | The gold standard (440 sentences, senior researcher's labels) with the previous / next sentence as context |
| [Prompting/](Prompting/) | Zero-shot prompt selection (3 prompts x 3 models) on 70% of the data, then the final test of the winning prompts on the held-out 30% |
| [Finetuning/](Finetuning/) | QLoRA fine-tuning of Llama-3.3-70B on the same 70% and its test on the same held-out 30% |
| [Human_vs_LLM/](Human_vs_LLM/) | Agreement between linguists in training, the senior researcher and fine-tuned Llama on the held-out test set (Cohen's κ / Krippendorff's α + heatmap) |
| [make_result_figures.py](make_result_figures.py) | Builds every figure and summary table in `Prompting/Figures/` and `Finetuning/Figures/` from the saved results |

Each experiment folder has its own README with the exact number of sentences used.

## Data at a glance

| | Sentences | Non-evaluative | Evaluative | Affect | Judgement | Appreciation | Ambiguous* |
|---|---|---|---|---|---|---|---|
| Gold standard (all) | 440 | 172 | 268 | 41 | 109 | 91 | 27 |
| Prompt development (70%) | 308 | 118 | 190 | 32 | 76 | 62 | 20 |
| Held-out test (30%) | 132 | 54 | 78 | 9 | 33 | 29 | 7 |
| Held-out test, ambiguous removed | **125** | 54 | 71 | 9 | 33 | 29 | – |

\* Ambiguous = an evaluative sentence with more than one gold category (all 27 are also flagged
`Ambiguous = 1`). Category columns count only single-category sentences.

The 70 / 30 split is stratified by domain (11 domains, 2 talks each, 40 sentences per domain:
28 for development, 12 held out). **The held-out 30% was never used to choose a prompt, a model
or a fine-tuning epoch.**

## Main results (held-out test, 125 sentences)

| Model | Binary macro-F1 | Combined 4-class macro-F1 (none / affect / judgement / appreciation) | Combined Cohen's κ |
|---|---|---|---|
| Qwen3-30B (zero-shot) | 0.780 | 0.519 | 0.399 |
| Llama-3.3-70B (zero-shot) | 0.799 | 0.521 | 0.384 |
| **Mistral-Large (zero-shot)** | 0.814 | **0.655** | **0.536** |
| Llama-3.3-70B (fine-tuned) | **0.833** | 0.602 | 0.516 |

- Winning prompts: **binary V3** (indicative words and phrases), **appraisal V2** (zero-shot, short definitions).
- On the combined task Mistral is significantly better than Llama and Qwen (randomization test, Holm p = 0.037).
  The three models do not differ significantly on the binary step.
- Fine-tuning improves Llama on every step (binary +0.026, combined +0.034 macro-F1), but none of these
  differences is statistically significant (p ≈ 0.5).

## Statistical significance

### Why

The differences between models are often small (e.g. 0.764 vs 0.743 macro-F1), and the test sets are small
(96–236 sentences). A gap of that size could be chance. All models label **the same sentences**, so we use two
**paired** tests:

| Test | What it compares | Why |
|---|---|---|
| **Approximate randomization test** (Noreen, 1989; Yeh, 2000) | Macro-F1 | Macro-F1 is not an average over sentences, so a standard parametric test does not fit. The two models' answers are swapped at random, sentence by sentence, 10,000 times; p = how often the macro-F1 gap is at least as large as the real one. |
| **Exact McNemar test** (McNemar, 1947) | Whether each sentence is labelled correctly | Uses only the sentences where one model is right and the other wrong, and tests whether that split is lopsided. |
| **Holm–Bonferroni correction** (Holm, 1979) | – | 3 model pairs are tested at once, so the p-values are corrected for multiple comparisons. |

The two tests can disagree. Macro-F1 gives small classes (e.g. Affect) as much weight as large ones, while McNemar
only counts right vs wrong answers. A difference is called significant when its Holm-corrected p < 0.05.

### The 3 runs (prompt selection)

The development data is unbalanced: 190 evaluative vs 118 non-evaluative sentences, and only 32 Affect vs 76 Judgement
sentences. Prompts are therefore compared on **class-balanced subsets**. A single subset could favour one model by chance,
so the subset is drawn **3 times, with seeds 42, 43 and 44**, and the results are reported as mean ± sd over the 3 runs.

- The models were **not run again**. Each model answered each sentence once (temperature 0); the 3 runs re-sample
  which of these saved answers are scored.
- Binary: every run uses all 118 non-evaluative sentences + a different 118 of the 190 evaluative ones (same domain mix).
  Over the 3 runs, 298 of the 308 dev sentences are used.
- Appraisal: every run uses all 32 Affect sentences + a different 32 Judgement + 32 Appreciation (ambiguous sentences removed).
- The tests are run **separately in each run**. `Significant runs` = in how many of the 3 runs p < 0.05.
- Done in [Prompting/Prompt_selection/prompt_selection.py](Prompting/Prompt_selection/prompt_selection.py);
  results in [Prompting/Prompt_selection/selection_results/](Prompting/Prompt_selection/selection_results/).
  The full per-run table is in the [Prompt_selection README](Prompting/Prompt_selection/README.md).

The held-out test and the fine-tuning test use the full held-out set (125 sentences) **once**, so they have no runs.

### Results

**1. Prompt selection** (70% dev data, balanced sets, 3 runs; models compared within the same prompt; Holm-corrected; range over the 3 runs)

| Task | Prompt | Pair (A vs B) | Δ macro-F1 A − B (mean) | p randomization | p McNemar | Significant runs |
|---|---|---|---|---|---|---|
| Binary | V1 | Llama vs Mistral | +0.002 | 0.62–1.00 | 1.00 | 0/3 |
| Binary | V1 | Llama vs Qwen | −0.024 | 0.53–1.00 | 1.00 | 0/3 |
| Binary | V1 | Mistral vs Qwen | −0.026 | 0.49–1.00 | 1.00 | 0/3 |
| Binary | V2 | Llama vs Mistral | −0.001 | 0.57–0.86 | 0.56–0.86 | 0/3 |
| Binary | V2 | Llama vs Qwen | +0.071 | 0.052–0.24 | 0.07–0.25 | 0/3 |
| Binary | V2 | Mistral vs Qwen | +0.072 | 0.052–0.25 | 0.07–0.25 | 0/3 |
| Binary | **V3** | Llama vs Mistral | −0.022 | 0.83–1.00 | 1.00 | 0/3 |
| Binary | **V3** | Llama vs Qwen | +0.003 | 0.88–1.00 | 1.00 | 0/3 |
| Binary | **V3** | Mistral vs Qwen | +0.024 | 0.44–0.94 | 0.79–1.00 | 0/3 |
| Appraisal | V1 | Llama vs Mistral | −0.011 | 0.67–1.00 | 0.72–1.00 | 0/3 |
| Appraisal | V1 | Llama vs Qwen | +0.004 | 1.00 | 0.35–1.00 | 0/3 |
| Appraisal | V1 | Mistral vs Qwen | +0.016 | 1.00 | 0.72–1.00 | 0/3 |
| Appraisal | **V2** | Llama vs Mistral | +0.009 | 1.00 | 0.79–1.00 | 0/3 |
| Appraisal | **V2** | Llama vs Qwen | −0.011 | 1.00 | 0.85–1.00 | 0/3 |
| Appraisal | **V2** | Mistral vs Qwen | −0.021 | 1.00 | 0.85–1.00 | 0/3 |
| Appraisal | V3 | Llama vs Mistral | −0.032 | 0.23–0.90 | 0.71–0.86 | 0/3 |
| Appraisal | V3 | **Llama vs Qwen** | +0.139 | **0.02**–0.23 | 0.06–0.29 | **2/3** (randomization only) |
| Appraisal | V3 | **Mistral vs Qwen** | +0.171 | **0.02–0.04** | 0.06–0.29 | **3/3** (randomization only) |

Bold prompt = chosen prompt. With the chosen prompts (binary **V3**, appraisal **V2**), **no model is significantly
better than another**. The only significant result is that Qwen is worse with appraisal prompt V3, which was not chosen.
The choice between the prompts themselves (V1 vs V2 vs V3) was made on mean macro-F1 and was not tested.

**2. Held-out test, zero-shot** (125 sentences, binary V3 → appraisal V2; Holm-corrected)

| Step | Pair (A vs B) | Δ macro-F1 A − B | Only A / only B correct | p randomization | p McNemar |
|---|---|---|---|---|---|
| Binary | Llama vs Mistral | −0.014 | 7 / 9 | 1.000 | 1.000 |
| Binary | Llama vs Qwen | +0.019 | 12 / 10 | 1.000 | 1.000 |
| Binary | Mistral vs Qwen | +0.033 | 10 / 6 | 0.904 | 1.000 |
| Combined 4-class | **Mistral vs Llama** | +0.134 | 19 / 7 | **0.037** | 0.087 |
| Combined 4-class | **Mistral vs Qwen** | +0.136 | 20 / 8 | **0.037** | 0.087 |
| Combined 4-class | Llama vs Qwen | +0.002 | 14 / 14 | 0.965 | 1.000 |
| Combined 4-class | Mistral + Qwen vs Mistral | −0.128 | 4 / 15 | 0.055 | 0.058 |

The appraisal step on its own is not tested, because each model passes a different set of sentences to step 2.
Source: [heldout_significance.csv](Prompting/Heldout_pipeline/outputs/results/heldout_significance.csv).

**3. Fine-tuned vs zero-shot Llama-3.3-70B** (125 held-out sentences; one comparison, so no correction)

| Step | Δ macro-F1 (fine-tuned − zero-shot) | Only fine-tuned / only zero-shot correct | p randomization | p McNemar |
|---|---|---|---|---|
| Binary | +0.026 | 13 / 9 | 0.525 | 0.524 |
| Combined 4-class | +0.034 | 17 / 14 | 0.531 | 0.720 |

Fine-tuning improves every score, but **none of the improvements is significant**.
Source: [test_significance.csv](Finetuning/outputs/test_significance.csv).

**In short:** the only robust difference between models is Mistral beating Llama and Qwen on the combined held-out
task, and only with the randomization test (McNemar p = 0.087).

### References

- Dror, R., Baumer, G., Shlomov, S., & Reichart, R. (2018). The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing. *ACL 2018*.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
- Noreen, E. W. (1989). *Computer-Intensive Methods for Testing Hypotheses: An Introduction*. Wiley.
- Yeh, A. (2000). More accurate tests for the statistical significance of result differences. *COLING 2000*.

## Running

All model runs were done on a GPU server (vLLM or Ollama for prompting; one 48 GB GPU for
fine-tuning). The figures can be rebuilt from this repository alone, without a GPU:

```bash
python make_result_figures.py
```

This repository holds the scripts, splits, summary metrics, significance tests and figures. The per-sentence
model answers (`pred_*.csv`) and the server logs are not included. Re-scoring the saved answers
(`evaluate`, `prompt_selection.py`, `heldout_pipeline.py evaluate`) needs those files to be copied back in first.

Order of the experiment: `Prompting/Binary_prompt_selection` → `Prompting/Appraisal_prompt_selection`
→ `Prompting/Prompt_selection` → `Prompting/Heldout_pipeline` and `Finetuning`.

The fine-tuned LoRA adapters (2 x 267 MB) are not included because of their size.
