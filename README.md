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

## Running

All model runs were done on a GPU server (vLLM or Ollama for prompting; one 48 GB GPU for
fine-tuning). The figures can be rebuilt from this folder alone, without a GPU:

```bash
python make_result_figures.py
```

This folder holds the scripts, splits, summary metrics and figures. The per-sentence model answers,
per-run and per-domain tables and server logs are in **`Final_extra/`**, which has the same folder
structure. To re-score the saved answers (`evaluate`, `prompt_selection.py`, `compare`), first merge
`Final_extra/` into this folder.

Order of the experiment: `Prompting/Binary_prompt_selection` → `Prompting/Appraisal_prompt_selection`
→ `Prompting/Prompt_selection` → `Prompting/Heldout_pipeline` and `Finetuning`.

The fine-tuned LoRA adapters (2 x 267 MB) are not included because of their size.
