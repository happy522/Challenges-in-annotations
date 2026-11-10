# Humans vs LLM: agreement on the held-out test set

How well do the linguists in training, the senior researcher (gold standard) and Llama-3.3-70B agree with each other,
**on the held-out test data only**?

## Data

- **125 held-out sentences**: the 30% test split, with ambiguous sentences removed (54 non-evaluative,
  71 evaluative: 9 Affect, 33 Judgement, 29 Appreciation). These are the same sentences as in `Finetuning/` and `Prompting/Heldout_pipeline/`.
- **Linguists in training**: every sentence was annotated by the **2 students** who annotated its talk
  (`Data/Students data/`, 22 students, 2 per talk). As in the human study, a sentence a student marked *Uncertain*
  is left out for that student (5 student labels).
- **Senior researcher**: the gold standard (`Data/gold_standard_with_context.csv`).
- **Llama**: Llama-3.3-70B **fine-tuned** with QLoRA. Its binary step labels all sentences, then its appraisal step labels
  the sentences it called evaluative (`Finetuning/outputs/test_finetuned/predictions.csv`).
  Zero-shot Llama is also in the CSV.

## Method

The same measures as the human annotation study:

- **Evaluative**: Cohen's κ on all sentences. Krippendorff's α is also in the CSV and is nearly identical.
- **Affect, Judgement, Appreciation**: Krippendorff's α (nominal), each category as a yes / no label, on the
  sentences that **both** raters of a pair call evaluative.

The two students of a sentence are different people for different talks, so the pairs are **pooled** over all sentences:

| Row | Pairs | Number of pairs | Both call it evaluative |
|---|---|---|---|
| Linguists in training | student A vs student B of the same talk | 120 | 43 |
| Linguists in training & senior researcher | each student vs gold | 245 | 98 |
| Fine-tuned Llama & linguists in training | Llama vs each student | 245 | 105 |
| Fine-tuned Llama & senior researcher | Llama vs gold | 125 | 65 |

## Results (held-out test, fine-tuned Llama)

| | Evaluative (κ) | Affect (α) | Judgement (α) | Appreciation (α) |
|---|---|---|---|---|
| Linguists in training | 0.27 | 0.39 | 0.04 | 0.40 |
| Linguists in training and senior researcher | 0.38 | 0.37 | 0.18 | 0.27 |
| Fine-tuned Llama and linguists in training | 0.36 | 0.46 | 0.34 | 0.31 |
| **Fine-tuned Llama and senior researcher** | **0.67** | **0.46** | **0.47** | **0.39** |

![Heatmap](outputs/Heatmap%20for%20the%20agreement%20between%20linguists%20in%20training,%20senior%20researcher%20and%20fine-tuned%20Llama-3.3-70B%20(held-out%20test).png)

Zero-shot Llama, for comparison:

| | Evaluative (κ) | Affect (α) | Judgement (α) | Appreciation (α) |
|---|---|---|---|---|
| Llama zero-shot and linguists in training | 0.41 | 0.45 | 0.24 | 0.27 |
| Llama zero-shot and senior researcher | 0.62 | 0.47 | 0.38 | 0.24 |

**Reading the table**
- Llama agrees with the senior researcher much more than the students do, for every category. On Evaluative κ it is
  0.67 vs 0.38 for the students.
- Llama agrees with the students about as well as the students agree with the senior researcher, and on Judgement better (0.34 vs 0.18).
- **Judgement** is the hardest category for the students (0.04 between students, 0.18 with the senior researcher).
- The numbers are based on few sentences, especially for the subclasses (43–105 pairs), so small differences should not be over-interpreted.

**Note on the full-data table:** the paper's values for all 440 sentences (students: 0.38 / 0.23 / 0.19 / 0.20;
students and senior researcher: 0.30 / 0.25 / 0.16 / 0.25) average over the 22 students instead of pooling the pairs.
For the students and senior researcher row, its Evaluative value matches the mean Krippendorff's α (0.30), not κ (0.33).
The held-out table above is pooled because each student has only ~12 held-out sentences.

## Files

| File | Content |
|---|---|
| `human_vs_llm_agreement.py` | The analysis (needs `xlrd` to read the two `.xls` student files) |
| `outputs/agreement_heldout.csv` | All rows (fine-tuned and zero-shot Llama), with κ and α for Evaluative and the number of pairs |
| `outputs/Heatmap ... (held-out test).png` | The heatmap above |

```bash
python Human_vs_LLM/human_vs_llm_agreement.py
```
