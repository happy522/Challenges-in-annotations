# Data

`gold_standard_with_context.csv`: the gold standard annotated by the trained linguist.
Every experiment in `Prompting/` and `Finetuning/` reads this file.

## Contents

- **440 sentences** from **22 TED talks** (EmotionalizTED corpus), across **11 domains** with 2 talks
  and 40 sentences each: Art, Business, Education, Entertainment, History, Medicine, Philosophy,
  Politics, Psychology, Science, Technology.
- Columns:

| Column | Meaning |
|---|---|
| `Text_ID`, `Sentence_ID` | Talk (e.g. `Art5.`) and sentence (e.g. `S6` = 6th sentence of the transcript) |
| `Sentence` | The sentence to label |
| `Evaluative` | 1 = evaluative, 0 = not evaluative |
| `Affect`, `Judgement`, `Appreciation` | 1 if the sentence has that Appraisal category (empty otherwise) |
| `Ambiguous` | 1 = sentence with more than one gold category |
| `Prev_Sentence`, `Next_Sentence` | The sentences before and after it in the transcript (context for the prompts) |

## Label counts

| | Sentences |
|---|---|
| Non-evaluative | 172 |
| Evaluative | 268 |
| – one category: Affect | 41 |
| – one category: Judgement | 109 |
| – one category: Appreciation | 91 |
| – two categories (ambiguous) | 27 |

Category totals when multi-label sentences are counted in each of their categories:
Affect 61, Judgement 127, Appreciation 107.

Evaluative / non-evaluative per domain:

| | Art | Bus | Edu | Ent | His | Med | Phi | Pol | Psy | Sci | Tech | All |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Non-evaluative | 13 | 15 | 7 | 10 | 14 | 18 | 17 | 17 | 13 | 25 | 23 | 172 |
| Evaluative | 27 | 25 | 33 | 30 | 26 | 22 | 23 | 23 | 27 | 15 | 17 | 268 |

## 70 / 30 split

The split is made once, in `Prompting/Binary_prompt_selection` (seed 42, stratified by domain), and saved in
`Prompting/Binary_prompt_selection/outputs/gold_split.csv`. Every later step reuses it.

| Split | Sentences | Non-evaluative | Evaluative | Affect | Judgement | Appreciation | Ambiguous |
|---|---|---|---|---|---|---|---|
| `prompt_dev` (70%) | 308 | 118 | 190 | 32 | 76 | 62 | 20 |
| `held_out` (30%) | 132 | 54 | 78 | 9 | 33 | 29 | 7 |
