# Binary prompt selection (Evaluative 0 / 1)

Runs the 3 binary prompts on the 70% development split, with all 3 models, and scores them against the gold standard.

## Data

- Input: `../../Data/gold_standard_with_context.csv` (440 sentences).
- **Split** (made here, seed 42, stratified by domain): saved in `outputs/gold_split.csv`.

| Split | Sentences | Non-evaluative | Evaluative | Per domain |
|---|---|---|---|---|
| `prompt_dev` (used here) | **308** | **118** | **190** | 28 |
| `held_out` (never used here) | 132 | 54 | 78 | 12 |

- Runs: 3 prompts x 3 models x 308 sentences = 2,772 model answers.

## Files

| File | Content |
|---|---|
| `binary_prompt_selection.py` | Split, prompts (`PROMPT_BINARY_V1..V3`), inference (vLLM or Ollama), answer parsing, scoring |
| `prompts/V1.txt`, `V2.txt`, `V3.txt` | Read-only copies of the three prompts (the script uses its own copies) |
| `run_all.sh` / `run_all_ollama.sh` | All models with vLLM / Ollama, then `evaluate` |
| `outputs/gold_split.csv` | The 70 / 30 split used by every later step |
| `outputs/evaluative_metrics_summary.csv` | Metrics per model x prompt (`evaluate` writes this as `metrics_summary.csv`) |

Not included in this repository: `outputs/pred_<model>_<prompt>.csv` (the raw and parsed answer for each sentence),
`outputs/metrics_by_domain.csv` and `logs/`.

## Results on all 308 dev sentences (macro-F1)

| Model | V1 | V2 | V3 |
|---|---|---|---|
| Qwen3-30B | 0.696 | 0.645 (28 unparsed) | 0.732 |
| Llama-3.3-70B | 0.656 | 0.684 | 0.729 |
| Mistral-Large | 0.645 | 0.675 | **0.741** |

The final choice is made on class-balanced subsets in `../Prompt_selection` (winner: **V3**).

## Usage

```bash
python binary_prompt_selection.py run --backend ollama --model mistral-large-instruct
python binary_prompt_selection.py run --model llama-3.3-70b-instruct --tp 4      # vLLM
python binary_prompt_selection.py evaluate
```
