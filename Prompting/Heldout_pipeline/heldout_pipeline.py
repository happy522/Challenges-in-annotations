"""
Final test on the held-out 30% of the gold standard (132 sentences, never used for
prompt selection): a two-step pipeline with the winning prompts.

    Step 1  binary     (Evaluative 0/1)               prompt V3 (Binary_prompt_selection)
    Step 2  appraisal  (Affect/Judgement/Appreciation) prompt V2 (Appraisal_prompt_selection)
            only for sentences step 1 marked as evaluative

Step 2 is also run on gold-evaluative sentences that step 1 missed, so the appraisal
step can be scored on its own too; the pipeline scores only use what step 1 passed on.

Scoring matches the fine-tuning test: ambiguous sentences are removed (125 left), and
binary / appraisal (gold-evaluative) / 4-class pipeline get macro-F1, accuracy, Cohen's
kappa and Krippendorff's alpha, with paired significance tests between the models.
The fine-tuned Llama predictions are added as an extra row when FINETUNED_PRED exists.

Prompts, parsers and the split are imported from the two prompt-selection folders,
which must sit next to this one (or set BINARY_DIR / APPRAISAL_DIR).

Usage (on the GPU server):
    export GOLD_CSV=/path/to/gold_standard_with_context.csv
    python heldout_pipeline.py run --model llama-3.3-70b-instruct --ollama-url http://127.0.0.1:11500
    python heldout_pipeline.py evaluate

    # or all 3 models in one go: bash run_all_ollama.sh
"""

import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getenv("BINARY_DIR", os.path.join(HERE, "..", "Binary_prompt_selection")))
sys.path.insert(0, os.getenv("APPRAISAL_DIR", os.path.join(HERE, "..", "Appraisal_prompt_selection")))
import binary_prompt_selection as B      # noqa: E402
import appraisal_prompt_selection as A   # noqa: E402

OUT_DIR = os.getenv("PIPELINE_OUT_DIR", os.path.join(HERE, "outputs"))
KEYS = ["Text_ID", "Sentence_ID"]
CATEGORIES = A.CATEGORIES

BINARY_PROMPT = "V3"     # winner of the binary prompt selection
APPRAISAL_PROMPT = "V2"  # winner of the appraisal prompt selection

# ==========================
# DATA
# ==========================

def held_out():
    """The held-out sentences of the binary split, with gold labels and context."""
    gold = pd.read_csv(A.GOLD_CSV, encoding="utf-8-sig")
    split = pd.read_csv(A.BINARY_SPLIT, encoding="utf-8-sig")[KEYS + ["split"]]
    df = gold.merge(split, on=KEYS, how="inner")
    df = df[df["split"] == "held_out"].reset_index(drop=True)
    df["Domain"] = df["Text_ID"].map(A.domain_of)
    df["Evaluative"] = df["Evaluative"].astype(int)
    for cat, col in A.GOLD_COLS.items():
        df[cat] = df[col].fillna(0).astype(int) * df["Evaluative"]
    print(f"[INFO] Held-out: {len(df)} sentences "
          f"({df['Evaluative'].sum()} evaluative / {(df['Evaluative'] == 0).sum()} not)")
    return B.add_context(df)

# ==========================
# RUN
# ==========================

def generator(args, max_model_len):
    step = argparse.Namespace(**{**vars(args), "max_model_len": max_model_len})
    return A.ollama_generator(step) if args.backend == "ollama" else A.vllm_generator(step)


def generate_cached(rows, prompts, cache, make_generate):
    """Run only the rows without a saved non-empty answer (same model, temperature 0)."""
    keys = list(zip(rows["Text_ID"], rows["Sentence_ID"]))
    raw = [cache.get(k, "") for k in keys]
    todo = [i for i, r in enumerate(raw) if not str(r).strip()]
    print(f"[INFO] {len(keys) - len(todo)} reused, {len(todo)} to run")
    if todo:
        for i, text in zip(todo, make_generate()([prompts[i] for i in todo])):
            raw[i] = text
    return raw


def run(args):
    data = held_out()
    if args.limit:
        data = data.head(args.limit)
    # --binary-from: step 1 answers of another model (its saved run), step 2 by --model
    name = f"{args.binary_from} + {args.model}" if args.binary_from else args.model
    path = os.path.join(OUT_DIR, f"pred_{name.replace(' + ', '+')}.csv")

    def load(p):
        return pd.read_csv(p, encoding="utf-8-sig") if os.path.exists(p) and not args.no_cache else None

    def answers(df, col):
        if df is None:
            return {}
        rows = df[df[col].fillna("").astype(str).str.strip() != ""]
        return {(r.Text_ID, r.Sentence_ID): getattr(r, col) for r in rows.itertuples()}

    old = load(path)
    own = load(os.path.join(OUT_DIR, f"pred_{args.model}.csv")) if args.binary_from else None

    def cache(col):
        if col == "binary_raw" and args.binary_from:
            src = os.path.join(OUT_DIR, f"pred_{args.binary_from}.csv")
            saved = answers(pd.read_csv(src, encoding="utf-8-sig"), col)
            if len(saved) < len(data):
                raise SystemExit(f"{src} is missing step-1 answers - run {args.binary_from} first")
            return saved
        # appraisal answers of --model from its own run, so only new sentences are generated
        return {**answers(own, col), **answers(old, col)}

    vllm = {}  # one vLLM engine for both steps (context length does not change outputs)

    def make_generate(max_len):
        if args.backend == "vllm":
            if "gen" not in vllm:
                vllm["gen"] = generator(args, max(args.binary_ctx, args.appraisal_ctx))
            return lambda: vllm["gen"]
        return lambda: generator(args, max_len)

    # Step 1: evaluative or not
    print(f"[INFO] {args.binary_from or args.model} step 1: binary {BINARY_PROMPT}")
    prompts = [[{"role": "user", "content": B.PROMPTS[BINARY_PROMPT].format(
                    prev_text=r.prev_text, target_text=r.Sentence, next_text=r.next_text)}]
               for r in data.itertuples()]
    data["binary_raw"] = generate_cached(data, prompts, cache("binary_raw"),
                                         make_generate(args.binary_ctx))
    data["pred_binary_raw"] = [B.parse_output(t)[0] for t in data["binary_raw"]]
    # unparsed binary answers are not passed on to step 2
    data["pred_evaluative"] = data["pred_binary_raw"].fillna(0).astype(int)

    # Step 2: appraisal for predicted-evaluative (+ gold-evaluative, for step-2-only scores)
    step2 = data[(data["pred_evaluative"] == 1) | (data["Evaluative"] == 1)].reset_index(drop=True)
    print(f"[INFO] {args.model} step 2: appraisal {APPRAISAL_PROMPT} on {len(step2)} sentences "
          f"({data['pred_evaluative'].sum()} predicted evaluative)")
    prompts = [[{"role": "user", "content": A.fill(A.PROMPTS[APPRAISAL_PROMPT], r.prev_text,
                                                   r.Sentence, r.next_text)}]
               for r in step2.itertuples()]
    step2["appraisal_raw"] = generate_cached(step2, prompts, cache("appraisal_raw"),
                                             make_generate(args.appraisal_ctx))
    data = data.merge(step2[KEYS + ["appraisal_raw"]], on=KEYS, how="left")
    data["appraisal_raw"] = data["appraisal_raw"].fillna("")

    parsed = [A.parse_output(t) if t else (None, None, None) for t in data["appraisal_raw"]]
    data["appraisal_parsed"] = [p[1] is not None for p in parsed]
    data["pred_label"] = [p[0] for p in parsed]
    for c in CATEGORIES:
        data[f"pred_{c}"] = [p[1][c] if p[1] else 0 for p in parsed]
    data["explanation"] = [(p[2] or {}).get("explanation", "") for p in parsed]
    data["model"] = name
    data["binary_prompt"], data["appraisal_prompt"] = BINARY_PROMPT, APPRAISAL_PROMPT

    os.makedirs(OUT_DIR, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[INFO] {name}: saved {path} "
          f"({data['pred_binary_raw'].isna().sum()} binary unparsed, "
          f"{((data['appraisal_raw'] != '') & ~data['appraisal_parsed']).sum()} appraisal unparsed)")
    evaluate()

# ==========================
# EVALUATION
# ==========================

# Scored like the fine-tuning test (Finetuning), on the same sentences: the held-out
# 30% without ambiguous ones (several gold categories, or Ambiguous = 1) -> 125 sentences.
# Labels are strings; an unreadable answer is "invalid" and an answer with several or no
# categories is "ambiguous" - both count as wrong.

N_PERM = 10_000
# Step 2 is scored only on the sentences the model's own step 1 called evaluative; a gold
# non-evaluative sentence among them has gold label "none" and is always wrong. The combined
# step scores both prompts together on all test sentences.
STEP_1 = "1 binary (all test sentences)"
STEP_2 = "2 appraisal (sentences the model called evaluative)"
COMBINED = "combined binary + appraisal (none / affect / judgment / appreciation)"
STEPS = {  # step -> (gold column, prediction column, labels for macro-F1)
    STEP_1: ("gold_binary", "pred_binary", ["0", "1"]),
    STEP_2: ("label4", "pred_category", CATEGORIES),
    COMBINED: ("label4", "pred_label4", ["none"] + CATEGORIES),
}
# Finetuning (or Finetune_llama) next to this folder (server) or next to Prompting/ (Final)
FINETUNED = os.getenv("FINETUNED_PRED", next(
    (p for p in (os.path.join(HERE, up, name, "outputs", "test_finetuned", "predictions.csv")
                 for up in ("..", os.path.join("..", ".."))
                 for name in ("Finetuning", "Finetune_llama")) if os.path.exists(p)), ""))


def krippendorff_alpha(a, b):
    """Krippendorff's alpha (nominal), two raters, no missing values."""
    a, b = np.asarray(a).astype(str), np.asarray(b).astype(str)
    labels = np.unique(np.concatenate([a, b]))
    if len(labels) < 2:
        return 1.0
    idx = {l: i for i, l in enumerate(labels)}
    o = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        o[idx[x], idx[y]] += 1
        o[idx[y], idx[x]] += 1
    n_c = o.sum(axis=1)
    n = n_c.sum()
    return 1 - (n - 1) * (o.sum() - np.trace(o)) / (n ** 2 - (n_c ** 2).sum())


def pipeline_labels(df):
    """Gold and predicted string labels for the three steps, ambiguous sentences removed."""
    df = df.copy()
    n_cats = df[CATEGORIES].astype(int).sum(axis=1)
    amb = ((df["Evaluative"] == 1) & (n_cats != 1))
    if "Ambiguous" in df.columns:
        amb |= df["Ambiguous"].fillna(0).astype(int) == 1
    df = df[~amb].reset_index(drop=True)

    df["gold_binary"] = df["Evaluative"].astype(int).astype(str)
    df["category"] = [next((c for c in CATEGORIES if r[c]), "none") for _, r in df.iterrows()]
    df["label4"] = df["category"]
    df["pred_binary"] = df["pred_binary_raw"].map(lambda v: "invalid" if pd.isna(v) else str(int(v)))
    flags = df[[f"pred_{c}" for c in CATEGORIES]].fillna(0).astype(int)
    df["pred_category"] = [
        "invalid" if not ok else A.single_label(l if isinstance(l, str) else None, dict(zip(CATEGORIES, f)))
        for ok, l, f in zip(df["appraisal_parsed"].astype(bool), df["pred_label"], flags.values)]
    df["pred_label4"] = np.where(df["pred_binary"] == "1", df["pred_category"], "none")
    return df, int(amb.sum())


def finetuned_labels(path):
    """The fine-tuned Llama predictions on the same 125 sentences (Finetuning test)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["gold_binary"] = df["Evaluative"].astype(int).astype(str)
    df["pred_binary"] = df["pred_binary"].map(lambda v: "invalid" if pd.isna(v) else str(int(v)))
    for col in ["category", "label4", "pred_category", "pred_label4"]:
        df[col] = df[col].fillna("none" if col in ("category", "label4") else "invalid").astype(str)
    df["model"] = "llama-3.3-70b-instruct (fine-tuned)"
    return df


def step_scores(gold, pred, labels):
    gold, pred = pd.Series(gold).astype(str), pd.Series(pred).astype(str)
    p, r, f, _ = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    row = {"n": len(gold), "invalid": int(pred.isin(["invalid", "ambiguous"]).sum()),
           "macro_f1": f.mean(), "accuracy": (gold == pred).mean(),
           "cohen_kappa": cohen_kappa_score(gold, pred),
           "krippendorff_alpha": krippendorff_alpha(gold, pred)}
    for i, l in enumerate(labels):
        name = {"0": "noneval", "1": "eval"}.get(l, l)
        row[f"precision_{name}"], row[f"recall_{name}"], row[f"f1_{name}"] = p[i], r[i], f[i]
        if l in CATEGORIES:  # agreement on "this category: yes / no"
            row[f"kappa_{l}"] = cohen_kappa_score(gold == l, pred == l)
    return row


def macro_f1_batch(gold, pred, labels):
    """Macro-F1 for a batch of label arrays pred (P, n)."""
    g = np.stack([gold == l for l in labels], axis=-1)[None]
    p = np.stack([pred == l for l in labels], axis=-1)
    tp, fp, fn = (p & g).sum(1), (p & ~g).sum(1), (~p & g).sum(1)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean(-1)


def randomization_test(gold, a, b, labels, rng):
    """Paired approximate randomization on macro-F1 (swap the two answers per sentence)."""
    diff = np.diff(macro_f1_batch(gold, np.stack([b, a]), labels))[0]
    swap = rng.random((N_PERM, len(gold))) < 0.5
    perm = (macro_f1_batch(gold, np.where(swap, b, a), labels)
            - macro_f1_batch(gold, np.where(swap, a, b), labels))
    return diff, (np.sum(np.abs(perm) >= abs(diff) - 1e-12) + 1) / (N_PERM + 1)


def mcnemar(gold, a, b):
    right_a, right_b = a == gold, b == gold
    only_a, only_b = int((right_a & ~right_b).sum()), int((~right_a & right_b).sum())
    return only_a, only_b, (binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0)


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    adj, running = np.empty_like(pvals), 0.0
    for rank, i in enumerate(np.argsort(pvals)):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def evaluate(pred_dir=OUT_DIR, out_dir=None):
    out_dir = out_dir or os.path.join(pred_dir, "results")
    files = sorted(glob.glob(os.path.join(pred_dir, "pred_*.csv")))
    if not files:
        print("[WARN] No prediction files found in", pred_dir)
        return
    os.makedirs(out_dir, exist_ok=True)

    data, n_amb = {}, 0
    for path in files:
        df, n_amb = pipeline_labels(pd.read_csv(path, encoding="utf-8-sig"))
        data[df["model"].iloc[0]] = df.set_index(KEYS).sort_index()
    zero_shot = sorted(data)
    if os.path.exists(FINETUNED):
        ft = finetuned_labels(FINETUNED).set_index(KEYS).sort_index()
        if ft.index.equals(data[zero_shot[0]].index):
            data[ft["model"].iloc[0]] = ft
        else:
            print(f"[WARN] {FINETUNED} has other sentences than the held-out test, not added")

    rows = []
    for step, (g, p, labels) in STEPS.items():
        for model, df in data.items():
            part = df[df["pred_binary"] == "1"] if step == STEP_2 else df
            row = {"step": step, "model": model, **step_scores(part[g], part[p], labels)}
            if step == STEP_2:
                row["gold_noneval"] = int((part["Evaluative"] == 0).sum())
                # agreement per subclass as in the human annotation study: Krippendorff's alpha
                # on the sentences that both the gold standard and the model call evaluative
                both = part[part["Evaluative"] == 1]
                row["n_both_eval"] = len(both)
                for c in CATEGORIES:
                    row[f"alpha_{c}"] = krippendorff_alpha(both["category"] == c, both["pred_category"] == c)
            rows.append(row)
    metrics = pd.DataFrame(rows)

    # paired tests per step (families and Holm below). Step 2 has
    # different sentences per model (each model's own step 1), so it gets no paired test.
    # Families (Holm within each): the single models against each other, and every
    # combination (binary of one model + appraisal of another) against the single models.
    singles = [m for m in zero_shot if " + " not in m]
    families = [list(itertools.combinations(singles, 2))]
    families += [[(c, s) for s in singles] for c in zero_shot if " + " in c]
    rng, tests = np.random.default_rng(0), []
    for pairs in families:  # single models first, so their p-values do not depend on the combinations
        for step, (g, p, labels) in STEPS.items():
            if step == STEP_2:
                continue
            family = []
            for a, b in pairs:
                da, db = data[a], data[b]
                gold = da[g].astype(str).values
                pa, pb = da[p].astype(str).values, db[p].astype(str).values
                diff, p_rand = randomization_test(gold, pa, pb, labels, rng)
                only_a, only_b, p_mc = mcnemar(gold, pa, pb)
                family.append({"step": step, "model_a": a, "model_b": b, "macro_f1_diff (a - b)": diff,
                               "p_randomization": p_rand, "only_a_correct": only_a,
                               "only_b_correct": only_b, "p_mcnemar": p_mc})
            for key in ("p_randomization", "p_mcnemar"):
                for r, v in zip(family, holm([r[key] for r in family])):
                    r[f"{key}_holm"] = v
            tests += family
    tests = pd.DataFrame(tests)

    metrics.to_csv(os.path.join(out_dir, "heldout_metrics.csv"), index=False)
    tests.to_csv(os.path.join(out_dir, "heldout_significance.csv"), index=False)
    pd.concat([df.reset_index().assign(model=m) for m, df in data.items()]).to_csv(
        os.path.join(out_dir, "heldout_predictions.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    main_cols = ["step", "model", "n", "gold_noneval", "invalid", "macro_f1", "accuracy",
                 "cohen_kappa", "krippendorff_alpha"]
    lines = [f"=== Held-out test (30%, {n_amb} ambiguous removed): binary {BINARY_PROMPT} -> "
             f"appraisal {APPRAISAL_PROMPT}, zero-shot ===", "",
             metrics[main_cols].round(3).to_string(index=False), "",
             "Per class", metrics[["step", "model"] + [c for c in metrics.columns
                                                     if c.startswith(("f1_", "kappa_", "alpha_"))]]
             .round(3).to_string(index=False), "",
             "=== Significance between zero-shot models (paired, same sentences; Holm within: single models, and each combination vs the single models) ===",
             tests.round(4).to_string(index=False), "",
             "p_randomization: approximate randomization test on macro-F1 (10,000 permutations);",
             "p_mcnemar: exact McNemar test on per-sentence correctness.",
             "Step 2: only the sentences each model's step 1 called evaluative (n differs per model;",
             "gold_noneval = how many of them are not evaluative in the gold standard - always wrong).",
             "Combined: both prompts together on all test sentences (step-1 'no' = none).",
             "Invalid / ambiguous answers count as wrong. The fine-tuned row is for comparison only",
             "(its test against zero-shot Llama is in Finetuning/outputs/test_significance.csv)."]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "heldout_results.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"\n[INFO] Saved results to {out_dir}")

# ==========================
# CLI
# ==========================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the two-step pipeline for one model")
    r.add_argument("--model", required=True, choices=list(A.MODELS))
    r.add_argument("--binary-from", default=None, choices=list(A.MODELS),
                   help="take step 1 from this model's saved run; --model does step 2 "
                        "(e.g. best binary model + best appraisal model)")
    r.add_argument("--backend", choices=["vllm", "ollama"], default="ollama")
    r.add_argument("--ollama-url", default=os.getenv("OLLAMA_HOST_URL", "http://localhost:11434"))
    r.add_argument("--ollama-model", default=None, help="Ollama tag overriding the default one")
    r.add_argument("--workers", type=int, default=4,
                   help="parallel Ollama requests (match OLLAMA_NUM_PARALLEL)")
    r.add_argument("--hf-path", default=None, help="local path / HF id overriding the default checkpoint")
    r.add_argument("--tp", type=int, default=None, help="vLLM tensor parallel size (default: all GPUs)")
    # same context sizes as in the prompt selection runs
    r.add_argument("--binary-ctx", type=int, default=8192)
    r.add_argument("--appraisal-ctx", type=int, default=4096)
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--gpu-mem", type=float, default=0.90)
    r.add_argument("--seed", type=int, default=A.SEED)
    r.add_argument("--no-cache", action="store_true",
                   help="rerun every sentence instead of reusing saved outputs")
    r.add_argument("--limit", type=int, default=None, help="only first N rows (smoke test)")

    e = sub.add_parser("evaluate", help="compute metrics for all prediction files")
    e.add_argument("--pred-dir", default=OUT_DIR, help="folder with the pred_*.csv files")
    e.add_argument("--out-dir", default=None, help="where to write results (default: <pred-dir>/results)")

    args = ap.parse_args()
    if args.cmd == "run":
        if args.backend == "vllm" and args.tp is None:
            import torch
            args.tp = max(torch.cuda.device_count(), 1)
        run(args)
    else:
        evaluate(args.pred_dir, args.out_dir)


if __name__ == "__main__":
    main()
