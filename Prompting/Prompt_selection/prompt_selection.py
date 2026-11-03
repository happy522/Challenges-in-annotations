"""
Final zero-shot prompt + model selection on class-balanced sets, over 3 runs,
with significance tests between models.

Uses the saved model outputs on the 70% prompt_dev split (../Binary_prompt_selection/outputs,
../Appraisal_prompt_selection/outputs); no model is run. The held-out 30% is not used anywhere.

Binary:    from the 308 prompt_dev sentences, 118 evaluative + 118 non-evaluative
           (all non-evaluative; evaluative sampled with the same domain mix).
Appraisal: from the 190 evaluative prompt_dev sentences, ambiguous ones are removed
           (more than one gold category, or Ambiguous = 1), then 32 Affect +
           32 Judgement + 32 Appreciation (all Affect; the others sampled with the
           same domain mix).

Runs: the balanced set is drawn 3 times (seeds 42, 43, 44); results are averaged.
Best prompt = highest macro-F1 averaged over models and runs; best model = highest
mean macro-F1 with that prompt.

Significance (per run, every model pair, same prompt, same sentences):
- paired approximate randomization test on macro-F1 (10,000 permutations)
- exact McNemar test on per-sentence correctness
p-values are Holm-corrected over the 3 model pairs.

Usage:
    python prompt_selection.py
Writes selection_results/ (selection sets, metrics per run, mean +- sd,
significance tests, summary).
"""

import glob
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY_DIR = os.path.join(HERE, "..", "Binary_prompt_selection", "outputs")
APPRAISAL_DIR = os.path.join(HERE, "..", "Appraisal_prompt_selection", "outputs")
OUT_DIR = os.path.join(HERE, "selection_results")

SEEDS = [42, 43, 44]
N_PERM = 10_000
ALPHA = 0.05
KEYS = ["Text_ID", "Sentence_ID"]
CATEGORIES = ["affect", "judgment", "appreciation"]
APPRAISAL_PER_CLASS = 32

# ==========================
# DATA
# ==========================

def load_preds(folder, n_expected):
    preds = {}
    for path in sorted(glob.glob(os.path.join(folder, "pred_*.csv"))):
        df = pd.read_csv(path, encoding="utf-8-sig")
        # only prompt_dev sentences may be used
        if "split" in df.columns and (df["split"] != "prompt_dev").any():
            raise ValueError(f"{path} contains held-out sentences")
        if len(df) != n_expected:
            raise ValueError(f"{path}: {len(df)} rows, expected {n_expected}")
        preds[(df["model"].iloc[0], df["prompt"].iloc[0])] = df
    if not preds:
        raise FileNotFoundError(f"No pred_*.csv in {folder}")
    return preds


def take(rows, n, seed):
    """n rows with the same domain mix as rows."""
    if n >= len(rows):
        return rows
    picked, _ = train_test_split(rows, train_size=n, random_state=seed, stratify=rows["Domain"])
    return picked


def binary_set(preds, seed):
    sents = next(iter(preds.values()))[KEYS + ["Domain", "Evaluative"]]
    n = int(sents["Evaluative"].value_counts().min())  # 118 = all non-evaluative
    return pd.concat([take(rows, n, seed) for _, rows in sents.groupby("Evaluative")])


def appraisal_set(preds, seed):
    sents = next(iter(preds.values()))[KEYS + ["Domain", "gold_label", "Ambiguous"]]
    ambiguous = (sents["gold_label"] == "multi") | (sents["Ambiguous"].fillna(0) == 1)
    single = sents[~ambiguous]
    return pd.concat([take(rows, APPRAISAL_PER_CLASS, seed)
                      for _, rows in single.groupby("gold_label")])

# ==========================
# PER-SENTENCE PREDICTIONS (gold, pred as arrays)
# ==========================

def binary_arrays(df):
    gold = df["Evaluative"].astype(int).values
    pred = df["pred_binary"].fillna(1 - df["Evaluative"]).astype(int).values  # unparsed = wrong
    return gold, pred


def appraisal_arrays(df):
    gold = df[CATEGORIES].astype(int).values
    pred = df[[f"pred_{c}" for c in CATEGORIES]].fillna(0).astype(int).values  # unparsed = nothing
    return gold, pred

# ==========================
# SCORES
# ==========================

def krippendorff_alpha(a, b):
    """Krippendorff's alpha (nominal) for two raters with no missing values:
    alpha = 1 - (n - 1) * sum_{c != k} o_ck / sum_{c != k} n_c * n_k
    with o the coincidence matrix and n = 2 * number of units."""
    a, b = np.asarray(a).astype(str), np.asarray(b).astype(str)
    labels = np.unique(np.concatenate([a, b]))
    if len(labels) < 2:
        return 1.0
    idx = {l: i for i, l in enumerate(labels)}
    o = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):  # each unit adds both ordered pairs
        o[idx[x], idx[y]] += 1
        o[idx[y], idx[x]] += 1
    n_c = o.sum(axis=1)
    n = n_c.sum()
    observed = o.sum() - np.trace(o)
    expected = n ** 2 - (n_c ** 2).sum()
    return 1 - (n - 1) * observed / expected


def binary_scores(df):
    gold, pred = binary_arrays(df)
    p, r, f, _ = precision_recall_fscore_support(gold, pred, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(gold, pred, labels=[0, 1]).ravel()
    return {"n": len(df), "unparsed": int(df["pred_binary"].isna().sum()),
            "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(gold, pred),
            "precision_eval": p[0], "recall_eval": r[0], "f1_eval": f[0],
            "cohen_kappa": cohen_kappa_score(gold, pred),
            "krippendorff_alpha": krippendorff_alpha(gold, pred),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def single_label(label, flags):
    """One predicted class ('ambiguous' if several or none)."""
    on = [c for c in CATEGORIES if flags[c]]
    if label in CATEGORIES and len(on) == 1:
        return label
    return on[0] if len(on) == 1 else "ambiguous"


def appraisal_scores(df):
    gold, pred = appraisal_arrays(df)
    p, r, f, _ = precision_recall_fscore_support(gold, pred, zero_division=0)
    pred_single = [single_label(l if isinstance(l, str) else None, dict(zip(CATEGORIES, fl)))
                   for l, fl in zip(df["pred_label"], pred)]
    row = {"n": len(df), "unparsed": int((~df["parsed"].astype(bool)).sum()),
           "macro_f1": f.mean(),
           "accuracy": accuracy_score(df["gold_label"], pred_single),
           "exact_match": (gold == pred).all(axis=1).mean(),
           # multiclass Cohen's kappa: gold category vs predicted category
           # (an "ambiguous" prediction is its own label, so it never agrees)
           "cohen_kappa": cohen_kappa_score(df["gold_label"], pred_single),
           "krippendorff_alpha": krippendorff_alpha(df["gold_label"], pred_single),
           "pred_ambiguous": pred_single.count("ambiguous")}
    for i, c in enumerate(CATEGORIES):
        row[f"f1_{c}"], row[f"p_{c}"], row[f"r_{c}"] = f[i], p[i], r[i]
        # per-category kappa: category present yes/no
        row[f"kappa_{c}"] = cohen_kappa_score(gold[:, i], pred[:, i])
        row[f"alpha_{c}"] = krippendorff_alpha(gold[:, i], pred[:, i])
    return row

# ==========================
# SIGNIFICANCE
# ==========================

def macro_f1_batch(gold, pred):
    """Macro-F1 for a batch of predictions: pred (P, n) binary labels or (P, n, 3) flags.
    Binary: F1 of class 0 and class 1 averaged; appraisal: F1 per category averaged."""
    if pred.ndim == 2:  # binary -> one-hot (P, n, 2)
        gold = np.stack([gold == 0, gold == 1], axis=-1)
        pred = np.stack([pred == 0, pred == 1], axis=-1)
    gold, pred = gold.astype(bool), pred.astype(bool)
    tp = (pred & gold).sum(axis=1)
    fp = (pred & ~gold).sum(axis=1)
    fn = (~pred & gold).sum(axis=1)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1.mean(axis=-1)


def randomization_test(gold, pred_a, pred_b, rng):
    """Paired approximate randomization: swap the two models' predictions per sentence."""
    observed = macro_f1_batch(gold, np.stack([pred_a, pred_b]))
    diff = observed[0] - observed[1]
    swap = rng.random((N_PERM, len(gold))) < 0.5
    if pred_a.ndim == 2:
        swap = swap[:, :, None]
    perm_a = np.where(swap, pred_b[None], pred_a[None])
    perm_b = np.where(swap, pred_a[None], pred_b[None])
    perm_diff = macro_f1_batch(gold, perm_a) - macro_f1_batch(gold, perm_b)
    p = (np.sum(np.abs(perm_diff) >= abs(diff) - 1e-12) + 1) / (N_PERM + 1)
    return diff, p


def mcnemar(gold, pred_a, pred_b):
    """Exact McNemar on per-sentence correctness (all categories right for appraisal)."""
    right_a = (pred_a == gold) if pred_a.ndim == 1 else (pred_a == gold).all(axis=1)
    right_b = (pred_b == gold) if pred_b.ndim == 1 else (pred_b == gold).all(axis=1)
    b = int((right_a & ~right_b).sum())
    c = int((~right_a & right_b).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def significance(preds, selected, arrays, seed, rng):
    rows = []
    models = sorted({m for m, _ in preds})
    prompts = sorted({p for _, p in preds})
    for prompt in prompts:
        data = {}
        for m in models:
            part = preds[(m, prompt)].merge(selected[KEYS], on=KEYS, how="inner")
            part = part.set_index(KEYS).loc[selected.set_index(KEYS).index]  # same order
            data[m] = arrays(part)
        family = []
        for a, b in itertools.combinations(models, 2):
            gold = data[a][0]
            diff, p_rand = randomization_test(gold, data[a][1], data[b][1], rng)
            only_a, only_b, p_mc = mcnemar(gold, data[a][1], data[b][1])
            family.append({"seed": seed, "prompt": prompt, "model_a": a, "model_b": b,
                           "macro_f1_diff": diff, "p_randomization": p_rand,
                           "only_a_correct": only_a, "only_b_correct": only_b,
                           "p_mcnemar": p_mc})
        for key in ("p_randomization", "p_mcnemar"):
            adj = holm([r[key] for r in family])
            for r, v in zip(family, adj):
                r[f"{key}_holm"] = v
        rows += family
    return pd.DataFrame(rows)

# ==========================
# MAIN
# ==========================

def score(preds, selected, scorer, seed):
    rows = []
    for (model, prompt), df in preds.items():
        part = df.merge(selected[KEYS], on=KEYS, how="inner")
        assert len(part) == len(selected)
        rows.append({"seed": seed, "model": model, "prompt": prompt, **scorer(part)})
    return rows


def short(model):
    return model.split("-")[0]


def run(name, folder, n_expected, make_set, label_col, scorer, arrays, cols, lines):
    preds = load_preds(folder, n_expected)
    rng = np.random.default_rng(0)
    runs, tests, sets = [], [], []
    for seed in SEEDS:
        selected = make_set(preds, seed)
        sets.append(selected.assign(seed=seed))
        runs += score(preds, selected, scorer, seed)
        tests.append(significance(preds, selected, arrays, seed, rng))
    runs = pd.DataFrame(runs).sort_values(["seed", "model", "prompt"])
    tests = pd.concat(tests)
    sets = pd.concat(sets)
    runs.to_csv(os.path.join(OUT_DIR, f"{name}_metrics_per_run.csv"), index=False)
    tests.to_csv(os.path.join(OUT_DIR, f"{name}_significance.csv"), index=False)
    sets.to_csv(os.path.join(OUT_DIR, f"{name}_selection_sets.csv"), index=False, encoding="utf-8-sig")

    mean = runs.groupby(["model", "prompt"])[cols].mean()
    sd = runs.groupby(["model", "prompt"])[cols].std(ddof=1)
    summary = mean.add_suffix("_mean").join(sd.add_suffix("_sd")).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, f"{name}_metrics_mean_sd.csv"), index=False)

    by_prompt = runs.groupby("prompt")["macro_f1"].mean()
    best_prompt = by_prompt.idxmax()
    with_prompt = mean.xs(best_prompt, level="prompt")["macro_f1"].sort_values(ascending=False)
    best_model = with_prompt.index[0]

    first = sets[sets["seed"] == SEEDS[0]]
    counts = first[label_col].value_counts().sort_index().to_dict()
    overlap = sets.groupby(KEYS).size()
    fmt = lambda m, s: f"{m:.3f} ± {s:.3f}"
    pivot = pd.DataFrame({p: [fmt(mean.loc[(m, p), "macro_f1"], sd.loc[(m, p), "macro_f1"])
                              for m in mean.index.levels[0]] for p in mean.index.levels[1]},
                         index=mean.index.levels[0])
    table = summary[["model", "prompt"]].copy()
    for c in cols:
        table[c] = [fmt(m, s) for m, s in zip(summary[f"{c}_mean"], summary[f"{c}_sd"])]

    t = tests[tests["prompt"] == best_prompt].copy()
    t["pair"] = [f"{short(a)} - {short(b)}" for a, b in zip(t["model_a"], t["model_b"])]
    t["sig"] = np.where(t["p_randomization_holm"] < ALPHA, "yes", "no")
    sig_view = t[["seed", "pair", "macro_f1_diff", "p_randomization_holm",
                  "only_a_correct", "only_b_correct", "p_mcnemar_holm", "sig"]]

    lines += [f"=== {name.capitalize()}: {len(first)} sentences per run {counts}, "
              f"{len(SEEDS)} runs (seeds {SEEDS}) ===",
              f"Sentences in all {len(SEEDS)} runs: {(overlap == len(SEEDS)).sum()}, "
              f"different sentences used overall: {len(overlap)}",
              "", "Mean ± sd over runs:", table.to_string(index=False), "",
              "Macro-F1 mean ± sd (rows: model, columns: prompt):", pivot.to_string(), "",
              f"Mean macro-F1 per prompt: {by_prompt.round(3).to_dict()}",
              f"Best prompt: {best_prompt}",
              f"Best model with {best_prompt}: {best_model} "
              f"(macro-F1 {fmt(mean.loc[(best_model, best_prompt), 'macro_f1'], sd.loc[(best_model, best_prompt), 'macro_f1'])})",
              "",
              f"Model comparison with {best_prompt} (Holm-corrected p; sig = randomization p < {ALPHA}):",
              sig_view.round(4).to_string(index=False), "", ""]
    summary = summary.assign(task=name, best=(summary["model"] == best_model)
                             & (summary["prompt"] == best_prompt))
    return summary, tests.assign(task=name)


def final_table(summary, tests):
    """One row per task x model x prompt: mean ± sd of the main metrics, and the
    significance test against the top model for the same task and prompt."""
    fmt = lambda m, s: f"{m:.3f} ± {s:.3f}"
    rows = []
    for (task, prompt), group in summary.groupby(["task", "prompt"], sort=False):
        top = group.sort_values("macro_f1_mean", ascending=False)["model"].iloc[0]
        for rank, r in enumerate(group.sort_values("macro_f1_mean", ascending=False).itertuples(), 1):
            row = {"task": task, "prompt": prompt, "model": r.model, "rank_in_prompt": rank}
            for m in ("macro_f1", "accuracy", "cohen_kappa", "krippendorff_alpha"):
                row[m] = fmt(getattr(r, f"{m}_mean"), getattr(r, f"{m}_sd"))
            if r.model == top:
                row.update({"compared_to": "(top model)", "macro_f1_diff": "", "p_randomization_holm": "",
                            "p_mcnemar_holm": "", "significant_runs": ""})
            else:
                t = tests[(tests["task"] == task) & (tests["prompt"] == prompt)
                          & tests[["model_a", "model_b"]].isin([r.model, top]).all(axis=1)]
                sign = np.where(t["model_a"] == r.model, 1, -1)  # diff as this model - top model
                diff = sign * t["macro_f1_diff"]
                pr, pm = t["p_randomization_holm"], t["p_mcnemar_holm"]
                row.update({"compared_to": top,
                            "macro_f1_diff": fmt(diff.mean(), diff.std(ddof=1)),
                            "p_randomization_holm": f"{pr.min():.3f}-{pr.max():.3f}",
                            "p_mcnemar_holm": f"{pm.min():.3f}-{pm.max():.3f}",
                            "significant_runs": f"{int((pr < ALPHA).sum())}/{len(pr)}"})
            row["chosen"] = "yes" if r.best else ""
            rows.append(row)
    return pd.DataFrame(rows)


METHOD_NOTES = f"""How to read the table
- Values are mean ± sd over {len(SEEDS)} runs (seeds {SEEDS}); each run draws a different balanced set
  from the 70% prompt_dev split (binary 118 + 118, appraisal 32 + 32 + 32, ambiguous removed).
  The held-out 30% is not used.
- Cohen's kappa and Krippendorff's alpha (nominal): model vs gold. Appraisal uses the predicted
  category; an "ambiguous" prediction counts as disagreement.
- Significance: every non-top model is compared with the top model of the same task and prompt,
  on the same sentences, in each run.
  - p_randomization_holm: paired approximate randomization test on macro-F1 ({N_PERM:,} permutations).
  - p_mcnemar_holm: exact McNemar test on per-sentence correctness.
  - Both are Holm-Bonferroni corrected over the 3 model pairs of that prompt and run; the range over
    the runs is shown. significant_runs = runs with randomization p < {ALPHA}.
- chosen = best prompt (highest mean macro-F1 over models and runs) with its best model.
"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = ["Zero-shot prompt + model selection on balanced sets (70% prompt_dev split),",
             f"{len(SEEDS)} runs with different balanced samples. The held-out 30% is not used.",
             "", ""]
    binary, binary_tests = run("binary", BINARY_DIR, 308, binary_set, "Evaluative", binary_scores, binary_arrays,
                 ["macro_f1", "accuracy", "precision_eval", "recall_eval", "f1_eval",
                  "cohen_kappa", "krippendorff_alpha"], lines)
    appraisal, appraisal_tests = run("appraisal", APPRAISAL_DIR, 190, appraisal_set, "gold_label", appraisal_scores,
                    appraisal_arrays,
                    ["macro_f1", "accuracy", "cohen_kappa", "krippendorff_alpha",
                     "f1_affect", "f1_judgment", "f1_appreciation",
                     "kappa_affect", "kappa_judgment", "kappa_appreciation",
                     "alpha_affect", "alpha_judgment", "alpha_appreciation"], lines)

    # final results: one table with the main metrics and the significance tests
    final = final_table(pd.concat([binary, appraisal]), pd.concat([binary_tests, appraisal_tests]))
    final.to_csv(os.path.join(OUT_DIR, "final_results.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT_DIR, "final_results.txt"), "w", encoding="utf-8") as f:
        f.write("Final results: zero-shot prompt + model selection\n\n")
        f.write(final.to_string(index=False) + "\n\n" + METHOD_NOTES)

    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "selection_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
