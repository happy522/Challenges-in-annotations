"""
Figures and tables for the zero-shot prompt analysis and the Llama fine-tuning,
in the same style as the paper's other figures.

Reads:
    Prompting/Prompt_selection/selection_results/  (prompt selection: balanced 70% sets, mean of 3 runs)
    Prompting/Heldout_pipeline/outputs/results/    (held-out 30% test: binary V3 -> appraisal V2, 3 models)
    Finetuning/outputs/                            (held-out 30% test, fine-tuned vs zero-shot Llama)
Writes:
    Prompting/Figures/*.png, *.csv
    Finetuning/Figures/*.png, *.csv

Usage:  python make_result_figures.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
SEL = os.path.join(ROOT, "Prompting", "Prompt_selection", "selection_results")
HELDOUT = os.getenv("HELDOUT_RESULTS", os.path.join(ROOT, "Prompting", "Heldout_pipeline", "outputs", "results"))
FT = os.path.join(ROOT, "Finetuning", "outputs")
OUT_PROMPT = os.getenv("PROMPT_FIG_DIR", os.path.join(ROOT, "Prompting", "Figures"))
OUT_FT = os.path.join(ROOT, "Finetuning", "Figures")

# same models, colours and hatches as the existing LLM figures
MODELS = {
    "qwen3-30b-a3b-instruct-2507": ("Qwen3-30B", "Qwen", "#4C72B0", "//"),
    "llama-3.3-70b-instruct": ("Llama-3.3-70B", "Llama", "#55A868", "\\\\"),
    "mistral-large-instruct": ("Mistral-Large", "Mistral", "#C44E52", "xx"),
}
FINETUNED = ("llama-3.3-70b-instruct (fine-tuned)", "Llama-3.3-70B fine-tuned", "#8172B2", "..")
# first two words of the step names in the result files -> axis label
STEP_NAMES = {"1 binary": "Evaluative\n(binary)", "2 appraisal": "Subclasses\n(model-evaluative)",
              "combined binary": "Combined\n(binary + appraisal)"}
SETUPS = {  # fine-tuning comparison
    "zeroshot": ("Zero-shot", "#55A868", "\\\\"),
    "finetuned": ("Fine-tuned", "#8172B2", ".."),
}
CATEGORIES = ["affect", "judgment", "appreciation"]
CAT_NAMES = ["Affect", "Judgement", "Appreciation"]
BINARY_PROMPT, APPRAISAL_PROMPT = "V3", "V2"
DPI = 600

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "hatch.linewidth": 1.0,
})

# ==========================
# PLOT HELPERS
# ==========================

def grouped_bars(groups, series, ylabel, path, ylim=(0, 1.0), legend_loc="upper right",
                 errors=None, pvalues=None, box=False):
    """groups: x labels; series: list of (label, colour, hatch, values)."""
    fig, ax = plt.subplots(figsize=(6.2, 3.45))
    x = np.arange(len(groups))
    width = 0.8 / len(series) if len(series) > 2 else 0.3
    for i, (label, colour, hatch, values) in enumerate(series):
        pos = x + (i - (len(series) - 1) / 2) * width
        err = errors[i] if errors is not None else None
        ax.bar(pos, values, width, label=label, color=colour, hatch=hatch,
               edgecolor="black", linewidth=0.8, zorder=3,
               yerr=err, capsize=2.5 if err is not None else 0,
               error_kw={"elinewidth": 0.8, "ecolor": "black", "zorder": 4})
        for p, v, e in zip(pos, values, err if err is not None else [0] * len(values)):
            ax.text(p, v + e + (0.028 if err is not None else 0.02) * (ylim[1] - ylim[0]), f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5, zorder=5)
    if pvalues is not None:
        for xi, p, top in zip(x, pvalues, np.max([s[3] for s in series], axis=0)):
            if p is None:  # no paired test for this group
                continue
            ax.text(xi, top + 0.11 * (ylim[1] - ylim[0]), f"p = {p:.2f}",
                    ha="center", va="bottom", fontsize=8.5, style="italic", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=10)
    ax.yaxis.grid(True, linestyle="--", color="#aaaaaa", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if not box:
        ax.spines[["top", "right"]].set_visible(False)
    if legend_loc == "above":  # outside the axes, for figures with many bars
        leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9.5,
                        frameon=True, fancybox=True, edgecolor="black")
    else:
        leg = ax.legend(loc=legend_loc, fontsize=9.5, frameon=True, fancybox=True, edgecolor="black")
    leg.get_frame().set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"[SAVED] {os.path.relpath(path, ROOT)}")


def heatmap(values, rows, cols, path, vmin=0.2, vmax=0.6, label="Cohen's κ"):
    with plt.rc_context({"font.family": "serif", "font.serif": ["DejaVu Serif"]}):
        fig, ax = plt.subplots(figsize=(10.4, 4.65))
        im = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=21, color="white")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=17)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=17)
        ax.tick_params(length=6, width=1.2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.spines[["left", "bottom"]].set_visible(True)
        ax.spines[["left", "bottom"]].set_linewidth(1.2)
        cb = fig.colorbar(im, ax=ax, pad=0.04)
        cb.set_label(label, fontsize=17)
        cb.ax.tick_params(labelsize=15)
        fig.tight_layout()
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
    print(f"[SAVED] {os.path.relpath(path, ROOT)}")

# ==========================
# PROMPT ANALYSIS (zero-shot, balanced 70% sets, mean ± SD over 3 runs)
# ==========================

def prompt_analysis():
    os.makedirs(OUT_PROMPT, exist_ok=True)
    binary = pd.read_csv(os.path.join(SEL, "binary_metrics_mean_sd.csv")).set_index(["model", "prompt"])
    appraisal = pd.read_csv(os.path.join(SEL, "appraisal_metrics_mean_sd.csv")).set_index(["model", "prompt"])
    prompts = ["V1", "V2", "V3"]

    # macro-F1 of every prompt, both tasks
    for name, table, title in [("evaluative class", binary, "binary"),
                               ("subclasses Affect, Judgement, and Appreciation", appraisal, "appraisal")]:
        grouped_bars(
            [f"Prompt {p}" for p in prompts],
            [(lab, col, hat, [table.loc[(m, p), "macro_f1_mean"] for p in prompts])
             for m, (lab, _, col, hat) in MODELS.items()],
            "Macro-F1", os.path.join(OUT_PROMPT, f"Zero-shot macro-F1 per prompt for the {name}.png"),
            legend_loc="lower right" if title == "binary" else "upper right", box=title == "binary")

    # the mean ± SD of the winning prompts on the selection sets
    rows = []
    for task, table, prompt in [("Evaluative (binary)", binary, BINARY_PROMPT),
                                ("Subclasses (appraisal)", appraisal, APPRAISAL_PROMPT)]:
        for m, (lab, _, _, _) in MODELS.items():
            r = table.loc[(m, prompt)]
            rows.append({"task": task, "prompt": prompt, "model": lab,
                         **{k: f"{r[f'{k}_mean']:.3f} ± {r[f'{k}_sd']:.3f}"
                            for k in ["macro_f1", "accuracy", "cohen_kappa", "krippendorff_alpha"]}})
    path = os.path.join(OUT_PROMPT, "prompt_selection_results_winning_prompts.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {os.path.relpath(path, ROOT)}")


def heldout_test():
    """Held-out 30% (ambiguous removed): binary V3, then appraisal V2, for every model."""
    path = os.path.join(HELDOUT, "heldout_metrics.csv")
    if not os.path.exists(path):
        print(f"[SKIP] held-out figures: {os.path.relpath(path, ROOT)} not found "
              f"(run heldout_pipeline.py on the server first)")
        return
    os.makedirs(OUT_PROMPT, exist_ok=True)
    res = pd.read_csv(path)
    res["step"] = res["step"].str.split(" ").str[:2].str.join(" ")
    res = res.set_index(["step", "model"])
    models = [m for m in MODELS if ("1 binary", m) in res.index]
    series = [(MODELS[m][0], MODELS[m][2], MODELS[m][3], m) for m in models]
    # combinations "binary model + appraisal model", e.g. the best binary and best appraisal setup
    combos = [m for m in res.index.get_level_values("model").unique() if " + " in m]
    combo_series = [(f"{MODELS[c.split(' + ')[0]][1]} {BINARY_PROMPT} + {MODELS[c.split(' + ')[1]][1]} {APPRAISAL_PROMPT}",
                     "#DD8452", "--", c) for c in combos]
    with_ft = (series + combo_series
               + ([FINETUNED[1:] + (FINETUNED[0],)] if ("1 binary", FINETUNED[0]) in res.index else []))

    # macro-F1 and kappa of every step (fine-tuned Llama as a reference bar)
    for metric, ylabel, fname in [("macro_f1", "Macro-F1", "Held-out test macro-F1 of the LLMs (binary V3, appraisal V2).png"),
                                  ("cohen_kappa", "Cohen's κ", "Held-out test Cohen's kappa of the LLMs (binary V3, appraisal V2).png")]:
        grouped_bars(list(STEP_NAMES.values()),
                     [(lab, col, hat, [res.loc[(s, m), metric] for s in STEP_NAMES]) for lab, col, hat, m in with_ft],
                     ylabel, os.path.join(OUT_PROMPT, fname), legend_loc="above")

    # step 1: the same four metrics as the earlier evaluative-class figure
    metrics = [("accuracy", "Accuracy"), ("precision_eval", "Precision (eval)"),
               ("recall_eval", "Recall (eval)"), ("f1_eval", "F1 (eval)")]
    grouped_bars([lab for _, lab in metrics],
                 [(lab, col, hat, [res.loc[("1 binary", m), k] for k, _ in metrics]) for lab, col, hat, m in series],
                 "Score", os.path.join(OUT_PROMPT, f"Held-out test results for the evaluative class (prompt {BINARY_PROMPT}).png"),
                 legend_loc="lower right", box=True)

    # step 2: F1 per subclass
    grouped_bars(CAT_NAMES,
                 [(lab, col, hat, [res.loc[("2 appraisal", m), f"f1_{c}"] for c in CATEGORIES])
                  for lab, col, hat, m in series + combo_series],
                 "F1-score", os.path.join(OUT_PROMPT, f"Held-out test results for the subclasses Affect, Judgement, and Appreciation (prompt {APPRAISAL_PROMPT}).png"),
                 legend_loc="above" if combos else "upper right")

    # agreement with the gold standard
    kappa = np.array([[res.loc[("1 binary", m), "cohen_kappa"]]
                      + [res.loc[("2 appraisal", m), f"kappa_{c}"] for c in CATEGORIES] for m in models + combos])
    heatmap(kappa, [MODELS[m][1] for m in models]
            + ["+".join(MODELS[x][1] for x in c.split(" + ")) for c in combos], ["Evaluative"] + CAT_NAMES,
            os.path.join(OUT_PROMPT, "Heatmap for the agreement between gold standard and LLMs (held-out test).png"),
            vmin=0.0, vmax=0.8)

    # table: scores + significance against the best zero-shot model of each step
    sig = pd.read_csv(os.path.join(HELDOUT, "heldout_significance.csv"))
    sig["step"] = sig["step"].str.split(" ").str[:2].str.join(" ")
    rows = []
    for s, name in STEP_NAMES.items():
        best = max(models, key=lambda m: res.loc[(s, m), "macro_f1"])
        for lab, _, _, m in with_ft:
            r = res.loc[(s, m)]
            row = {"step": name.replace("\n", " "), "model": lab, "n": int(r["n"]),
                   "gold_noneval": int(r["gold_noneval"]) if pd.notna(r.get("gold_noneval")) else "",
                   "invalid": int(r["invalid"]),
                   **{k: round(r[k], 3) for k in ["macro_f1", "accuracy", "cohen_kappa", "krippendorff_alpha"]},
                   "compared_to": "", "p_randomization_holm": "", "p_mcnemar_holm": ""}
            if s == "2 appraisal":  # different sentences per model: no paired test
                pass
            elif m in combos:  # combination vs the best single model
                t = sig[(sig["step"] == s) & (sig["model_a"] == m) & (sig["model_b"] == best)].iloc[0]
                row.update({"compared_to": MODELS[best][0],
                            "p_randomization_holm": round(t["p_randomization_holm"], 3),
                            "p_mcnemar_holm": round(t["p_mcnemar_holm"], 3)})
            elif m in models and m != best:
                t = sig[(sig["step"] == s) & sig["model_a"].isin([m, best]) & sig["model_b"].isin([m, best])].iloc[0]
                row.update({"compared_to": MODELS[best][0],
                            "p_randomization_holm": round(t["p_randomization_holm"], 3),
                            "p_mcnemar_holm": round(t["p_mcnemar_holm"], 3)})
            elif m == best and m in models:
                row["compared_to"] = "(best zero-shot)"
            rows.append(row)
    path = os.path.join(OUT_PROMPT, "heldout_test_results.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {os.path.relpath(path, ROOT)}")

# ==========================
# FINE-TUNING (held-out 30%, ambiguous removed)
# ==========================

def ft_predictions(setup):
    return pd.read_csv(os.path.join(FT, f"test_{setup}", "predictions.csv"), encoding="utf-8-sig")


def binary_metrics(df):
    gold, pred = df["Evaluative"].astype(int), df["pred_binary"].fillna(-1).astype(int)
    p, r, f, _ = precision_recall_fscore_support(gold, pred, labels=[1], zero_division=0)
    return {"accuracy": (gold == pred).mean(), "precision_eval": p[0], "recall_eval": r[0], "f1_eval": f[0],
            "kappa_evaluative": cohen_kappa_score(gold, pred)}


def category_kappas(df):
    """One-vs-rest kappa per subclass on the sentences the model called evaluative (step 2)."""
    passed = df[df["pred_binary"] == 1]
    gold, pred = passed["label4"].fillna("none"), passed["pred_category"].fillna("invalid")
    return {f"kappa_{c}": cohen_kappa_score(gold == c, pred == c) for c in CATEGORIES}


def fine_tuning():
    os.makedirs(OUT_FT, exist_ok=True)
    res = pd.read_csv(os.path.join(FT, "test_results.csv"))
    sig = pd.read_csv(os.path.join(FT, "test_significance.csv"))
    res["step"] = res["step"].str.split(" ").str[:2].str.join(" ")
    sig["step"] = sig["step"].str.split(" ").str[:2].str.join(" ")
    steps, names = list(STEP_NAMES), list(STEP_NAMES.values())
    res = res.set_index(["step", "model"])
    sig = sig.set_index("step")
    # step 2 has different sentences per setup, so no paired test
    pvals = [sig.loc[s, "p_randomization"] if s in sig.index else None for s in steps]

    # macro-F1 and kappa per step
    for metric, ylabel, fname in [("macro_f1", "Macro-F1", "Macro-F1 of zero-shot and fine-tuned Llama-3.3-70B on the held-out test set.png"),
                                  ("cohen_kappa", "Cohen's κ", "Cohen's kappa of zero-shot and fine-tuned Llama-3.3-70B on the held-out test set.png")]:
        grouped_bars(names,
                     [(lab, col, hat, [res.loc[(s, k), metric] for s in steps]) for k, (lab, col, hat) in SETUPS.items()],
                     ylabel, os.path.join(OUT_FT, fname), legend_loc="upper right",
                     pvalues=pvals if metric == "macro_f1" else None)

    preds = {k: ft_predictions(k) for k in SETUPS}
    extra = {k: {**binary_metrics(d), **category_kappas(d)} for k, d in preds.items()}

    # binary: same four metrics as the evaluative-class figure
    metrics = [("accuracy", "Accuracy"), ("precision_eval", "Precision (eval)"),
               ("recall_eval", "Recall (eval)"), ("f1_eval", "F1 (eval)")]
    grouped_bars([lab for _, lab in metrics],
                 [(lab, col, hat, [extra[k][m] for m, _ in metrics]) for k, (lab, col, hat) in SETUPS.items()],
                 "Score", os.path.join(OUT_FT, "Fine-tuning results for the evaluative class.png"),
                 legend_loc="lower right", box=True)

    # subclasses: F1 per category
    grouped_bars(CAT_NAMES,
                 [(lab, col, hat, [res.loc[("2 appraisal", k), f"f1_{c}"] for c in CATEGORIES])
                  for k, (lab, col, hat) in SETUPS.items()],
                 "F1-score", os.path.join(OUT_FT, "Fine-tuning results for the subclasses Affect, Judgement, and Appreciation.png"))

    # agreement heatmap
    kappa = np.array([[extra[k]["kappa_evaluative"]] + [extra[k][f"kappa_{c}"] for c in CATEGORIES] for k in SETUPS])
    heatmap(kappa, [lab for lab, _, _ in SETUPS.values()], ["Evaluative"] + CAT_NAMES,
            os.path.join(OUT_FT, "Heatmap for the agreement between gold standard and Llama-3.3-70B (zero-shot vs fine-tuned).png"),
            vmin=0.2, vmax=0.8)

    # validation macro-F1 per epoch (best epoch is kept)
    fig, ax = plt.subplots(figsize=(6.2, 3.45))
    tasks = [("binary", "Evaluative (binary)", "#4C72B0", "o"),
             ("appraisal", "Subclasses (appraisal)", "#C44E52", "s")]
    logs = {t: pd.read_csv(os.path.join(FT, f"train_log_{t}.csv")).set_index("epoch") for t, *_ in tasks}
    for task, label, colour, marker in tasks:
        log = logs[task]
        other = logs["appraisal" if task == "binary" else "binary"]["val_macro_f1"]
        ax.plot(log.index, log["val_macro_f1"], marker=marker, color=colour, label=label,
                markeredgecolor="black", linewidth=1.6, markersize=7, zorder=3)
        best = log["val_macro_f1"].idxmax()
        for e, v in log["val_macro_f1"].items():
            # the higher of the two points at an epoch is labelled above, the lower one below
            below = v < other.get(e, -1)
            ax.text(e, v - 0.04 if below else v + 0.03, f"{v:.2f}", ha="center",
                    va="top" if below else "bottom", fontsize=8.5,
                    fontweight="bold" if e == best else "normal")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation macro-F1", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0.7, 3.3)
    ax.yaxis.grid(True, linestyle="--", color="#aaaaaa", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    leg = ax.legend(loc="lower right", fontsize=9.5, frameon=True, fancybox=True, edgecolor="black")
    leg.get_frame().set_linewidth(1.0)
    fig.tight_layout()
    path = os.path.join(OUT_FT, "Validation macro-F1 per epoch during fine-tuning.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"[SAVED] {os.path.relpath(path, ROOT)}")

    # table
    rows = []
    for s, name in zip(steps, names):
        for k, (lab, _, _) in SETUPS.items():
            r = res.loc[(s, k)]
            test = k == "finetuned" and s in sig.index
            rows.append({"step": name.replace("\n", " "), "model": lab, "n": int(r["n"]),
                         "gold_noneval": int(r["gold_noneval"]) if pd.notna(r.get("gold_noneval")) else "",
                         "macro_f1": round(r["macro_f1"], 3), "accuracy": round(r["accuracy"], 3),
                         "cohen_kappa": round(r["cohen_kappa"], 3),
                         "krippendorff_alpha": round(r["krippendorff_alpha"], 3),
                         "macro_f1_diff (ft - zs)": round(sig.loc[s, "macro_f1_diff (ft - zs)"], 3) if test else "",
                         "p_randomization": round(sig.loc[s, "p_randomization"], 3) if test else "",
                         "p_mcnemar": round(sig.loc[s, "p_mcnemar"], 3) if test else ""})
    path = os.path.join(OUT_FT, "fine_tuning_results.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {os.path.relpath(path, ROOT)}")
    for k in SETUPS:
        print(k, {m: round(v, 3) for m, v in extra[k].items()})


if __name__ == "__main__":
    prompt_analysis()
    heldout_test()
    fine_tuning()
