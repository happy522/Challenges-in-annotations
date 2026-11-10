"""
Agreement between the linguists in training (students), the senior researcher (gold
standard) and Llama-3.3-70B on the held-out test set (125 sentences, ambiguous removed).

Measures as in the human annotation study:
    Evaluative                           Cohen's kappa (and Krippendorff's alpha), all sentences
    Affect / Judgement / Appreciation    Krippendorff's alpha (nominal), each as a yes / no label,
                                         on the sentences BOTH raters of a pair call evaluative

Every held-out sentence was annotated by 2 students (the two who annotated its talk).
Pairs are pooled over all sentences:
    Linguists in training                       student A vs student B             (1 pair per sentence)
    Linguists in training and senior researcher each student vs gold                 (2 pairs per sentence)
    Llama and linguists in training             Llama vs each student                (2 pairs per sentence)
    Llama and senior researcher                 Llama vs gold                        (1 pair per sentence)
Sentences a student marked "Uncertain" are left out for that student (as in the human study).

Llama = the fine-tuned model (binary adapter, then appraisal adapter on the sentences it calls
evaluative); the zero-shot model is reported as well.

Usage:  python Human_vs_LLM/human_vs_llm_agreement.py      (needs xlrd for the two .xls files)
Writes: Human_vs_LLM/outputs/agreement_heldout.csv and the heatmap.
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from make_result_figures import heatmap, krippendorff_alpha  # noqa: E402

GOLD_CSV = os.path.join(ROOT, "Data", "gold_standard_with_context.csv")
STUDENT_DIR = os.path.join(ROOT, "Data", "Students data")
FT_OUT = os.path.join(ROOT, "Finetuning", "outputs")
OUT_DIR = os.path.join(HERE, "outputs")

KEYS = ["Text_ID", "Sentence_ID"]
COLS = ["Evaluative", "Affect", "Judgement", "Appreciation"]
CATS = {"Affect": "affect", "Judgement": "judgment", "Appreciation": "appreciation"}
LLAMA = {"finetuned": "Llama-3.3-70B fine-tuned", "zeroshot": "Llama-3.3-70B zero-shot"}

# ==========================
# RATERS (one 0/1 table each, indexed by sentence)
# ==========================

def test_sentences():
    """The 125 held-out test sentences (ambiguous removed), the same as in Finetuning."""
    split = pd.read_csv(os.path.join(FT_OUT, "ft_split.csv"), encoding="utf-8-sig")
    return split.loc[split["ft_split"] == "test", KEYS]


def gold_labels(test):
    gold = pd.read_csv(GOLD_CSV, encoding="utf-8-sig").merge(test, on=KEYS)
    gold[COLS] = gold[COLS].fillna(0).astype(int)
    return gold.set_index(KEYS)[COLS]


def llama_labels(setup):
    pred = pd.read_csv(os.path.join(FT_OUT, f"test_{setup}", "predictions.csv"), encoding="utf-8-sig")
    out = pd.DataFrame({"Evaluative": pred["pred_binary"].fillna(-1).astype(int)})  # -1 = unreadable
    for col, cat in CATS.items():
        out[col] = ((pred["pred_binary"] == 1) & (pred["pred_category"] == cat)).astype(int)
    out.index = pd.MultiIndex.from_frame(pred[KEYS])
    return out


def student_labels():
    """{talk: {student: labels}}; a student's 'Uncertain' sentences are dropped."""
    talks = {}
    for path in sorted(glob.glob(os.path.join(STUDENT_DIR, "*.xls*"))):
        m = re.match(r"([A-Za-z]+)_?(\d+)_(A\d+)", os.path.basename(path))  # e.g. Art5_A15, Pol_2_A3
        talk, student = m.group(1) + m.group(2), m.group(3)
        df = pd.read_excel(path)
        for c in [c for c in df.columns if c.lower() == "uncertain"]:
            df = df[df[c] != 1]
        df["Text_ID"] = df["Text_ID"].astype(str).str.replace("_", "")
        df[COLS] = df[COLS].fillna(0).astype(int)
        talks.setdefault(talk, {})[student] = df.drop_duplicates(KEYS).set_index(KEYS)[COLS]
    return talks

# ==========================
# AGREEMENT
# ==========================

def agreement(pairs):
    """pairs: list of (labels_a, labels_b) row pairs (Series over COLS), pooled into one score."""
    a = pd.DataFrame([p[0] for p in pairs]).reset_index(drop=True)
    b = pd.DataFrame([p[1] for p in pairs]).reset_index(drop=True)
    row = {"pairs": len(a),
           "Evaluative (kappa)": cohen_kappa_score(a["Evaluative"].astype(str), b["Evaluative"].astype(str)),
           "Evaluative (alpha)": krippendorff_alpha(a["Evaluative"], b["Evaluative"])}
    both = (a["Evaluative"] == 1) & (b["Evaluative"] == 1)
    row["pairs_both_evaluative"] = int(both.sum())
    for col in CATS:
        row[f"{col} (alpha)"] = krippendorff_alpha(a.loc[both, col], b.loc[both, col])
    return row


def main():
    test = test_sentences()
    gold = gold_labels(test)
    talks = student_labels()
    llama = {k: llama_labels(k) for k in LLAMA}

    # students of each sentence (the two who annotated its talk); skip a student without that sentence
    def students_of(key):
        talk = key[0].rstrip(".")
        return [s.loc[key] for s in talks[talk].values() if key in s.index]

    missing = sum(2 - len(students_of(k)) for k in gold.index)
    comparisons = {"Linguists in training": [], "Linguists in training and senior researcher": []}
    for k in LLAMA:
        comparisons[f"{LLAMA[k]} and linguists in training"] = []
        comparisons[f"{LLAMA[k]} and senior researcher"] = []

    for key in gold.index:
        studs = students_of(key)
        g = gold.loc[key]
        if len(studs) == 2:
            comparisons["Linguists in training"].append((studs[0], studs[1]))
        for s in studs:
            comparisons["Linguists in training and senior researcher"].append((s, g))
        for k, pred in llama.items():
            m = pred.loc[key]
            comparisons[f"{LLAMA[k]} and senior researcher"].append((m, g))
            for s in studs:
                comparisons[f"{LLAMA[k]} and linguists in training"].append((m, s))

    table = pd.DataFrame([{"comparison": name, **agreement(pairs)} for name, pairs in comparisons.items()])
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "agreement_heldout.csv")
    table.round(3).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{len(gold)} held-out sentences; {missing} student labels missing (Uncertain)")
    print(table.round(3).to_string(index=False))
    print(f"[SAVED] {os.path.relpath(path, ROOT)}")

    # heatmap: the four rows of the paper's table, with the fine-tuned Llama
    rows = ["Linguists in training", "Linguists in training and senior researcher",
            f"{LLAMA['finetuned']} and linguists in training", f"{LLAMA['finetuned']} and senior researcher"]
    labels = ["Linguists in training", "Linguists in training\n& senior researcher",
              "Fine-tuned Llama &\nlinguists in training", "Fine-tuned Llama &\nsenior researcher"]
    cols = ["Evaluative (kappa)"] + [f"{c} (alpha)" for c in CATS]
    values = table.set_index("comparison").loc[rows, cols].to_numpy(dtype=float)
    heatmap(values, labels, ["Evaluative (κ)"] + [f"{c} (α)" for c in CATS],
            os.path.join(OUT_DIR, "Heatmap for the agreement between linguists in training, senior researcher "
                                  "and fine-tuned Llama-3.3-70B (held-out test).png"),
            vmin=0.0, vmax=0.8, label="Agreement")


if __name__ == "__main__":
    main()
