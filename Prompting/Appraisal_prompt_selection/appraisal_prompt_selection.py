"""
Appraisal prompt selection (Affect / Judgement / Appreciation) for evaluative sentences.

Runs the 3 multiclass prompt variants (prompts/V1.txt, V2.txt, V3.txt - same as
the multiclass prompts of the paper) on the evaluative sentences (Evaluative = 1) of the binary
experiment's prompt_dev split, and reports metrics per model x prompt. The binary
held-out 30% is never seen by any model.

Usage (on the GPU server):
    export GOLD_CSV=/path/to/gold_standard_with_context.csv
    export BINARY_SPLIT=/path/to/Binary_prompt_selection/outputs/gold_split.csv  # if not ../Binary_prompt_selection
    python appraisal_prompt_selection.py run --backend ollama --model qwen3-30b-a3b-instruct-2507
    python appraisal_prompt_selection.py evaluate

    # or everything in one go: bash run_all_ollama.sh
"""

import argparse
import glob
import html
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# ==========================
# CONFIGURATION
# ==========================

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# Override with env vars (GOLD_CSV, RAW_DIR, OUT_DIR) when the folder is copied
# outside the repo, e.g. on the GPU server.
GOLD_CSV = os.getenv("GOLD_CSV", os.path.join(REPO, "Data", "gold_standard_with_context.csv"))
RAW_DIR = os.getenv("RAW_DIR", os.path.join(REPO, "Classified_data", "Raw data"))
OUT_DIR = os.getenv("OUT_DIR", os.path.join(HERE, "outputs"))
PROMPT_DIR = os.path.join(HERE, "prompts")
# The binary experiment's split: its held-out 30% stays unseen here too.
BINARY_SPLIT = os.getenv("BINARY_SPLIT", os.path.join(HERE, "..", "Binary_prompt_selection",
                                                      "outputs", "gold_split.csv"))

KEYS = ["Text_ID", "Sentence_ID"]
SEED = 42

CATEGORIES = ["affect", "judgment", "appreciation"]
GOLD_COLS = {"affect": "Affect", "judgment": "Judgement", "appreciation": "Appreciation"}

# Same models as the multiclass experiments (GWDG names -> HF checkpoints).
MODELS = {
    "llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "mistral-large-instruct": "mistralai/Mistral-Large-Instruct-2411",
    "qwen3-30b-a3b-instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}

# Same models as Ollama tags (--backend ollama). Override with --ollama-model.
OLLAMA_MODELS = {
    "llama-3.3-70b-instruct": "llama3.3:70b",
    "mistral-large-instruct": "mistral-large:123b",
    "qwen3-30b-a3b-instruct-2507": "qwen3:30b-a3b-instruct-2507-q4_K_M",
}

# ==========================
# PROMPTS
# ==========================

# V1: few-shot, definitions with cues, multi-label + ambiguous
# V2: shorter definitions, explicit/implicit guidance
# V3: minimal prompt (Aenne, based on the literature review), step-by-step
PROMPTS = {v: open(os.path.join(PROMPT_DIR, f"{v}.txt"), encoding="utf-8").read()
           for v in ("V1", "V2", "V3")}


def fill(template, prev_text, target_text, next_text):
    # plain replace: V3 contains literal JSON braces, so str.format would fail
    return (template.replace("{prev_text}", prev_text)
                    .replace("{target_text}", target_text)
                    .replace("{next_text}", next_text))

# ==========================
# DATA
# ==========================

def domain_of(text_id):
    """'Tech21.' -> 'Tech'"""
    return re.match(r"[A-Za-z]+", text_id).group(0)


def gold_label(row):
    cats = [c for c in CATEGORIES if row[c] == 1]
    return cats[0] if len(cats) == 1 else "multi"


def make_split():
    """
    Use the binary split (Binary_prompt_selection/outputs/gold_split.csv): the
    evaluative sentences (Evaluative = 1) of its prompt_dev part. The binary held-out
    30% is never shown to any model, in either experiment.
    """
    gold = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")
    gold = gold[gold["Evaluative"] == 1].copy()
    gold["Domain"] = gold["Text_ID"].map(domain_of)
    for cat, col in GOLD_COLS.items():
        gold[cat] = gold[col].fillna(0).astype(int)
    gold["gold_label"] = gold.apply(gold_label, axis=1)

    binary = pd.read_csv(BINARY_SPLIT, encoding="utf-8-sig")[KEYS + ["split"]]
    gold = gold.drop(columns="split", errors="ignore").merge(binary, on=KEYS, how="left")
    if gold["split"].isna().any():
        raise ValueError(f"{gold['split'].isna().sum()} evaluative sentences missing from {BINARY_SPLIT}")
    dev = gold[gold["split"] == "prompt_dev"]

    os.makedirs(OUT_DIR, exist_ok=True)
    gold.to_csv(os.path.join(OUT_DIR, "appraisal_split.csv"), index=False, encoding="utf-8-sig")
    print(f"[INFO] Split (from binary split): {len(dev)} prompt_dev / {len(gold) - len(dev)} "
          f"held_out evaluative sentences")
    print("[INFO] prompt_dev per domain:", dev.groupby("Domain").size().to_dict())
    print("[INFO] prompt_dev labels:", dev["gold_label"].value_counts().to_dict())
    return dev.reset_index(drop=True)


def add_context(df):
    """Add PREVIOUS / NEXT sentences. Uses the Prev_Sentence / Next_Sentence columns
    if the gold file has them (gold_standard_with_context.csv), otherwise looks them
    up in the full transcript (Sentence_ID 'S6' = row 5)."""
    if {"Prev_Sentence", "Next_Sentence"} <= set(df.columns):
        df = df.copy()
        df["prev_text"] = df["Prev_Sentence"].fillna("").astype(str).map(html.unescape).str.strip()
        df["next_text"] = df["Next_Sentence"].fillna("").astype(str).map(html.unescape).str.strip()
        return df

    transcripts = {}
    prevs, nexts = [], []
    for _, row in df.iterrows():
        name = row["Text_ID"].rstrip(".")
        if name not in transcripts:
            raw = pd.read_excel(os.path.join(RAW_DIR, f"{name}.xlsx"))
            transcripts[name] = [html.unescape(str(t)).strip() for t in raw["text"].fillna("")]
        sents = transcripts[name]
        i = int(row["Sentence_ID"].lstrip("S")) - 1
        prevs.append(sents[i - 1] if i - 1 >= 0 else "")
        nexts.append(sents[i + 1] if i + 1 < len(sents) else "")
    df = df.copy()
    df["prev_text"], df["next_text"] = prevs, nexts
    return df

# ==========================
# OUTPUT PARSING
# ==========================

LABEL_ALIASES = {"judgement": "judgment", "appraisal": "appreciation"}


def to_flag(value):
    try:
        return int(float(value) >= 0.5)
    except (TypeError, ValueError):
        return 0


def parse_output(text):
    """Return (label, {category: 0/1}, parsed_json) or (None, None, None).
    Uses the last JSON object in the output, since V3 may reason before answering.
    Flags missing from the output are derived from "label"."""
    for cand in reversed(re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        obj = {str(k).lower(): v for k, v in obj.items()}
        if "judgement" in obj and "judgment" not in obj:
            obj["judgment"] = obj["judgement"]
        label = str(obj.get("label", "")).strip().lower()
        label = LABEL_ALIASES.get(label, label)
        if label not in CATEGORIES + ["ambiguous"] and not any(c in obj for c in CATEGORIES):
            continue

        flags = {c: to_flag(obj.get(c, 0)) for c in CATEGORIES}
        if label in CATEGORIES:
            flags[label] = 1
        if label not in CATEGORIES + ["ambiguous"]:
            label = None
        return label, flags, obj
    return parse_fields(text)


def parse_fields(text):
    """Fallback for almost-JSON (e.g. unescaped quotes in top_spans/explanation,
    output cut off by num_predict): read "label" and the 0/1 flags with regexes."""
    text = str(text)
    m = re.search(r'"label"\s*:\s*"(\w+)"', text, flags=re.IGNORECASE)
    label = LABEL_ALIASES.get(m.group(1).lower(), m.group(1).lower()) if m else ""
    found = {}
    for c in CATEGORIES:
        key = "judge?ment" if c == "judgment" else c
        f = re.search(rf'"{key}"\s*:\s*"?([01](?:\.\d+)?)', text, flags=re.IGNORECASE)
        if f:
            found[c] = to_flag(f.group(1))
    if label not in CATEGORIES + ["ambiguous"] and not found:
        return None, None, None
    flags = {c: found.get(c, 0) for c in CATEGORIES}
    if label in CATEGORIES:
        flags[label] = 1
    e = re.search(r'"explanation"\s*:\s*"(.*)', text, flags=re.DOTALL)
    obj = {"explanation": e.group(1).strip().rstrip('}"').strip() if e else ""}
    return (label if label in CATEGORIES + ["ambiguous"] else None), flags, obj


def single_label(label, flags):
    """One predicted class for single-label scoring ('ambiguous' if several/none)."""
    on = [c for c in CATEGORIES if flags[c]]
    if label in CATEGORIES and len(on) == 1:
        return label
    return on[0] if len(on) == 1 else "ambiguous"

# ==========================
# RUN
# ==========================

def vllm_generator(args):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.hf_path or MODELS[args.model],
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
    )
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed)

    def generate(messages):
        return [o.outputs[0].text for o in llm.chat(messages, params, use_tqdm=True)]
    return generate


def ollama_generator(args):
    """Ollama server (ollama serve) - it splits the model over all visible GPUs itself."""
    from concurrent.futures import ThreadPoolExecutor
    import requests

    tag = args.ollama_model or OLLAMA_MODELS[args.model]
    url = args.ollama_url.rstrip("/") + "/api/chat"
    options = {"temperature": args.temperature, "seed": args.seed,
               "num_predict": args.max_tokens, "num_ctx": args.max_model_len}

    def one(msgs):
        for attempt in range(3):
            try:
                r = requests.post(url, json={"model": tag, "messages": msgs, "stream": False,
                                             "options": options}, timeout=600)
                r.raise_for_status()
                return r.json()["message"]["content"]
            except requests.RequestException as e:
                print(f"[WARN] Ollama request failed (attempt {attempt + 1}): {e}")
        return ""

    def generate(messages):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = []
            for i, text in enumerate(pool.map(one, messages), 1):
                results.append(text)
                if i % 25 == 0 or i == len(messages):
                    print(f"[INFO] {i}/{len(messages)}", flush=True)
            return results
    return generate


def cached_outputs(path):
    """Raw outputs of an earlier run of the same model x prompt, keyed by sentence
    (temperature 0 + fixed seed, so rerunning them would give the same answers)."""
    if not os.path.exists(path):
        return {}
    old = pd.read_csv(path, encoding="utf-8-sig")
    old = old[old["raw_output"].fillna("").astype(str).str.strip() != ""]
    return {(r.Text_ID, r.Sentence_ID): r.raw_output for r in old.itertuples()}


def run(args):
    data = add_context(make_split())
    if args.limit:
        data = data.head(args.limit)

    generate = None
    for version in args.prompts:
        path = os.path.join(OUT_DIR, f"pred_{args.model}_{version}.csv")
        cache = {} if args.no_cache else cached_outputs(path)
        keys = list(zip(data["Text_ID"], data["Sentence_ID"]))
        todo = [i for i, k in enumerate(keys) if k not in cache]
        print(f"[INFO] {args.model} {version}: {len(keys) - len(todo)} reused, {len(todo)} to run")

        raw = [cache.get(k, "") for k in keys]
        if todo:
            if generate is None:
                generate = ollama_generator(args) if args.backend == "ollama" else vllm_generator(args)
            messages = [[{"role": "user", "content": fill(PROMPTS[version], data.prev_text[i],
                                                          data.Sentence[i], data.next_text[i])}]
                        for i in todo]
            for i, text in zip(todo, generate(messages)):
                raw[i] = text

        out = data.copy()
        out["model"], out["prompt"] = args.model, version
        out["raw_output"] = raw
        out = add_parsed(out)

        out.to_csv(path, index=False, encoding="utf-8-sig")
        n_empty = (out["raw_output"].str.strip() == "").sum()
        print(f"[INFO] {args.model} {version}: saved {path} "
              f"({(~out['parsed']).sum()} unparsed outputs, {n_empty} empty = failed requests)")

    evaluate()


def add_parsed(out):
    out = out.copy()
    out["raw_output"] = out["raw_output"].fillna("").astype(str)
    parsed = [parse_output(t) for t in out["raw_output"]]
    out["parsed"] = [p[1] is not None for p in parsed]
    out["pred_label"] = [p[0] for p in parsed]
    for c in CATEGORIES:
        out[f"pred_{c}"] = [p[1][c] if p[1] else np.nan for p in parsed]
    out["explanation"] = [(p[2] or {}).get("explanation", (p[2] or {}).get("reasoning", ""))
                          for p in parsed]
    return out


def reparse():
    """Parse the saved raw outputs again (no model needed), then evaluate."""
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "pred_*.csv"))):
        df = add_parsed(pd.read_csv(path, encoding="utf-8-sig"))
        df.to_csv(path, index=False, encoding="utf-8-sig")
        n_empty = (df["raw_output"].str.strip() == "").sum()
        print(f"[INFO] {os.path.basename(path)}: {(~df['parsed']).sum()} unparsed "
              f"({n_empty} empty = failed requests)")
    evaluate()

# ==========================
# EVALUATION
# ==========================

def evaluate():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "pred_*.csv")))
    if not files:
        print("[WARN] No prediction files found in", OUT_DIR)
        return

    summary, by_domain = [], []
    for path in files:
        df = pd.read_csv(path, encoding="utf-8-sig")
        gold = df[CATEGORIES].astype(int).values
        # unparsed outputs predict nothing (all flags 0)
        pred = df[[f"pred_{c}" for c in CATEGORIES]].fillna(0).astype(int).values

        p, r, f, _ = precision_recall_fscore_support(gold, pred, zero_division=0)
        row = {"model": df["model"].iloc[0], "prompt": df["prompt"].iloc[0], "n": len(df),
               "unparsed": int((~df["parsed"]).sum()),
               # multi-label scores (gold can have several categories)
               "macro_f1": f.mean(),
               "micro_f1": f1_score(gold, pred, average="micro", zero_division=0),
               "exact_match": (gold == pred).all(axis=1).mean()}
        for i, c in enumerate(CATEGORIES):
            row[f"f1_{c}"], row[f"p_{c}"], row[f"r_{c}"] = f[i], p[i], r[i]

        # single-label scores on sentences with exactly one gold category
        single = df["gold_label"] != "multi"
        pred_single = [single_label(l if isinstance(l, str) else None,
                                    dict(zip(CATEGORIES, fl)))
                       for l, fl in zip(df["pred_label"], pred)]
        pred_single = pd.Series(pred_single, index=df.index)[single]
        row["n_single"] = int(single.sum())
        row["single_accuracy"] = accuracy_score(df.loc[single, "gold_label"], pred_single)
        row["single_macro_f1"] = f1_score(df.loc[single, "gold_label"], pred_single,
                                          labels=CATEGORIES, average="macro", zero_division=0)
        row["pred_ambiguous"] = int((pred_single == "ambiguous").sum())
        summary.append(row)

        for dom, idx in df.groupby("Domain").groups.items():
            ix = df.index.get_indexer(idx)
            by_domain.append({"model": row["model"], "prompt": row["prompt"], "domain": dom,
                              "n": len(ix),
                              "macro_f1": f1_score(gold[ix], pred[ix], average="macro",
                                                   zero_division=0)})

    summary = pd.DataFrame(summary).sort_values(["model", "prompt"])
    summary.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)
    pd.DataFrame(by_domain).to_csv(os.path.join(OUT_DIR, "metrics_by_domain.csv"), index=False)

    pd.set_option("display.width", 250)
    cols = ["model", "prompt", "n", "unparsed", "macro_f1", "micro_f1", "exact_match",
            "f1_affect", "f1_judgment", "f1_appreciation", "single_accuracy",
            "single_macro_f1", "pred_ambiguous"]
    print("\n=== Appraisal prompt comparison (prompt_dev split) ===")
    print(summary[cols].round(3).to_string(index=False))

    mean = summary.groupby("prompt")[["macro_f1", "single_macro_f1", "exact_match"]].mean()
    print("\n=== Mean over models ===")
    print(mean.round(3).to_string())
    print(f"\nBest prompt by mean macro-F1: {mean['macro_f1'].idxmax()}")

# ==========================
# CLI
# ==========================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run prompts for one model with vLLM or Ollama")
    r.add_argument("--model", required=True, choices=list(MODELS))
    r.add_argument("--backend", choices=["vllm", "ollama"], default="ollama")
    r.add_argument("--ollama-url", default=os.getenv("OLLAMA_HOST_URL", "http://localhost:11434"))
    r.add_argument("--ollama-model", default=None, help="Ollama tag overriding the default one")
    r.add_argument("--workers", type=int, default=4,
                   help="parallel Ollama requests (match OLLAMA_NUM_PARALLEL)")
    r.add_argument("--hf-path", default=None, help="local path / HF id overriding the default checkpoint")
    r.add_argument("--prompts", nargs="+", default=list(PROMPTS), choices=list(PROMPTS))
    r.add_argument("--tp", type=int, default=None, help="vLLM tensor parallel size (default: all GPUs)")
    r.add_argument("--max-model-len", type=int, default=4096)
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--gpu-mem", type=float, default=0.90)
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--no-cache", action="store_true",
                   help="rerun every sentence instead of reusing saved outputs")
    r.add_argument("--limit", type=int, default=None, help="only first N rows (smoke test)")

    sub.add_parser("evaluate", help="compute metrics for all prediction files")
    sub.add_parser("reparse", help="re-parse saved raw outputs (no model needed), then evaluate")
    sub.add_parser("split", help="only write the split of evaluative sentences")

    args = ap.parse_args()
    if args.cmd == "run":
        if args.backend == "vllm" and args.tp is None:
            import torch
            args.tp = max(torch.cuda.device_count(), 1)
        run(args)
    elif args.cmd == "evaluate":
        evaluate()
    elif args.cmd == "reparse":
        reparse()
    else:
        make_split()


if __name__ == "__main__":
    main()
