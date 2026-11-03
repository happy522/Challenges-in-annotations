"""
Binary prompt selection (Evaluative = 0/1) for Appraisal Theory annotation.

Runs the 3 binary prompt variants on a 70% split of the gold standard
(same number of sentences from every domain), with local vLLM inference
across multiple GPUs, and reports metrics per model x prompt.

Usage (on the GPU server):
    # one model per call (vLLM frees GPU memory reliably only on process exit)
    python binary_prompt_selection.py run --model qwen3-30b-a3b-instruct-2507 --tp 2
    python binary_prompt_selection.py run --model llama-3.3-70b-instruct --tp 4
    python binary_prompt_selection.py run --model mistral-large-instruct --tp 8

    # metrics over every predictions file in the output folder
    python binary_prompt_selection.py evaluate

    # or everything in one go: bash run_all.sh
"""

import argparse
import glob
import html
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

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

SPLIT_FRACTION = 0.70
SEED = 42

# Same models as the multiclass experiments (GWDG names -> HF checkpoints).
# Override the checkpoint with --hf-path if your server has a local copy.
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

# V1: Few-shot prompting, chain-of-thought prompting, avoidance of ambiguity
# (same as the first multilabel prompt)
PROMPT_BINARY_V1 = """
You are an annotation assistant for Appraisal Theory (Martin & White, 2005).

Task:
Decide whether the TARGET text contains any evaluative content.

You are given:
- PREVIOUS: text from the row above (may be empty)
- TARGET: the current text to label
- NEXT: text from the row below (may be empty)

Use PREVIOUS and NEXT only as context (e.g., to resolve pronouns or topics).
The label must refer ONLY to the TARGET text.

Coding rule “Evaluative” (binary):

- Evaluative = 1
  -> If the TARGET text expresses an emotion, a judgment of people/behaviour,
     or an aesthetic/quality/value evaluation of things, performances or phenomena.

- Evaluative = 0
  -> If the TARGET text is purely factual, descriptive, or neutral, with no clear
     emotional stance, no judgment of people/behaviour, and no quality/value
     evaluation of things or events.

Examples (illustrative only):
- “The room is beautiful.” -> evaluative (1): aesthetic evaluation of a thing.
- “She was very unfair to her team.” -> evaluative (1): judgment of behaviour.
- “The concert made me really happy.” -> evaluative (1): emotional reaction.
- “The meeting starts at 3 pm.” -> non-evaluative (0): purely factual.

Important constraints:
- Do NOT invent or assume information that is not explicitly present in the
  TARGET text or necessary to interpret it in context. No hallucinations.
- If you are unsure, choose the more conservative option and explain briefly.

Output format:
Return ONLY a single valid JSON object (no extra text, no explanations outside JSON).
Use exactly these fields:

- "binary": 0 or 1
    - 0 = non-evaluative TARGET
    - 1 = evaluative TARGET
- "label": "no_eval" or "eval"
- "justification": a short justification for the decision (max 40 words), based ONLY on the given texts.
- "evidence_span": a short quote from the TARGET that supports the decision (or "" if none).

Context:

PREVIOUS:
\"\"\"{prev_text}\"\"\"

TARGET:
\"\"\"{target_text}\"\"\"

NEXT:
\"\"\"{next_text}\"\"\"

Let's think step-by-step.
"""

# V2: Zero-shot prompting, less detail, shorter class description
# (same as the 2nd multilabel prompt)
PROMPT_BINARY_V2 = """
You are an annotation assistant for Appraisal Theory.

Task:
Decide whether the TARGET text contains any evaluative content.

You are given:
- PREVIOUS: text from the row above (may be empty)
- TARGET: the current text to label
- NEXT: text from the row below (may be empty)

Use PREVIOUS and NEXT only as context (e.g., to resolve pronouns or topics).
The label must refer ONLY to the TARGET text.

Coding rule “Evaluative” (binary):

- Evaluative = 1
  -> If the TARGET text expresses an emotion, a judgment,
     or an evaluation of things.

- Evaluative = 0
  -> If the TARGET text is purely factual, descriptive, or neutral, and no evaluation of things.

Important constraints:
- Do NOT invent or assume information that is not explicitly present in the
  TARGET text or necessary to interpret it in context. No hallucinations.
- If you are unsure, choose the more conservative option and explain briefly.

Output format:
Return ONLY a single valid JSON object (no extra text, no explanations outside JSON).
Use exactly these fields:

- "binary": 0 or 1
    - 0 = non-evaluative TARGET
    - 1 = evaluative TARGET
- "label": "no_eval" or "eval"
- "justification": a short justification for the decision (max 40 words), based ONLY on the given texts.
- "evidence_span": a short quote from the TARGET that supports the decision (or "" if none).

Context:

PREVIOUS:
\"\"\"{prev_text}\"\"\"

TARGET:
\"\"\"{target_text}\"\"\"

NEXT:
\"\"\"{next_text}\"\"\"

"""

# V3: Few-shot prompting - rather than real examples, we provide indicative
# adjectives and n-grams (same as the first multilabel prompt)
PROMPT_BINARY_V3 = """
You are an annotation assistant for Appraisal Theory (Martin & White, 2005).

Task:
Decide whether the TARGET text contains any evaluative content.

You are given:
- PREVIOUS: text from the row above (may be empty)
- TARGET: the current text to label
- NEXT: text from the row below (may be empty)

Use PREVIOUS and NEXT only as context (e.g., to resolve pronouns or topics).
The label must refer ONLY to the TARGET text.

Coding rule “Evaluative” (binary):

- Evaluative = 1
  -> If the TARGET text expresses an emotion, a judgment of people/behaviour,
     or an aesthetic/quality/value evaluation of things, performances or phenomena.

- Evaluative = 0
  -> If the TARGET text is purely factual, descriptive, or neutral, with no clear
     emotional stance, no judgment of people/behaviour, and no quality/value
     evaluation of things or events.

Examples:

Evaluative = 1 - typical cues:
- Emotions (Affect): love, hate, fear, afraid, scared, worried, happy, sad,
  angry, proud, excited, grateful, nervous; "I love", "I feel", "fear of",
  "I don't want to", "my heart", "I was terrified"
- Judgement of people/behaviour: brave, honest, fair/unfair, corrupt, lazy,
  talented, responsible, wrong; "be brave", "the right thing", "deserve",
  "failed to", "should have", "suffered enough"
- Appreciation of things/events: good, great, bad, amazing, beautiful,
  important, interesting, perfect, terrible, extraordinary, valuable,
  pointless; "the most important", "the best", "a big problem",
  "incredibly difficult", "hard choices"
- Implicit / graded evaluation: intensifiers (really, so, too, incredibly,
  pretty), superlatives (the most, the only), counter-expectation (even,
  only, not only ... but, at least, (not) enough), obligation (need to,
  should, must, have to), opinion framing (I think, I believe),
  difficulty (hard, easy, difficult)

Evaluative = 0 - typical cues:
- Neutral reporting of facts, numbers, dates, places ("in 2008",
  "three years ago", "200 of these")
- Presentational moves ("here's a", "this is a", "let me show you",
  "I'm going to talk about")
- Information questions ("how do we ...?", "what is ...?")
- Plain narration of actions ("I asked", "we moved to")

Important constraints:
- Do NOT invent or assume information that is not explicitly present in the
  TARGET text or necessary to interpret it in context. No hallucinations.
- If you are unsure, choose the more conservative option and explain briefly.

Output format:
Return ONLY a single valid JSON object (no extra text, no explanations outside JSON).
Use exactly these fields:

- "binary": 0 or 1
    - 0 = non-evaluative TARGET
    - 1 = evaluative TARGET
- "label": "no_eval" or "eval"
- "justification": a short justification for the decision (max 40 words), based ONLY on the given texts.
- "evidence_span": a short quote from the TARGET that supports the decision (or "" if none).

Context:

PREVIOUS:
\"\"\"{prev_text}\"\"\"

TARGET:
\"\"\"{target_text}\"\"\"

NEXT:
\"\"\"{next_text}\"\"\"

"""

PROMPTS = {"V1": PROMPT_BINARY_V1, "V2": PROMPT_BINARY_V2, "V3": PROMPT_BINARY_V3}

# ==========================
# DATA
# ==========================

def domain_of(text_id):
    """'Tech21.' -> 'Tech'"""
    return re.match(r"[A-Za-z]+", text_id).group(0)


def make_split(seed=SEED):
    """
    Take 70% of the gold standard with the same number of sentences per domain.
    Within a domain, sentences are drawn equally from each transcript and
    stratified by the gold Evaluative label. The split is saved so the held-out
    30% stays untouched for the final evaluation.
    """
    gold = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")
    gold["Domain"] = gold["Text_ID"].map(domain_of)

    per_domain = int(round(gold.groupby("Domain").size().min() * SPLIT_FRACTION))
    parts = []
    for _, dom in gold.groupby("Domain"):
        texts = sorted(dom["Text_ID"].unique())
        sizes = [per_domain // len(texts)] * len(texts)
        for i in range(per_domain % len(texts)):
            sizes[i] += 1
        for text_id, n in zip(texts, sizes):
            rows = dom[dom["Text_ID"] == text_id]
            try:
                picked, _ = train_test_split(rows, train_size=n, random_state=seed,
                                             stratify=rows["Evaluative"])
            except ValueError:  # a class too small to stratify
                picked = rows.sample(n=n, random_state=seed)
            parts.append(picked)

    dev = pd.concat(parts)
    gold["split"] = np.where(gold.index.isin(dev.index), "prompt_dev", "held_out")

    os.makedirs(OUT_DIR, exist_ok=True)
    gold.to_csv(os.path.join(OUT_DIR, "gold_split.csv"), index=False, encoding="utf-8-sig")
    print(f"[INFO] Split: {len(dev)} prompt_dev rows ({per_domain} per domain), "
          f"{len(gold) - len(dev)} held_out rows")
    return gold[gold["split"] == "prompt_dev"].reset_index(drop=True)


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

def parse_output(text):
    """Return (binary, parsed_json_or_None). Uses the last JSON object in the output,
    since the chain-of-thought prompt may reason before answering."""
    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if "binary" in obj:
            try:
                b = int(obj["binary"])
                if b in (0, 1):
                    return b, obj
            except (TypeError, ValueError):
                pass
        label = str(obj.get("label", "")).lower()
        if label in ("eval", "no_eval"):
            return int(label == "eval"), obj

    m = re.findall(r'"binary"\s*:\s*"?([01])', text)
    if m:
        return int(m[-1]), None
    return None, None

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


def run(args):
    data = add_context(make_split(args.seed))
    if args.limit:
        data = data.head(args.limit)

    generate = ollama_generator(args) if args.backend == "ollama" else vllm_generator(args)

    for version in args.prompts:
        template = PROMPTS[version]
        messages = [
            [{"role": "user", "content": template.format(prev_text=r.prev_text,
                                                         target_text=r.Sentence,
                                                         next_text=r.next_text)}]
            for r in data.itertuples()
        ]

        out = data.copy()
        out["model"], out["prompt"] = args.model, version
        out["raw_output"] = generate(messages)
        parsed = [parse_output(t) for t in out["raw_output"]]
        out["pred_binary"] = [p[0] for p in parsed]
        out["justification"] = [(p[1] or {}).get("justification", "") for p in parsed]
        out["evidence_span"] = [(p[1] or {}).get("evidence_span", "") for p in parsed]

        path = os.path.join(OUT_DIR, f"pred_{args.model}_{version}.csv")
        out.to_csv(path, index=False, encoding="utf-8-sig")
        n_fail = out["pred_binary"].isna().sum()
        print(f"[INFO] {args.model} {version}: saved {path} ({n_fail} unparsed outputs)")

    evaluate(args)

# ==========================
# EVALUATION
# ==========================

def scores(gold, pred):
    p, r, f, _ = precision_recall_fscore_support(gold, pred, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(gold, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(gold, pred),
        "precision_eval": p[0],
        "recall_eval": r[0],
        "f1_eval": f[0],
        "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(gold, pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def evaluate(args=None):
    files = sorted(glob.glob(os.path.join(OUT_DIR, "pred_*.csv")))
    if not files:
        print("[WARN] No prediction files found in", OUT_DIR)
        return

    summary, by_domain = [], []
    for path in files:
        df = pd.read_csv(path, encoding="utf-8-sig")
        n_unparsed = int(df["pred_binary"].isna().sum())
        # unparsed outputs count as wrong (predict the opposite of gold)
        pred = df["pred_binary"].fillna(1 - df["Evaluative"]).astype(int)
        gold = df["Evaluative"].astype(int)

        row = {"model": df["model"].iloc[0], "prompt": df["prompt"].iloc[0], "n": len(df),
               "unparsed": n_unparsed, **scores(gold, pred)}
        summary.append(row)

        for dom, idx in df.groupby("Domain").groups.items():
            by_domain.append({"model": row["model"], "prompt": row["prompt"], "domain": dom,
                              "n": len(idx),
                              "accuracy": accuracy_score(gold[idx], pred[idx]),
                              "macro_f1": f1_score(gold[idx], pred[idx], average="macro",
                                                   zero_division=0)})

    summary = pd.DataFrame(summary).sort_values(["model", "prompt"])
    summary.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)
    pd.DataFrame(by_domain).to_csv(os.path.join(OUT_DIR, "metrics_by_domain.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n=== Binary prompt comparison (prompt_dev split) ===")
    print(summary.round(3).to_string(index=False))

    mean = summary.groupby("prompt")[["macro_f1", "cohen_kappa", "accuracy"]].mean()
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
    r.add_argument("--backend", choices=["vllm", "ollama"], default="vllm")
    r.add_argument("--ollama-url", default=os.getenv("OLLAMA_HOST_URL", "http://localhost:11434"))
    r.add_argument("--ollama-model", default=None, help="Ollama tag overriding the default one")
    r.add_argument("--workers", type=int, default=4,
                   help="parallel Ollama requests (match OLLAMA_NUM_PARALLEL)")
    r.add_argument("--hf-path", default=None, help="local path / HF id overriding the default checkpoint")
    r.add_argument("--prompts", nargs="+", default=list(PROMPTS), choices=list(PROMPTS))
    r.add_argument("--tp", type=int, default=None, help="tensor parallel size (default: all visible GPUs)")
    r.add_argument("--max-model-len", type=int, default=8192)
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--gpu-mem", type=float, default=0.90)
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--limit", type=int, default=None, help="only first N rows (smoke test)")

    sub.add_parser("evaluate", help="compute metrics for all prediction files")
    sub.add_parser("split", help="only write the 70/30 gold split")

    args = ap.parse_args()
    if args.cmd == "run":
        if args.backend == "vllm" and args.tp is None:
            import torch
            args.tp = max(torch.cuda.device_count(), 1)
        run(args)
    elif args.cmd == "evaluate":
        evaluate()
    else:
        make_split()


if __name__ == "__main__":
    main()
