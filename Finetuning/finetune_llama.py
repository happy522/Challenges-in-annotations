"""
QLoRA fine-tuning of Llama 3.3 70B Instruct on the two Appraisal tasks:

    binary     Evaluative 0/1                        input: binary prompt V3
    appraisal  Affect / Judgement / Appreciation     input: appraisal prompt V2
               (evaluative sentences only, one category each)

Data (the same 70 / 30 split as the prompt selection):
    - Ambiguous sentences are removed everywhere (train, val and test): evaluative
      sentences with more than one gold category (or none), and Ambiguous = 1.
    - train + val = the 70% prompt_dev split, split 85 / 15, stratified by
      class (non-evaluative / affect / judgment / appreciation) and domain.
      Both tasks use the same train / val sentences (appraisal: the evaluative ones).
    - test = the 30% held-out split, used only by `test`.
    - Training classes are balanced by oversampling the smaller classes
      (val and test are never changed). Turn off with --no-balance.

Training: one LoRA adapter per task on the 4-bit base model. The model learns to
answer with a short JSON object ({"binary": 1, "label": "eval"} or
{"label": "judgment"}); the loss is only on these answer tokens. After every epoch
the val set is scored and the epoch with the best val macro-F1 is kept.

Test (held-out 30%, two steps, like a real annotation run):
    step 1  binary model on every test sentence
    step 2  appraisal model on the sentences step 1 calls evaluative
    The fine-tuned answers on the test sentences are generated inside `train`, by the
    best-val model while it is still in memory (the epoch is chosen on val only);
    `test` combines them, without loading the model again.
    `test --zero-shot` runs the same steps with the plain base model (no adapters),
    so `compare` shows the effect of fine-tuning on exactly the same sentences.

Hardware: by default the whole 4-bit model runs on ONE 48 GB GPU (LoRA on the attention
layers), because splitting the model over GPUs gave broken numbers on our server
(corrupted GPU-to-GPU copies; the script checks this at start). --multi-gpu splits it.

Usage (on the GPU server; needs one free 48 GB GPU):
    export GOLD_CSV=/path/to/gold_standard_with_context.csv
    python finetune_llama.py split
    python finetune_llama.py train --task binary
    python finetune_llama.py train --task appraisal
    python finetune_llama.py test
    python finetune_llama.py test --zero-shot
    python finetune_llama.py compare

    # or everything in one go: bash run_all.sh
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
# the prompt-selection folders sit next to this one (server) or in ../Prompting (Final)
for _up in (os.path.join(HERE, "..", "Prompting"), os.path.join(HERE, "..")):
    sys.path.insert(0, os.path.join(_up, "Binary_prompt_selection"))
    sys.path.insert(0, os.path.join(_up, "Appraisal_prompt_selection"))
import binary_prompt_selection as B      # noqa: E402  (binary prompts, parser, context)
import appraisal_prompt_selection as A   # noqa: E402  (appraisal prompts, parser, split paths)

OUT_DIR = os.getenv("FT_OUT_DIR", os.path.join(HERE, "outputs"))
# gold file: GOLD_CSV if set, else gold_standard_with_context.csv next to this script or
# one folder up, else the repo default
GOLD_CSV = os.getenv("GOLD_CSV") or next(
    (p for p in (os.path.join(HERE, "gold_standard_with_context.csv"),
                 os.path.join(HERE, "..", "gold_standard_with_context.csv"))
     if os.path.exists(p)), A.GOLD_CSV)

BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
BINARY_PROMPT = "V3"     # winner of the binary prompt selection
APPRAISAL_PROMPT = "V2"  # winner of the appraisal prompt selection
VAL_FRACTION = 0.15
SEED = 42
N_PERM = 10_000

KEYS = ["Text_ID", "Sentence_ID"]
CATEGORIES = A.CATEGORIES                  # affect, judgment, appreciation
PIPE_LABELS = ["none"] + CATEGORIES        # final label of a sentence after both steps

# ==========================
# DATA
# ==========================

def read_gold():
    """Gold sentences with their 70/30 split, single category and ambiguity flag."""
    if not os.path.exists(GOLD_CSV):
        raise FileNotFoundError(f"gold file not found: {GOLD_CSV} - put gold_standard_with_context.csv "
                                f"in {HERE} or set GOLD_CSV")
    gold = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")
    split = pd.read_csv(A.BINARY_SPLIT, encoding="utf-8-sig")[KEYS + ["split"]]
    gold = gold.drop(columns="split", errors="ignore").merge(split, on=KEYS, how="inner")
    gold["Domain"] = gold["Text_ID"].map(B.domain_of)
    gold["Evaluative"] = gold["Evaluative"].astype(int)
    for cat, col in A.GOLD_COLS.items():
        gold[cat] = gold[col].fillna(0).astype(int) * gold["Evaluative"]
    n_cats = gold[CATEGORIES].sum(axis=1)
    amb_col = gold["Ambiguous"].fillna(0).astype(int) if "Ambiguous" in gold else 0
    gold["ambiguous"] = ((gold["Evaluative"] == 1) & (n_cats != 1)) | (amb_col == 1)
    gold["category"] = [next((c for c in CATEGORIES if r[c]), "") for _, r in gold.iterrows()]
    gold["label4"] = np.where(gold["Evaluative"] == 1, gold["category"], "none")
    return gold


def make_split(seed=SEED):
    gold = read_gold()
    clean = gold[~gold["ambiguous"]]
    dev = clean[clean["split"] == "prompt_dev"]
    try:
        _, val = train_test_split(dev, test_size=VAL_FRACTION, random_state=seed,
                                  stratify=dev["label4"] + "_" + dev["Domain"])
    except ValueError:  # some class x domain cells too small
        _, val = train_test_split(dev, test_size=VAL_FRACTION, random_state=seed,
                                  stratify=dev["label4"])

    gold["ft_split"] = np.select(
        [gold["ambiguous"], gold["split"] == "held_out", gold.index.isin(val.index)],
        ["removed_ambiguous", "test", "val"], "train")
    os.makedirs(OUT_DIR, exist_ok=True)
    gold[KEYS + ["Domain", "Evaluative", "label4", "ft_split"]].to_csv(
        os.path.join(OUT_DIR, "ft_split.csv"), index=False, encoding="utf-8-sig")
    table = pd.crosstab(gold["ft_split"], np.where(gold["ambiguous"], "ambiguous", gold["label4"]),
                        margins=True)
    print("[INFO] Split (rows: part, columns: gold class):\n" + table.to_string())


def load_data():
    """Non-ambiguous gold sentences with context, labels and ft_split."""
    path = os.path.join(OUT_DIR, "ft_split.csv")
    if not os.path.exists(path):
        make_split()
    split = pd.read_csv(path, encoding="utf-8-sig")[KEYS + ["ft_split"]]
    gold = read_gold().merge(split, on=KEYS, how="inner")
    gold = gold[gold["ft_split"] != "removed_ambiguous"]
    return B.add_context(gold).reset_index(drop=True)


def task_rows(data, task, part):
    rows = data[data["ft_split"] == part]
    if task == "appraisal":
        rows = rows[rows["Evaluative"] == 1]
    return rows.reset_index(drop=True)


def class_of(task, rows):
    return rows["Evaluative"].astype(str) if task == "binary" else rows["category"]


def oversample(rows, task, seed):
    """Repeat examples of the smaller classes until every class has as many as the largest."""
    groups = list(rows.groupby(class_of(task, rows)))
    target = max(len(g) for _, g in groups)
    parts = [g.sample(n=target, replace=len(g) < target, random_state=seed)
             if len(g) < target else g for _, g in groups]
    return pd.concat(parts).reset_index(drop=True)


def prompt_text(task, r):
    if task == "binary":
        return B.PROMPTS[BINARY_PROMPT].format(prev_text=r.prev_text, target_text=r.Sentence,
                                               next_text=r.next_text)
    return A.fill(A.PROMPTS[APPRAISAL_PROMPT], r.prev_text, r.Sentence, r.next_text)


def target_text(task, r):
    """The gold answer as the short JSON object the model learns to produce."""
    if task == "binary":
        return json.dumps({"binary": int(r.Evaluative),
                           "label": "eval" if r.Evaluative else "no_eval"})
    return json.dumps({"label": r.category})

# ==========================
# PARSING + SCORES
# ==========================

def parse_binary(raw):
    return [B.parse_output(t)[0] if t else None for t in raw]


def parse_category(raw):
    """One category per answer; 'ambiguous' / unreadable answers become 'invalid'."""
    out = []
    for t in raw:
        label, flags, _ = A.parse_output(t) if t else (None, None, None)
        on = [c for c in CATEGORIES if flags and flags[c]]
        if label in CATEGORIES:
            out.append(label)
        elif len(on) == 1:
            out.append(on[0])
        else:
            out.append("invalid")
    return out


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


def scores(gold, pred, labels):
    """Single-label scores. Predictions outside `labels` (invalid) count as wrong."""
    gold, pred = list(map(str, gold)), list(map(str, pred))
    row = {"n": len(gold), "invalid": sum(p not in labels for p in pred),
           "macro_f1": f1_score(gold, pred, labels=labels, average="macro", zero_division=0),
           "accuracy": accuracy_score(gold, pred),
           "cohen_kappa": cohen_kappa_score(gold, pred),
           "krippendorff_alpha": krippendorff_alpha(gold, pred)}
    for l, f in zip(labels, f1_score(gold, pred, labels=labels, average=None, zero_division=0)):
        row[f"f1_{l}"] = f
    return row


def val_score(task, rows, raw):
    if task == "binary":
        pred = [str(p) if p is not None else "invalid" for p in parse_binary(raw)]
        return scores(rows["Evaluative"].astype(str), pred, ["0", "1"])
    return scores(rows["category"], parse_category(raw), CATEGORIES)

# ==========================
# MODEL
# ==========================

def load_model(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if not torch.cuda.is_available():
        raise RuntimeError("no GPU visible - set CUDA_VISIBLE_DEVICES")
    n_gpu = torch.cuda.device_count()
    free = [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(n_gpu)]
    print("[INFO] free GPU memory (GiB): " + ", ".join(f"GPU{i} {f:.1f}" for i, f in enumerate(free)),
          flush=True)
    copies_ok = check_gpu_copies() if n_gpu > 1 else True

    if args.multi_gpu:
        # split over all visible GPUs, GPUs only (layers that do not fit would otherwise go
        # to the CPU); keep ~25% of each GPU free for activations and the KV cache
        if not copies_ok:
            raise RuntimeError("GPU-to-GPU copies are corrupted on this server: the model cannot "
                               "be split over GPUs. Run without --multi-gpu (one GPU).")
        max_memory = {i: f"{int(f * 0.75)}GiB" for i, f in enumerate(free)}
        kwargs = {"device_map": "auto", "max_memory": max_memory, "dtype": torch.bfloat16}
    else:
        # default: the whole 4-bit model (~40 GB) on one 48 GB GPU, so no data moves
        # between GPUs (splitting the model gave broken numbers on this server)
        if free[0] < 44:
            raise RuntimeError(f"GPU0 has only {free[0]:.1f} GiB free, ~44 GiB needed: "
                               "set CUDA_VISIBLE_DEVICES to one GPU that no other job is using.")
        kwargs = {"device_map": {"": 0}, "dtype": torch.bfloat16}
    # pre-quantized checkpoints (e.g. unsloth/...-bnb-4bit) bring their own config
    if not hasattr(AutoConfig.from_pretrained(args.base), "quantization_config"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.base, **kwargs)
    placement = getattr(model, "hf_device_map", {"": 0})
    off_gpu = sorted({str(d) for d in placement.values() if not isinstance(d, int)})
    if off_gpu:
        raise RuntimeError(f"part of the model was placed on {off_gpu}, not on the GPUs: "
                           "use GPUs that no other job is using.")
    counts = pd.Series(list(placement.values())).value_counts().sort_index()
    print("[INFO] model layers per GPU: " + ", ".join(f"GPU{d} {n}" for d, n in counts.items())
          + f"; memory used on GPU0 {torch.cuda.memory_allocated(0) / 2**30:.1f} GiB", flush=True)
    return model, tok


def check_gpu_copies():
    """Copy random data between every pair of visible GPUs and compare. On some servers
    peer-to-peer copies are silently corrupted, which breaks a model split over GPUs."""
    import torch

    n = torch.cuda.device_count()
    bad = []
    for i in range(n):
        x = torch.randn(16 * 2**20, device=f"cuda:{i}")
        for j in range(n):
            if i != j and not torch.equal(x.to(f"cuda:{j}").to(f"cuda:{i}"), x):
                bad.append(f"GPU{i}<->GPU{j}")
    del x
    torch.cuda.empty_cache()
    print(f"[INFO] GPU-to-GPU copy check: {'OK' if not bad else 'CORRUPTED ' + ', '.join(bad)}",
          flush=True)
    return not bad


def chat_prefix(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)


def encode(tok, task, rows, max_len):
    """(sentence id, input_ids, labels), with the loss only on the answer tokens."""
    examples, too_long = [], 0
    for r in rows.itertuples():
        p = tok(chat_prefix(tok, prompt_text(task, r)), add_special_tokens=False)["input_ids"]
        a = tok(target_text(task, r) + tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(p) + len(a) > max_len:
            too_long += 1
            p = p[-(max_len - len(a)):]
        examples.append((f"{r.Text_ID}{r.Sentence_ID}", p + a, [-100] * len(p) + a))
    if too_long:
        print(f"[WARN] {too_long} examples cut to --max-len {max_len}")
    return examples


def answer_loss(model, ids, labels):
    """Cross-entropy on the answer tokens only. The output layer is applied only at the
    answer positions, which saves the memory of logits over the whole prompt."""
    import torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    device = base.get_input_embeddings().weight.device
    hidden = base.model(input_ids=torch.tensor([ids], device=device),
                        use_cache=False).last_hidden_state[0, :-1]
    target = torch.tensor(labels[1:], device=hidden.device)  # token t predicts token t+1
    pos = (target != -100).nonzero(as_tuple=True)[0]
    head = base.get_output_embeddings()
    logits = head(hidden[pos].to(head.weight.device)).float()
    return torch.nn.functional.cross_entropy(logits, target[pos].to(logits.device))


def generate(model, tok, task, rows, batch_size, max_new_tokens):
    import torch

    tok.padding_side = "left"
    texts = [chat_prefix(tok, prompt_text(task, r)) for r in rows.itertuples()]
    outs = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, use_cache=True)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"[INFO] {task}: generated {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    return outs

# ==========================
# TRAIN
# ==========================

def train(args):
    import torch
    from peft import LoraConfig, get_peft_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = load_data()
    train_rows, val_rows = task_rows(data, args.task, "train"), task_rows(data, args.task, "val")
    # all test sentences for both tasks: the pipeline needs appraisal answers for every
    # sentence the binary model may call evaluative
    test_rows = task_rows(data, "binary", "test")
    print(f"[INFO] {args.task}: {len(train_rows)} train {class_of(args.task, train_rows).value_counts().to_dict()}"
          f" / {len(val_rows)} val {class_of(args.task, val_rows).value_counts().to_dict()}")
    if not args.no_balance:
        train_rows = oversample(train_rows, args.task, args.seed)
        print(f"[INFO] {args.task}: train balanced by oversampling -> {len(train_rows)} "
              f"{class_of(args.task, train_rows).value_counts().to_dict()}")

    model, tok = load_model(args)
    # QLoRA on one GPU: no fp32 copies of the embeddings / output layer (they do not fit
    # next to the 4-bit model on 48 GB); gradient checkpointing keeps activations small
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.lora_mlp:
        targets += ["gate_proj", "up_proj", "down_proj"]
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=targets))
    model.print_trainable_parameters()

    examples = encode(tok, args.task, train_rows, args.max_len)
    vocab = model.get_output_embeddings().weight.shape[0]
    top = max(max(ids) for _, ids, _ in examples)
    if top >= vocab:
        raise ValueError(f"token id {top} >= model vocabulary {vocab}: tokenizer and model "
                         f"of {args.base} do not match")
    print(f"[INFO] {len(examples)} training examples, max length "
          f"{max(len(ids) for _, ids, _ in examples)} tokens, vocabulary {vocab}")
    # sanity check: the untrained model must give a normal loss (seen: binary ~2.8,
    # appraisal ~6.4, as its short answer differs more from what prompt V2 asks for).
    # Random guessing is ln(128256) = 11.8; the broken runs (model split over GPUs with
    # corrupted GPU-to-GPU copies) gave 10.6-11.8.
    model.eval()
    with torch.no_grad():
        start = [answer_loss(model, ids, labels).item() for _, ids, labels in examples[:8]]
    print(f"[INFO] loss before training (8 examples): {np.mean(start):.3f}", flush=True)
    if not np.isfinite(start).all() or np.mean(start) > 9:
        raise RuntimeError(f"loss before training is {np.mean(start):.3f} (random guessing: 11.8) - "
                           "the model gives broken numbers on this GPU setup.")
    steps = math.ceil(len(examples) / args.grad_accum) * args.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / max(1, steps - warmup))))

    adapter_dir = os.path.join(OUT_DIR, f"adapter_{args.task}")
    log, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(examples)
        total, counted, skipped, t0 = 0.0, 0, 0, time.time()
        for i, (key, ids, labels) in enumerate(examples, 1):
            loss = answer_loss(model, ids, labels)
            # one nan / inf loss would turn every LoRA weight into nan: skip that example
            if torch.isfinite(loss):
                (loss / args.grad_accum).backward()
                total += loss.item()
                counted += 1
            else:
                skipped += 1
                print(f"[WARN] epoch {epoch}: non-finite loss on {key}, example skipped", flush=True)
            if i % args.grad_accum == 0 or i == len(examples):
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if torch.isfinite(norm):
                    opt.step()
                else:
                    skipped += 1
                    print(f"[WARN] epoch {epoch}: non-finite gradient at example {i}, "
                          f"update skipped", flush=True)
                sched.step()
                opt.zero_grad()
            if i % 25 == 0:
                print(f"[INFO] epoch {epoch} {i}/{len(examples)} loss {total / max(counted, 1):.4f}",
                      flush=True)

        raw = generate(model, tok, args.task, val_rows, args.batch_size, max_new_tokens=32)
        val = val_score(args.task, val_rows, raw)
        row = {"epoch": epoch, "train_loss": total / max(counted, 1), "skipped": skipped,
               "minutes": (time.time() - t0) / 60, **{f"val_{k}": v for k, v in val.items()}}
        log.append(row)
        print(f"[INFO] epoch {epoch}: train loss {row['train_loss']:.4f}, val macro-F1 "
              f"{val['macro_f1']:.3f}, kappa {val['cohen_kappa']:.3f} ({val['invalid']} invalid)",
              flush=True)

        if best is None or val["macro_f1"] > best:
            best = val["macro_f1"]
            model.save_pretrained(adapter_dir)
            tok.save_pretrained(adapter_dir)
            out = val_rows.copy()
            out["raw_output"] = raw
            out.to_csv(os.path.join(OUT_DIR, f"val_pred_{args.task}.csv"),
                       index=False, encoding="utf-8-sig")
            print(f"[INFO] saved best adapter (epoch {epoch}) to {adapter_dir}")
            # answers on every test sentence from this (best-val) model, in memory: the
            # adapter is not reloaded for testing. The epoch is chosen on val only.
            test_raw = generate(model, tok, args.task, test_rows, args.batch_size, max_new_tokens=32)
            test_rows[KEYS].assign(raw_output=test_raw, epoch=epoch).to_csv(
                os.path.join(OUT_DIR, f"test_raw_{args.task}.csv"), index=False, encoding="utf-8-sig")
            print(f"[INFO] test answers of epoch {epoch} saved; first: {test_raw[0]!r}", flush=True)

    pd.DataFrame(log).to_csv(os.path.join(OUT_DIR, f"train_log_{args.task}.csv"), index=False)
    print(f"[INFO] {args.task}: best val macro-F1 {best:.3f}")

# ==========================
# TEST (held-out 30%, two steps)
# ==========================

def test(args):
    data = task_rows(load_data(), "binary", "test")
    if args.limit:  # quick check on a few sentences; results go to a separate folder
        data = data.head(args.limit)
    name = ("zeroshot" if args.zero_shot else "finetuned") + (f"_limit{args.limit}" if args.limit else "")
    print(f"[INFO] Test ({name}): {len(data)} held-out sentences "
          f"{data['label4'].value_counts().to_dict()}")
    if args.zero_shot:
        model, tok = load_model(args)
        # zero-shot answers follow the full prompt format (with explanations), so they are longer
        # (smaller batches: long answers need more KV-cache memory next to the 40 GB model)
        answer = lambda task, rows: generate(model, tok, task, rows, min(args.batch_size, 4), 512)
    else:
        # fine-tuned answers were written by `train` (best-val epoch, model in memory)
        saved = {}
        for task in ("binary", "appraisal"):
            path = os.path.join(OUT_DIR, f"test_raw_{task}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} missing - run `train --task {task}` first")
            df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
            saved[task] = dict(zip(zip(df["Text_ID"], df["Sentence_ID"]), df["raw_output"]))
            print(f"[INFO] {task}: test answers from epoch {df['epoch'].iloc[0]}")
        answer = lambda task, rows: [saved[task][k] for k in zip(rows["Text_ID"], rows["Sentence_ID"])]

    # step 1: evaluative or not, every sentence
    data["binary_raw"] = answer("binary", data)
    for t in data["binary_raw"][:3]:
        print(f"[INFO] binary raw answer: {t!r}")
    step1 = parse_binary(data["binary_raw"])
    data["pred_binary"] = [str(p) if p is not None else "invalid" for p in step1]

    # step 2: category, for every gold- or predicted-evaluative sentence
    # (gold-evaluative ones give the appraisal score on its own; the pipeline only uses
    # the answers for sentences step 1 called evaluative)
    need = data[(data["pred_binary"] == "1") | (data["Evaluative"] == 1)]
    raw = dict(zip(need.index, answer("appraisal", need)))
    data["appraisal_raw"] = [raw.get(i, "") for i in data.index]
    for t in list(raw.values())[:3]:
        print(f"[INFO] appraisal raw answer: {t!r}")
    data["pred_category"] = [c if i in raw else "" for i, c in
                             zip(data.index, parse_category(data["appraisal_raw"]))]
    data["pred_label4"] = np.where(data["pred_binary"] == "1", data["pred_category"],
                                   np.where(data["pred_binary"] == "0", "none", "invalid"))

    out_dir = os.path.join(OUT_DIR, f"test_{name}")
    os.makedirs(out_dir, exist_ok=True)
    data.drop(columns=["prev_text", "next_text"]).to_csv(
        os.path.join(out_dir, "predictions.csv"), index=False, encoding="utf-8-sig")
    table = test_scores(data).assign(model=name)
    table.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    print(table.round(3).to_string(index=False))


STEP_1 = "1 binary (all test sentences)"
STEP_2 = "2 appraisal (sentences the model called evaluative)"
COMBINED = "combined binary + appraisal (none / affect / judgment / appreciation)"


def test_scores(data):
    # step 2 only sees what step 1 passed on; a gold non-evaluative sentence among them
    # has gold label "none" and is always wrong
    passed = data[data["pred_binary"].astype(str) == "1"]
    return pd.DataFrame([
        {"step": STEP_1, **scores(data["Evaluative"].astype(str), data["pred_binary"], ["0", "1"])},
        {"step": STEP_2, **scores(passed["label4"], passed["pred_category"], CATEGORIES),
         "gold_noneval": int((passed["Evaluative"] == 0).sum())},
        {"step": COMBINED, **scores(data["label4"], data["pred_label4"], PIPE_LABELS)},
    ])

# ==========================
# COMPARE fine-tuned vs zero-shot
# ==========================

def randomization_test(gold, pred_a, pred_b, labels, rng):
    """Paired approximate randomization test on macro-F1."""
    gold, pred_a, pred_b = (np.asarray(x).astype(str) for x in (gold, pred_a, pred_b))
    f = lambda p: f1_score(gold, p, labels=labels, average="macro", zero_division=0)
    diff = f(pred_a) - f(pred_b)
    hits = 0
    for _ in range(N_PERM):
        swap = rng.random(len(gold)) < 0.5
        hits += abs(f(np.where(swap, pred_b, pred_a)) - f(np.where(swap, pred_a, pred_b))) >= abs(diff) - 1e-12
    return diff, (hits + 1) / (N_PERM + 1)


def mcnemar(gold, pred_a, pred_b):
    right_a = np.asarray(pred_a).astype(str) == np.asarray(gold).astype(str)
    right_b = np.asarray(pred_b).astype(str) == np.asarray(gold).astype(str)
    b, c = int((right_a & ~right_b).sum()), int((~right_a & right_b).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)


def compare(args):
    preds, tables = {}, []
    for name in ("finetuned", "zeroshot"):
        path = os.path.join(OUT_DIR, f"test_{name}", "predictions.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing - run `test`{' --zero-shot' if name == 'zeroshot' else ''} first")
        preds[name] = pd.read_csv(path, encoding="utf-8-sig", dtype={"pred_binary": str}).fillna("")
        tables.append(test_scores(preds[name]).assign(model=name))
    ft, zs = preds["finetuned"], preds["zeroshot"]
    assert (ft[KEYS].values == zs[KEYS].values).all(), "different test sentences"

    rng = np.random.default_rng(0)
    tests = []
    for step, gold, col, labels in [(STEP_1, ft["Evaluative"].astype(str), "pred_binary", ["0", "1"]),
                                    (COMBINED, ft["label4"], "pred_label4", PIPE_LABELS)]:
        g, a, b = np.asarray(gold), np.asarray(ft[col]), np.asarray(zs[col])
        diff, p_rand = randomization_test(g, a, b, labels, rng)
        only_ft, only_zs, p_mc = mcnemar(g, a, b)
        tests.append({"step": step, "macro_f1_diff (ft - zs)": diff, "p_randomization": p_rand,
                      "only_finetuned_correct": only_ft, "only_zeroshot_correct": only_zs,
                      "p_mcnemar": p_mc})
    # step 2 is scored on different sentences per model (each model's own step 1),
    # so there is no paired test for it; the combined step covers both prompts

    metrics = pd.concat(tables)
    cols = ["step", "model", "n", "gold_noneval", "invalid", "macro_f1", "accuracy", "cohen_kappa",
            "krippendorff_alpha"] + [c for c in metrics.columns if c.startswith("f1_")]
    metrics = metrics[cols].sort_values(["step", "model"])
    tests = pd.DataFrame(tests)
    metrics.to_csv(os.path.join(OUT_DIR, "test_results.csv"), index=False)
    tests.to_csv(os.path.join(OUT_DIR, "test_significance.csv"), index=False)

    pd.set_option("display.width", 250)
    text = ("=== Held-out test (30%, ambiguous removed): fine-tuned vs zero-shot Llama 3.3 70B ===\n\n"
            + metrics.round(3).to_string(index=False)
            + "\n\n=== Significance (paired, same sentences) ===\n"
            + tests.round(4).to_string(index=False)
            + f"\n\np_randomization: approximate randomization test on macro-F1 ({N_PERM:,} permutations);"
            + "\np_mcnemar: exact McNemar test on per-sentence correctness."
            + "\nInvalid answers (unreadable / 'ambiguous') count as wrong."
            + "\nStep 2 is scored on the sentences each model's step 1 called evaluative (n differs per"
            + "\nmodel; gold_noneval = how many of them are not evaluative in the gold standard, always wrong)."
            + "\nThe combined step scores both prompts together on all test sentences.\n")
    with open(os.path.join(OUT_DIR, "test_results.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)

# ==========================
# CLI
# ==========================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split", help="write the train / val / test split (ambiguous removed)")

    def model_args(p):
        p.add_argument("--base", default=BASE_MODEL,
                       help="HF id or local path (e.g. unsloth/Llama-3.3-70B-Instruct-bnb-4bit)")
        p.add_argument("--batch-size", type=int, default=8, help="generation batch size")
        p.add_argument("--multi-gpu", action="store_true",
                       help="split the model over all visible GPUs (default: one GPU)")

    t = sub.add_parser("train", help="fine-tune one task, keep the best epoch by val macro-F1")
    t.add_argument("--task", required=True, choices=["binary", "appraisal"])
    model_args(t)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--grad-accum", type=int, default=8)
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--max-len", type=int, default=2048)
    t.add_argument("--no-balance", action="store_true", help="do not oversample smaller classes")
    t.add_argument("--lora-mlp", action="store_true",
                   help="LoRA also on the MLP layers (more memory; e.g. with --multi-gpu)")
    t.add_argument("--seed", type=int, default=SEED)

    s = sub.add_parser("test", help="two-step test on the held-out 30%%")
    model_args(s)
    s.add_argument("--zero-shot", action="store_true", help="base model without adapters (baseline)")
    s.add_argument("--limit", type=int, default=None, help="only the first N test sentences (quick check)")

    sub.add_parser("compare", help="fine-tuned vs zero-shot on the test set, with significance")

    args = ap.parse_args()
    if args.cmd == "split":
        make_split()
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "test":
        test(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
