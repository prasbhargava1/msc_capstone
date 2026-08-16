"""
Shared functions used by every stage of the pipeline: loading the model,
loading data, running the zero-shot sentiment prompt, computing metrics,
bootstrap comparisons, and a generic "compute this or load it from disk
if it already exists" helper that all the caching in this project uses.
"""

import os
import pickle

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import config


# -----------------------------------------------------------------------
# generic caching helper
# -----------------------------------------------------------------------
def cache_or_compute(path, compute_fn, saver=None, loader=None):
    """
    If `path` already exists, load and return it. Otherwise call
    compute_fn(), save the result to `path`, and return it.

    This is the one function that gives the whole pipeline its
    "reuse if it exists, else compute" behaviour. Every stage calls
    this instead of doing its own if-exists check.
    """
    if os.path.exists(path):
        print(f"  [cache hit] {path}")
        if loader is not None:
            return loader(path)
        return np.load(path, allow_pickle=True)

    print(f"  [computing] {path}")
    result = compute_fn()

    if saver is not None:
        saver(result, path)
    else:
        np.save(path, result)

    return result


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------
# model loading
# -----------------------------------------------------------------------
def load_model_and_tokenizer(model_name=None, device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_name or config.MODEL_NAME
    device = device or (config.DEVICE if torch.cuda.is_available() else "cpu")

    dtype = torch.float16 if config.DTYPE == "float16" and device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map={"": 0} if device == "cuda" else None
    ).eval()

    if device == "cpu":
        model = model.to("cpu")

    return model, tokenizer


def get_model_dims(model):
    """
    Read the architecture dimensions we need straight from the model's
    own config, instead of hardcoding them. Works for any Llama-family
    model (Mistral, Llama, TinyLlama, ...).
    """
    cfg = model.config
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    return {
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "n_layers": cfg.num_hidden_layers,
        "n_heads": cfg.num_attention_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": cfg.hidden_size // cfg.num_attention_heads,
        "heads_per_group": cfg.num_attention_heads // n_kv_heads,
    }


# -----------------------------------------------------------------------
# data loading
# -----------------------------------------------------------------------
def load_split(dataset_cfg, split_name, n, seed):
    """
    Load n examples (balanced classes) from one split of a HF dataset,
    shuffled first so the seed actually changes which examples are picked
    and not just their order.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_cfg["hf_name"])[split_name]
    ds = ds.shuffle(seed=seed)

    text_field = dataset_cfg["text_field"]
    label_field = dataset_cfg["label_field"]
    pos_val = dataset_cfg["positive_value"]

    pos = [x for x in ds if x[label_field] == pos_val][: n // 2]
    neg = [x for x in ds if x[label_field] != pos_val][: n // 2]
    examples = pos + neg

    rng = np.random.RandomState(seed)
    rng.shuffle(examples)

    texts = [x[text_field] for x in examples]
    labels = np.array([1 if x[label_field] == pos_val else 0 for x in examples])
    return texts, labels


def load_disjoint_splits(dataset_cfg, n_eval, n_val, seed):
    """
    Build two subsets of the same split that are guaranteed not to
    overlap: shuffle once, then carve non-overlapping index ranges
    per class, instead of drawing each subset with its own independent
    shuffle (which does not guarantee disjointness).
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_cfg["hf_name"])["test"]
    ds = ds.shuffle(seed=seed)

    text_field = dataset_cfg["text_field"]
    label_field = dataset_cfg["label_field"]
    pos_val = dataset_cfg["positive_value"]

    pos = [x for x in ds if x[label_field] == pos_val]
    neg = [x for x in ds if x[label_field] != pos_val]

    n_eval_half, n_val_half = n_eval // 2, n_val // 2

    eval_examples = pos[:n_eval_half] + neg[:n_eval_half]
    val_examples = (
        pos[n_eval_half : n_eval_half + n_val_half]
        + neg[n_eval_half : n_eval_half + n_val_half]
    )

    def to_texts_labels(examples, seed):
        rng = np.random.RandomState(seed)
        rng.shuffle(examples)
        texts = [x[text_field] for x in examples]
        labels = np.array([1 if x[label_field] == pos_val else 0 for x in examples])
        return texts, labels

    eval_texts, eval_labels = to_texts_labels(eval_examples, seed)
    val_texts, val_labels = to_texts_labels(val_examples, seed + 1)

    overlap = len(set(eval_texts) & set(val_texts))
    print(f"  disjointness check: eval={len(eval_texts)} val={len(val_texts)} overlap={overlap}")
    assert overlap == 0, "eval and validation sets overlap - this should never happen"

    return eval_texts, eval_labels, val_texts, val_labels


# -----------------------------------------------------------------------
# zero-shot sentiment prompting
# -----------------------------------------------------------------------
def make_prompt(text):
    return f"Review: {text[:config.MAX_PROMPT_LEN].strip()}\nSentiment (positive/negative):"


def get_pos_neg_token_ids(tokenizer):
    pos_id = tokenizer.encode(" positive", add_special_tokens=False)
    neg_id = tokenizer.encode(" negative", add_special_tokens=False)
    assert len(pos_id) == 1 and len(neg_id) == 1, (
        "this pipeline assumes 'positive'/'negative' are single tokens "
        "for this tokenizer - check before running further"
    )
    return pos_id[0], neg_id[0]


@torch.no_grad()
def get_logodds(model, tokenizer, texts, pos_id, neg_id, batch_size=8):
    device = next(model.parameters()).device
    prompts = [make_prompt(t) for t in texts]
    logodds = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        logits = model(**enc).logits[:, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)
        lo = (logprobs[:, pos_id] - logprobs[:, neg_id]).float().cpu().numpy()
        logodds.extend(lo)

    return np.array(logodds)


# -----------------------------------------------------------------------
# metrics
# -----------------------------------------------------------------------
def compute_metrics(labels, logodds, threshold=0.0):
    preds = (logodds > threshold).astype(int)
    pos_mask = labels == 1
    neg_mask = labels == 0
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, logodds),
        "pos_acc": accuracy_score(labels[pos_mask], preds[pos_mask]) if pos_mask.any() else 0.0,
        "neg_acc": accuracy_score(labels[neg_mask], preds[neg_mask]) if neg_mask.any() else 0.0,
    }


def print_metrics(name, m):
    print(
        f"[{name}] acc={m['accuracy']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f} "
        f"pos_acc={m['pos_acc']:.3f} neg_acc={m['neg_acc']:.3f}"
    )


# -----------------------------------------------------------------------
# paired bootstrap
# -----------------------------------------------------------------------
def paired_bootstrap(labels, logodds_a, logodds_b, n_resamples=None, seed=None):
    """
    Compare two conditions evaluated on the same examples. Each resample
    draws the same indices for both conditions, so shared sampling noise
    cancels out instead of being counted for each condition separately.
    """
    n_resamples = n_resamples or config.N_BOOTSTRAP_RESAMPLES
    seed = seed if seed is not None else config.SEED
    rng = np.random.RandomState(seed)
    n = len(labels)

    diffs, wins, valid = [], 0, 0
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        bl = labels[idx]
        if len(np.unique(bl)) < 2:
            continue
        valid += 1
        auc_a = roc_auc_score(bl, logodds_a[idx])
        auc_b = roc_auc_score(bl, logodds_b[idx])
        diffs.append(auc_a - auc_b)
        if auc_a > auc_b:
            wins += 1

    diffs = np.array(diffs)
    return {
        "mean_diff": diffs.mean(),
        "ci95": (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)),
        "p_a_wins": wins / valid,
        "n_valid": valid,
    }
