"""
Stage 7 (OPTIONAL, supplementary): fine-tune a pruned model with LoRA and
check whether it recovers, and whether that recovery generalises to a
second dataset. This is not the main comparison in this pipeline - see
Methodology / Fine-Tuning Recovery for why it is kept small.

Usage:
    python stage7_finetune.py --condition causal --n_examples 4000
"""

import argparse
import os

from peft import LoraConfig, get_peft_model

import config
import stage4_build_masks as stage4
import utils


def run(condition, n_examples):
    print(f"Stage 7 (optional): fine-tune {condition}-pruned model on {n_examples} examples")

    masks, model, tokenizer, dims = stage4.run(condition)
    stage4.apply_masks_to_model(model, masks, dims)

    lora_cfg = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_texts, train_labels = utils.load_split(config.DATASET, "train", n_examples, seed=config.SEED + 2)
    _, _, val_texts, val_labels = utils.load_disjoint_splits(
        config.DATASET, config.N_EVAL_EXAMPLES, config.N_VAL_EXAMPLES, seed=config.SEED
    )

    pos_id, neg_id = utils.get_pos_neg_token_ids(tokenizer)
    optimizer = __import__("torch").optim.AdamW(model.parameters(), lr=config.FT_LR)

    best_val_auc = -1
    best_state = None

    # NOTE: this is a minimal, readable training loop, not a
    # production-grade Trainer setup - fine for an MSc-scale experiment
    import torch

    device = next(model.parameters()).device
    n_steps = n_examples // config.FT_BATCH_SIZE

    for step in range(n_steps):
        batch_start = (step * config.FT_BATCH_SIZE) % len(train_texts)
        batch_texts = train_texts[batch_start : batch_start + config.FT_BATCH_SIZE]
        batch_labels = train_labels[batch_start : batch_start + config.FT_BATCH_SIZE]
        if len(batch_texts) == 0:
            continue

        prompts = [utils.make_prompt(t) + (" positive" if y == 1 else " negative")
                   for t, y in zip(batch_texts, batch_labels)]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        out = model(**enc, labels=enc["input_ids"])
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.FT_VAL_CHECK_EVERY == 0:
            val_logodds = utils.get_logodds(model, tokenizer, val_texts, pos_id, neg_id)
            val_auc = utils.compute_metrics(val_labels, val_logodds)["auc"]
            print(f"  step {step}/{n_steps}: loss={loss.item():.4f} val_auc={val_auc:.4f}")
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    adapter_path = os.path.join(config.PATHS["adapters"], f"{condition}_{n_examples}ex")
    model.save_pretrained(adapter_path)
    print(f"  saved adapter -> {adapter_path} (best val AUC = {best_val_auc:.4f})")

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["causal", "magnitude", "random"], required=True)
    parser.add_argument("--n_examples", type=int, default=config.FT_EXAMPLES_LARGE)
    args = parser.parse_args()
    run(args.condition, args.n_examples)
