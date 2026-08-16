"""
Stage 5: load the model, optionally apply a pruning mask, run the
zero-shot sentiment eval, and save both the metrics and the raw
log-odds (the raw log-odds are needed later for the bootstrap
comparison in stage 6 - metrics alone are not enough for that).

Usage:
    python stage5_evaluate.py --condition fresh       (no pruning at all)
    python stage5_evaluate.py --condition causal
    python stage5_evaluate.py --condition magnitude
    python stage5_evaluate.py --condition random
"""

import argparse
import os

import numpy as np

import config
import stage4_build_masks as stage4
import utils


def run(condition):
    print(f"Stage 5: evaluate condition = {condition}")

    eval_texts, eval_labels, _, _ = utils.load_disjoint_splits(
        config.DATASET, config.N_EVAL_EXAMPLES, config.N_VAL_EXAMPLES, seed=config.SEED
    )

    if condition == "fresh":
        model, tokenizer = utils.load_model_and_tokenizer()
    else:
        masks, model, tokenizer, dims = stage4.run(condition)
        stage4.apply_masks_to_model(model, masks, dims)

    pos_id, neg_id = utils.get_pos_neg_token_ids(tokenizer)

    logodds_path = os.path.join(config.PATHS["results"], f"{condition}_logodds.npy")
    logodds = utils.cache_or_compute(
        logodds_path,
        lambda: utils.get_logodds(model, tokenizer, eval_texts, pos_id, neg_id),
    )

    metrics = utils.compute_metrics(eval_labels, logodds)
    utils.print_metrics(condition, metrics)

    metrics_path = os.path.join(config.PATHS["results"], f"{condition}_metrics.pkl")
    utils.save_pickle(metrics, metrics_path)

    return logodds, eval_labels, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", choices=["fresh", "causal", "magnitude", "random"], required=True
    )
    args = parser.parse_args()
    run(args.condition)
