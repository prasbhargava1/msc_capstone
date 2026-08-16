"""
Stage 8: cross-domain evaluation.

Evaluates a pruning condition on datasets that were never used at any
stage of the pipeline - not for SAE training, not for causal feature
discovery, not for pruning. This tests whether an advantage measured on
IMDB (the dataset the pruning decision was derived from) also holds on
data from a different distribution.

Rotten Tomatoes is same-domain (movie reviews) but a different style
(short critic snippets rather than long user reviews). Yelp Polarity is
a different domain entirely (restaurant and business reviews).

Usage:
    python stage8_cross_domain.py --condition causal
    python stage8_cross_domain.py --condition magnitude
"""

import argparse
import os

import numpy as np

import config
import stage4_build_masks as stage4
import utils


def evaluate_on_dataset(model, tokenizer, pos_id, neg_id, dataset_name, dataset_cfg, condition):
    """Evaluate one already-pruned model on one held-out dataset."""
    # Yelp has no validation split, so both datasets are sampled from their
    # test split here - these are evaluation-only datasets, so no validation
    # subset is needed.
    texts, labels = utils.load_split(
        dataset_cfg, "test", config.N_EVAL_EXAMPLES, seed=config.SEED
    )

    logodds_path = os.path.join(
        config.PATHS["results"], f"{condition}_{dataset_name}_logodds.npy"
    )
    logodds = utils.cache_or_compute(
        logodds_path,
        lambda: utils.get_logodds(model, tokenizer, texts, pos_id, neg_id),
    )

    metrics = utils.compute_metrics(labels, logodds)
    utils.print_metrics(f"{condition} on {dataset_name}", metrics)

    labels_path = os.path.join(config.PATHS["results"], f"{dataset_name}_labels.npy")
    if not os.path.exists(labels_path):
        np.save(labels_path, labels)

    return logodds, labels, metrics


def run(condition):
    print(f"Stage 8: cross-domain evaluation, condition = {condition}")

    if condition == "fresh":
        model, tokenizer = utils.load_model_and_tokenizer()
    else:
        masks, model, tokenizer, dims = stage4.run(condition)
        stage4.apply_masks_to_model(model, masks, dims)
        sparsity = stage4.measure_real_sparsity(model)
        print(f"  actual parameter sparsity: {sparsity:.2%}")

    pos_id, neg_id = utils.get_pos_neg_token_ids(tokenizer)

    results = {}
    for dataset_name, dataset_cfg in config.GENERALIZATION_DATASETS.items():
        logodds, labels, metrics = evaluate_on_dataset(
            model, tokenizer, pos_id, neg_id, dataset_name, dataset_cfg, condition
        )
        results[dataset_name] = metrics

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", choices=["fresh", "causal", "magnitude", "random"], required=True
    )
    args = parser.parse_args()
    run(args.condition)
