"""
Stage 4: turn the causal features from stage 3 into an actual pruning
plan - one binary mask per layer, per condition (causal / magnitude /
random). A mask says which MLP neurons and which attention heads survive.

Usage:
    python stage4_build_masks.py --condition causal
    python stage4_build_masks.py --condition magnitude
    python stage4_build_masks.py --condition random
"""

import argparse
import os

import config
import pruning
import utils


def build_masks_for_condition(model, feature_results, dims, condition, random_seed=1001):
    """
    Returns {layer_idx: {"mlp": mask, "heads": mask}} for every layer
    that gets pruned under this condition (layers with 0 causal features
    are skipped entirely, matching the causal condition's own tiering).
    """
    masks = {}

    for layer_idx, layer in enumerate(model.model.layers):
        n_causal = len(feature_results[layer_idx]["causal_idx"])
        prune_pct = pruning.get_tier(n_causal)
        if prune_pct is None:
            continue  # this layer is left untouched under every condition,
                      # so the comparison stays apples-to-apples on sparsity

        if condition == "causal":
            directions = feature_results[layer_idx]["causal_directions"]
            mlp_importance = pruning.causal_mlp_importance(layer, directions)
            head_importance = pruning.causal_head_importance(
                layer, directions, dims["n_heads"], dims["head_dim"]
            )
        elif condition == "magnitude":
            mlp_importance = pruning.magnitude_mlp_importance(layer)
            head_importance = pruning.magnitude_head_importance(
                layer, dims["n_heads"], dims["head_dim"]
            )
        elif condition == "random":
            mlp_importance = pruning.random_importance(dims["intermediate_size"], seed=random_seed + layer_idx)
            head_importance = pruning.random_importance(dims["n_heads"], seed=random_seed + layer_idx + 1000)
        else:
            raise ValueError(f"unknown condition: {condition}")

        masks[layer_idx] = {
            "mlp": pruning.build_binary_mask(mlp_importance, prune_pct),
            "heads": pruning.build_binary_mask(head_importance, prune_pct),
        }

    return masks


def apply_masks_to_model(model, masks, dims):
    for layer_idx, m in masks.items():
        layer = model.model.layers[layer_idx]
        pruning.apply_mlp_mask(layer, m["mlp"])
        pruning.apply_head_mask(
            layer, m["heads"], dims["n_heads"], dims["n_kv_heads"],
            dims["head_dim"], dims["heads_per_group"]
        )


def measure_real_sparsity(model):
    """
    Count what fraction of the model's parameters are exactly zero after
    pruning. This is the number reported in Results - it measures actual
    parameters, not just how many neurons/heads were selected.
    """
    total, zeroed = 0, 0
    for param in model.parameters():
        total += param.numel()
        zeroed += (param == 0).sum().item()
    return zeroed / total


def sparsity_of(masks, dims):
    """Rough unit-level sparsity (what fraction of neurons/heads were
    selected for pruning). Use measure_real_sparsity() for the actual
    parameter-level figure."""
    total, pruned = 0, 0
    for m in masks.values():
        total += len(m["mlp"]) + len(m["heads"])
        pruned += (m["mlp"] == 0).sum() + (m["heads"] == 0).sum()
    return pruned / total if total > 0 else 0.0


def run(condition, seed=None):
    import stage3_feature_discovery as stage3

    print(f"Stage 4: build masks for condition = {condition}")
    feature_results, saes = stage3.run()

    model, tokenizer = utils.load_model_and_tokenizer()
    dims = utils.get_model_dims(model)

    path = os.path.join(config.PATHS["masks"], f"{condition}.pkl")

    def compute():
        return build_masks_for_condition(model, feature_results, dims, condition, random_seed=seed or 1001)

    masks = utils.cache_or_compute(path, compute, saver=utils.save_pickle, loader=utils.load_pickle)

    print(f"  masks built for {len(masks)} layers, "
          f"proportion of units pruned (rough, unit-count basis) = {sparsity_of(masks, dims):.3f}")
    return masks, model, tokenizer, dims


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["causal", "magnitude", "random"], required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run(args.condition, seed=args.seed)
