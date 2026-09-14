"""
Stage 9: live-model verification.

The causal verification in stage 3 measures each feature's effect on a
linear probe reading cached SAE reconstructions. That is a stand-in for
the model itself, used because running every ablation through Mistral
would be far too slow.

This stage checks that the stand-in was reasonable. For the strongest
surviving features in each layer, it repeats the ablation inside a real
Mistral forward pass, using a hook that removes the feature's
contribution from the residual stream before the remaining layers run.

Two things are recorded per feature: the flip rate, and the mean
absolute change in the model's predicted sentiment log-odds. The second
matters because a feature can shift the model's confidence without
pushing any prediction across the decision boundary.

This does not filter the causal features used for pruning. Stage 4 uses
everything stage 3 verified. This stage is a check, not a gate.

Usage:
    python stage9_live_verification.py
    python stage9_live_verification.py --top_k 5 --n_examples 200
"""

import argparse
import os

import numpy as np
import torch

import config
import sae as sae_module
import utils


def make_ablation_hook(sae_model, feature_idx):
    """
    Returns a forward hook that removes one SAE feature's contribution
    from the residual stream as it passes through the target layer.
    """
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output

        original_dtype = hidden.dtype
        hidden_f32 = hidden.float()

        # encode the live residual stream, take this feature's activation,
        # and subtract its contribution back out
        features = sae_model.encode(hidden_f32)
        activation = features[..., feature_idx : feature_idx + 1]
        direction = sae_model.W_dec[feature_idx, :]
        modified = hidden_f32 - activation * direction

        modified = modified.to(original_dtype)

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    return hook


@torch.no_grad()
def logodds_with_ablation(model, tokenizer, texts, pos_id, neg_id,
                          layer_idx, sae_model, feature_idx):
    """Run the eval set through the model with one feature ablated."""
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(make_ablation_hook(sae_model, feature_idx))
    try:
        return utils.get_logodds(model, tokenizer, texts, pos_id, neg_id)
    finally:
        handle.remove()


def verify_layer(model, tokenizer, texts, pos_id, neg_id, layer_idx,
                 sae_model, feature_indices, baseline_logodds):
    """Ablate each given feature in turn and compare against baseline."""
    results = []
    baseline_preds = (baseline_logodds > 0).astype(int)

    for feature_idx in feature_indices:
        ablated = logodds_with_ablation(
            model, tokenizer, texts, pos_id, neg_id,
            layer_idx, sae_model, feature_idx
        )
        ablated_preds = (ablated > 0).astype(int)

        flip_rate = float((baseline_preds != ablated_preds).mean())
        mean_abs_shift = float(np.abs(baseline_logodds - ablated).mean())

        results.append({
            "feature": int(feature_idx),
            "flip_rate": flip_rate,
            "mean_abs_logodds_shift": mean_abs_shift,
        })
        print(f"    feature {feature_idx}: flip_rate={flip_rate:.4f} "
              f"mean|delta logodds|={mean_abs_shift:.4f}")

    return results


def run(top_k=5, n_examples=None):
    import stage3_feature_discovery as stage3

    n_examples = n_examples or 200
    print(f"Stage 9: live-model verification (top {top_k} features per layer, "
          f"{n_examples} examples)")

    feature_results, saes = stage3.run()

    texts, labels = utils.load_split(
        config.DATASET, "test", n_examples, seed=config.SEED
    )

    model, tokenizer = utils.load_model_and_tokenizer()
    pos_id, neg_id = utils.get_pos_neg_token_ids(tokenizer)

    # one unablated pass, reused as the reference for every feature
    baseline_path = os.path.join(config.PATHS["results"], "live_baseline_logodds.npy")
    baseline = utils.cache_or_compute(
        baseline_path,
        lambda: utils.get_logodds(model, tokenizer, texts, pos_id, neg_id),
    )
    baseline_auc = utils.compute_metrics(labels, baseline)["auc"]
    print(f"  unablated baseline AUC: {baseline_auc:.4f}")

    all_results = {}
    for layer_idx, result in sorted(feature_results.items()):
        causal_idx = result["causal_idx"]
        if len(causal_idx) == 0:
            continue

        # stage 3 keeps causal_idx in descending flip-rate order, so the
        # first entries are the strongest candidates for this layer
        to_test = list(causal_idx)[:top_k]
        print(f"  layer {layer_idx} ({len(causal_idx)} causal features, "
              f"testing {len(to_test)}):")

        path = os.path.join(config.PATHS["results"], f"live_verify_layer_{layer_idx}.pkl")

        def compute(layer_idx=layer_idx, to_test=to_test):
            return verify_layer(
                model, tokenizer, texts, pos_id, neg_id,
                layer_idx, saes[layer_idx], to_test, baseline
            )

        all_results[layer_idx] = utils.cache_or_compute(
            path, compute, saver=utils.save_pickle, loader=utils.load_pickle
        )

    print()
    print("=" * 60)
    print("Summary: strongest feature per layer")
    print("=" * 60)
    print(f"{'Layer':>6} {'flip rate':>12} {'mean |d logodds|':>18}")
    for layer_idx, results in sorted(all_results.items()):
        if not results:
            continue
        best = max(results, key=lambda r: r["flip_rate"])
        print(f"{layer_idx:>6} {best['flip_rate']:>12.4f} "
              f"{best['mean_abs_logodds_shift']:>18.4f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5,
                        help="how many of each layer's strongest features to test")
    parser.add_argument("--n_examples", type=int, default=200,
                        help="evaluation examples per ablation")
    args = parser.parse_args()
    run(top_k=args.top_k, n_examples=args.n_examples)
