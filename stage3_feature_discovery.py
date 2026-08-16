"""
Stage 3: find which SAE features are causally linked to sentiment.

Two steps per layer:
  1. correlation - score every feature by how differently it fires on
     positive vs negative examples, keep the top N as candidates.
  2. causal verification - ablate each candidate (and, separately, an
     equal number of random non-candidate features) and measure how
     often a linear probe's prediction flips. A candidate is kept only
     if it flips predictions at least as often as the strongest random
     feature did, in that same layer.

Usage:
    python stage3_feature_discovery.py
"""

import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

import config
import stage2_train_saes as stage2
import utils


def correlational_scores(sae_model, activations, labels, device):
    x = torch.tensor(activations, dtype=torch.float32, device=device)
    with torch.no_grad():
        features, _ = sae_model(x)
    features = features.cpu().numpy()

    pos_mean = features[labels == 1].mean(axis=0)
    neg_mean = features[labels == 0].mean(axis=0)
    return np.abs(pos_mean - neg_mean)


def flip_rate(probe, x_hat, x_hat_ablated):
    base_pred = probe.predict(x_hat)
    ablated_pred = probe.predict(x_hat_ablated)
    return (base_pred != ablated_pred).mean()


def ablate_feature(x_hat, features, feature_idx, decoder_row):
    """x_hat with one feature's contribution subtracted out."""
    contribution = features[:, feature_idx : feature_idx + 1] * decoder_row[None, :]
    return x_hat - contribution


def causal_verification(sae_model, activations, labels, candidate_idx, device):
    x = torch.tensor(activations, dtype=torch.float32, device=device)
    with torch.no_grad():
        features, x_hat = sae_model(x)
    features = features.cpu().numpy()
    x_hat = x_hat.cpu().numpy()
    decoder = sae_model.W_dec.detach().cpu().numpy()

    probe = LogisticRegression(max_iter=1000).fit(x_hat, labels)

    candidate_flip_rates = {}
    for idx in candidate_idx:
        ablated = ablate_feature(x_hat, features, idx, decoder[idx])
        candidate_flip_rates[idx] = flip_rate(probe, x_hat, ablated)

    rng = np.random.RandomState(config.SEED)
    n_features = features.shape[1]
    non_candidates = [i for i in range(n_features) if i not in set(candidate_idx)]
    random_idx = rng.choice(non_candidates, size=min(config.N_RANDOM_CONTROLS, len(non_candidates)), replace=False)

    random_flip_rates = []
    for idx in random_idx:
        ablated = ablate_feature(x_hat, features, idx, decoder[idx])
        random_flip_rates.append(flip_rate(probe, x_hat, ablated))

    threshold = max(max(random_flip_rates), config.CAUSAL_THRESHOLD_FLOOR)
    causal_idx = [idx for idx, fr in candidate_flip_rates.items() if fr >= threshold]

    return causal_idx, threshold


def run():
    print("Stage 3: feature discovery")
    saes, activations, labels = stage2.run()
    device = next(iter(saes.values())).W_dec.device

    results = {}
    for layer_idx, sae_model in saes.items():
        path = os.path.join(config.PATHS["features"], f"layer_{layer_idx}.pkl")

        def compute(layer_idx=layer_idx, sae_model=sae_model):
            acts = activations[layer_idx]
            scores = correlational_scores(sae_model, acts, labels, device)
            candidate_idx = np.argsort(scores)[::-1][: config.N_CANDIDATES_PER_LAYER]
            causal_idx, threshold = causal_verification(sae_model, acts, labels, candidate_idx, device)
            decoder = sae_model.W_dec.detach().cpu().numpy()
            return {
                "causal_idx": causal_idx,
                "causal_directions": decoder[causal_idx] if len(causal_idx) > 0 else np.zeros((0, decoder.shape[1])),
                "threshold": threshold,
                "n_candidates": len(candidate_idx),
            }

        result = utils.cache_or_compute(path, compute, saver=utils.save_pickle, loader=utils.load_pickle)
        results[layer_idx] = result
        print(f"layer {layer_idx}: {len(result['causal_idx'])} causal features "
              f"(threshold={result['threshold']:.3f})")

    return results, saes


if __name__ == "__main__":
    run()
