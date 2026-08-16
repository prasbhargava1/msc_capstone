"""
Stage 2: train one sparse autoencoder per layer on the activations
captured in stage 1. Each layer's SAE is cached separately, so if you
add more layers or rerun later, layers already trained are skipped.

Usage:
    python stage2_train_saes.py
"""

import os

import config
import sae
import stage1_capture_activations as stage1
import utils


def run():
    print("Stage 2: train SAEs")
    activations, labels = stage1.run()

    saes = {}
    for layer_idx, acts in activations.items():
        path = os.path.join(config.PATHS["sae_models"], f"layer_{layer_idx}.pt")
        print(f"layer {layer_idx}:")
        saes[layer_idx] = sae.train_sae(acts, path)

    print(f"  done - {len(saes)} SAEs trained/loaded")
    return saes, activations, labels


if __name__ == "__main__":
    run()
