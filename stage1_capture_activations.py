"""
Stage 1: run a batch of training examples through the model and save the
residual-stream activation (the layer output, mean-pooled over tokens)
at every layer. This is the data the SAEs train on in stage 2.

Usage:
    python stage1_capture_activations.py
"""

import os

import numpy as np
import torch

import config
import utils


@torch.no_grad()
def capture_activations(model, tokenizer, texts, n_layers):
    device = next(model.parameters()).device
    activations = {layer_idx: [] for layer_idx in range(n_layers)}

    hooks = []
    captured = {}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = hidden.mean(dim=1).float().cpu().numpy()  # mean-pool over tokens
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    for i in range(0, len(texts), 8):
        batch = texts[i : i + 8]
        prompts = [utils.make_prompt(t) for t in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        model(**enc)
        for layer_idx in range(n_layers):
            activations[layer_idx].append(captured[layer_idx])

    for h in hooks:
        h.remove()

    return {layer_idx: np.concatenate(v, axis=0) for layer_idx, v in activations.items()}


def run():
    print("Stage 1: capture activations")
    model, tokenizer = utils.load_model_and_tokenizer()
    dims = utils.get_model_dims(model)
    print(f"  model dims: {dims}")

    texts, labels = utils.load_split(
        config.DATASET, "train", config.N_ACTIVATION_EXAMPLES, seed=config.SEED
    )

    def compute():
        return capture_activations(model, tokenizer, texts, dims["n_layers"])

    path = os.path.join(config.PATHS["activations"], "all_layers.pkl")
    activations = utils.cache_or_compute(
        path, compute, saver=utils.save_pickle, loader=utils.load_pickle
    )

    print(f"  captured activations for {dims['n_layers']} layers, "
          f"shape per layer: {activations[0].shape}")
    return activations, labels


if __name__ == "__main__":
    run()
