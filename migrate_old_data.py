"""
One-time migration: converts data already sitting in your old Drive
folder (/content/drive/MyDrive/sae_pruning) into the format and file
layout this new pipeline expects, so stage2 and stage3 get real cache
hits instead of silently retraining/rediscovering everything.

What gets migrated (expensive to recompute, format-compatible):
  - All 32 SAE checkpoints
  - All 32 layers' causal feature sets (indices + threshold)

What does NOT get migrated, on purpose:
  - Pruning masks / pruned models - these need rebuilding under this
    pipeline's binary pruning anyway, so there is nothing to reuse here.
  - Correlational top-200 candidates - cheap to recompute, not worth
    the extra migration complexity.

Run this once, before stage2/stage3, with SAE_PRUNING_ROOT already set.
"""

import json
import os

import numpy as np
import torch

import config
from sae import SparseAutoencoder

OLD_ROOT = "/content/drive/MyDrive/sae_pruning"
OLD_SAE_DIR = os.path.join(OLD_ROOT, "sae_models")
OLD_CAUSAL_DIR = os.path.join(OLD_ROOT, "causal_features")

N_LAYERS = 32


def migrate_sae(layer_idx, device="cpu"):
    """Old checkpoint: {"input_dim", "dict_size", "state_dict"}, filename
    sae_layer_{N}.pt. New pipeline expects a raw state_dict at layer_{N}.pt.
    Parameter names (W_enc/b_enc/W_dec/b_dec) already match, confirmed
    against the actual training script, so this is a pure format
    conversion, not a parameter remapping."""
    old_path = os.path.join(OLD_SAE_DIR, f"sae_layer_{layer_idx}.pt")
    new_path = os.path.join(config.PATHS["sae_models"], f"layer_{layer_idx}.pt")

    if os.path.exists(new_path):
        print(f"  layer {layer_idx}: SAE already migrated, skipping")
        return True
    if not os.path.exists(old_path):
        print(f"  layer {layer_idx}: no old SAE checkpoint found at {old_path}")
        return False

    ckpt = torch.load(old_path, map_location=device)
    sae = SparseAutoencoder(ckpt["input_dim"], expansion_factor=1)
    # rebuild with the correct dict_size directly, expansion_factor above is a
    # placeholder since dict_size is set explicitly on the next line
    sae.W_enc = torch.nn.Parameter(torch.empty(ckpt["input_dim"], ckpt["dict_size"]))
    sae.W_dec = torch.nn.Parameter(torch.empty(ckpt["dict_size"], ckpt["input_dim"]))
    sae.b_enc = torch.nn.Parameter(torch.zeros(ckpt["dict_size"]))
    sae.load_state_dict(ckpt["state_dict"])

    torch.save(sae.state_dict(), new_path)
    print(f"  layer {layer_idx}: SAE migrated ({ckpt['input_dim']} -> {ckpt['dict_size']})")
    return True


def migrate_causal_features(layer_idx, device="cpu"):
    """Old format: plain index array at layer_{N}_causal_refined_v2.npy,
    threshold in a separate layer_{N}_refine_meta_v2.json. New pipeline
    expects one pickle bundling {causal_idx, causal_directions, threshold,
    n_candidates}. Directions are reconstructed from the (already-migrated)
    SAE's W_dec, indexed by the old causal indices."""
    old_causal_path = os.path.join(OLD_CAUSAL_DIR, f"layer_{layer_idx}_causal_refined_v2.npy")
    old_meta_path = os.path.join(OLD_CAUSAL_DIR, f"layer_{layer_idx}_refine_meta_v2.json")
    new_path = os.path.join(config.PATHS["features"], f"layer_{layer_idx}.pkl")

    if os.path.exists(new_path):
        print(f"  layer {layer_idx}: features already migrated, skipping")
        return
    if not os.path.exists(old_causal_path):
        print(f"  layer {layer_idx}: no old causal features found, skipping "
              f"(this layer will be treated as having 0 causal features)")
        return

    causal_idx = np.load(old_causal_path)
    threshold = 0.02
    if os.path.exists(old_meta_path):
        with open(old_meta_path) as f:
            meta = json.load(f)
        threshold = meta.get("adaptive_threshold", 0.02)

    new_sae_path = os.path.join(config.PATHS["sae_models"], f"layer_{layer_idx}.pt")
    if not os.path.exists(new_sae_path):
        print(f"  layer {layer_idx}: SAE not migrated yet, run migrate_sae first - skipping")
        return

    ckpt = torch.load(os.path.join(OLD_SAE_DIR, f"sae_layer_{layer_idx}.pt"), map_location=device)
    decoder = ckpt["state_dict"]["W_dec"].numpy()
    causal_directions = decoder[causal_idx] if len(causal_idx) > 0 else np.zeros((0, ckpt["input_dim"]))

    import utils
    utils.save_pickle(
        {
            "causal_idx": list(causal_idx),
            "causal_directions": causal_directions.astype(np.float32),
            "threshold": threshold,
            "n_candidates": 200,  # matches this project's fixed N_CANDIDATES_PER_LAYER throughout
        },
        new_path,
    )
    print(f"  layer {layer_idx}: {len(causal_idx)} causal features migrated")


def run():
    if not os.path.exists(OLD_ROOT):
        print(f"Old data folder not found at {OLD_ROOT} - nothing to migrate.")
        return

    print("Migrating SAE checkpoints...")
    for layer_idx in range(N_LAYERS):
        migrate_sae(layer_idx)

    print("\nMigrating causal features...")
    for layer_idx in range(N_LAYERS):
        migrate_causal_features(layer_idx)

    print("\nDone. stage2 and stage3 should now show cache hits for every migrated layer.")


if __name__ == "__main__":
    run()
