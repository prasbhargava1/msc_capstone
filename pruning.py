"""
Turns causal features (or plain weight magnitude, or nothing at all) into
a binary pruning mask, and applies that mask to a model.

Three importance methods, one for each condition compared in the thesis:
  - causal:    cosine similarity / projection onto causally verified SAE features
  - magnitude: weight-norm only, no interpretability signal
  - random:    uniformly random, used as a no-information control

Pruning is binary: a neuron or head is either kept (scale 1) or removed
(scale 0). No partial scaling - see Methodology for why.
"""

import numpy as np
import torch
import torch.nn.functional as F

import config


def get_tier(n_causal):
    for min_causal, percentile in config.PRUNE_TIERS:
        if n_causal >= min_causal:
            return percentile
    return None  # 0 causal features -> layer is skipped


# -----------------------------------------------------------------------
# importance scores
# -----------------------------------------------------------------------
def causal_mlp_importance(layer, causal_feature_directions):
    """
    causal_feature_directions: [n_causal, hidden_size] array of decoder
    directions for this layer's causally verified features.
    Returns one importance score per MLP neuron (cosine similarity, summed
    over all causal features).
    """
    down_proj = layer.mlp.down_proj.weight.data.float()          # [hidden, intermediate]
    neuron_dirs = F.normalize(down_proj.T, dim=1)                 # [intermediate, hidden]
    feature_dirs = F.normalize(
        torch.tensor(causal_feature_directions, dtype=torch.float32), dim=1
    )
    cos_sim = neuron_dirs @ feature_dirs.T                        # [intermediate, n_causal]
    importance = cos_sim.abs().sum(dim=1)
    return (importance / (importance.max() + 1e-8)).cpu().numpy()


def causal_head_importance(layer, causal_feature_directions, n_heads, head_dim):
    """
    Returns one importance score per attention head: the average norm of
    each causal feature's direction after being projected through that
    head's output-projection slice.
    """
    o_proj = layer.self_attn.o_proj.weight.data.float()
    feature_dirs = F.normalize(
        torch.tensor(causal_feature_directions, dtype=torch.float32), dim=1
    )
    importance = torch.zeros(n_heads)
    for h in range(n_heads):
        w_o_h = o_proj[:, h * head_dim : (h + 1) * head_dim]
        proj = feature_dirs @ w_o_h
        importance[h] = proj.norm(dim=1).mean()
    return (importance / (importance.max() + 1e-8)).cpu().numpy()


def magnitude_mlp_importance(layer):
    gate = layer.mlp.gate_proj.weight.data.abs().norm(dim=1)
    up = layer.mlp.up_proj.weight.data.abs().norm(dim=1)
    down = layer.mlp.down_proj.weight.data.abs().norm(dim=0)
    return (gate + up + down).cpu().numpy()


def magnitude_head_importance(layer, n_heads, head_dim):
    q_proj = layer.self_attn.q_proj.weight.data.float()
    o_proj = layer.self_attn.o_proj.weight.data.float()
    importance = torch.zeros(n_heads)
    for h in range(n_heads):
        s, e = h * head_dim, (h + 1) * head_dim
        importance[h] = q_proj[s:e, :].norm() + o_proj[:, s:e].norm()
    return importance.cpu().numpy()


def random_importance(n_units, seed):
    rng = np.random.RandomState(seed)
    return rng.rand(n_units)


# -----------------------------------------------------------------------
# turning importance scores into a binary keep/prune decision
# -----------------------------------------------------------------------
def build_binary_mask(importance, prune_percentile):
    """
    Returns a same-length array of 1s (keep) and 0s (prune). Units at or
    below the given percentile of this layer's own importance
    distribution are pruned.
    """
    threshold = np.percentile(importance, prune_percentile)
    return (importance > threshold).astype(np.float32)


# -----------------------------------------------------------------------
# applying a mask to the actual model weights
# -----------------------------------------------------------------------
@torch.no_grad()
def apply_mlp_mask(layer, mask):
    """mask: [intermediate_size] array of 0/1, one entry per MLP neuron."""
    mask_t = torch.tensor(mask, dtype=layer.mlp.gate_proj.weight.dtype,
                           device=layer.mlp.gate_proj.weight.device)
    layer.mlp.gate_proj.weight.data *= mask_t.unsqueeze(1)
    layer.mlp.up_proj.weight.data *= mask_t.unsqueeze(1)
    layer.mlp.down_proj.weight.data *= mask_t.unsqueeze(0)


@torch.no_grad()
def apply_head_mask(layer, mask, n_heads, n_kv_heads, head_dim, heads_per_group):
    """mask: [n_heads] array of 0/1, one entry per query head."""
    device = layer.self_attn.q_proj.weight.device
    dtype = layer.self_attn.q_proj.weight.dtype

    for h in range(n_heads):
        s, e = h * head_dim, (h + 1) * head_dim
        scale = float(mask[h])
        layer.self_attn.q_proj.weight.data[s:e, :] *= scale
        layer.self_attn.o_proj.weight.data[:, s:e] *= scale

    # a key/value head is shared by a group of query heads (grouped-query
    # attention) - only zero it if every query head in its group is pruned
    for kv in range(n_kv_heads):
        group = range(kv * heads_per_group, (kv + 1) * heads_per_group)
        if all(mask[h] == 0 for h in group):
            s, e = kv * head_dim, (kv + 1) * head_dim
            layer.self_attn.k_proj.weight.data[s:e, :] = 0
            layer.self_attn.v_proj.weight.data[s:e, :] = 0
