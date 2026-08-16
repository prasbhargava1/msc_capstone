"""
Central config for the SAE-guided pruning pipeline.

To run this pipeline on a different model, change MODEL_NAME below.
This code assumes a Llama-family architecture (Mistral, Llama, TinyLlama,
etc.) since it looks for layer.mlp.gate_proj / up_proj / down_proj and
layer.self_attn.q_proj / k_proj / v_proj / o_proj. A different
architecture (e.g. GPT-2) would need those attribute names changed in
utils.py's get_model_dims() and in pruning.py.

To run on a different dataset, change DATASET below. It must be a
Hugging Face dataset with a text field and a binary label field.
"""

import os

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DTYPE = "float16"          # use "float32" if running on CPU
DEVICE = "cuda"             # falls back to cpu automatically in utils.py

# ---------------------------------------------------------------------
# Dataset (primary dataset used for SAE training, causal discovery, and
# the main pruning evaluation)
# ---------------------------------------------------------------------
DATASET = {
    "hf_name": "stanfordnlp/imdb",
    "text_field": "text",
    "label_field": "label",
    "positive_value": 1,
}

# datasets used only for the optional generalisation check (not required
# to run the core pipeline)
GENERALIZATION_DATASETS = {
    "rotten_tomatoes": {
        "hf_name": "cornell-movie-review-data/rotten_tomatoes",
        "text_field": "text",
        "label_field": "label",
        "positive_value": 1,
    },
    "yelp_polarity": {
        "hf_name": "fancyzhx/yelp_polarity",
        "text_field": "text",
        "label_field": "label",
        "positive_value": 1,
    },
}

SEED = 42

# ---------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------
N_ACTIVATION_EXAMPLES = 2000     # how many training examples to capture activations for
MAX_PROMPT_LEN = 400              # characters of review text kept in the prompt

# ---------------------------------------------------------------------
# SAE training
# ---------------------------------------------------------------------
SAE_EXPANSION_FACTOR = 4          # feature dict size = hidden_size * this
SAE_L1_COEFF = 8e-4
SAE_LR = 1e-3
SAE_EPOCHS = 50
SAE_BATCH_SIZE = 64
SAE_EARLY_STOP_PATIENCE = 5

# ---------------------------------------------------------------------
# Feature discovery
# ---------------------------------------------------------------------
N_CANDIDATES_PER_LAYER = 200      # top-N correlational candidates per layer
N_RANDOM_CONTROLS = 200           # random features used to set the causal threshold
CAUSAL_THRESHOLD_FLOOR = 0.02     # minimum flip-rate threshold, guards against a
                                   # degenerate (too-low) random control draw

# ---------------------------------------------------------------------
# Pruning tiers: (min_causal_features, prune_percentile)
# checked in order, first match wins. Binary pruning: below the
# percentile -> zeroed, at or above -> kept. No partial scaling.
# ---------------------------------------------------------------------
PRUNE_TIERS = [
    (15, 50),   # aggressive
    (5, 35),    # moderate
    (1, 20),    # conservative
]                # n = 0 causal features -> layer is skipped entirely

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
N_EVAL_EXAMPLES = 1000
N_VAL_EXAMPLES = 60               # used for fine-tuning checkpoint selection only
N_BOOTSTRAP_RESAMPLES = 2000

# ---------------------------------------------------------------------
# Fine-tuning (optional supplementary check, not the thesis's main focus)
# ---------------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
FT_LR = 1e-4
FT_BATCH_SIZE = 8
FT_EXAMPLES_SMALL = 400
FT_EXAMPLES_LARGE = 4000
FT_VAL_CHECK_EVERY = 50

# ---------------------------------------------------------------------
# Paths - everything gets cached under this root. If a file already
# exists here, the pipeline loads it instead of recomputing it.
# ---------------------------------------------------------------------
ROOT = os.environ.get("SAE_PRUNING_ROOT", "./sae_pruning_cache")

PATHS = {
    "activations":   os.path.join(ROOT, "activations"),
    "sae_models":    os.path.join(ROOT, "sae_models"),
    "features":      os.path.join(ROOT, "causal_features"),
    "importance":    os.path.join(ROOT, "importance_scores"),
    "masks":         os.path.join(ROOT, "masks"),
    "results":       os.path.join(ROOT, "results"),
    "adapters":      os.path.join(ROOT, "lora_adapters"),
}

for p in PATHS.values():
    os.makedirs(p, exist_ok=True)
