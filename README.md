# SAE-Guided Causal Pruning Pipeline

Small, readable pipeline for the thesis's core comparison: does pruning
guided by causally-verified sparse autoencoder features beat magnitude-
based and random pruning, at matched sparsity, with no retraining.

## Structure

```
config.py                       all settings live here - model name, dataset,
                                 hyperparameters, cache paths
utils.py                        shared functions: model loading, data loading,
                                 prompting, metrics, bootstrap, caching
sae.py                          the SAE model class and its training loop
pruning.py                      importance scoring (causal / magnitude / random)
                                 and binary mask building/application

stage1_capture_activations.py   capture residual-stream activations
stage2_train_saes.py            train one SAE per layer
stage3_feature_discovery.py     correlational candidates -> causal verification
stage4_build_masks.py           turn causal features (or magnitude, or random)
                                 into a binary pruning mask
stage5_evaluate.py              apply a mask, run the zero-shot eval, save results
stage6_bootstrap_compare.py     paired bootstrap between two saved conditions
stage7_finetune.py              OPTIONAL supplementary fine-tuning check
                                 (not part of the main comparison, see Methodology)

run_pipeline.py                 runs the main comparison end to end
colab_runner.ipynb              same pipeline, set up for Google Colab with
                                 Drive-backed caching (see "Running on Colab" below)
```

## How the caching works

Every stage checks its cache folder first. If the file it needs already
exists (an SAE checkpoint, a set of causal features, a pruning mask, a
set of log-odds), it loads that instead of recomputing it. This means:

- Re-running `run_pipeline.py` after an interruption picks up where it
  left off instead of starting over.
- If you only change something in stage 4 onward (e.g. try a different
  pruning tier schedule), stages 1-3 are not repeated.
- To force a full recompute, delete the relevant folder under
  `./sae_pruning_cache/` (or point `SAE_PRUNING_ROOT` somewhere new).

## Running it

```bash
pip install torch transformers datasets peft scikit-learn numpy

python run_pipeline.py
```

This runs the fresh baseline, then causal / magnitude / random pruning,
all with no fine-tuning, then the paired bootstrap comparisons between
them. Needs a GPU with enough memory for whichever model you set in
`config.py` (the thesis uses Mistral-7B).

To try the optional fine-tuning check on top of an already-pruned model:

```bash
python stage7_finetune.py --condition causal --n_examples 4000
```

## Running on Colab

Open `colab_runner.ipynb` in Colab and run the cells in order. The only
real difference from running locally is that the cache folder gets
pointed at Google Drive instead of Colab's own local disk - Colab wipes
local disk on every disconnect, so without this every stage would
recompute from scratch each session instead of reusing what was already
built. This is done with one line, before `config.py` is imported:

```python
os.environ["SAE_PRUNING_ROOT"] = "/content/drive/MyDrive/sae_pruning_cache"
```

Everything else in the notebook just imports and calls the same stage
functions used by `run_pipeline.py` - it is not a separate copy of the
pipeline logic, only a different way of running it. The notebook clones
your GitHub repo by default (edit the URL in the cell); if you have not
pushed to GitHub yet, it also has a commented-out fallback cell for
uploading `sae_pruning_pipeline.zip` directly instead.

## Using a different model or dataset

Change `MODEL_NAME` in `config.py` to any Llama-family model on Hugging
Face (Mistral, Llama, TinyLlama, etc. all work, since they share the
same layer attribute names this code looks for: `gate_proj`, `up_proj`,
`down_proj`, `q_proj`, `k_proj`, `v_proj`, `o_proj`). A different
architecture family (e.g. GPT-2) would need those attribute names
updated in `pruning.py`.

Change `DATASET` in `config.py` to any Hugging Face dataset with a text
field and a binary label field.

## What has actually been tested, and what has not

Being upfront about this rather than implying everything has been run
end to end:

**Tested directly** (unit tests with synthetic data, no GPU or model
download needed - see the test commands used during development, not
included here to keep the repo small): the SAE's forward pass, its loss
function, and its cache-then-reuse behaviour; every importance-scoring
function in `pruning.py`; the binary mask logic; mask application to
model weights, including the grouped-query-attention key/value sharing
rule; the correlational scoring and ablation-based causal verification
logic in stage 3; the mask-building logic in stage 4, including the
"skip layers with zero causal features" rule; `cache_or_compute`;
`compute_metrics`; `paired_bootstrap`. All of this passed against hand-
checkable synthetic examples.

**Not run end to end**: actually downloading and running Mistral-7B
through the full pipeline. That needs a GPU and a Hugging Face model
download, neither of which is available in the environment this code
was written in. The pipeline logic has been checked as thoroughly as
possible without that - but the first real run on your own GPU (Colab
or otherwise) is the actual, final test. Watch stage 1 and stage 2
particularly closely the first time, since those involve the model
itself rather than pure Python/NumPy logic.

## Notes on specific design choices

- **Binary pruning, not partial scaling.** A neuron scaled to 0.4 still
  occupies full memory - only an exact zero can be physically removed to
  shrink the model. Earlier experiments in this project also found
  binary pruning outperforms partial scaling at matched sparsity, so
  this is not just a simplification.
- **Layers with zero causally-verified features are left untouched**,
  under every condition (causal, magnitude, random) - this keeps the
  three conditions comparable on which layers get pruned, differing
  only in which specific neurons/heads within a pruned layer survive.
- **Fine-tuning is deliberately kept separate and optional** (stage 7),
  reflecting that the thesis's primary contribution is the pruning-
  criterion comparison, not the recovery question.
