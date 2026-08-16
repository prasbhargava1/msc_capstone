"""
Stage 6: compare two conditions that have already been evaluated in
stage 5, using a paired bootstrap on their saved log-odds.

Usage:
    python stage6_bootstrap_compare.py --a causal --b magnitude
"""

import argparse
import os

import numpy as np

import config
import utils


def run(condition_a, condition_b):
    print(f"Stage 6: paired bootstrap, {condition_a} vs {condition_b}")

    logodds_a = np.load(os.path.join(config.PATHS["results"], f"{condition_a}_logodds.npy"))
    logodds_b = np.load(os.path.join(config.PATHS["results"], f"{condition_b}_logodds.npy"))

    # both conditions must have been evaluated on the identical eval set for
    # a paired comparison to be valid - rebuild it the same way stage 5 did
    _, eval_labels, _, _ = utils.load_disjoint_splits(
        config.DATASET, config.N_EVAL_EXAMPLES, config.N_VAL_EXAMPLES, seed=config.SEED
    )

    assert len(logodds_a) == len(eval_labels), "condition A's log-odds do not match the eval set size"
    assert len(logodds_b) == len(eval_labels), "condition B's log-odds do not match the eval set size"

    result = utils.paired_bootstrap(eval_labels, logodds_a, logodds_b)

    print(f"  mean AUC diff ({condition_a} - {condition_b}) = {result['mean_diff']:+.4f}")
    print(f"  95% CI = [{result['ci95'][0]:+.4f}, {result['ci95'][1]:+.4f}]")
    print(f"  P({condition_a} > {condition_b}) = {result['p_a_wins']:.3f}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    args = parser.parse_args()
    run(args.a, args.b)
