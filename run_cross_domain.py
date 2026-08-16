"""
Runs the cross-domain evaluation end to end: fresh, causal, and
magnitude conditions evaluated on Rotten Tomatoes and Yelp Polarity,
followed by a paired bootstrap comparing causal against magnitude on
each dataset separately.

Neither dataset is used at any other stage of the pipeline, so this
tests whether the causal advantage measured on IMDB also holds on data
the pruning decision was not derived from.

Usage:
    python run_cross_domain.py
"""

import os

import numpy as np

import config
import stage8_cross_domain as stage8
import utils


def main():
    print("=" * 70)
    print("Cross-domain evaluation: Rotten Tomatoes and Yelp Polarity")
    print("=" * 70)

    all_results = {}
    for condition in ["fresh", "causal", "magnitude"]:
        print()
        all_results[condition] = stage8.run(condition)

    print()
    print("=" * 70)
    print("Summary (AUC)")
    print("=" * 70)
    header = f"{'Dataset':<20} {'Fresh':>10} {'Causal':>10} {'Magnitude':>10} {'Diff':>10}"
    print(header)
    for dataset_name in config.GENERALIZATION_DATASETS:
        fresh_auc = all_results["fresh"][dataset_name]["auc"]
        causal_auc = all_results["causal"][dataset_name]["auc"]
        mag_auc = all_results["magnitude"][dataset_name]["auc"]
        diff = causal_auc - mag_auc
        print(f"{dataset_name:<20} {fresh_auc:>10.3f} {causal_auc:>10.3f} "
              f"{mag_auc:>10.3f} {diff:>+10.3f}")

    print()
    print("=" * 70)
    print("Paired bootstrap: causal vs magnitude, per dataset")
    print("=" * 70)

    for dataset_name in config.GENERALIZATION_DATASETS:
        labels = np.load(os.path.join(config.PATHS["results"], f"{dataset_name}_labels.npy"))
        causal_lo = np.load(
            os.path.join(config.PATHS["results"], f"causal_{dataset_name}_logodds.npy")
        )
        mag_lo = np.load(
            os.path.join(config.PATHS["results"], f"magnitude_{dataset_name}_logodds.npy")
        )

        result = utils.paired_bootstrap(labels, causal_lo, mag_lo)
        print()
        print(f"{dataset_name}:")
        print(f"  mean AUC diff (causal - magnitude) = {result['mean_diff']:+.4f}")
        print(f"  95% CI = [{result['ci95'][0]:+.4f}, {result['ci95'][1]:+.4f}]")
        print(f"  P(causal > magnitude) = {result['p_a_wins']:.3f}")


if __name__ == "__main__":
    main()
