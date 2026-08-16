"""
Runs the thesis's main comparison end to end: fresh baseline, then
causal / magnitude / random pruning, all evaluated with no fine-tuning,
followed by the paired bootstrap comparisons between them.

Every stage checks the cache first, so re-running this after an
interruption (or after changing config.py) only recomputes what is
missing or has changed.

Fine-tuning (stage 7) is NOT run by default, since it is a supplementary
check, not the main comparison - run it separately if you want it:
    python stage7_finetune.py --condition causal --n_examples 4000

Usage:
    python run_pipeline.py
"""

import stage5_evaluate as stage5
import stage6_bootstrap_compare as stage6


def main():
    print("=" * 70)
    print("Running main comparison: fresh / causal / magnitude / random")
    print("=" * 70)

    for condition in ["fresh", "causal", "magnitude", "random"]:
        print()
        stage5.run(condition)

    print()
    print("=" * 70)
    print("Paired bootstrap comparisons")
    print("=" * 70)
    print()
    stage6.run("causal", "magnitude")
    print()
    stage6.run("causal", "random")


if __name__ == "__main__":
    main()
