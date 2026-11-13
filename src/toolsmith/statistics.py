"""Evaluation statistics for paired tool-calling experiments.

Bootstrap uses paired resampling to preserve per-example comparison structure.
This does not train a model; use predictions from a held-out evaluation set.
"""
from __future__ import annotations
from random import Random
from statistics import mean


def paired_accuracy_delta(baseline_correct: list[bool], tuned_correct: list[bool], *,
                          resamples: int = 2000, seed: int = 42) -> dict[str, float | int]:
    """Return percentage-point difference as a fraction and percentile bootstrap CI."""
    if not baseline_correct or len(baseline_correct) != len(tuned_correct):
        raise ValueError("Paired nonempty predictions are required")
    if resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    paired = [int(t) - int(b) for b, t in zip(baseline_correct, tuned_correct)]
    rng = Random(seed)
    n = len(paired)
    estimates = sorted(mean(paired[rng.randrange(n)] for _ in range(n)) for _ in range(resamples))
    lower = estimates[int(.025 * (resamples - 1))]
    upper = estimates[int(.975 * (resamples - 1))]
    return {"n": n, "baseline_accuracy": mean(baseline_correct),
            "tuned_accuracy": mean(tuned_correct), "delta": mean(paired),
            "ci_lower": lower, "ci_upper": upper}
