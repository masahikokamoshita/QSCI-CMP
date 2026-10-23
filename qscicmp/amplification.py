"""Exact simulation of amplitude amplification on the ground-state distribution.

After k applications of the amplification operator with marked set A, the
amplitudes of the marked and unmarked determinants are rescaled uniformly
(Eq. (4) of the paper), so sampling from the amplified state reduces to
choosing the marked or unmarked sector with the amplified weights and then
rejection sampling from the original distribution within that sector.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def choose_grover_iterations(unmarked_probability: float, policy: str = "maximize_unknown") -> int:
    """Iteration count k for a given weight of the unmarked (target) states.

    ``maximize_unknown`` picks the k near pi/(4 theta) that maximizes the
    success probability p(k) = sin^2((2k+1) theta).  ``min_query_scaling``
    stops earlier at (2k+1) theta ~ 1.17, which minimizes the expected
    number of queries per new sample.
    """
    if unmarked_probability <= 0.0 or unmarked_probability >= 1.0:
        return 0
    theta = np.arcsin(np.sqrt(unmarked_probability))
    if policy == "min_query_scaling":
        return max(0, int(np.rint(1.1656 / (2.0 * theta) - 0.5)))
    estimate = (np.pi / (4.0 * theta)) - 0.5
    candidates = {0, max(0, int(np.floor(estimate))), max(0, int(np.ceil(estimate)))}
    best_k, best_p = 0, unmarked_probability
    for k in sorted(candidates):
        p = float(np.sin((2 * k + 1) * theta) ** 2)
        if p > best_p + 1e-15 or (abs(p - best_p) <= 1e-15 and k < best_k):
            best_k, best_p = k, p
    return best_k


def amplitude_scales(marked_probability: float, unmarked_probability: float, iterations: int) -> tuple[float, float]:
    """Factors by which the marked and unmarked amplitudes are rescaled after ``iterations`` rounds."""
    if iterations <= 0:
        return 1.0, 1.0
    if np.isclose(unmarked_probability, 0.0):
        return (-1.0) ** iterations, 0.0
    if np.isclose(marked_probability, 0.0):
        return 0.0, (-1.0) ** iterations
    theta = np.arcsin(np.sqrt(unmarked_probability))
    return (
        float(np.cos((2 * iterations + 1) * theta) / np.cos(theta)),
        float(np.sin((2 * iterations + 1) * theta) / np.sin(theta)),
    )


def sample_amplified(
    rng: np.random.Generator,
    shots: int,
    *,
    cumulative_probabilities: np.ndarray,
    indices: np.ndarray,
    marked: np.ndarray,
    marked_probability: float,
    unmarked_probability: float,
    marked_scale: float,
    unmarked_scale: float,
) -> Counter[int]:
    """Draw ``shots`` samples from the amplified state.

    ``cumulative_probabilities`` and ``indices`` describe the ground-state
    distribution over its nonzero determinants and ``marked`` is the array
    of marked determinant indices.
    """
    if shots <= 0:
        return Counter()
    marked_weight = (marked_scale ** 2) * marked_probability
    unmarked_weight = (unmarked_scale ** 2) * unmarked_probability
    total_weight = marked_weight + unmarked_weight
    if total_weight <= 0.0:
        return Counter()
    sample_marked = rng.random(shots) < (marked_weight / total_weight)
    counts: Counter[int] = Counter()
    for want_marked, multiplicity in Counter(bool(flag) for flag in sample_marked).items():
        remaining = int(multiplicity)
        accept_probability = marked_probability if want_marked else unmarked_probability
        if accept_probability <= 0.0:
            continue
        while remaining > 0:
            draw_size = max(remaining + 8, int(np.ceil(remaining / accept_probability * 1.2)))
            candidates = indices[np.searchsorted(cumulative_probabilities, rng.random(draw_size), side="right")]
            membership = np.isin(candidates, marked, assume_unique=False)
            accepted = candidates[membership if want_marked else ~membership]
            if accepted.size == 0:
                continue
            selected = accepted[:remaining]
            values, multiplicities = np.unique(selected, return_counts=True)
            counts.update({int(i): int(c) for i, c in zip(values, multiplicities)})
            remaining -= int(selected.size)
    return counts
