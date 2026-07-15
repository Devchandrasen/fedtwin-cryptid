"""Paired uncertainty estimates and multiple-comparison correction."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


def paired_bootstrap_delta(
    y: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    metric: Metric,
    groups: np.ndarray | None = None,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 31,
) -> dict[str, float | int]:
    y = np.asarray(y)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if not (len(y) == len(left) == len(right)) or len(y) == 0:
        raise ValueError("paired arrays must have the same nonzero length")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    if groups is None:
        unique_groups = np.arange(len(y))
        row_indices = [np.asarray([idx]) for idx in range(len(y))]
    else:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError("groups must have the same length as y")
        unique_groups = np.unique(groups)
        row_indices = [np.flatnonzero(groups == group) for group in unique_groups]
    observed = float(metric(y, left) - metric(y, right))
    if not np.isfinite(observed):
        raise ValueError("metric produced a non-finite observed delta")
    valid_deltas: list[float] = []
    attempts = 0
    max_attempts = resamples * 10
    while len(valid_deltas) < resamples and attempts < max_attempts:
        attempts += 1
        sampled_groups = rng.integers(0, len(unique_groups), size=len(unique_groups))
        sampled_rows = np.concatenate([row_indices[int(group_idx)] for group_idx in sampled_groups])
        try:
            delta = float(
                metric(y[sampled_rows], left[sampled_rows])
                - metric(y[sampled_rows], right[sampled_rows])
            )
        except ValueError:
            continue
        if np.isfinite(delta):
            valid_deltas.append(delta)
    if len(valid_deltas) != resamples:
        raise ValueError(
            f"metric produced only {len(valid_deltas)} finite bootstrap resamples "
            f"after {attempts} attempts"
        )
    deltas = np.asarray(valid_deltas, dtype=float)
    alpha = (1.0 - confidence) / 2.0
    lower_tail = int(np.count_nonzero(deltas <= 0.0))
    upper_tail = int(np.count_nonzero(deltas >= 0.0))
    p_two_sided = min(1.0, 2.0 * (min(lower_tail, upper_tail) + 1) / (resamples + 1))
    return {
        "delta": float(observed),
        "ci_low": float(np.quantile(deltas, alpha)),
        "ci_high": float(np.quantile(deltas, 1.0 - alpha)),
        "p_two_sided": float(p_two_sided),
        "resamples": int(resamples),
        "discarded_resamples": int(attempts - resamples),
        "clusters": int(len(unique_groups)),
    }


def paired_permutation_delta(
    y: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    metric: Metric,
    permutations: int = 5000,
    seed: int = 31,
) -> dict[str, float | int]:
    y = np.asarray(y)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if not (len(y) == len(left) == len(right)) or len(y) == 0:
        raise ValueError("paired arrays must have the same nonzero length")
    if permutations < 100:
        raise ValueError("permutations must be at least 100")
    rng = np.random.default_rng(seed)
    observed = float(metric(y, left) - metric(y, right))
    if not np.isfinite(observed):
        raise ValueError("metric produced a non-finite observed delta")
    extreme = 0
    for _ in range(permutations):
        swap = rng.random(len(y)) < 0.5
        perm_left = np.where(swap, right, left)
        perm_right = np.where(swap, left, right)
        delta = float(metric(y, perm_left) - metric(y, perm_right))
        if not np.isfinite(delta):
            raise ValueError("metric produced a non-finite permutation delta")
        extreme += int(abs(delta) >= abs(observed))
    return {
        "delta": float(observed),
        "p_two_sided": float((extreme + 1) / (permutations + 1)),
        "permutations": int(permutations),
    }


def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if (
        values.ndim != 1
        or len(values) == 0
        or np.any(~np.isfinite(values))
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("p-values must be a finite one-dimensional array in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, original_idx in enumerate(order):
        candidate = min(1.0, (total - rank) * values[original_idx])
        running = max(running, candidate)
        adjusted[original_idx] = running
    return adjusted
