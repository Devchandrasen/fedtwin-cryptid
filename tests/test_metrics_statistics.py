from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fedtwin.metrics import (
    brier_score,
    detection_metrics,
    expected_calibration_error,
    expected_precision_at_prevalence,
    review_workload_at_budget,
)
from fedtwin.statistics import holm_adjust, paired_bootstrap_delta, paired_permutation_delta


def test_detection_calibration_and_workload_metrics() -> None:
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.3, 0.7, 0.9])
    metrics = detection_metrics(y, score, method="test", modality="visual")
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["brier"] == pytest.approx(0.05)
    assert brier_score(y, score) == pytest.approx(0.05)
    assert 0 <= expected_calibration_error(y, score, bins=4) <= 1
    workload = review_workload_at_budget(y, score, budget=2)
    assert workload["recall_at_budget"] == 1.0
    assert workload["reviews_per_true_positive"] == 1.0
    assert expected_precision_at_prevalence(prevalence=0.01, recall=0.95, fpr=0.05) < 0.2


def test_metric_input_validation() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        detection_metrics(np.array([0, 1]), np.array([0.2, np.nan]), method="x", modality="x")
    with pytest.raises(ValueError, match="binary"):
        detection_metrics(np.array([0, 2]), np.array([0.2, 0.8]), method="x", modality="x")
    with pytest.raises(ValueError, match="prevalence"):
        expected_precision_at_prevalence(prevalence=0, recall=0.9, fpr=0.1)
    with pytest.raises(ValueError, match="budget"):
        review_workload_at_budget(np.array([0, 1]), np.array([0.2, 0.8]), budget=0)


def test_clustered_paired_statistics_and_holm_correction() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    left = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.85, 0.2, 0.9])
    right = np.array([0.4, 0.6, 0.45, 0.55, 0.3, 0.6, 0.4, 0.65])
    groups = np.repeat(np.arange(4), 2)
    bootstrap = paired_bootstrap_delta(
        y,
        left,
        right,
        metric=lambda labels, values: float(average_precision_score(labels, values)),
        groups=groups,
        resamples=200,
        seed=7,
    )
    permutation = paired_permutation_delta(
        y,
        left,
        right,
        metric=lambda labels, values: float(average_precision_score(labels, values)),
        permutations=200,
        seed=7,
    )
    assert bootstrap["clusters"] == 4
    assert bootstrap["delta"] >= 0
    assert 0 < bootstrap["p_two_sided"] <= 1
    assert bootstrap["discarded_resamples"] >= 0
    assert 0 <= permutation["p_two_sided"] <= 1
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.all(adjusted >= np.array([0.01, 0.04, 0.03]))
    with pytest.raises(ValueError):
        holm_adjust([np.nan])
    with pytest.raises(ValueError):
        holm_adjust([])
    with pytest.raises(ValueError, match="confidence"):
        paired_bootstrap_delta(y, left, right, metric=lambda labels, values: 0.0, confidence=1.0)
