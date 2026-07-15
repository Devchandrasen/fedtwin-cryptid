"""Metrics for detection, client fairness, and scenario robustness."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _validated_binary_inputs(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y, dtype=int)
    scores = np.asarray(score, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("y and score must be non-empty one-dimensional arrays of equal length")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("y must contain binary labels")
    if not np.isfinite(scores).all():
        raise ValueError("score contains non-finite values")
    return labels, scores


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y, score = _validated_binary_inputs(y, score)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    y, score = _validated_binary_inputs(y, score)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def fpr_at_recall(y: np.ndarray, score: np.ndarray, recall_target: float = 0.95) -> float:
    y, score = _validated_binary_inputs(y, score)
    if not 0.0 < recall_target <= 1.0:
        raise ValueError("recall_target must lie in (0, 1]")
    if len(np.unique(y)) < 2:
        return float("nan")
    precision, recall, thresholds = precision_recall_curve(y, score)
    candidates = np.where(recall[:-1] >= recall_target)[0]
    if len(candidates) == 0:
        return 1.0
    threshold = thresholds[candidates[-1]]
    pred = score >= threshold
    fp = np.sum((pred == 1) & (y == 0))
    tn = np.sum((pred == 0) & (y == 0))
    return float(fp / max(fp + tn, 1))


def brier_score(y: np.ndarray, score: np.ndarray) -> float:
    y, score = _validated_binary_inputs(y, score)
    return float(np.mean((np.clip(score, 0.0, 1.0) - y) ** 2))


def expected_calibration_error(y: np.ndarray, score: np.ndarray, *, bins: int = 15) -> float:
    y, score = _validated_binary_inputs(y, score)
    if bins < 2:
        raise ValueError("bins must be at least two")
    clipped = np.clip(score, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (clipped >= lower) & (clipped < upper if index < bins - 1 else clipped <= upper)
        if not np.any(mask):
            continue
        error += float(mask.mean()) * abs(float(clipped[mask].mean()) - float(y[mask].mean()))
    return float(error)


def expected_precision_at_prevalence(*, prevalence: float, recall: float, fpr: float) -> float:
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must lie in (0, 1)")
    if not 0.0 <= recall <= 1.0 or not 0.0 <= fpr <= 1.0:
        raise ValueError("recall and fpr must lie in [0, 1]")
    numerator = prevalence * recall
    return float(numerator / max(numerator + (1.0 - prevalence) * fpr, 1e-15))


def review_workload_at_budget(y: np.ndarray, score: np.ndarray, *, budget: int) -> dict[str, float | int]:
    y, score = _validated_binary_inputs(y, score)
    if budget <= 0:
        raise ValueError("budget must be positive")
    take = min(int(budget), len(y))
    selected = np.argsort(-score, kind="stable")[:take]
    true_positives = int(y[selected].sum())
    total_positives = int(y.sum())
    return {
        "review_budget": int(take),
        "true_positives_found": true_positives,
        "recall_at_budget": float(true_positives / max(total_positives, 1)),
        "precision_at_budget": float(true_positives / max(take, 1)),
        "reviews_per_true_positive": float(take / max(true_positives, 1)),
    }


def detection_metrics(y: np.ndarray, score: np.ndarray, *, method: str, modality: str, privacy_mode: str = "none") -> dict:
    y, score = _validated_binary_inputs(y, score)
    pred = score >= 0.5
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return {
        "method": method,
        "modality": modality,
        "privacy_mode": privacy_mode,
        "roc_auc": safe_auc(y, score),
        "pr_auc": safe_ap(y, score),
        "fpr_at_95_recall": fpr_at_recall(y, score),
        "brier": brier_score(y, score),
        "ece_15": expected_calibration_error(y, score, bins=15),
        "accuracy": float(np.mean(pred == y)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": int(len(y)),
    }


def scenario_metrics(y: np.ndarray, score: np.ndarray, scenarios: np.ndarray, *, method: str) -> pd.DataFrame:
    rows = []
    for scenario in sorted(set(map(str, scenarios))):
        idx = scenarios == scenario
        row = detection_metrics(y[idx], score[idx], method=method, modality="multimodal")
        row["scenario"] = scenario
        rows.append(row)
    return pd.DataFrame(rows)


def client_metrics(y: np.ndarray, score: np.ndarray, clients: np.ndarray, *, method: str) -> pd.DataFrame:
    rows = []
    for client in sorted(set(map(int, clients))):
        idx = clients == client
        row = detection_metrics(y[idx], score[idx], method=method, modality="multimodal")
        row["client_id"] = client
        rows.append(row)
    return pd.DataFrame(rows)
