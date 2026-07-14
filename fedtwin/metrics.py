"""Metrics for detection, client fairness, and scenario robustness."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def fpr_at_recall(y: np.ndarray, score: np.ndarray, recall_target: float = 0.95) -> float:
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


def detection_metrics(y: np.ndarray, score: np.ndarray, *, method: str, modality: str, privacy_mode: str = "none") -> dict:
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
