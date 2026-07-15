"""Small scoring-head models used by the executable benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import sigmoid


@dataclass
class LogisticHead:
    weights: np.ndarray

    @classmethod
    def zeros(cls, n_features: int) -> LogisticHead:
        return cls(weights=np.zeros(n_features + 1, dtype=float))

    def copy(self) -> LogisticHead:
        return LogisticHead(self.weights.copy())

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        xb = add_bias(x)
        return sigmoid(xb @ self.weights)


def add_bias(x: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((x.shape[0], 1), dtype=float), x])


def train_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    initial: np.ndarray | None = None,
    lr: float = 0.08,
    epochs: int = 80,
    l2: float = 1e-4,
    prox_center: np.ndarray | None = None,
    prox_mu: float = 0.0,
) -> LogisticHead:
    xb = add_bias(x)
    if initial is None:
        w = np.zeros(xb.shape[1], dtype=float)
    else:
        w = initial.astype(float).copy()
    y_f = y.astype(float)
    n = max(1, len(y_f))
    for _ in range(epochs):
        pred = sigmoid(xb @ w)
        grad = xb.T @ (pred - y_f) / n
        reg = l2 * w
        reg[0] = 0.0
        grad += reg
        if prox_center is not None and prox_mu > 0:
            prox = prox_mu * (w - prox_center)
            prox[0] = 0.0
            grad += prox
        w -= lr * grad
    return LogisticHead(w)


def model_digest(weights: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(weights, dtype=np.float64).tobytes()).hexdigest()
