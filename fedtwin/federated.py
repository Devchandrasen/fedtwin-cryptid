"""Federated training loops."""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np

from .crypto import aggregate_updates
from .models import LogisticHead, train_logistic


def train_centralized(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 120,
    lr: float = 0.08,
) -> tuple[LogisticHead, dict]:
    start = time.perf_counter()
    model = train_logistic(x, y, epochs=epochs, lr=lr)
    return model, {"method": "centralized", "rounds": 1, "runtime_sec": time.perf_counter() - start}


def train_local_models(
    x: np.ndarray,
    y: np.ndarray,
    clients: np.ndarray,
    *,
    epochs: int = 100,
    lr: float = 0.08,
) -> tuple[dict[int, LogisticHead], dict]:
    start = time.perf_counter()
    out: dict[int, LogisticHead] = {}
    for client_id in sorted(set(map(int, clients))):
        idx = clients == client_id
        out[client_id] = train_logistic(x[idx], y[idx], epochs=epochs, lr=lr)
    return out, {"method": "local", "rounds": 1, "runtime_sec": time.perf_counter() - start}


def train_personalized_models(
    x: np.ndarray,
    y: np.ndarray,
    clients: np.ndarray,
    base_weights: np.ndarray,
    *,
    epochs: int = 20,
    lr: float = 0.04,
    prox_mu: float = 0.01,
) -> tuple[dict[int, LogisticHead], dict]:
    start = time.perf_counter()
    out: dict[int, LogisticHead] = {}
    for client_id in sorted(set(map(int, clients))):
        idx = clients == client_id
        out[client_id] = train_logistic(
            x[idx],
            y[idx],
            initial=base_weights,
            epochs=epochs,
            lr=lr,
            prox_center=base_weights,
            prox_mu=prox_mu,
        )
    return out, {"method": "personalized_fedavg", "rounds": 1, "runtime_sec": time.perf_counter() - start}


def train_federated(
    x: np.ndarray,
    y: np.ndarray,
    clients: np.ndarray,
    *,
    method: str = "fedavg",
    privacy_mode: str = "plain",
    rounds: int = 20,
    local_epochs: int = 5,
    lr: float = 0.08,
    prox_mu: float = 0.0,
    dropout_rate: float = 0.0,
    seed: int = 31,
) -> tuple[LogisticHead, dict, list[dict]]:
    if method not in {"fedavg", "fedprox"}:
        raise ValueError("method must be 'fedavg' or 'fedprox'")
    if rounds <= 0 or local_epochs <= 0:
        raise ValueError("rounds and local_epochs must be positive")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must lie in [0, 1)")
    start = time.perf_counter()
    client_ids = sorted(set(map(int, clients)))
    global_model = LogisticHead.zeros(x.shape[1])
    round_logs: list[dict] = []

    for round_id in range(rounds):
        updates: list[np.ndarray] = []
        weights: list[float] = []
        round_loss = []
        base_w = global_model.weights.copy()
        round_rng = np.random.default_rng(np.random.SeedSequence([seed, round_id]))
        active_client_ids = [client_id for client_id in client_ids if round_rng.random() >= dropout_rate]
        if not active_client_ids:
            active_client_ids = [client_ids[int(round_rng.integers(0, len(client_ids)))]]
        for client_id in active_client_ids:
            idx = clients == client_id
            if idx.sum() == 0:
                continue
            local = train_logistic(
                x[idx],
                y[idx],
                initial=base_w,
                epochs=local_epochs,
                lr=lr,
                prox_center=base_w if method == "fedprox" else None,
                prox_mu=prox_mu if method == "fedprox" else 0.0,
            )
            updates.append(local.weights - base_w)
            weights.append(float(idx.sum()))
            pred = local.predict_proba(x[idx])
            eps = 1e-9
            loss = -np.mean(y[idx] * np.log(pred + eps) + (1 - y[idx]) * np.log(1 - pred + eps))
            round_loss.append(loss)

        aggregate, report = aggregate_updates(updates, weights, mode=privacy_mode)
        global_model.weights = base_w + aggregate
        log = asdict(report)
        log.update({
            "round": round_id + 1,
            "method": method,
            "privacy_mode": privacy_mode,
            "mean_client_loss": float(np.mean(round_loss)) if round_loss else float("nan"),
            "active_clients": int(len(active_client_ids)),
            "dropped_clients": int(len(client_ids) - len(active_client_ids)),
            "dropout_rate_requested": float(dropout_rate),
        })
        round_logs.append(log)

    summary = {
        "method": method,
        "privacy_mode": privacy_mode,
        "rounds": rounds,
        "runtime_sec": time.perf_counter() - start,
        "mean_encryption_time_sec": float(np.mean([r["encryption_time_sec"] for r in round_logs])),
        "mean_aggregation_time_sec": float(np.mean([r["aggregation_time_sec"] for r in round_logs])),
        "mean_ciphertext_expansion": float(np.mean([r["ciphertext_expansion"] for r in round_logs])),
        "dropout_rate": float(dropout_rate),
        "seed": int(seed),
    }
    return global_model, summary, round_logs
