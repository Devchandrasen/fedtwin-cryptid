from __future__ import annotations

import numpy as np
import pytest

from fedtwin.crypto import (
    aggregate_updates,
    generate_paillier_keypair,
    paillier_decrypt_int,
    paillier_encrypt_int,
    secureagg_simulate,
)
from fedtwin.federated import train_centralized, train_federated, train_local_models, train_personalized_models


def _updates() -> tuple[list[np.ndarray], list[float]]:
    return (
        [
            np.array([0.1, -0.2, 0.3]),
            np.array([0.4, 0.1, -0.1]),
            np.array([-0.2, 0.3, 0.2]),
            np.array([0.05, -0.1, 0.25]),
        ],
        [5.0, 3.0, 2.0, 4.0],
    )


def test_secureagg_pairwise_masks_and_dropout_match_active_plain() -> None:
    updates, weights = _updates()
    aggregate, report, details = secureagg_simulate(updates, weights, active_clients=[0, 2, 3], mask_seed=19)
    plain, _ = aggregate_updates([updates[i] for i in [0, 2, 3]], [weights[i] for i in [0, 2, 3]], mode="plain")
    assert np.allclose(aggregate, plain, atol=1e-10)
    assert details["masks_cancel"]
    assert report.dropped_clients == 1
    assert report.mode == "secureagg_sim"
    recovered, recovered_report, recovered_details = secureagg_simulate(
        updates,
        weights,
        active_clients=[0, 2, 3],
        mask_seed=19,
        dropout_stage="after_mask_setup",
    )
    assert np.allclose(recovered, plain, atol=1e-10)
    assert recovered_details["recovery_applied"]
    assert recovered_details["pre_recovery_residual_norm"] > 0
    assert recovered_details["reconstructed_mask_norm"] > 0
    assert recovered_report.protocol_scope.startswith("pairwise-mask and dropout-reconstruction")
    with pytest.raises(ValueError, match="dropout_stage"):
        secureagg_simulate(updates, weights, dropout_stage="mid_round")
    alias, alias_report = aggregate_updates(updates, weights, mode="secureagg")
    canonical, _ = aggregate_updates(updates, weights, mode="secureagg_sim")
    assert np.allclose(alias, canonical)
    assert alias_report.protocol_scope.startswith("pairwise-mask")


def test_quantized_proxy_is_explicit_and_input_validation_is_strict() -> None:
    updates, weights = _updates()
    plain, _ = aggregate_updates(updates, weights, mode="plain")
    proxy, report = aggregate_updates(updates, weights, mode="quantized_transport_proxy", he_scale=1e6)
    assert np.max(np.abs(plain - proxy)) < 1e-6
    assert "not homomorphic encryption" in report.protocol_scope
    with pytest.raises(ValueError, match="same length"):
        aggregate_updates(updates, [1.0], mode="plain")
    with pytest.raises(ValueError, match="finite"):
        aggregate_updates([np.array([np.nan])], [1.0], mode="plain")
    with pytest.raises(ValueError, match="unknown"):
        aggregate_updates(updates, weights, mode="mystery")


def test_paillier_signed_round_trip_and_key_floor() -> None:
    with pytest.raises(ValueError, match="at least 512"):
        generate_paillier_keypair(256)
    public, private = generate_paillier_keypair(512)
    for value in (-123, 0, 456):
        assert paillier_decrypt_int(paillier_encrypt_int(value, public), private) == value


def test_federated_training_modes_dropout_and_guards() -> None:
    x = np.array(
        [
            [-2.0, -1.0],
            [-1.0, -0.5],
            [1.0, 0.5],
            [2.0, 1.0],
            [-1.5, -0.7],
            [1.5, 0.8],
        ]
    )
    y = np.array([0, 0, 1, 1, 0, 1])
    clients = np.array([0, 0, 0, 1, 1, 1])
    centralized, central_summary = train_centralized(x, y, epochs=5)
    assert central_summary["method"] == "centralized"
    local, _ = train_local_models(x, y, clients, epochs=5)
    assert set(local) == {0, 1}
    personalized, _ = train_personalized_models(x, y, clients, centralized.weights, epochs=3)
    assert set(personalized) == {0, 1}
    first, summary, logs = train_federated(
        x,
        y,
        clients,
        method="fedprox",
        privacy_mode="secureagg_sim",
        rounds=3,
        local_epochs=2,
        prox_mu=0.01,
        dropout_rate=0.5,
        seed=9,
    )
    second, _, second_logs = train_federated(
        x,
        y,
        clients,
        method="fedprox",
        privacy_mode="secureagg_sim",
        rounds=3,
        local_epochs=2,
        prox_mu=0.01,
        dropout_rate=0.5,
        seed=9,
    )
    assert np.allclose(first.weights, second.weights)
    assert [row["dropped_clients"] for row in logs] == [row["dropped_clients"] for row in second_logs]
    assert summary["dropout_rate"] == 0.5
    with pytest.raises(ValueError, match="method"):
        train_federated(x, y, clients, method="unknown")
    with pytest.raises(ValueError, match="dropout"):
        train_federated(x, y, clients, dropout_rate=1.0)
