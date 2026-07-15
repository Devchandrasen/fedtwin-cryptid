from __future__ import annotations

import numpy as np

from fedtwin.crypto import aggregate_updates, paillier_aggregate_updates
from fedtwin.ledger import HashChainLedger, canonical_hash, sha256_text


def test_secureagg_style_simulation_matches_plain_aggregate() -> None:
    updates = [
        np.array([0.10, -0.20, 0.30], dtype=float),
        np.array([0.40, 0.15, -0.10], dtype=float),
        np.array([-0.05, 0.35, 0.20], dtype=float),
    ]
    weights = [5.0, 3.0, 2.0]
    plain, plain_report = aggregate_updates(updates, weights, mode="plain")
    secure, secure_report = aggregate_updates(updates, weights, mode="secureagg")
    assert np.allclose(plain, secure)
    assert secure_report.ciphertext_expansion == 2.0
    assert plain_report.ciphertext_expansion == 1.0


def test_he_proxy_quantization_error_is_small_for_compact_updates() -> None:
    updates = [
        np.array([0.1234567, -0.2345678, 0.3456789], dtype=float),
        np.array([-0.0456789, 0.1567891, -0.2678912], dtype=float),
    ]
    weights = [7.0, 4.0]
    plain, _ = aggregate_updates(updates, weights, mode="plain")
    he_proxy, report = aggregate_updates(updates, weights, mode="heagg", he_scale=1e6)
    assert np.max(np.abs(plain - he_proxy)) < 1e-6
    assert report.ciphertext_expansion == 16.0


def test_real_paillier_additive_aggregation_matches_plain_compact_sum() -> None:
    updates = [
        np.array([0.001234, -0.002345, 0.003456], dtype=float),
        np.array([-0.000987, 0.001876, -0.002765], dtype=float),
        np.array([0.002222, 0.000333, -0.001111], dtype=float),
    ]
    weights = [9.0, 5.0, 7.0]
    plain, _ = aggregate_updates(updates, weights, mode="plain")
    paillier, report, details = paillier_aggregate_updates(updates, weights, key_bits=512, he_scale=1e6)
    assert np.max(np.abs(plain - paillier)) < 1e-6
    assert report.mode == "paillier"
    assert report.ciphertext_expansion > 1.0
    assert report.max_abs_error_vs_plain < 1e-6
    assert details["scheme"] == "Paillier"
    assert int(details["max_safe_abs_aggregate_integer"]) > int(details["observed_max_abs_aggregate_integer"])
    assert details["max_safe_quantization_scale"]


def test_hash_chain_detects_payload_and_previous_hash_tampering() -> None:
    ledger = HashChainLedger()
    for idx in range(5):
        payload = {
            "receipt_id": f"r-{idx}",
            "previous_chain_hash": ledger.previous_hash,
            "decision": "review",
            "confidence": float(idx) / 10.0,
        }
        ledger.anchor(payload)
    assert ledger.verify()

    ledger.records[2]["payload"]["decision"] = "tampered"
    assert not ledger.verify()
    ledger.records[2]["payload"]["decision"] = "review"
    assert ledger.verify()

    old_prev = ledger.records[3]["previous_chain_hash"]
    ledger.records[3]["previous_chain_hash"] = "f" * 64
    assert not ledger.verify()
    ledger.records[3]["previous_chain_hash"] = old_prev
    assert ledger.verify()


def test_canonical_receipt_chain_hash_construction() -> None:
    previous = "0" * 64
    payload = {"receipt_id": "test", "previous_chain_hash": previous, "decision": "review"}
    receipt_hash = canonical_hash(payload)
    chain_hash = sha256_text(previous + receipt_hash)
    ledger = HashChainLedger()
    anchored_receipt_hash, anchored_chain_hash, _ = ledger.anchor(payload)
    assert anchored_receipt_hash == receipt_hash
    assert anchored_chain_hash == chain_hash
