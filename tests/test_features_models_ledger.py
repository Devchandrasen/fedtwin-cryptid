from __future__ import annotations

import numpy as np
import pytest

from fedtwin.features import apply_standardization, cosine, l2_normalize, sigmoid, standardize_train_test
from fedtwin.ledger import HashChainLedger, canonical_hash, keyed_commitment, make_receipts, sha256_text
from fedtwin.models import LogisticHead, add_bias, model_digest, train_logistic
from fedtwin.transformations import get_transformation, transformation_names


def test_feature_and_model_utilities_are_stable() -> None:
    vectors = np.array([[3.0, 4.0], [0.0, 0.0]])
    normalized = l2_normalize(vectors)
    assert np.linalg.norm(normalized[0]) == pytest.approx(1.0)
    assert cosine(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]))[0] == pytest.approx(1.0)
    sigmoid_values = sigmoid(np.array([-100.0, 0.0, 100.0]))
    assert np.all((sigmoid_values >= 0) & (sigmoid_values <= 1))
    assert sigmoid_values[1] == 0.5
    train = np.array([[1.0, 2.0], [3.0, 2.0]])
    test = np.array([[5.0, 2.0]])
    train_std, test_std, mean, std = standardize_train_test(train, test)
    assert np.allclose(apply_standardization(test, mean, std), test_std)
    assert np.isfinite(train_std).all()
    assert add_bias(test).shape == (1, 3)
    model = train_logistic(train, np.array([0, 1]), epochs=10)
    assert isinstance(model.copy(), LogisticHead)
    assert model.predict_proba(test).shape == (1,)
    assert model_digest(model.weights) == model_digest(model.weights.copy())


def test_receipt_signatures_chain_and_commitments() -> None:
    assert len(sha256_text("value")) == 64
    assert keyed_commitment("secret", "value") != keyed_commitment("other", "value")
    receipts, summary = make_receipts(
        scores=np.array([0.9, 0.2, 0.7]),
        y=np.array([1, 0, 1]),
        clients=np.array([0, 1, 0]),
        model_digest=sha256_text("model"),
        limit=3,
        sign_receipts=True,
    )
    assert len(receipts) == 3
    assert summary["verification_success"]
    assert summary["tamper_detection_success"]
    ledger = HashChainLedger()
    payload = {"previous_chain_hash": ledger.previous_hash, "value": 1}
    receipt_hash, _, _ = ledger.anchor(payload)
    assert receipt_hash == canonical_hash(payload)
    assert ledger.verify()
    ledger.records[0]["payload"]["value"] = 2
    assert not ledger.verify()


def test_transformation_catalog() -> None:
    names = transformation_names()
    assert "partial_segment" in names
    assert get_transformation("partial_segment").modality == "multimodal"
    with pytest.raises(KeyError):
        get_transformation("missing")
