from __future__ import annotations

import pandas as pd
import pytest

from fedtwin.pairs import assert_asset_disjoint, construct_asset_disjoint_pairs, validate_pair_frame


def test_asset_disjoint_pair_construction_is_deterministic_and_balanced() -> None:
    positives = pd.DataFrame(
        {
            "query_id": ["a", "c", "e", "g", "i", "k"],
            "reference_id": ["b", "d", "f", "h", "j", "l"],
        }
    )
    first = construct_asset_disjoint_pairs(positives, negative_ratio=1.0, test_fraction=0.4, seed=31)
    second = construct_asset_disjoint_pairs(positives, negative_ratio=1.0, test_fraction=0.4, seed=31)
    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.test, second.test)
    assert first.manifest["asset_overlap"] == 0
    assert first.train["label"].value_counts().nunique() == 1
    assert first.test["label"].value_counts().nunique() == 1
    assert_asset_disjoint(first.train, first.test)


def test_pair_validation_rejects_schema_duplicates_self_pairs_and_leakage() -> None:
    with pytest.raises(ValueError, match="missing required"):
        construct_asset_disjoint_pairs(pd.DataFrame({"query_id": ["a"]}))
    with pytest.raises(ValueError, match="self-pairs"):
        construct_asset_disjoint_pairs(pd.DataFrame({"query_id": ["a"], "reference_id": ["a"]}))
    frame = pd.DataFrame(
        {
            "query_id": ["a", "a"],
            "reference_id": ["b", "b"],
            "label": [1, 1],
            "split": ["train", "train"],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_pair_frame(frame)
    train = frame.iloc[:1]
    test = pd.DataFrame({"query_id": ["a"], "reference_id": ["c"], "label": [0], "split": ["test"]})
    with pytest.raises(ValueError, match="asset leakage"):
        assert_asset_disjoint(train, test)


def test_pair_configuration_guards_impossible_sampling() -> None:
    positives = pd.DataFrame({"query_id": ["a"], "reference_id": ["b"]})
    with pytest.raises(ValueError, match="negative_ratio"):
        construct_asset_disjoint_pairs(positives, negative_ratio=-1)
    with pytest.raises(ValueError, match="test_fraction"):
        construct_asset_disjoint_pairs(positives, test_fraction=1.0)
