"""Deterministic, asset-disjoint pair construction.

The public benchmark adapters ultimately produce pair-feature rows, but split
construction belongs in one auditable module.  Positive-pair connected
components are assigned wholly to train or test before negatives are sampled,
so an asset identifier cannot cross the split boundary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

PAIR_COLUMNS = ("query_id", "reference_id")


@dataclass(frozen=True)
class PairSplits:
    train: pd.DataFrame
    test: pd.DataFrame
    manifest: dict[str, object]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _normalize_positive_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in PAIR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"positive_pairs is missing required columns: {missing}")
    out = frame.copy()
    for column in PAIR_COLUMNS:
        if out[column].isna().any():
            raise ValueError(f"positive_pairs contains missing {column} values")
        out[column] = out[column].astype(str)
    if (out["query_id"] == out["reference_id"]).any():
        raise ValueError("positive_pairs must not contain self-pairs")
    out["label"] = 1
    return out.drop_duplicates(["query_id", "reference_id"], keep="first").reset_index(drop=True)


def validate_pair_frame(frame: pd.DataFrame) -> None:
    """Raise when a pair frame violates the public pair schema."""

    missing = [column for column in (*PAIR_COLUMNS, "label", "split") if column not in frame.columns]
    if missing:
        raise ValueError(f"pair frame is missing required columns: {missing}")
    if frame[list(PAIR_COLUMNS)].isna().any().any():
        raise ValueError("pair identifiers must be non-null")
    if not set(frame["label"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("pair labels must be binary")
    if frame.duplicated(list(PAIR_COLUMNS)).any():
        raise ValueError("duplicate directional pairs are not allowed")
    if (frame["query_id"].astype(str) == frame["reference_id"].astype(str)).any():
        raise ValueError("self-pairs are not allowed")


def _sample_negatives(
    positive: pd.DataFrame,
    assets: Sequence[str],
    *,
    count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if count <= 0:
        return pd.DataFrame(columns=[*PAIR_COLUMNS, "label"])
    unique_assets = np.asarray(sorted(set(map(str, assets))), dtype=object)
    if len(unique_assets) < 2:
        raise ValueError("at least two assets are required to sample negatives")
    forbidden = set(zip(positive["query_id"].astype(str), positive["reference_id"].astype(str), strict=True))
    forbidden.update((right, left) for left, right in list(forbidden))
    selected: set[tuple[str, str]] = set()
    max_attempts = max(10_000, count * 100)
    attempts = 0
    while len(selected) < count and attempts < max_attempts:
        batch = min(max((count - len(selected)) * 3, 64), 100_000)
        left_idx = rng.integers(0, len(unique_assets), size=batch)
        right_idx = rng.integers(0, len(unique_assets), size=batch)
        for left_pos, right_pos in zip(left_idx, right_idx, strict=True):
            attempts += 1
            left = str(unique_assets[int(left_pos)])
            right = str(unique_assets[int(right_pos)])
            pair = (left, right)
            if left != right and pair not in forbidden and pair not in selected:
                selected.add(pair)
                if len(selected) == count:
                    break
            if attempts >= max_attempts:
                break
    if len(selected) != count:
        raise ValueError(
            f"unable to sample {count} unique negatives from {len(unique_assets)} assets; "
            f"sampled {len(selected)}"
        )
    rows = sorted(selected)
    return pd.DataFrame(rows, columns=list(PAIR_COLUMNS)).assign(label=0)


def assert_asset_disjoint(train: pd.DataFrame, test: pd.DataFrame) -> None:
    train_assets = set(train["query_id"].astype(str)) | set(train["reference_id"].astype(str))
    test_assets = set(test["query_id"].astype(str)) | set(test["reference_id"].astype(str))
    overlap = sorted(train_assets & test_assets)
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ValueError(f"asset leakage across train/test split: {preview}")


def construct_asset_disjoint_pairs(
    positive_pairs: pd.DataFrame,
    *,
    asset_ids: Iterable[str] | None = None,
    negative_ratio: float = 1.0,
    test_fraction: float = 0.2,
    seed: int = 31,
) -> PairSplits:
    """Construct deterministic train/test pairs with asset-level isolation.

    Connected positive components are the indivisible split units.  Optional
    unpaired assets are assigned using the same deterministic hash rule.
    Negative pairs are sampled only within a split.
    """

    if negative_ratio < 0:
        raise ValueError("negative_ratio must be nonnegative")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one")
    positive = _normalize_positive_pairs(positive_pairs)
    uf = _UnionFind()
    for row in positive.itertuples(index=False):
        uf.union(str(row.query_id), str(row.reference_id))
    all_assets = set(positive["query_id"]) | set(positive["reference_id"])
    if asset_ids is not None:
        all_assets.update(map(str, asset_ids))
    for asset in all_assets:
        uf.find(str(asset))

    positive_roots = sorted({uf.find(str(asset)) for asset in set(positive["query_id"]) | set(positive["reference_id"])})
    component_split: dict[str, str] = {}
    if len(positive_roots) >= 2:
        test_components = min(
            len(positive_roots) - 1,
            max(1, int(round(len(positive_roots) * test_fraction))),
        )
        ranked_roots = sorted(positive_roots, key=lambda root: (_stable_fraction(root, seed), root))
        test_roots = set(ranked_roots[:test_components])
        component_split.update({root: "test" if root in test_roots else "train" for root in positive_roots})
    elif positive_roots:
        raise ValueError("asset-disjoint splitting requires at least two positive connected components")
    for asset in sorted(all_assets):
        root = uf.find(str(asset))
        component_split.setdefault(root, "test" if _stable_fraction(root, seed) < test_fraction else "train")
    split_by_asset = {asset: component_split[uf.find(str(asset))] for asset in all_assets}
    positive["split"] = positive["query_id"].map(split_by_asset)
    if (positive["split"] != positive["reference_id"].map(split_by_asset)).any():
        raise AssertionError("connected-component split construction failed")

    rng = np.random.default_rng(seed)
    output: dict[str, pd.DataFrame] = {}
    for split in ("train", "test"):
        positives = positive[positive["split"] == split].copy()
        split_assets = sorted(asset for asset, assigned in split_by_asset.items() if assigned == split)
        negative_count = int(round(len(positives) * negative_ratio))
        negatives = _sample_negatives(positives, split_assets, count=negative_count, rng=rng)
        negatives["split"] = split
        frame = pd.concat([positives, negatives], ignore_index=True, sort=False)
        frame = frame.sample(frac=1.0, random_state=seed + (0 if split == "train" else 1)).reset_index(drop=True)
        validate_pair_frame(frame)
        output[split] = frame

    assert_asset_disjoint(output["train"], output["test"])
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "seed": int(seed),
        "test_fraction": float(test_fraction),
        "negative_ratio": float(negative_ratio),
        "split_policy": "positive-connected-component asset-disjoint",
        "train_pairs": int(len(output["train"])),
        "test_pairs": int(len(output["test"])),
        "train_positive_pairs": int(output["train"]["label"].sum()),
        "test_positive_pairs": int(output["test"]["label"].sum()),
        "train_assets": int(len(set(output["train"]["query_id"]) | set(output["train"]["reference_id"]))),
        "test_assets": int(len(set(output["test"]["query_id"]) | set(output["test"]["reference_id"]))),
        "asset_overlap": 0,
    }
    return PairSplits(train=output["train"], test=output["test"], manifest=manifest)
