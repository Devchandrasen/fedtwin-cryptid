"""Validated experiment configuration for portable reproduction commands."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

TIERS = {"tier0", "tier1", "vcsl_public", "vcsl_isc", "fma_audio"}
AGGREGATION_MODES = {"plain", "secureagg_sim", "quantized_transport_proxy", "paillier"}


@dataclass(frozen=True)
class ExperimentConfig:
    tier: str = "tier0"
    seeds: tuple[int, ...] = (31, 37, 41)
    clients: int = 5
    assets: int = 1000
    queries: int = 1000
    dim: int = 32
    rounds: int = 20
    local_epochs: int = 5
    noniid_alpha: float = 0.3
    difficulty: float = 1.0
    max_train_pairs: int = 60_000
    max_test_pairs: int = 30_000
    negative_ratio: float = 1.0
    feature_map: str = "poly2"
    feature_policy: str = "invariant"
    federated_methods: tuple[str, ...] = (
        "centralized",
        "local",
        "fedavg",
        "fedprox",
        "secureagg_sim",
        "quantized_transport_proxy",
    )
    modalities: tuple[str, ...] = ("video", "audio", "multimodal")
    ledger_modes: tuple[str, ...] = ("hashchain",)
    attacker_fractions: tuple[float, ...] = (0.0, 0.1, 0.2)
    attacks: tuple[str, ...] = ("label_flip", "sign_flip")
    dropout_rates: tuple[float, ...] = (0.0, 0.1, 0.2)
    fma_max_decode_failure_fraction: float = 0.05
    paillier_key_bits: int = 2048
    output_tag: str = "paper"

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}; expected one of {sorted(TIERS)}")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must contain nonnegative integers")
        for name in ("clients", "assets", "queries", "dim", "rounds", "local_epochs"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_train_pairs <= 0 or self.max_test_pairs <= 0:
            raise ValueError("pair caps must be positive")
        if self.negative_ratio < 0:
            raise ValueError("negative_ratio must be nonnegative")
        if self.feature_map not in {"linear", "poly2"}:
            raise ValueError("feature_map must be 'linear' or 'poly2'")
        if self.feature_policy not in {"all", "invariant"}:
            raise ValueError("feature_policy must be 'all' or 'invariant'")
        if self.paillier_key_bits < 2048:
            raise ValueError("reported Paillier experiments require at least 2048-bit keys")
        for values, name in ((self.attacker_fractions, "attacker_fractions"), (self.dropout_rates, "dropout_rates")):
            if any(value < 0 or value >= 1 for value in values):
                raise ValueError(f"{name} values must lie in [0, 1)")
        if not 0 <= self.fma_max_decode_failure_fraction < 1:
            raise ValueError("fma_max_decode_failure_fraction must lie in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ExperimentConfig:
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown experiment configuration keys: {unknown}")
        tuple_fields = {
            "seeds",
            "federated_methods",
            "modalities",
            "ledger_modes",
            "attacker_fractions",
            "attacks",
            "dropout_rates",
        }
        values = dict(payload)
        for name in tuple_fields & values.keys():
            values[name] = tuple(values[name])
        return cls(**values)

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
        else:
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("experiment configuration must be a mapping")
        return cls.from_mapping(payload)
