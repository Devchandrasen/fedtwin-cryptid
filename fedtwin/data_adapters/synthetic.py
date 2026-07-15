"""Synthetic correctness/stress adapter."""

from __future__ import annotations

from fedtwin.data import BenchmarkData, generate_synthetic_benchmark

from .base import DataSourceReport


class SyntheticAdapter:
    tier = "tier0"

    def validate(self) -> DataSourceReport:
        return DataSourceReport(
            tier=self.tier,
            root="generated-in-memory",
            license_note="Repository-generated synthetic evidence; no third-party media.",
        )

    def load(self, **kwargs: object) -> BenchmarkData:
        return generate_synthetic_benchmark(**kwargs)
