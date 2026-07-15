"""Shared validation contracts for benchmark data adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from fedtwin.data import BenchmarkData
from fedtwin.manifests import sha256_file


@dataclass(frozen=True)
class DataSourceReport:
    tier: str
    root: str
    files: tuple[dict[str, object], ...] = field(default_factory=tuple)
    license_note: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)


class DatasetAdapter(Protocol):
    tier: str

    def validate(self) -> DataSourceReport: ...

    def load(self, **kwargs: object) -> BenchmarkData: ...


def require_files(root: str | Path, relative_paths: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    base = Path(root).resolve()
    missing = [relative for relative in relative_paths if not (base / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files under {base}: {', '.join(missing)}")
    return tuple(
        {
            "path": relative,
            "bytes": (base / relative).stat().st_size,
            "sha256": sha256_file(base / relative),
        }
        for relative in relative_paths
    )
