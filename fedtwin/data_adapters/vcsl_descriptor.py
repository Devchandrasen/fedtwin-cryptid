"""VCSL released ISC frame-descriptor adapter."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from fedtwin.data import BenchmarkData, generate_vcsl_isc_benchmark
from fedtwin.manifests import sha256_file

from .base import DataSourceReport
from .vcsl_metadata import VCSLMetadataAdapter


class VCSLDescriptorAdapter:
    tier = "vcsl_isc"

    def __init__(self, metadata_dir: str | Path, feature_dir: str | Path) -> None:
        self.metadata_dir = Path(metadata_dir).resolve()
        self.feature_dir = Path(feature_dir).resolve()

    def validate(self) -> DataSourceReport:
        metadata = VCSLMetadataAdapter(self.metadata_dir).validate()
        common_root = Path(os.path.commonpath((self.metadata_dir, self.feature_dir))).resolve()
        feature_files = sorted(path.resolve() for path in self.feature_dir.rglob("*.npy"))
        if not feature_files:
            raise FileNotFoundError(f"no .npy descriptor files found under {self.feature_dir}")
        dimensions: set[int] = set()
        records: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        for path in feature_files:
            try:
                path.relative_to(self.feature_dir)
            except ValueError as exc:
                raise ValueError(f"VCSL descriptor path escapes configured root: {path}") from exc
            if path.stem in seen_ids:
                raise ValueError(f"duplicate VCSL descriptor identifier: {path.stem}")
            seen_ids.add(path.stem)
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError(f"invalid frame-descriptor shape {array.shape} in {path}")
            dimensions.add(int(array.shape[1]))
            if not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"non-numeric descriptor dtype {array.dtype} in {path}")
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite descriptor values in {path}")
            records.append(
                {
                    "path": path.relative_to(common_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        if len(dimensions) != 1:
            raise ValueError(f"descriptor dimensions are inconsistent: {sorted(dimensions)}")
        metadata_records = tuple(
            {
                **record,
                "path": (self.metadata_dir / str(record["path"])).relative_to(common_root).as_posix(),
            }
            for record in metadata.files
        )
        return DataSourceReport(
            tier=self.tier,
            root=str(common_root),
            files=metadata_records + tuple(records),
            license_note=(
                "Descriptors and metadata are not redistributed; users must obtain the released VCSL/ISC "
                "resources and comply with upstream terms."
            ),
        )

    def load(self, **kwargs: object) -> BenchmarkData:
        self.validate()
        return generate_vcsl_isc_benchmark(
            metadata_dir=self.metadata_dir,
            feature_dir=self.feature_dir,
            **kwargs,
        )
