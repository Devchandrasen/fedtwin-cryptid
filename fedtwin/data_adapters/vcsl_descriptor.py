"""VCSL released ISC frame-descriptor adapter."""

from __future__ import annotations

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
        feature_files = sorted(self.feature_dir.rglob("*.npy"))
        if not feature_files:
            raise FileNotFoundError(f"no .npy descriptor files found under {self.feature_dir}")
        dimensions: set[int] = set()
        warnings: list[str] = []
        for path in feature_files[:16]:
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError(f"invalid frame-descriptor shape {array.shape} in {path}")
            dimensions.add(int(array.shape[1]))
            if not np.isfinite(array[: min(len(array), 8)]).all():
                raise ValueError(f"non-finite descriptor values in {path}")
        if len(dimensions) != 1:
            raise ValueError(f"sampled descriptor dimensions are inconsistent: {sorted(dimensions)}")
        if len(feature_files) > 16:
            warnings.append("shape and finite-value checks sampled the first 16 descriptor files")
        sampled = tuple(
            {
                "path": path.relative_to(self.feature_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in feature_files[:16]
        )
        return DataSourceReport(
            tier=self.tier,
            root=str(self.feature_dir),
            files=metadata.files + sampled,
            license_note=(
                "Descriptors and metadata are not redistributed; users must obtain the released VCSL/ISC "
                "resources and comply with upstream terms."
            ),
            warnings=tuple(warnings),
        )

    def load(self, **kwargs: object) -> BenchmarkData:
        self.validate()
        return generate_vcsl_isc_benchmark(
            metadata_dir=self.metadata_dir,
            feature_dir=self.feature_dir,
            **kwargs,
        )
