"""VCSL public metadata/label adapter."""

from __future__ import annotations

from pathlib import Path

from fedtwin.data import BenchmarkData, generate_vcsl_public_benchmark

from .base import DataSourceReport, require_files

VCSL_METADATA_FILES = (
    "frames_all.csv",
    "pair_file_test.csv",
    "pair_file_train.csv",
    "pair_file_val.csv",
    "videos_url_uuid.csv",
    "video_categories.json",
)


class VCSLMetadataAdapter:
    tier = "vcsl_public"

    def __init__(self, metadata_dir: str | Path) -> None:
        self.metadata_dir = Path(metadata_dir).resolve()

    def validate(self) -> DataSourceReport:
        records = require_files(self.metadata_dir, VCSL_METADATA_FILES)
        return DataSourceReport(
            tier=self.tier,
            root=str(self.metadata_dir),
            files=records,
            license_note=(
                "Metadata is not redistributed. Users must obtain VCSL resources from the upstream project "
                "and comply with its dataset and source-video terms."
            ),
        )

    def load(self, **kwargs: object) -> BenchmarkData:
        self.validate()
        return generate_vcsl_public_benchmark(metadata_dir=self.metadata_dir, **kwargs)
