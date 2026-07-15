"""FMA audio-transformation adapter."""

from __future__ import annotations

import os
from pathlib import Path

from fedtwin.data import (
    BenchmarkData,
    _discover_fma_audio_files,
    _load_fma_metadata_fields,
    _resolve_fma_tracks_path,
    generate_fma_audio_benchmark,
)
from fedtwin.manifests import sha256_file

from .base import DataSourceReport


class FMAAudioAdapter:
    tier = "fma_audio"

    def __init__(self, audio_dir: str | Path, metadata_dir: str | Path, cache_dir: str | Path) -> None:
        self.audio_dir = Path(audio_dir).resolve()
        self.metadata_dir = Path(metadata_dir).resolve()
        self.cache_dir = Path(cache_dir).resolve()

    def validate(self) -> DataSourceReport:
        tracks = _resolve_fma_tracks_path(self.metadata_dir)
        if tracks is None:
            raise FileNotFoundError(f"missing FMA tracks.csv under {self.metadata_dir}")
        metadata = _load_fma_metadata_fields(self.metadata_dir)
        audio_by_id = _discover_fma_audio_files(self.audio_dir)
        audio_files = [audio_by_id[track_id] for track_id in sorted(audio_by_id)]
        if not audio_files:
            raise FileNotFoundError(f"no FMA .mp3 files found under {self.audio_dir}")
        missing_metadata = sorted(set(audio_by_id) - set(metadata))
        if missing_metadata:
            preview = ", ".join(map(str, missing_metadata[:8]))
            raise ValueError(f"FMA audio tracks are absent from tracks.csv: {preview}")
        wrong_subset = sorted(track_id for track_id in audio_by_id if metadata[track_id]["subset"] != "small")
        if wrong_subset:
            preview = ", ".join(map(str, wrong_subset[:8]))
            raise ValueError(f"FMA staged audio includes tracks outside the 'small' subset: {preview}")
        missing_genres = sorted(track_id for track_id in audio_by_id if not metadata[track_id]["genre_top"])
        if missing_genres:
            preview = ", ".join(map(str, missing_genres[:8]))
            raise ValueError(f"FMA staged audio has missing top-level genres: {preview}")
        missing_licenses = sorted(track_id for track_id in audio_by_id if not metadata[track_id]["license"])
        if missing_licenses:
            preview = ", ".join(map(str, missing_licenses[:8]))
            raise ValueError(f"FMA staged audio has missing per-track licenses: {preview}")
        empty_files = [path for path in audio_files if path.stat().st_size <= 0]
        if empty_files:
            raise ValueError(f"FMA staged audio contains an empty file: {empty_files[0]}")

        common_root = Path(os.path.commonpath((self.audio_dir, self.metadata_dir))).resolve()
        records = [
            {
                "path": tracks.relative_to(common_root).as_posix(),
                "bytes": tracks.stat().st_size,
                "sha256": sha256_file(tracks),
            }
        ]
        records.extend(
            {
                "path": path.relative_to(common_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in audio_files
        )
        return DataSourceReport(
            tier=self.tier,
            root=str(common_root),
            files=tuple(records),
            license_note=(
                "FMA audio is not redistributed. Each track retains its upstream Creative Commons license; "
                "users must obtain FMA from the official release and respect per-track terms."
            ),
        )

    def load(self, **kwargs: object) -> BenchmarkData:
        self.validate()
        return generate_fma_audio_benchmark(
            audio_dir=self.audio_dir,
            metadata_dir=self.metadata_dir,
            cache_dir=self.cache_dir,
            **kwargs,
        )
