"""FMA audio-transformation adapter."""

from __future__ import annotations

from pathlib import Path

from fedtwin.data import BenchmarkData, generate_fma_audio_benchmark
from fedtwin.manifests import sha256_file

from .base import DataSourceReport


class FMAAudioAdapter:
    tier = "fma_audio"

    def __init__(self, audio_dir: str | Path, metadata_dir: str | Path, cache_dir: str | Path) -> None:
        self.audio_dir = Path(audio_dir).resolve()
        self.metadata_dir = Path(metadata_dir).resolve()
        self.cache_dir = Path(cache_dir).resolve()

    def validate(self) -> DataSourceReport:
        tracks = self.metadata_dir / "tracks.csv"
        if not tracks.is_file():
            raise FileNotFoundError(f"missing FMA metadata file: {tracks}")
        audio_files = sorted(self.audio_dir.rglob("*.mp3"))
        if not audio_files:
            raise FileNotFoundError(f"no FMA .mp3 files found under {self.audio_dir}")
        sample = audio_files[:16]
        records = (
            {"path": str(tracks.name), "bytes": tracks.stat().st_size, "sha256": sha256_file(tracks)},
            *(
                {
                    "path": path.relative_to(self.audio_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sample
            ),
        )
        return DataSourceReport(
            tier=self.tier,
            root=str(self.audio_dir),
            files=tuple(records),
            license_note=(
                "FMA audio is not redistributed. Each track retains its upstream Creative Commons license; "
                "users must obtain FMA from the official release and respect per-track terms."
            ),
            warnings=("audio hashes cover the first 16 discovered tracks; the run manifest records selected tracks",),
        )

    def load(self, **kwargs: object) -> BenchmarkData:
        self.validate()
        return generate_fma_audio_benchmark(
            audio_dir=self.audio_dir,
            metadata_dir=self.metadata_dir,
            cache_dir=self.cache_dir,
            **kwargs,
        )
