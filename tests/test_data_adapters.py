from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fedtwin.data as data_module
from fedtwin.data import (
    FMA_AUDIO_TRANSFORMS,
    _audio_descriptor,
    _audio_transform,
    _frame_audio,
    _frame_match_scores,
    _prepare_fma_audio_cache,
    _sample_vcsl_pairs,
    _UnionFind,
    dataframe_from_split,
    generate_fma_audio_benchmark,
    generate_vcsl_isc_benchmark,
)
from fedtwin.data_adapters import FMAAudioAdapter, SyntheticAdapter, VCSLDescriptorAdapter, VCSLMetadataAdapter


def test_synthetic_and_vcsl_metadata_adapters(vcsl_metadata_dir: Path) -> None:
    synthetic = SyntheticAdapter()
    assert synthetic.validate().root == "generated-in-memory"
    generated = synthetic.load(clients=3, assets=30, queries=40, dim=8, seed=31)
    assert generated.x_train.shape[1] == len(generated.feature_names)
    assert set(np.unique(generated.y_test)).issubset({0, 1})

    adapter = VCSLMetadataAdapter(vcsl_metadata_dir)
    report = adapter.validate()
    assert len(report.files) == 6
    data = adapter.load(clients=2, dim=8, max_train_pairs=3, max_test_pairs=2, negative_ratio=1.0, seed=31)
    assert data.manifest["generator"] == "vcsl_public_label_topology"
    assert len(data.y_train) == 6
    frame = dataframe_from_split(
        data.x_test,
        data.y_test,
        data.client_test,
        data.scenario_test,
        data.feature_names,
    )
    assert {"label", "client_id", "scenario"}.issubset(frame.columns)


def test_vcsl_descriptor_adapter_and_generator(vcsl_metadata_dir: Path, tmp_path: Path) -> None:
    feature_dir = tmp_path / "features" / "eff256d"
    feature_dir.mkdir(parents=True)
    ids = pd.read_csv(vcsl_metadata_dir / "frames_all.csv")["uuid"].astype(str)
    rng = np.random.default_rng(5)
    for video_id in ids:
        np.save(feature_dir / f"{video_id}.npy", rng.normal(size=(6, 8)).astype(np.float32))
    adapter = VCSLDescriptorAdapter(vcsl_metadata_dir, feature_dir.parent)
    report = adapter.validate()
    assert report.tier == "vcsl_isc"
    data = generate_vcsl_isc_benchmark(
        metadata_dir=vcsl_metadata_dir,
        feature_dir=feature_dir.parent,
        clients=2,
        max_train_pairs=2,
        max_test_pairs=1,
        negative_ratio=1.0,
        max_frames=4,
        seed=31,
    )
    assert data.manifest["generator"] == "vcsl_released_isc_frame_features"
    assert "released ISC" in data.manifest["source"]
    left = np.eye(3, dtype=float)
    mean_score, max_score, topk, mutual = _frame_match_scores(left, left)
    assert max_score == 1.0
    assert mean_score <= topk <= max_score
    assert mutual == 1.0


def test_fma_adapter_validation_and_feature_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio_dir = tmp_path / "audio"
    metadata_dir = tmp_path / "metadata"
    cache_dir = tmp_path / "cache"
    audio_dir.mkdir()
    metadata_dir.mkdir()
    (metadata_dir / "tracks.csv").write_text("track_id,genre\n1,Rock\n", encoding="utf-8")
    for track_id in range(1, 7):
        (audio_dir / f"{track_id:06d}.mp3").write_bytes(b"fixture")
    report = FMAAudioAdapter(audio_dir, metadata_dir, cache_dir).validate()
    assert report.tier == "fma_audio"

    rng = np.random.default_rng(12)
    descriptors = rng.normal(size=(6, len(FMA_AUDIO_TRANSFORMS) + 1, 10)).astype(np.float32)
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=2, keepdims=True), 1e-12)
    fake_cache = {
        "track_ids": np.arange(1, 7),
        "genres": np.array(["Rock", "Rock", "Jazz", "Jazz", "Folk", "Folk"], dtype=object),
        "clients": np.arange(6),
        "descriptors": descriptors,
        "transforms": ["clean", *FMA_AUDIO_TRANSFORMS],
        "cache_path": str(cache_dir / "fake.npz"),
    }
    monkeypatch.setattr(data_module, "_prepare_fma_audio_cache", lambda **_: fake_cache)
    benchmark = generate_fma_audio_benchmark(
        audio_dir=audio_dir,
        metadata_dir=metadata_dir,
        cache_dir=cache_dir,
        clients=3,
        max_tracks=6,
        max_train_pairs=4,
        max_test_pairs=2,
        negative_ratio=1.0,
        sample_rate=8000,
        max_seconds=1.0,
        seed=31,
    )
    assert benchmark.manifest["generator"] == "fma_small_audio_transform_features"
    assert len(benchmark.y_test) == 4


def test_audio_transform_and_descriptor_branches() -> None:
    sample_rate = 8000
    time = np.linspace(0, 1, sample_rate, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    rng = np.random.default_rng(3)
    for name in FMA_AUDIO_TRANSFORMS:
        transformed = _audio_transform(audio, name, rng)
        assert transformed.ndim == 1
        assert np.isfinite(transformed).all()
    assert _frame_audio(audio).shape[1] == 512
    descriptor = _audio_descriptor(audio, sample_rate)
    assert descriptor.ndim == 1
    assert np.linalg.norm(descriptor) == pytest.approx(1.0, abs=1e-5)


def test_fma_cache_is_pickle_free_hashed_and_schema_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio"
    metadata = tmp_path / "metadata"
    cache = tmp_path / "cache"
    audio.mkdir()
    metadata.mkdir()
    (metadata / "tracks.csv").write_text("fixture", encoding="utf-8")
    genres = {1: "Rock", 2: "Jazz", 3: "Folk"}
    for track_id in genres:
        (audio / f"{track_id:06d}.mp3").write_bytes(f"audio-{track_id}".encode())
    monkeypatch.setattr(data_module, "_load_fma_tracks", lambda _path: genres)
    monkeypatch.setattr(
        data_module,
        "_decode_mp3",
        lambda _path, **_kwargs: np.linspace(-1.0, 1.0, 2048, dtype=np.float32),
    )
    monkeypatch.setattr(
        data_module,
        "_audio_descriptor",
        lambda _audio, _sample_rate: np.ones(8, dtype=np.float32) / np.sqrt(8),
    )

    generated = _prepare_fma_audio_cache(
        audio_dir=audio,
        metadata_dir=metadata,
        cache_dir=cache,
        max_tracks=3,
        sample_rate=8000,
        max_seconds=1.0,
        seed=31,
    )
    assert generated["genres"].dtype.kind == "U"
    assert Path(generated["cache_manifest_path"]).is_file()
    with np.load(generated["cache_path"], allow_pickle=False) as loaded:
        assert loaded["genres"].dtype.kind == "U"
        assert loaded["transforms"].dtype.kind == "U"

    reloaded = _prepare_fma_audio_cache(
        audio_dir=audio,
        metadata_dir=metadata,
        cache_dir=cache,
        max_tracks=3,
        sample_rate=8000,
        max_seconds=1.0,
        seed=31,
    )
    assert reloaded["input_records"] == generated["input_records"]


def test_fma_decode_failures_are_recorded_and_thresholded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio"
    metadata = tmp_path / "metadata"
    cache = tmp_path / "cache"
    audio.mkdir()
    metadata.mkdir()
    (metadata / "tracks.csv").write_text("fixture", encoding="utf-8")
    genres = {1: "Rock", 2: "Jazz", 3: "Folk"}
    for track_id in genres:
        (audio / f"{track_id:06d}.mp3").write_bytes(b"audio")
    monkeypatch.setattr(data_module, "_load_fma_tracks", lambda _path: genres)

    def decode(path: Path, **_kwargs: object) -> np.ndarray:
        if path.stem == "000002":
            raise RuntimeError("fixture decode failure")
        return np.linspace(-1.0, 1.0, 2048, dtype=np.float32)

    monkeypatch.setattr(data_module, "_decode_mp3", decode)
    monkeypatch.setattr(
        data_module,
        "_audio_descriptor",
        lambda _audio, _sample_rate: np.ones(8, dtype=np.float32) / np.sqrt(8),
    )
    with pytest.raises(RuntimeError, match="decode failure fraction"):
        _prepare_fma_audio_cache(
            audio_dir=audio,
            metadata_dir=metadata,
            cache_dir=cache,
            max_tracks=3,
            sample_rate=8000,
            max_seconds=1.0,
            seed=31,
            max_decode_failure_fraction=0.2,
        )
    failure_report = cache / "fma_audio_sr8000_sec1_tracks3.decode_failures.json"
    payload = json.loads(failure_report.read_text(encoding="utf-8"))
    assert payload["decode_failures"] == 1
    assert payload["failures"][0]["track_id"] == 2


def test_adapter_missing_source_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        VCSLMetadataAdapter(tmp_path).validate()
    metadata = tmp_path / "meta"
    audio = tmp_path / "audio"
    metadata.mkdir()
    audio.mkdir()
    (metadata / "tracks.csv").write_text("track_id,genre\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no FMA"):
        FMAAudioAdapter(audio, metadata, tmp_path / "cache").validate()


def test_vcsl_sampler_guarantees_distinct_positive_queries() -> None:
    positives = pd.DataFrame(
        {
            "query_id": [f"q{query}" for query in range(12) for _ in range(2)],
            "reference_id": [f"r{row}" for row in range(24)],
        }
    )
    all_ids = sorted(set(positives["query_id"]) | set(positives["reference_id"]) | {"negative-ref"})
    categories = {identifier: "one" for identifier in all_ids}
    sampled = _sample_vcsl_pairs(
        positives,
        all_ids,
        categories,
        _UnionFind(),
        np.random.default_rng(31),
        max_positive_pairs=10,
        negative_ratio=2.0,
        min_positive_queries=10,
    )
    selected = sampled.loc[sampled["label"].eq(1), "query_id"]
    assert selected.nunique() == 10
    assert len(sampled) == 30
