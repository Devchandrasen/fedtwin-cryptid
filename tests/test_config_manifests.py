from __future__ import annotations

import json
from pathlib import Path

import pytest

from fedtwin.config import ExperimentConfig
from fedtwin.manifests import (
    build_run_manifest,
    canonical_json_hash,
    file_records,
    sha256_file,
    verify_file_records,
    write_json,
)


def test_config_loads_yaml_json_and_rejects_unsafe_claim_settings(tmp_path: Path) -> None:
    yaml_path = tmp_path / "paper.yaml"
    yaml_path.write_text("tier: tier0\nseeds: [31, 37, 41]\nrounds: 2\n", encoding="utf-8")
    config = ExperimentConfig.from_file(yaml_path)
    assert config.seeds == (31, 37, 41)
    assert config.rounds == 2
    json_path = tmp_path / "paper.json"
    json_path.write_text(json.dumps({"tier": "tier1", "seeds": [7], "paillier_key_bits": 2048}), encoding="utf-8")
    assert ExperimentConfig.from_file(json_path).tier == "tier1"
    with pytest.raises(ValueError, match="unknown experiment"):
        ExperimentConfig.from_mapping({"unknown": 1})
    with pytest.raises(ValueError, match="2048"):
        ExperimentConfig(paillier_key_bits=1024)
    with pytest.raises(ValueError, match="dropout_rates"):
        ExperimentConfig(dropout_rates=(1.0,))
    with pytest.raises(ValueError, match="fma_max_seconds"):
        ExperimentConfig(fma_max_seconds=0.0)
    with pytest.raises(ValueError, match="fma_sample_rate"):
        ExperimentConfig(fma_sample_rate=0)


def test_manifest_hashing_records_and_detects_tampering(tmp_path: Path) -> None:
    file_path = tmp_path / "value.txt"
    file_path.write_text("stable", encoding="utf-8")
    records = file_records([file_path], relative_to=tmp_path)
    assert records[0]["sha256"] == sha256_file(file_path)
    assert verify_file_records(tmp_path, records) == []
    file_path.write_text("tampered", encoding="utf-8")
    errors = verify_file_records(tmp_path, records)
    assert any("mismatch" in error for error in errors)
    assert canonical_json_hash({"b": 2, "a": 1}) == canonical_json_hash({"a": 1, "b": 2})


def test_manifest_verifier_rejects_paths_outside_artifact_root_and_duplicates(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    inside = artifact / "inside.txt"
    inside.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    valid = file_records([inside], relative_to=artifact)[0]

    traversal = {"path": "../outside.txt", "bytes": outside.stat().st_size, "sha256": sha256_file(outside)}
    absolute = {"path": str(outside.resolve()), "bytes": outside.stat().st_size, "sha256": sha256_file(outside)}
    errors = verify_file_records(artifact, [traversal, absolute, valid, dict(valid)])

    assert sum("escapes artifact root" in error or "must be relative" in error for error in errors) == 2
    assert any("duplicate file record" in error for error in errors)


def test_run_manifest_and_strict_json_writer(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text("a\n1\n", encoding="utf-8")
    output_path.write_text("b\n2\n", encoding="utf-8")
    manifest = build_run_manifest(
        repository_root=Path(__file__).resolve().parents[1],
        config={"seed": 31},
        inputs=[input_path],
        outputs=[output_path],
        counts={"rows": 1},
        feature_schema=["score"],
        timing={"fit": 0.1},
    )
    assert manifest["config_sha256"]
    assert manifest["runtime"]["python"]
    target = tmp_path / "nested" / "manifest.json"
    write_json(target, manifest)
    assert json.loads(target.read_text(encoding="utf-8"))["counts"]["rows"] == 1
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"value": float("nan")})
