"""Portable command-line interface for reproduction and artifact checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifact import regenerate_paper_assets, verify_artifact
from .config import ExperimentConfig
from .data_adapters import FMAAudioAdapter, SyntheticAdapter, VCSLDescriptorAdapter, VCSLMetadataAdapter
from .manifests import build_run_manifest, file_records, write_json


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _source_report(config: ExperimentConfig, data_root: Path) -> object:
    if config.tier in {"tier0", "tier1"}:
        return SyntheticAdapter().validate()
    metadata_dir = data_root / "public_data" / "vcsl_metadata"
    if config.tier == "vcsl_public":
        return VCSLMetadataAdapter(metadata_dir).validate()
    if config.tier == "vcsl_isc":
        return VCSLDescriptorAdapter(metadata_dir, data_root / "data" / "vcsl_features" / "isc_extracted").validate()
    return FMAAudioAdapter(
        data_root / "data" / "fma" / "fma_small",
        data_root / "data" / "fma" / "fma_metadata",
        data_root / "data" / "fma" / "cache",
    ).validate()


def _namespace(config: ExperimentConfig, *, data_root: Path, output_dir: Path) -> argparse.Namespace:
    from run_benchmark import build_parser

    args = build_parser().parse_args([])
    for name, value in config.to_dict().items():
        if hasattr(args, name):
            setattr(args, name, list(value) if isinstance(value, tuple) else value)
    args.run_dir = str(output_dir)
    args.output_tag = config.output_tag
    args.data_dir = str(data_root / "data")
    args.vcsl_metadata_dir = str(data_root / "public_data" / "vcsl_metadata")
    args.vcsl_feature_dir = str(data_root / "data" / "vcsl_features" / "isc_extracted")
    args.fma_audio_dir = str(data_root / "data" / "fma" / "fma_small")
    args.fma_metadata_dir = str(data_root / "data" / "fma" / "fma_metadata")
    args.fma_cache_dir = str(data_root / "data" / "fma" / "cache")
    return args


def _replace_path_prefixes(value: object, mappings: list[tuple[str, str]]) -> object:
    if isinstance(value, dict):
        return {key: _replace_path_prefixes(item, mappings) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_path_prefixes(item, mappings) for item in value]
    if isinstance(value, str):
        normalized = value.replace("/", "\\")
        for prefix, replacement in mappings:
            if normalized.lower().startswith(prefix.lower()):
                suffix = normalized[len(prefix) :].lstrip("\\")
                portable_suffix = suffix.replace("\\", "/")
                return replacement if not suffix else f"{replacement}/{portable_suffix}"
    return value


def _sanitize_run_json_paths(run_dir: Path, *, data_root: Path) -> None:
    mappings = [
        (str(run_dir.resolve()).replace("/", "\\"), "${RUN_DIR}"),
        (str(data_root.resolve()).replace("/", "\\"), "${DATA_ROOT}"),
        (str(_repository_root().resolve()).replace("/", "\\"), "${REPOSITORY_ROOT}"),
    ]
    mappings.sort(key=lambda item: len(item[0]), reverse=True)
    for path in sorted(run_dir.rglob("*.json")):
        if path.name == "run_manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        sanitized = _replace_path_prefixes(payload, mappings)
        if not isinstance(sanitized, dict):
            raise ValueError(f"run JSON root must be an object: {path}")
        write_json(path, sanitized)


def _run_reproduction(config: ExperimentConfig, *, data_root: Path, output_dir: Path) -> Path:
    from run_benchmark import run_benchmark

    source = _source_report(config, data_root)
    args = _namespace(config, data_root=data_root, output_dir=output_dir)
    run_dir = run_benchmark(args)
    _sanitize_run_json_paths(run_dir, data_root=data_root)
    outputs = [path for path in run_dir.rglob("*") if path.is_file() and path.name != "run_manifest.json"]
    dataset_manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    transformation_manifest = json.loads((run_dir / "transformation_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = build_run_manifest(
        repository_root=_repository_root(),
        config=config.to_dict(),
        counts={
            "seeds": len(config.seeds),
            "clients": int(dataset_manifest.get("clients", config.clients)),
            "train_rows_per_seed": int(dataset_manifest.get("train_rows", 0)),
            "test_rows_per_seed": int(dataset_manifest.get("test_rows", 0)),
            "train_assets": int(dataset_manifest.get("train_assets", 0)),
            "test_assets": int(dataset_manifest.get("test_assets", 0)),
            "train_positive_rows_per_seed": int(dataset_manifest.get("train_positive_rows", 0)),
            "train_negative_rows_per_seed": int(dataset_manifest.get("train_negative_rows", 0)),
            "test_positive_rows_per_seed": int(dataset_manifest.get("test_positive_rows", 0)),
            "test_negative_rows_per_seed": int(dataset_manifest.get("test_negative_rows", 0)),
            "train_queries_per_seed": int(dataset_manifest.get("train_query_count", 0)),
            "test_queries_per_seed": int(dataset_manifest.get("test_query_count", 0)),
        },
        feature_schema=list(map(str, transformation_manifest.get("feature_names", []))),
        timing={"total_runtime": float(summary.get("total_runtime_sec", 0.0))},
    )
    manifest["data_source"] = {
        "tier": source.tier,
        "root": "${DATA_ROOT}",
        "files": list(source.files),
        "license_note": source.license_note,
        "warnings": list(source.warnings),
    }
    recorded_inputs = dataset_manifest.get("input_records")
    manifest["inputs"] = list(recorded_inputs) if isinstance(recorded_inputs, list) else list(source.files)
    manifest["outputs"] = file_records(outputs, relative_to=run_dir)
    write_json(run_dir / "run_manifest.json", manifest)
    return run_dir


def _cmd_smoke(args: argparse.Namespace) -> int:
    config = ExperimentConfig(
        tier="tier0",
        seeds=(int(args.seed),),
        clients=int(args.clients),
        assets=int(args.assets),
        queries=int(args.queries),
        rounds=int(args.rounds),
        local_epochs=int(args.local_epochs),
        output_tag="smoke",
        paillier_key_bits=2048,
    )
    run_dir = _run_reproduction(config, data_root=_repository_root(), output_dir=Path(args.output_dir).resolve())
    print(json.dumps({"status": "ok", "run_dir": str(run_dir)}, indent=2))
    return 0


def _cmd_reproduce(args: argparse.Namespace) -> int:
    config = ExperimentConfig.from_file(args.config)
    data_root = Path(args.data_root).resolve() if args.data_root else _repository_root()
    run_dir = _run_reproduction(config, data_root=data_root, output_dir=Path(args.output_dir).resolve())
    print(json.dumps({"status": "ok", "run_dir": str(run_dir)}, indent=2))
    return 0


def _cmd_regenerate(args: argparse.Namespace) -> int:
    summary = regenerate_paper_assets(args.results_dir, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    summary = verify_artifact(args.manifest)
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fedtwin", description="FedTwin-CryptID reproducibility tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run the deterministic synthetic smoke benchmark")
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--seed", type=int, default=31)
    smoke.add_argument("--clients", type=int, default=5)
    smoke.add_argument("--assets", type=int, default=80)
    smoke.add_argument("--queries", type=int, default=100)
    smoke.add_argument("--rounds", type=int, default=1)
    smoke.add_argument("--local-epochs", type=int, default=1)
    smoke.set_defaults(handler=_cmd_smoke)

    reproduce = subparsers.add_parser("reproduce", help="run a validated experiment configuration")
    reproduce.add_argument("--config", required=True)
    reproduce.add_argument("--data-root", default=None, help="explicit root containing public_data/ and data/")
    reproduce.add_argument("--output-dir", required=True)
    reproduce.set_defaults(handler=_cmd_reproduce)

    regenerate = subparsers.add_parser("regenerate-paper", help="regenerate paper assets from a frozen artifact")
    regenerate.add_argument("--results-dir", required=True)
    regenerate.add_argument("--output-dir", required=True)
    regenerate.set_defaults(handler=_cmd_regenerate)

    verify = subparsers.add_parser("verify-artifact", help="verify every file in a frozen artifact")
    verify.add_argument("--manifest", required=True)
    verify.set_defaults(handler=_cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
