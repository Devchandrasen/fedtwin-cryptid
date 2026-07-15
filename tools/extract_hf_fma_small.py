"""Extract a pinned, hash-verified FMA-small mirror into a licensed data root."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

TRACK_NAME = re.compile(r"^[0-9]{6}\.mp3$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported extraction-config schema")
    if not isinstance(payload.get("dataset"), dict) or not isinstance(payload.get("shards"), list):
        raise ValueError("extraction config requires dataset and shards")
    return payload


def extract(config_path: Path, shard_dir: Path, output_dir: Path, manifest_path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required; run this tool with uv run --with pyarrow") from exc

    config = load_config(config_path)
    shard_root = shard_dir.resolve()
    output_root = output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    seen_names: set[str] = set()
    input_records: list[dict[str, Any]] = []
    output_records: list[dict[str, Any]] = []

    for expected in config["shards"]:
        relative = Path(str(expected["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe shard path: {relative}")
        shard = (shard_root / relative).resolve()
        try:
            shard.relative_to(shard_root)
        except ValueError as exc:
            raise ValueError(f"shard escapes configured root: {relative}") from exc
        if not shard.is_file():
            raise FileNotFoundError(f"missing configured shard: {shard}")
        actual_bytes = shard.stat().st_size
        actual_hash = sha256_file(shard)
        if actual_bytes != int(expected["bytes"]) or actual_hash != str(expected["sha256"]).lower():
            raise ValueError(f"shard size/hash mismatch: {relative}")
        input_records.append({"path": relative.as_posix(), "bytes": actual_bytes, "sha256": actual_hash})

        parquet = pq.ParquetFile(shard)
        if parquet.schema_arrow.get_field_index("audio") < 0:
            raise ValueError(f"shard is missing the audio field: {relative}")
        for batch in parquet.iter_batches(batch_size=16, columns=["audio"]):
            audio = batch.column(0)
            paths = audio.field("path")
            payloads = audio.field("bytes")
            for index in range(len(audio)):
                source_name = Path(str(paths[index].as_py())).name
                if not TRACK_NAME.fullmatch(source_name):
                    raise ValueError(f"unexpected audio filename in {relative}: {source_name!r}")
                if source_name in seen_names:
                    raise ValueError(f"duplicate audio filename across shards: {source_name}")
                seen_names.add(source_name)
                payload = payloads[index].as_py()
                if not isinstance(payload, bytes) or not payload:
                    raise ValueError(f"empty or invalid audio payload: {source_name}")
                target = (output_root / source_name[:3] / source_name).resolve()
                try:
                    target.relative_to(output_root)
                except ValueError as exc:
                    raise ValueError(f"audio target escapes configured root: {target}") from exc
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(payload).hexdigest()
                if target.exists():
                    if target.stat().st_size != len(payload) or sha256_file(target) != digest:
                        raise ValueError(f"existing extracted audio differs: {target}")
                else:
                    target.write_bytes(payload)
                output_records.append(
                    {
                        "path": target.relative_to(output_root).as_posix(),
                        "bytes": len(payload),
                        "sha256": digest,
                    }
                )

    output_records.sort(key=lambda record: str(record["path"]))
    manifest = {
        "schema_version": "1.0",
        "dataset": config["dataset"],
        "inputs": input_records,
        "outputs": output_records,
        "output_track_count": len(output_records),
        "output_bytes": sum(int(record["bytes"]) for record in output_records),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = extract(args.config, args.shard_dir, args.output_dir, args.manifest)
    print(json.dumps({"status": "ok", "tracks": manifest["output_track_count"]}, indent=2))


if __name__ == "__main__":
    main()
