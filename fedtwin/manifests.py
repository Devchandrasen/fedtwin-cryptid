"""Run-manifest and artifact-integrity helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def dependency_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in sorted(set(names)):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def file_records(paths: Iterable[str | Path], *, relative_to: str | Path | None = None) -> list[dict[str, Any]]:
    base = Path(relative_to).resolve() if relative_to is not None else None
    records: list[dict[str, Any]] = []
    for raw_path in sorted({Path(path).resolve() for path in paths}, key=lambda value: str(value).lower()):
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if base is None:
            display = raw_path.name
        else:
            try:
                display = raw_path.relative_to(base).as_posix()
            except ValueError:
                display = raw_path.name
        records.append({"path": display, "bytes": raw_path.stat().st_size, "sha256": sha256_file(raw_path)})
    return records


def resolve_within_root(root: str | Path, relative_path: str | Path) -> Path:
    """Resolve a manifest path without permitting reads outside ``root``.

    Artifact manifests are untrusted input. Absolute paths, parent traversal,
    and symlinks that escape the artifact directory are rejected before a file
    is opened or hashed.
    """

    base = Path(root).resolve()
    raw = str(relative_path)
    if not raw.strip():
        raise ValueError("manifest path must be a non-empty relative path")
    relative = Path(raw)
    if relative.is_absolute() or relative.drive or relative.anchor:
        raise ValueError(f"manifest path must be relative: {raw}")
    if any(part == ".." for part in relative.parts):
        raise ValueError(f"manifest path escapes artifact root: {raw}")
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"manifest path escapes artifact root: {raw}") from exc
    return candidate


def build_run_manifest(
    *,
    repository_root: str | Path,
    config: dict[str, Any],
    inputs: Iterable[str | Path] = (),
    outputs: Iterable[str | Path] = (),
    counts: dict[str, int] | None = None,
    feature_schema: list[str] | None = None,
    timing: dict[str, float] | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    status = _git_value(root, "status", "--porcelain")
    return {
        "schema_version": "1.0",
        "git": {
            "commit": _git_value(root, "rev-parse", "HEAD"),
            "branch": _git_value(root, "branch", "--show-current"),
            "dirty": bool(status),
        },
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "executable": Path(sys.executable).name,
            "dependencies": dependency_versions(
                ["cryptography", "matplotlib", "numpy", "pandas", "PyYAML", "scikit-learn", "seaborn", "tqdm"]
            ),
        },
        "hardware": {
            "processor": platform.processor() or "unknown",
            "logical_cpu_count": os.cpu_count(),
            "machine": platform.machine(),
        },
        "config": config,
        "config_sha256": canonical_json_hash(config),
        "inputs": file_records(inputs),
        "outputs": file_records(outputs),
        "counts": counts or {},
        "feature_schema": feature_schema or [],
        "timing_sec": timing or {},
    }


def verify_file_records(root: str | Path, records: Iterable[dict[str, Any]]) -> list[str]:
    base = Path(root).resolve()
    errors: list[str] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or "path" not in record:
            errors.append("invalid file record: missing path")
            continue
        display = str(record["path"])
        try:
            path = resolve_within_root(base, display)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        normalized = path.relative_to(base).as_posix().casefold()
        if normalized in seen:
            errors.append(f"duplicate file record: {display}")
            continue
        seen.add(normalized)
        if not path.is_file():
            errors.append(f"missing: {display}")
            continue
        try:
            expected_bytes = int(record.get("bytes", -1))
        except (TypeError, ValueError):
            errors.append(f"invalid byte count: {display}")
            continue
        if expected_bytes != path.stat().st_size:
            errors.append(f"size mismatch: {display}")
        expected = str(record.get("sha256", ""))
        actual = sha256_file(path)
        if expected != actual:
            errors.append(f"sha256 mismatch: {display}")
    return errors
