"""Frozen paper-artifact verification and deterministic regeneration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .manifests import canonical_json_hash, file_records, resolve_within_root, verify_file_records, write_json

ARTIFACT_MANIFEST = "manifest.json"


def build_artifact_manifest(
    root: str | Path,
    *,
    version: str,
    evidence_status: str,
    claims_file: str = "claims.csv",
) -> dict[str, Any]:
    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)
    files = [
        path
        for path in base.rglob("*")
        if path.is_file()
        and path.name != ARTIFACT_MANIFEST
        and "__pycache__" not in path.parts
        and path.suffix.lower() not in {".pyc", ".pyo"}
    ]
    manifest = {
        "schema_version": "1.0",
        "artifact_version": version,
        "evidence_status": evidence_status,
        "raw_media_included": False,
        "raw_embeddings_included": False,
        "claims_file": claims_file,
        "files": file_records(files, relative_to=base),
    }
    write_json(base / ARTIFACT_MANIFEST, manifest)
    return manifest


def verify_artifact(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "artifact_version", "evidence_status", "files"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"artifact manifest is missing keys: {missing}")
    if not isinstance(payload["files"], list) or not payload["files"]:
        raise ValueError("artifact manifest contains no files")
    errors = verify_file_records(path.parent, payload["files"])
    forbidden = []
    for record in payload["files"]:
        file_path = str(record["path"])
        parts = Path(file_path).parts
        if "__pycache__" in parts or Path(file_path).suffix.lower() in {".pyc", ".pyo"}:
            forbidden.append(file_path)
    if forbidden:
        errors.extend(f"forbidden generated file: {item}" for item in forbidden)
    try:
        claims = resolve_within_root(path.parent, str(payload.get("claims_file", "claims.csv")))
    except ValueError as exc:
        errors.append(str(exc))
        claims = path.parent / "__invalid_claims_path__"
    if claims.is_file():
        frame = pd.read_csv(claims)
        required_claim_columns = {"claim_id", "manuscript_location", "evidence_file", "status"}
        claims_schema_valid = required_claim_columns.issubset(frame.columns)
        if not claims_schema_valid:
            errors.append("claims.csv does not satisfy the claim-to-evidence schema")
        else:
            unresolved = (
                frame.loc[frame["status"].astype(str).eq("replacement-required"), "claim_id"].astype(str).tolist()
            )
            if unresolved:
                errors.append(f"claims still require replacement evidence: {', '.join(unresolved)}")
        for evidence in frame.get("evidence_file", pd.Series(dtype=str)).dropna().astype(str):
            try:
                evidence_path = resolve_within_root(path.parent, evidence)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if not evidence_path.is_file():
                errors.append(f"claim evidence is missing: {evidence}")
    if errors:
        raise ValueError("artifact verification failed:\n- " + "\n- ".join(errors))
    return {
        "artifact_version": payload["artifact_version"],
        "evidence_status": payload["evidence_status"],
        "files_verified": len(payload["files"]),
        "manifest": str(path),
    }


def _plot_feature_ablation(source: Path, destination: Path) -> None:
    frame = pd.read_csv(source)
    required = {"method", "pr_auc_mean", "pr_auc_std"}
    if not required.issubset(frame.columns):
        raise ValueError(f"feature ablation table is missing columns: {sorted(required - set(frame.columns))}")
    plot = frame[["method", "pr_auc_mean", "pr_auc_std"]].copy()
    for column in ("pr_auc_mean", "pr_auc_std"):
        plot[column] = pd.to_numeric(plot[column], errors="raise")
    if not np.isfinite(plot[["pr_auc_mean", "pr_auc_std"]].to_numpy(dtype=float)).all():
        raise ValueError("feature ablation source contains non-finite values")
    plot = plot.sort_values("pr_auc_mean", ascending=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    colors = ["#4C78A8" if method not in {"score only", "default invariant"} else "#E45756" for method in plot["method"]]
    ax.barh(
        np.arange(len(plot)),
        plot["pr_auc_mean"],
        xerr=plot["pr_auc_std"],
        color=colors,
        alpha=0.9,
        error_kw={"elinewidth": 0.8, "capsize": 2},
    )
    ax.set_yticks(np.arange(len(plot)))
    ax.set_yticklabels(plot["method"])
    ax.set_xlabel("PR-AUC (mean across seeds; error bars are sample SD)")
    ax.set_xlim(max(0.0, float(plot["pr_auc_mean"].min()) - 0.04), min(1.0, float(plot["pr_auc_mean"].max()) + 0.04))
    ax.grid(axis="x", linewidth=0.6, alpha=0.35)
    ax.set_title("Fixed-evidence feature-policy ablation", loc="left", fontweight="bold")
    fig.tight_layout()
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "fig10_feature_ablation.pdf", bbox_inches="tight")
    fig.savefig(destination / "fig10_feature_ablation.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def build_confirmatory_tier_summary(results_root: str | Path) -> pd.DataFrame:
    """Derive the four-tier paper table directly from checksum-verified seed-level runs."""

    root = Path(results_root).resolve()
    specs = [
        (
            "VCSL public labels",
            "vcsl_public_asset_disjoint",
            "video_similarity",
            "centralized_multimodal",
            "local_multimodal",
        ),
        (
            "VCSL ISC visual",
            "vcsl_isc_confirmatory",
            "video_similarity",
            "centralized_multimodal",
            "local_multimodal",
        ),
        (
            "FMA audio 20 s",
            "fma_audio_20s_confirmatory",
            "audio_similarity",
            "centralized_audio_evidence",
            "local_audio_evidence",
        ),
        (
            "FMA audio 5 s",
            "fma_audio_5s_confirmatory",
            "audio_similarity",
            "centralized_audio_evidence",
            "local_audio_evidence",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for tier, directory, single_method, centralized_method, local_method in specs:
        run = root / "new_runs" / directory
        metrics_path = run / "metrics_detection.csv"
        manifest_path = run / "run_manifest.json"
        dataset_path = run / "dataset_manifest.json"
        if not metrics_path.is_file() or not manifest_path.is_file() or not dataset_path.is_file():
            raise FileNotFoundError(f"confirmatory tier is incomplete: {run}")
        metrics = pd.read_csv(metrics_path)
        required_columns = {"method", "seed", "pr_auc"}
        if missing := sorted(required_columns - set(metrics.columns)):
            raise ValueError(f"{metrics_path} is missing columns: {missing}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        output_errors = verify_file_records(run, manifest.get("outputs", []))
        if output_errors:
            raise ValueError(f"{run} has invalid output records: {'; '.join(output_errors)}")
        if manifest.get("config_sha256") != canonical_json_hash(manifest.get("config", {})):
            raise ValueError(f"{run} has an invalid configuration hash")
        git = manifest.get("git", {})
        if not git.get("commit") or git.get("dirty") is not False:
            raise ValueError(f"{run} is not a clean, commit-identified confirmatory run")
        grouped = metrics.groupby("method")["pr_auc"].agg(["mean", "std", "count"])
        required_methods = {
            single_method,
            centralized_method,
            local_method,
            "fedavg_plain",
            "fedprox_plain",
            "fedavg_secureagg_sim",
            "fedavg_quantized_transport_proxy",
        }
        if missing := sorted(required_methods - set(grouped.index)):
            raise ValueError(f"{metrics_path} is missing confirmatory methods: {missing}")
        for method in required_methods:
            seeds = sorted(metrics.loc[metrics["method"].eq(method), "seed"].astype(int).unique())
            if seeds != [31, 37, 41] or int(grouped.at[method, "count"]) != 3:
                raise ValueError(f"{tier} method {method} does not contain exactly seeds 31, 37, and 41")

        adapted_candidates = [local_method]
        if "fedavgft_plain" in grouped.index:
            adapted_candidates.append("fedavgft_plain")
            seeds = sorted(metrics.loc[metrics["method"].eq("fedavgft_plain"), "seed"].astype(int).unique())
            if seeds != [31, 37, 41] or int(grouped.at["fedavgft_plain", "count"]) != 3:
                raise ValueError(f"{tier} method fedavgft_plain does not contain exactly seeds 31, 37, and 41")
        client_adapted = max(adapted_candidates, key=lambda name: float(grouped.at[name, "mean"]))
        privacy_method = max(
            ("fedavg_secureagg_sim", "fedavg_quantized_transport_proxy"),
            key=lambda name: float(grouped.at[name, "mean"]),
        )
        eligible = sorted(required_methods | set(adapted_candidates))
        best_method = max(eligible, key=lambda name: float(grouped.at[name, "mean"]))
        counts = manifest.get("counts", {})
        row = {
            "tier": tier,
            "run_dir": f"new_runs/{directory}",
            "run_commit": git["commit"],
            "config_sha256": manifest.get("config_sha256", ""),
            "seeds": "31;37;41",
            "train_rows": int(counts.get("train_rows_per_seed", dataset.get("train_rows", 0))),
            "test_rows": int(counts.get("test_rows_per_seed", dataset.get("test_rows", 0))),
            "train_assets": int(counts.get("train_assets", dataset.get("train_assets", 0))),
            "test_assets": int(counts.get("test_assets", dataset.get("test_assets", 0))),
            "clients": int(counts.get("clients", dataset.get("clients", 0))),
            "source": str(dataset.get("source", "")),
            "evidence_status": "verified-confirmatory",
            "best_single_method": single_method,
            "best_single_pr_auc": float(grouped.at[single_method, "mean"]),
            "best_single_pr_auc_std": float(grouped.at[single_method, "std"]),
            "centralized_method": centralized_method,
            "centralized_pr_auc": float(grouped.at[centralized_method, "mean"]),
            "centralized_pr_auc_std": float(grouped.at[centralized_method, "std"]),
            "fedavg_plain_pr_auc": float(grouped.at["fedavg_plain", "mean"]),
            "fedavg_plain_pr_auc_std": float(grouped.at["fedavg_plain", "std"]),
            "fedavg_secureagg_sim_pr_auc": float(grouped.at["fedavg_secureagg_sim", "mean"]),
            "fedavg_secureagg_sim_pr_auc_std": float(grouped.at["fedavg_secureagg_sim", "std"]),
            "fedavg_quantized_transport_proxy_pr_auc": float(
                grouped.at["fedavg_quantized_transport_proxy", "mean"]
            ),
            "fedavg_quantized_transport_proxy_pr_auc_std": float(
                grouped.at["fedavg_quantized_transport_proxy", "std"]
            ),
            "fedprox_pr_auc": float(grouped.at["fedprox_plain", "mean"]),
            "fedprox_pr_auc_std": float(grouped.at["fedprox_plain", "std"]),
            "client_adapted_method": client_adapted,
            "client_adapted_pr_auc": float(grouped.at[client_adapted, "mean"]),
            "client_adapted_pr_auc_std": float(grouped.at[client_adapted, "std"]),
            "best_privacy_preserving_method": privacy_method,
            "best_privacy_preserving_pr_auc": float(grouped.at[privacy_method, "mean"]),
            "best_privacy_preserving_pr_auc_std": float(grouped.at[privacy_method, "std"]),
            "best_overall_method": best_method,
            "best_overall_pr_auc": float(grouped.at[best_method, "mean"]),
            "best_overall_std": float(grouped.at[best_method, "std"]),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    numeric = out.select_dtypes(include="number").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("confirmatory tier summary contains non-finite values")
    return out


def _plot_multitier_results(tables: Path, destination: Path) -> None:
    """Regenerate the four-tier comparison from confirmatory evidence only."""

    summary = pd.read_csv(tables / "tier_result_summary.csv")
    if set(summary["evidence_status"].astype(str)) != {"verified-confirmatory"}:
        raise ValueError("multitier plot refuses non-confirmatory tier rows")

    columns = {
        "Best single branch": "best_single_pr_auc",
        "Centralized": "centralized_pr_auc",
        "FedAvg": "fedavg_plain_pr_auc",
        "SecureAgg sim.": "fedavg_secureagg_sim_pr_auc",
        "Quant. proxy": "fedavg_quantized_transport_proxy_pr_auc",
    }
    std_columns = {
        "Best single branch": "best_single_pr_auc_std",
        "Centralized": "centralized_pr_auc_std",
        "FedAvg": "fedavg_plain_pr_auc_std",
        "SecureAgg sim.": "fedavg_secureagg_sim_pr_auc_std",
        "Quant. proxy": "fedavg_quantized_transport_proxy_pr_auc_std",
    }
    values = summary[list(columns.values())].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    errors = summary[list(std_columns.values())].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("multitier source contains non-finite values")
    if not np.isfinite(errors).all():
        raise ValueError("multitier uncertainty source contains non-finite values")

    labels = list(summary["tier"].astype(str))
    x = np.arange(len(labels), dtype=float)
    width = 0.13
    colors = ["#9CA3AF", "#4C78A8", "#54A24B", "#B279A2", "#F58518"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    for index, ((label, column), color) in enumerate(zip(columns.items(), colors, strict=True)):
        offset = (index - (len(columns) - 1) / 2) * width
        ax.errorbar(
            x + offset,
            summary[column],
            yerr=summary[std_columns[label]],
            marker="o",
            linestyle="none",
            markersize=4,
            capsize=2,
            linewidth=0.8,
            label=label,
            color=color,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PR-AUC (three-seed mean)")
    ax.set_ylim(max(0.0, float(values.min()) - 0.06), 1.01)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35)
    ax.legend(ncol=3, frameon=False, loc="lower right")
    ax.set_title("Fixed-evidence calibration across evaluation tiers", loc="left", fontweight="bold")
    ax.text(
        0.0,
        -0.24,
        "All rows are three-seed confirmatory reruns; tiers are separate and not directly rank-comparable.",
        transform=ax.transAxes,
        fontsize=7,
    )
    fig.tight_layout()
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "fig01_multitier_pr_auc.pdf", bbox_inches="tight")
    fig.savefig(destination / "fig01_multitier_pr_auc.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_privacy_utility(source: Path, destination: Path) -> None:
    frame = pd.read_csv(source)
    required = {"tier", "method", "expansion", "pr_auc", "delta"}
    if not required.issubset(frame.columns):
        raise ValueError(f"privacy utility table is missing columns: {sorted(required - set(frame.columns))}")
    numeric = frame[["expansion", "pr_auc", "delta"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("privacy utility source contains non-finite values")
    frame[["expansion", "pr_auc", "delta"]] = numeric

    labels = {
        "fedavg_plain": "Plain FedAvg",
        "fedavg_secureagg_sim": "SecureAgg sim.",
        "fedavg_quantized_transport_proxy": "Quant. proxy",
    }
    unknown = sorted(set(frame["method"].astype(str)) - set(labels))
    if unknown:
        raise ValueError(f"privacy utility table contains legacy/unknown method labels: {unknown}")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    for tier, group in frame.groupby("tier", sort=False):
        ordered = group.sort_values("expansion")
        axes[0].plot(ordered["expansion"], ordered["pr_auc"], marker="o", linewidth=1.2, label=tier)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Compact-update byte expansion")
    axes[0].set_ylabel("PR-AUC")
    axes[0].grid(linewidth=0.5, alpha=0.35)
    axes[0].legend(fontsize=6, frameon=False)
    method_delta = frame.groupby("method", sort=False)["delta"].mean().reindex(labels)
    axes[1].bar(labels.values(), method_delta.to_numpy(dtype=float) * 1e8, color=["#4C78A8", "#72B7B2", "#F58518"])
    axes[1].axhline(0.0, color="#2F3A45", linewidth=0.7)
    axes[1].set_ylabel(r"Mean PR-AUC delta vs. plain ($\times10^{-8}$)")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "fig03_privacy_utility_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(destination / "fig03_privacy_utility_tradeoff.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _protected_expansions(tables: Path) -> tuple[list[str], np.ndarray, list[str]]:
    paillier_columns = [
        "proxy_ciphertext_expansion",
        "paillier_ciphertext_expansion",
        "proxy_max_abs_error_vs_plain",
        "paillier_max_abs_error_vs_plain",
    ]
    paillier = pd.read_csv(tables / "paillier_update_aggregation.csv", usecols=paillier_columns).iloc[0]
    secureagg = pd.read_csv(tables / "secureagg_dropout_or_proxy.csv")
    secure_expansion = float(pd.to_numeric(secureagg["byte_expansion"], errors="raise").iloc[0])
    values = np.asarray(
        [1.0, secure_expansion, float(paillier["proxy_ciphertext_expansion"]), float(paillier["paillier_ciphertext_expansion"])],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("protected aggregation source contains non-finite expansion values")
    errors = [
        "reference",
        f"err <= {pd.to_numeric(secureagg['max_abs_error_vs_active_plain'], errors='raise').max():.1e}",
        f"err = {float(paillier['proxy_max_abs_error_vs_plain']):.1e}",
        f"err = {float(paillier['paillier_max_abs_error_vs_plain']):.1e}",
    ]
    return ["Plain", "SecureAgg sim.", "Quant. proxy", "Paillier"], values, errors


def _plot_crypto_overhead(tables: Path, destination: Path) -> None:
    labels, values, _ = _protected_expansions(tables)
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    bars = ax.bar(labels, values, color=["#4C78A8", "#72B7B2", "#F58518", "#B279A2"])
    ax.bar_label(bars, labels=[f"{value:.0f}x" for value in values], padding=2, fontsize=7)
    ax.set_yscale("log", base=2)
    ax.set_ylabel("Compact-update byte expansion")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "fig06_crypto_overhead.pdf", bbox_inches="tight")
    fig.savefig(destination / "fig06_crypto_overhead.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_paillier_overhead(tables: Path, destination: Path) -> None:
    labels, values, errors = _protected_expansions(tables)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    bars = ax.bar(labels, values, color=["#4C78A8", "#72B7B2", "#F58518", "#B279A2"])
    ax.bar_label(bars, labels=errors, padding=2, fontsize=7, rotation=8)
    ax.set_ylabel("Ciphertext/transport byte expansion")
    ax.set_title("Compact-update protection accounting", loc="left", fontweight="bold")
    ax.set_ylim(0.0, float(values.max()) * 1.22)
    ax.grid(axis="y", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "paillier_update_overhead.pdf", bbox_inches="tight")
    fig.savefig(destination / "paillier_update_overhead.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def regenerate_paper_assets(results_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(results_dir).resolve()
    destination = Path(output_dir).resolve()
    manifest = source / ARTIFACT_MANIFEST
    verification = verify_artifact(manifest)
    tables_source = source / "tables"
    if not tables_source.is_dir():
        raise FileNotFoundError(tables_source)
    tables_destination = destination / "tables"
    figures_destination = destination / "figures"
    tables_destination.mkdir(parents=True, exist_ok=True)
    figures_destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(tables_source.glob("*.csv")):
        shutil.copy2(path, tables_destination / path.name)
    confirmatory_inputs = [
        source / "new_runs" / directory / filename
        for directory in (
            "vcsl_public_asset_disjoint",
            "vcsl_isc_confirmatory",
            "fma_audio_20s_confirmatory",
            "fma_audio_5s_confirmatory",
        )
        for filename in ("metrics_detection.csv", "run_manifest.json", "dataset_manifest.json")
    ]
    if all(path.is_file() for path in confirmatory_inputs):
        confirmatory = build_confirmatory_tier_summary(source)
        confirmatory.to_csv(tables_destination / "tier_result_summary.csv", index=False, lineterminator="\n")
    generated_figures = ["fig10_feature_ablation.pdf", "fig10_feature_ablation.png"]
    _plot_feature_ablation(tables_source / "feature_ablation.csv", figures_destination)
    if (tables_destination / "tier_result_summary.csv").is_file():
        _plot_multitier_results(tables_destination, figures_destination)
        generated_figures.extend(["fig01_multitier_pr_auc.pdf", "fig01_multitier_pr_auc.png"])
    if (tables_source / "privacy_utility_points.csv").is_file():
        _plot_privacy_utility(tables_source / "privacy_utility_points.csv", figures_destination)
        generated_figures.extend(["fig03_privacy_utility_tradeoff.pdf", "fig03_privacy_utility_tradeoff.png"])
    if (tables_source / "paillier_update_aggregation.csv").is_file() and (
        tables_source / "secureagg_dropout_or_proxy.csv"
    ).is_file():
        _plot_crypto_overhead(tables_source, figures_destination)
        _plot_paillier_overhead(tables_source, figures_destination)
        generated_figures.extend(
            [
                "fig06_crypto_overhead.pdf",
                "fig06_crypto_overhead.png",
                "paillier_update_overhead.pdf",
                "paillier_update_overhead.png",
            ]
        )
    summary = {
        **verification,
        "tables_copied": len(list(tables_destination.glob("*.csv"))),
        "figures_generated": generated_figures,
        "output_dir": str(destination),
    }
    write_json(destination / "regeneration_summary.json", summary)
    return summary
