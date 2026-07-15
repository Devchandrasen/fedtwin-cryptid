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

from .manifests import file_records, resolve_within_root, verify_file_records, write_json

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


def _plot_multitier_results(tables: Path, destination: Path) -> None:
    """Regenerate the main comparison while overriding the stale VCSL archive row."""

    archived = pd.read_csv(tables / "tier_result_summary.csv")
    vcsl = pd.read_csv(tables / "vcsl_asset_disjoint_detection_summary.csv").set_index("method")
    required_methods = {
        "video_similarity",
        "centralized_multimodal",
        "fedavg_plain",
        "fedavg_secureagg_sim",
        "fedavg_quantized_transport_proxy",
    }
    if missing := sorted(required_methods - set(vcsl.index)):
        raise ValueError(f"VCSL asset-disjoint summary is missing methods: {missing}")

    row = archived["tier"].astype(str).eq("VCSL public labels")
    if int(row.sum()) != 1:
        raise ValueError("tier result summary must contain exactly one VCSL public-label row")
    overrides = {
        "train_rows": 5000,
        "test_rows": 1800,
        "best_single_pr_auc": vcsl.at["video_similarity", "pr_auc_mean"],
        "centralized_multimodal_pr_auc": vcsl.at["centralized_multimodal", "pr_auc_mean"],
        "fedavg_plain_pr_auc": vcsl.at["fedavg_plain", "pr_auc_mean"],
        "fedavg_secureagg_sim_pr_auc": vcsl.at["fedavg_secureagg_sim", "pr_auc_mean"],
        "fedavg_quantized_transport_proxy_pr_auc": vcsl.at[
            "fedavg_quantized_transport_proxy", "pr_auc_mean"
        ],
        "best_overall_pr_auc": vcsl.at["fedavg_plain", "pr_auc_mean"],
        "best_overall_std": vcsl.at["fedavg_plain", "pr_auc_sd"],
    }
    for column, value in overrides.items():
        archived.loc[row, column] = value

    columns = {
        "Best single branch": "best_single_pr_auc",
        "Centralized": "centralized_multimodal_pr_auc",
        "FedAvg": "fedavg_plain_pr_auc",
        "SecureAgg sim.": "fedavg_secureagg_sim_pr_auc",
        "Quant. proxy": "fedavg_quantized_transport_proxy_pr_auc",
    }
    values = archived[list(columns.values())].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("multitier source contains non-finite values")

    labels = [tier if tier == "VCSL public labels" else f"{tier}*" for tier in archived["tier"].astype(str)]
    x = np.arange(len(labels), dtype=float)
    width = 0.16
    colors = ["#9CA3AF", "#4C78A8", "#54A24B", "#B279A2", "#F58518"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    for index, ((label, column), color) in enumerate(zip(columns.items(), colors, strict=True)):
        offset = (index - (len(columns) - 1) / 2) * width
        ax.bar(x + offset, archived[column], width=width, label=label, color=color, alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PR-AUC (three-seed mean)")
    ax.set_ylim(max(0.0, float(values.min()) - 0.04), 1.01)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35)
    ax.legend(ncol=3, frameon=False, loc="lower right")
    ax.set_title("Fixed-evidence calibration across evaluation tiers", loc="left", fontweight="bold")
    ax.text(
        0.0,
        -0.24,
        "* Archived exploratory tier; VCSL public labels is the checksum-verified asset-disjoint rerun.",
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
    paillier = pd.read_csv(tables / "paillier_update_aggregation.csv").iloc[0]
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
    generated_figures = ["fig10_feature_ablation.pdf", "fig10_feature_ablation.png"]
    _plot_feature_ablation(tables_source / "feature_ablation.csv", figures_destination)
    if (tables_source / "tier_result_summary.csv").is_file() and (
        tables_source / "vcsl_asset_disjoint_detection_summary.csv"
    ).is_file():
        _plot_multitier_results(tables_source, figures_destination)
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
