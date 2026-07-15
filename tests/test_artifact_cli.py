from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fedtwin.artifact import (
    _plot_crypto_overhead,
    _plot_multitier_results,
    _plot_paillier_overhead,
    _plot_privacy_utility,
    build_artifact_manifest,
    regenerate_paper_assets,
    verify_artifact,
)
from fedtwin.cli import main


def _artifact(root: Path) -> Path:
    tables = root / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        {
            "method": ["score only", "default invariant", "audio only"],
            "pr_auc_mean": [0.90, 0.89, 0.80],
            "pr_auc_std": [0.01, 0.02, 0.03],
        }
    ).to_csv(tables / "feature_ablation.csv", index=False)
    pd.DataFrame(
        {
            "claim_id": ["C1"],
            "manuscript_location": ["Results"],
            "evidence_file": ["tables/feature_ablation.csv"],
            "status": ["verified"],
        }
    ).to_csv(root / "claims.csv", index=False)
    build_artifact_manifest(root, version="v1", evidence_status="verified-derived-results")
    return root / "manifest.json"


def test_artifact_build_verify_regenerate_and_tamper_detection(tmp_path: Path) -> None:
    manifest = _artifact(tmp_path / "artifact")
    summary = verify_artifact(manifest)
    assert summary["files_verified"] == 2
    generated = regenerate_paper_assets(manifest.parent, tmp_path / "generated")
    assert generated["tables_copied"] == 1
    assert (tmp_path / "generated" / "figures" / "fig10_feature_ablation.pdf").is_file()
    assert b"nan" not in (tmp_path / "generated" / "figures" / "fig10_feature_ablation.pdf").read_bytes().lower()
    (manifest.parent / "claims.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="verification failed"):
        verify_artifact(manifest)


def test_cli_verify_regenerate_and_smoke(tmp_path: Path) -> None:
    manifest = _artifact(tmp_path / "artifact")
    assert main(["verify-artifact", "--manifest", str(manifest)]) == 0
    assert (
        main(
            [
                "regenerate-paper",
                "--results-dir",
                str(manifest.parent),
                "--output-dir",
                str(tmp_path / "paper"),
            ]
        )
        == 0
    )
    assert main(["smoke", "--output-dir", str(tmp_path / "runs"), "--assets", "20", "--queries", "24"]) == 0
    run_manifest = tmp_path / "runs" / "smoke" / "run_manifest.json"
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    assert payload["config"]["tier"] == "tier0"
    assert payload["outputs"]
    for csv_path in (tmp_path / "runs" / "smoke").rglob("*.csv"):
        assert b"\r\n" not in csv_path.read_bytes()


def test_artifact_rejects_unresolved_replacement_claim(tmp_path: Path) -> None:
    manifest = _artifact(tmp_path / "artifact")
    claims = pd.read_csv(manifest.parent / "claims.csv")
    claims["status"] = "replacement-required"
    claims.to_csv(manifest.parent / "claims.csv", index=False)
    build_artifact_manifest(manifest.parent, version="v1", evidence_status="revision-in-progress")
    with pytest.raises(ValueError, match="claims still require replacement evidence"):
        verify_artifact(manifest)


def test_multitier_plot_overrides_vcsl_archive_row(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    tables.mkdir()
    pd.DataFrame(
        {
            "tier": ["Synthetic twin", "VCSL public labels"],
            "best_single_pr_auc": [0.80, 0.10],
            "centralized_multimodal_pr_auc": [0.90, 0.10],
            "fedavg_plain_pr_auc": [0.91, 0.10],
            "fedavg_secureagg_sim_pr_auc": [0.91, 0.10],
            "fedavg_quantized_transport_proxy_pr_auc": [0.91, 0.10],
        }
    ).to_csv(tables / "tier_result_summary.csv", index=False)
    pd.DataFrame(
        {
            "method": [
                "video_similarity",
                "centralized_multimodal",
                "fedavg_plain",
                "fedavg_secureagg_sim",
                "fedavg_quantized_transport_proxy",
            ],
            "pr_auc_mean": [0.75, 0.88, 0.89, 0.89, 0.89],
            "pr_auc_sd": [0.01, 0.01, 0.02, 0.02, 0.02],
        }
    ).to_csv(tables / "vcsl_asset_disjoint_detection_summary.csv", index=False)

    _plot_multitier_results(tables, figures)
    assert (figures / "fig01_multitier_pr_auc.pdf").is_file()
    assert (figures / "fig01_multitier_pr_auc.png").is_file()

    pd.DataFrame(
        {
            "tier": ["Synthetic", "Synthetic", "Synthetic", "VCSL", "VCSL", "VCSL"],
            "method": [
                "fedavg_plain",
                "fedavg_secureagg_sim",
                "fedavg_quantized_transport_proxy",
            ]
            * 2,
            "expansion": [1.0, 2.0, 16.0] * 2,
            "pr_auc": [0.90, 0.90, 0.90, 0.80, 0.80, 0.80],
            "delta": [0.0, 0.0, 1e-8, 0.0, 0.0, -1e-8],
        }
    ).to_csv(tables / "privacy_utility_points.csv", index=False)
    pd.DataFrame(
        {
            "proxy_ciphertext_expansion": [16.0],
            "paillier_ciphertext_expansion": [64.0],
            "proxy_max_abs_error_vs_plain": [2.6e-7],
            "paillier_max_abs_error_vs_plain": [2.6e-7],
        }
    ).to_csv(tables / "paillier_update_aggregation.csv", index=False)
    pd.DataFrame(
        {
            "byte_expansion": [2.0, 2.0],
            "max_abs_error_vs_active_plain": [2e-15, 3e-15],
        }
    ).to_csv(tables / "secureagg_dropout_or_proxy.csv", index=False)

    _plot_privacy_utility(tables / "privacy_utility_points.csv", figures)
    _plot_crypto_overhead(tables, figures)
    _plot_paillier_overhead(tables, figures)
    assert (figures / "fig03_privacy_utility_tradeoff.pdf").is_file()
    assert (figures / "fig06_crypto_overhead.pdf").is_file()
    assert (figures / "paillier_update_overhead.pdf").is_file()
