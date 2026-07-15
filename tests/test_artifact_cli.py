from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fedtwin.artifact import (
    _plot_communication_round_ablation,
    _plot_crypto_overhead,
    _plot_multitier_results,
    _plot_paillier_overhead,
    _plot_privacy_utility,
    build_artifact_manifest,
    build_communication_round_ablation,
    build_confirmatory_tier_summary,
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
    pd.testing.assert_frame_equal(
        pd.read_csv(manifest.parent / "tables" / "feature_ablation.csv"),
        pd.read_csv(tmp_path / "generated" / "tables" / "feature_ablation.csv"),
        check_exact=False,
        atol=1e-4,
        rtol=0.0,
    )
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
    assert main(["smoke", "--output-dir", str(tmp_path / "repeat"), "--assets", "20", "--queries", "24"]) == 0
    run_manifest = tmp_path / "runs" / "smoke" / "run_manifest.json"
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    assert payload["config"]["tier"] == "tier0"
    assert payload["data_source"]["root"] == "generated-in-memory"
    assert payload["outputs"]
    for csv_path in (tmp_path / "runs" / "smoke").rglob("*.csv"):
        assert b"\r\n" not in csv_path.read_bytes()
    for json_path in (tmp_path / "runs" / "smoke").rglob("*.json"):
        assert b"\r\n" not in json_path.read_bytes()
    localization = pd.read_csv(tmp_path / "runs" / "smoke" / "metrics_localization.csv")
    assert list(localization.columns) == [
        "seed",
        "method",
        "segment_f1_proxy",
        "segment_precision_proxy",
        "partial_segment_n",
    ]

    first = tmp_path / "runs" / "smoke"
    second = tmp_path / "repeat" / "smoke"
    metric_paths = sorted(path.relative_to(first) for path in first.rglob("metrics_*.csv"))
    manifest_paths = sorted(
        path.relative_to(first)
        for path in first.rglob("*.json")
        if path.name in {"client_split_manifest.json", "dataset_manifest.json", "transformation_manifest.json"}
    )
    assert metric_paths and manifest_paths
    for relative in metric_paths:
        left = pd.read_csv(first / relative)
        right = pd.read_csv(second / relative)
        runtime_columns = [name for name in left.columns if "time" in name or "latency" in name]
        pd.testing.assert_frame_equal(
            left.drop(columns=runtime_columns),
            right.drop(columns=runtime_columns),
            check_exact=True,
            obj=str(relative),
        )
    for relative in manifest_paths:
        assert (first / relative).read_bytes() == (second / relative).read_bytes(), relative


def test_artifact_rejects_unresolved_replacement_claim(tmp_path: Path) -> None:
    manifest = _artifact(tmp_path / "artifact")
    claims = pd.read_csv(manifest.parent / "claims.csv")
    claims["status"] = "replacement-required"
    claims.to_csv(manifest.parent / "claims.csv", index=False)
    build_artifact_manifest(manifest.parent, version="v1", evidence_status="revision-in-progress")
    with pytest.raises(ValueError, match="claims still require replacement evidence"):
        verify_artifact(manifest)


def test_multitier_plot_requires_confirmatory_finite_rows(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    tables.mkdir()
    pd.DataFrame(
        {
            "tier": ["VCSL public labels", "FMA audio 5 s"],
            "evidence_status": ["verified-confirmatory", "verified-confirmatory"],
            "best_single_pr_auc": [0.80, 0.90],
            "best_single_pr_auc_std": [0.01, 0.01],
            "centralized_pr_auc": [0.90, 0.95],
            "centralized_pr_auc_std": [0.01, 0.01],
            "fedavg_plain_pr_auc": [0.91, 0.96],
            "fedavg_plain_pr_auc_std": [0.01, 0.01],
            "fedavg_secureagg_sim_pr_auc": [0.91, 0.96],
            "fedavg_secureagg_sim_pr_auc_std": [0.01, 0.01],
            "fedavg_quantized_transport_proxy_pr_auc": [0.91, 0.96],
            "fedavg_quantized_transport_proxy_pr_auc_std": [0.01, 0.01],
        }
    ).to_csv(tables / "tier_result_summary.csv", index=False)

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
            "seed": [31, 37, 41],
            "max_safe_abs_aggregate_integer": [str(10**620)] * 3,
            "proxy_ciphertext_expansion": [16.0] * 3,
            "paillier_ciphertext_expansion": [64.0] * 3,
            "proxy_max_abs_error_vs_plain": [2.5e-7, 2.6e-7, 2.4e-7],
            "paillier_max_abs_error_vs_plain": [2.5e-7, 2.6e-7, 2.4e-7],
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


def test_confirmatory_tier_summary_validates_seed_runs(tmp_path: Path) -> None:
    specs = [
        ("vcsl_public_asset_disjoint", "video_similarity", "centralized_multimodal", "local_multimodal"),
        ("vcsl_isc_confirmatory", "video_similarity", "centralized_multimodal", "local_multimodal"),
        (
            "fma_audio_20s_confirmatory",
            "audio_similarity",
            "centralized_audio_evidence",
            "local_audio_evidence",
        ),
        (
            "fma_audio_5s_confirmatory",
            "audio_similarity",
            "centralized_audio_evidence",
            "local_audio_evidence",
        ),
    ]
    common = [
        "fedavg_plain",
        "fedprox_plain",
        "fedavg_secureagg_sim",
        "fedavg_quantized_transport_proxy",
        "fedavgft_plain",
    ]
    for directory, single, centralized, local in specs:
        run = tmp_path / "new_runs" / directory
        run.mkdir(parents=True)
        methods = [single, centralized, local, *common]
        rows = [
            {"method": method, "seed": seed, "pr_auc": 0.80 + 0.001 * index + 0.0001 * seed}
            for index, method in enumerate(methods)
            for seed in (31, 37, 41)
        ]
        pd.DataFrame(rows).to_csv(run / "metrics_detection.csv", index=False, lineterminator="\n")
        (run / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "source": "test source",
                    "clients": 4,
                    "train_rows": 30,
                    "test_rows": 15,
                    "train_assets": 20,
                    "test_assets": 10,
                }
            ),
            encoding="utf-8",
        )
        config = {"seeds": [31, 37, 41], "tier": directory}
        from fedtwin.manifests import canonical_json_hash, file_records

        manifest = {
            "config": config,
            "config_sha256": canonical_json_hash(config),
            "git": {"commit": "a" * 40, "dirty": False},
            "counts": {"clients": 4, "train_rows_per_seed": 30, "test_rows_per_seed": 15},
            "outputs": file_records(
                [run / "metrics_detection.csv", run / "dataset_manifest.json"],
                relative_to=run,
            ),
        }
        (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = build_confirmatory_tier_summary(tmp_path)
    assert len(summary) == 4
    assert set(summary["evidence_status"]) == {"verified-confirmatory"}
    assert set(summary["seeds"]) == {"31;37;41"}
    assert summary.select_dtypes(include="number").map(pd.notna).all().all()


def test_communication_round_ablation_requires_complete_three_seed_trace(tmp_path: Path) -> None:
    run = tmp_path / "new_runs" / "vcsl_public_asset_disjoint"
    run.mkdir(parents=True)
    rows = []
    for seed in (31, 37, 41):
        for method, privacy_mode in (
            ("fedavg", "plain"),
            ("fedprox", "plain"),
            ("fedavg", "secureagg_sim"),
            ("fedavg", "quantized_transport_proxy"),
        ):
            for round_id in range(1, 21):
                rows.append(
                    {
                        "round": round_id,
                        "method": method,
                        "privacy_mode": privacy_mode,
                        "mean_client_loss": 1.0 / (round_id + seed),
                        "seed": seed,
                    }
                )
    pd.DataFrame(rows).to_csv(run / "metrics_federated.csv", index=False, lineterminator="\n")
    result = build_communication_round_ablation(tmp_path)
    assert len(result) == 80
    assert set(result["seed_count"]) == {3}
    assert set(result["seeds"]) == {"31;37;41"}
    figures = tmp_path / "figures"
    source = tmp_path / "communication_round_ablation.csv"
    result.to_csv(source, index=False, lineterminator="\n")
    _plot_communication_round_ablation(source, figures)
    assert (figures / "communication_round_ablation.pdf").is_file()
