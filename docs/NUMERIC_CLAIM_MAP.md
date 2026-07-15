# Numerical claim map

Every reported number must be recoverable from a released result row and a
manifested source. `paper_results/v1/manifest.json` is the integrity root, and
`paper_results/v1/claims.csv` is the compact claim registry. This map gives the
reviewer-facing route from manuscript content to released evidence.

| Manuscript claim or asset | Released table or source | Validation boundary |
|---|---|---|
| Abstract and main multi-tier PR-AUC results | `tables/tier_result_summary.csv`; per-tier `new_runs/*/metrics_detection.csv` | Three seeds; confirmatory asset-disjoint VCSL and FMA tiers |
| Dataset, pair, asset, query, client, and decode counts | `tables/tier_result_summary.csv`; per-tier `dataset_manifest.json`, `run_manifest.json`, and configuration | Counts are per recorded run; restricted media are not redistributed |
| Feature-policy comparison and pooled effect | `tables/fixed_evidence_baselines.csv`, `tables/feature_ablation.csv`, `tables/paired_bootstrap_tests.csv`, `tables/paired_permutation_tests.csv`, and `new_runs/vcsl_feature_policy_confirmatory/` | Fixed held-out examples; paired bootstrap/permutation with Holm correction |
| Query ranking and 1:100--1:10,000 stress | `tables/query_ranking_metrics.csv` | Candidate-instance pools; hardest tier has 100 positive-bearing queries and repeated references/noise realizations |
| FPR, recall-at-budget, calibration, and workload | `tables/operating_point_summary.csv`, `tables/prevalence_workload_metrics.csv`, and per-tier `metrics_detection.csv` | Human-review triage interpretation only |
| Threshold and reliability diagnostics | `tables/threshold_stability_by_client.csv` and `tables/calibration_reliability_curves.csv` | Diagnostic ablations, not a universal threshold claim |
| Client heterogeneity and personalization | `tables/per_client_metrics.csv`, `tables/client_variability_points.csv`, `tables/personalization_strengthening.csv`, and per-tier metrics | Frozen client assignment within each recorded tier |
| Communication-round convergence | `tables/communication_round_ablation.csv`; source `new_runs/vcsl_public_asset_disjoint/metrics_federated.csv` | Complete rounds 1--20 for seeds 31, 37, and 41 |
| Poisoning and robust aggregation | `tables/robustness_stress.csv` | Controlled 20-round compact-head stress with 0/10/20 percent attackers |
| SecureAgg dropout behavior | `tables/secureagg_dropout_or_proxy.csv` | Arithmetic mask-cancellation/recovery simulation at 0/10/20 percent dropout; not a security proof |
| Quantized transport accounting | `tables/quantized_transport_proxy.csv` and `tables/privacy_utility_points.csv` | Numeric/byte proxy only; not encryption |
| Real Paillier compact-update sums | `tables/paillier_update_aggregation.csv` | Three separately generated 2048-bit keys for seeds 31, 37, and 41; no encrypted media inference |
| Privacy/leakage diagnostics | `tables/privacy_attack_results.csv` and `tables/privacy_leakage.csv` | Empirical attacks and context leakage; no differential-privacy claim |
| Receipt verification and tampering | `tables/receipt_test_vector.csv`, `tables/receipt_v2_summary.csv`, and `tables/receipt_scaling.csv` | Signed, salted, media-free receipt chain; no ownership adjudication claim |
| Separate visual/audio evidence and synthetic modality conflict | `tables/tier_result_summary.csv`, `tables/av_sync_tier_metrics.csv`, and `tables/av_sync_operating_points.csv` | Separate natural visual/audio tiers plus synthetic stress; no synchronized natural A/V claim |

Figures are regenerated from these tables by
`python -m fedtwin.cli regenerate-paper`. A regenerated scientific table must
match the released table within `1e-4`; deterministic CSV/JSON scientific
fields are checked exactly, excluding wall-clock timing and latency provenance.
