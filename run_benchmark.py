from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from fedtwin.data import (
    dataframe_from_split,
    generate_fma_audio_benchmark,
    generate_synthetic_benchmark,
    generate_vcsl_isc_benchmark,
    generate_vcsl_public_benchmark,
)
from fedtwin.features import standardize_train_test
from fedtwin.federated import train_centralized, train_federated, train_local_models, train_personalized_models
from fedtwin.ledger import make_receipts
from fedtwin.manifests import write_json
from fedtwin.metrics import client_metrics, detection_metrics, scenario_metrics
from fedtwin.models import model_digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FedTwin-CryptID benchmark.")
    parser.add_argument("--tier", default="tier0", choices=["tier0", "tier1", "vcsl_public", "vcsl_isc", "fma_audio"], help="Benchmark tier.")
    parser.add_argument("--data-dir", default="data", help="Data/cache directory.")
    parser.add_argument("--run-dir", default="outputs", help="Output run directory.")
    parser.add_argument("--output-tag", default="smoke", help="Run output tag.")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--assets", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument(
        "--federated-methods",
        nargs="+",
        default=["centralized", "local", "fedavg", "fedprox", "secureagg_sim", "quantized_transport_proxy"],
    )
    parser.add_argument("--modalities", nargs="+", default=["video", "audio", "multimodal"])
    parser.add_argument("--ledger-modes", nargs="+", default=["none", "hashchain"])
    parser.add_argument("--noniid-alpha", type=float, default=0.3)
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--vcsl-metadata-dir", default="public_data/vcsl_metadata")
    parser.add_argument("--vcsl-feature-dir", default="data/vcsl_features/isc_extracted")
    parser.add_argument("--isc-max-frames", type=int, default=160)
    parser.add_argument("--fma-audio-dir", default="data/fma/fma_small")
    parser.add_argument("--fma-metadata-dir", default="data/fma/fma_metadata")
    parser.add_argument("--fma-cache-dir", default="data/fma/cache")
    parser.add_argument("--fma-max-tracks", type=int, default=1200)
    parser.add_argument("--fma-sample-rate", type=int, default=8000)
    parser.add_argument("--fma-max-seconds", type=float, default=25.0)
    parser.add_argument("--fma-max-decode-failure-fraction", type=float, default=0.05)
    parser.add_argument("--max-train-pairs", type=int, default=60000)
    parser.add_argument("--max-test-pairs", type=int, default=30000)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--feature-map", default="poly2", choices=["linear", "poly2"])
    parser.add_argument("--feature-policy", default="invariant", choices=["all", "invariant"])
    parser.add_argument("--write-pair-csv", action="store_true", help="Write expanded train/test pair feature CSV files.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _feature_atoms(name: str) -> list[str]:
    name = name.replace("__sq", "")
    return name.split("__x__")


def _feature_allowed(name: str, allowed: set[str]) -> bool:
    if name.startswith("transform_"):
        return True
    atoms = _feature_atoms(name)
    return all(atom in allowed for atom in atoms)


def _is_invariant_feature(name: str) -> bool:
    spurious_tokens = ("category_", "same_category", "metadata_score", "category_audio_score", "client_norm")
    return not any(token in name for token in spurious_tokens)


def modality_indices(feature_names: list[str], modality: str, feature_policy: str = "invariant") -> list[int]:
    names = set()
    if modality == "video":
        names.update(["video_score"])
    elif modality == "audio":
        names.update(["audio_score"])
    else:
        if feature_policy == "all":
            return list(range(len(feature_names)))
        invariant_multimodal = {
            "video_score",
            "audio_score",
            "video_rel_score",
            "audio_rel_score",
            "product_modality_score",
        }
        return [
            i
            for i, name in enumerate(feature_names)
            if _is_invariant_feature(name) and _feature_allowed(name, invariant_multimodal)
        ]
    names.update([name for name in feature_names if name.startswith("transform_")])
    names.add("client_norm")
    selected = [i for i, name in enumerate(feature_names) if _feature_allowed(name, names)]
    if feature_policy == "invariant":
        selected = [i for i in selected if _is_invariant_feature(feature_names[i])]
    return selected


def expand_feature_map(
    x_train: np.ndarray,
    x_test: np.ndarray,
    feature_names: list[str],
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if mode == "linear":
        return x_train, x_test, feature_names
    interaction_roots = [
        "video_score",
        "audio_score",
        "temporal_score",
        "video_reliability",
        "audio_reliability",
        "modality_gap",
        "fusion_prior",
        "frame_ratio",
        "video_rel_score",
        "audio_rel_score",
        "temporal_rel_score",
        "max_modality_score",
        "mean_modality_score",
        "product_modality_score",
        "frame_video_score",
    ]
    indices = [feature_names.index(name) for name in interaction_roots if name in feature_names]
    train_parts = [x_train]
    test_parts = [x_test]
    expanded_names = list(feature_names)
    for idx in indices:
        name = feature_names[idx]
        train_parts.append((x_train[:, idx] ** 2)[:, None])
        test_parts.append((x_test[:, idx] ** 2)[:, None])
        expanded_names.append(f"{name}__sq")
    for left, right in combinations(indices, 2):
        left_name = feature_names[left]
        right_name = feature_names[right]
        train_parts.append((x_train[:, left] * x_train[:, right])[:, None])
        test_parts.append((x_test[:, left] * x_test[:, right])[:, None])
        expanded_names.append(f"{left_name}__x__{right_name}")
    return np.hstack(train_parts), np.hstack(test_parts), expanded_names


def minmax_score(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


def run_one_seed(args: argparse.Namespace, seed: int, output_dir: Path) -> dict:
    start_seed = time.perf_counter()
    if args.tier == "vcsl_public":
        data = generate_vcsl_public_benchmark(
            metadata_dir=args.vcsl_metadata_dir,
            clients=args.clients,
            dim=args.dim,
            max_train_pairs=args.max_train_pairs,
            max_test_pairs=args.max_test_pairs,
            negative_ratio=args.negative_ratio,
            seed=seed,
        )
    elif args.tier == "vcsl_isc":
        data = generate_vcsl_isc_benchmark(
            metadata_dir=args.vcsl_metadata_dir,
            feature_dir=args.vcsl_feature_dir,
            clients=args.clients,
            max_train_pairs=args.max_train_pairs,
            max_test_pairs=args.max_test_pairs,
            negative_ratio=args.negative_ratio,
            max_frames=args.isc_max_frames,
            seed=seed,
        )
    elif args.tier == "fma_audio":
        data = generate_fma_audio_benchmark(
            audio_dir=args.fma_audio_dir,
            metadata_dir=args.fma_metadata_dir,
            cache_dir=args.fma_cache_dir,
            clients=args.clients,
            max_tracks=args.fma_max_tracks,
            max_train_pairs=args.max_train_pairs,
            max_test_pairs=args.max_test_pairs,
            negative_ratio=args.negative_ratio,
            sample_rate=args.fma_sample_rate,
            max_seconds=args.fma_max_seconds,
            max_decode_failure_fraction=args.fma_max_decode_failure_fraction,
            seed=seed,
        )
    else:
        data = generate_synthetic_benchmark(
            clients=args.clients,
            assets=args.assets,
            queries=args.queries,
            dim=args.dim,
            noniid_alpha=args.noniid_alpha,
            difficulty=args.difficulty,
            seed=seed,
        )

    x_train_raw, x_test_raw, feature_names = expand_feature_map(
        data.x_train,
        data.x_test,
        data.feature_names,
        mode=args.feature_map,
    )
    x_train_std, x_test_std, mean, std = standardize_train_test(x_train_raw, x_test_raw)

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if args.write_pair_csv:
        train_df = dataframe_from_split(x_train_raw, data.y_train, data.client_train, data.scenario_train, feature_names)
        test_df = dataframe_from_split(x_test_raw, data.y_test, data.client_test, data.scenario_test, feature_names)
        train_df.to_csv(seed_dir / "train_pairs.csv", index=False, lineterminator="\n")
        test_df.to_csv(seed_dir / "test_pairs.csv", index=False, lineterminator="\n")

    detection_rows: list[dict] = []
    federated_rows: list[dict] = []
    privacy_rows: list[dict] = []
    scenario_frames: list[pd.DataFrame] = []
    client_frames: list[pd.DataFrame] = []
    ledger_rows: list[dict] = []
    localization_rows: list[dict] = []
    runtime_rows: list[dict] = []

    y_test = data.y_test
    video_score = minmax_score(0.65 * x_test_raw[:, feature_names.index("video_score")] + 0.35 * x_test_raw[:, feature_names.index("temporal_score")])
    audio_score = minmax_score(x_test_raw[:, feature_names.index("audio_score")])
    fusion_score = minmax_score(0.5 * video_score + 0.5 * audio_score)

    simple_scores = {
        "random": np.full_like(y_test, float(np.mean(data.y_train)), dtype=float),
        "audio_similarity": audio_score,
    }
    if args.tier != "fma_audio":
        simple_scores.update(
            {
                "video_similarity": video_score,
                "early_fusion_similarity": fusion_score,
            }
        )
    for method, score in simple_scores.items():
        modality = (
            "audio"
            if args.tier == "fma_audio"
            else "video"
            if "video" in method
            else "audio"
            if "audio" in method
            else "multimodal"
        )
        row = detection_metrics(y_test, score, method=method, modality=modality)
        row["seed"] = seed
        detection_rows.append(row)
        scenario_df = scenario_metrics(y_test, score, data.scenario_test, method=method)
        scenario_df["seed"] = seed
        scenario_frames.append(scenario_df)

    trained_models: dict[str, tuple[np.ndarray, str]] = {}
    if "centralized" in args.federated_methods:
        for modality in args.modalities:
            idx = modality_indices(feature_names, modality, args.feature_policy)
            model, summary = train_centralized(x_train_std[:, idx], data.y_train, epochs=140)
            score = model.predict_proba(x_test_std[:, idx])
            method_name = (
                "centralized_audio_evidence"
                if args.tier == "fma_audio" and modality == "multimodal"
                else f"centralized_{modality}"
            )
            reported_modality = "audio" if args.tier == "fma_audio" else modality
            trained_models[method_name] = (score, model_digest(model.weights))
            row = detection_metrics(y_test, score, method=method_name, modality=reported_modality)
            row["seed"] = seed
            detection_rows.append(row)
            summary.update({"seed": seed, "method": method_name, "privacy_mode": "none"})
            runtime_rows.append(summary)

    if "local" in args.federated_methods:
        feature_idx = modality_indices(feature_names, "multimodal", args.feature_policy)
        model_map, summary = train_local_models(x_train_std[:, feature_idx], data.y_train, data.client_train)
        score = np.zeros_like(y_test, dtype=float)
        for client_id in sorted(model_map):
            client_idx = data.client_test == client_id
            score[client_idx] = model_map[client_id].predict_proba(x_test_std[client_idx][:, feature_idx])
        digest = model_digest(np.mean([m.weights for m in model_map.values()], axis=0))
        local_method_name = "local_audio_evidence" if args.tier == "fma_audio" else "local_multimodal"
        trained_models[local_method_name] = (score, digest)
        reported_modality = "audio" if args.tier == "fma_audio" else "multimodal"
        row = detection_metrics(y_test, score, method=local_method_name, modality=reported_modality)
        row["seed"] = seed
        detection_rows.append(row)
        summary.update({"seed": seed, "method": local_method_name, "privacy_mode": "local"})
        runtime_rows.append(summary)

    fed_specs = []
    if "fedavg" in args.federated_methods:
        fed_specs.append(("fedavg", "plain", 0.0))
    if "fedprox" in args.federated_methods:
        fed_specs.append(("fedprox", "plain", 0.02))
    if {"secureagg", "secureagg_sim"} & set(args.federated_methods):
        fed_specs.append(("fedavg", "secureagg_sim", 0.0))
    if {"heagg", "quantized_transport_proxy"} & set(args.federated_methods):
        fed_specs.append(("fedavg", "quantized_transport_proxy", 0.0))

    for method, privacy_mode, prox_mu in fed_specs:
        feature_idx = modality_indices(feature_names, "multimodal", args.feature_policy)
        model, summary, round_logs = train_federated(
            x_train_std[:, feature_idx],
            data.y_train,
            data.client_train,
            method=method,
            privacy_mode=privacy_mode,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            prox_mu=prox_mu,
            seed=seed,
        )
        score = model.predict_proba(x_test_std[:, feature_idx])
        method_name = f"{method}_{privacy_mode}"
        trained_models[method_name] = (score, model_digest(model.weights))
        reported_modality = "audio" if args.tier == "fma_audio" else "multimodal"
        row = detection_metrics(y_test, score, method=method_name, modality=reported_modality, privacy_mode=privacy_mode)
        row["seed"] = seed
        detection_rows.append(row)

        client_df = client_metrics(y_test, score, data.client_test, method=method_name)
        client_df["seed"] = seed
        client_frames.append(client_df)
        scenario_df = scenario_metrics(y_test, score, data.scenario_test, method=method_name)
        scenario_df["seed"] = seed
        scenario_frames.append(scenario_df)

        for log in round_logs:
            log = dict(log)
            log["seed"] = seed
            federated_rows.append(log)
        privacy = {
            "seed": seed,
            "method": method_name,
            "privacy_mode": privacy_mode,
            "raw_media_shared": 0,
            "raw_embeddings_on_ledger": 0,
            "mean_encryption_time_sec": summary["mean_encryption_time_sec"],
            "mean_aggregation_time_sec": summary["mean_aggregation_time_sec"],
            "ciphertext_expansion": summary["mean_ciphertext_expansion"],
        }
        privacy_rows.append(privacy)
        summary.update({"seed": seed, "method": method_name})
        runtime_rows.append(summary)

    if "fedavgft" in args.federated_methods:
        feature_idx = modality_indices(feature_names, "multimodal", args.feature_policy)
        base_model, base_summary, round_logs = train_federated(
            x_train_std[:, feature_idx],
            data.y_train,
            data.client_train,
            method="fedavg",
            privacy_mode="plain",
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            seed=seed,
        )
        model_map, ft_summary = train_personalized_models(
            x_train_std[:, feature_idx],
            data.y_train,
            data.client_train,
            base_model.weights,
        )
        score = np.zeros_like(y_test, dtype=float)
        for client_id in sorted(model_map):
            client_idx = data.client_test == client_id
            score[client_idx] = model_map[client_id].predict_proba(x_test_std[client_idx][:, feature_idx])
        digest = model_digest(np.mean([m.weights for m in model_map.values()], axis=0))
        trained_models["fedavgft_plain"] = (score, digest)
        reported_modality = "audio" if args.tier == "fma_audio" else "multimodal"
        row = detection_metrics(y_test, score, method="fedavgft_plain", modality=reported_modality, privacy_mode="plain")
        row["seed"] = seed
        detection_rows.append(row)
        client_df = client_metrics(y_test, score, data.client_test, method="fedavgft_plain")
        client_df["seed"] = seed
        client_frames.append(client_df)
        scenario_df = scenario_metrics(y_test, score, data.scenario_test, method="fedavgft_plain")
        scenario_df["seed"] = seed
        scenario_frames.append(scenario_df)
        for log in round_logs:
            log = dict(log)
            log["seed"] = seed
            log["method"] = "fedavgft_base"
            federated_rows.append(log)
        ft_summary.update(
            {
                "seed": seed,
                "method": "fedavgft_plain",
                "privacy_mode": "plain",
                "base_runtime_sec": base_summary["runtime_sec"],
            }
        )
        runtime_rows.append(ft_summary)

    if "hashchain" in args.ledger_modes:
        if "fedavg_secureagg_sim" in trained_models:
            ledger_score, digest = trained_models["fedavg_secureagg_sim"]
            ledger_method = "fedavg_secureagg_sim"
        elif "fedavg_quantized_transport_proxy" in trained_models:
            ledger_score, digest = trained_models["fedavg_quantized_transport_proxy"]
            ledger_method = "fedavg_quantized_transport_proxy"
        else:
            ledger_score, digest = fusion_score, "similarity"
            ledger_method = "early_fusion_similarity"
        _, ledger_summary = make_receipts(
            scores=ledger_score,
            y=y_test,
            clients=data.client_test,
            model_digest=digest,
            limit=min(250, len(y_test)),
        )
        ledger_summary.update({"seed": seed, "method": ledger_method})
        ledger_rows.append(ledger_summary)

    for method_name, (score, _) in trained_models.items():
        partial_idx = data.scenario_test == "partial_segment"
        if partial_idx.any():
            row = detection_metrics(y_test[partial_idx], score[partial_idx], method=method_name, modality="multimodal")
            localization_rows.append(
                {
                    "seed": seed,
                    "method": method_name,
                    "segment_f1_proxy": row["recall"],
                    "segment_precision_proxy": row["precision"],
                    "partial_segment_n": row["n"],
                }
            )

    write_json(seed_dir / "dataset_manifest.json", data.manifest)
    write_json(seed_dir / "client_split_manifest.json", {"clients": args.clients, "noniid_alpha": args.noniid_alpha, "seed": seed})
    write_json(
        seed_dir / "transformation_manifest.json",
        {
            "transformations": data.manifest.get("transformations", sorted(set(map(str, data.scenario_train)) | set(map(str, data.scenario_test)))),
            "source": data.manifest.get("generator", "unknown"),
            "feature_map": args.feature_map,
            "feature_policy": args.feature_policy,
            "expanded_feature_count": len(feature_names),
            "feature_names": feature_names,
        },
    )

    seed_summary = {
        "seed": seed,
        "runtime_sec": time.perf_counter() - start_seed,
        "train_rows": int(len(data.y_train)),
        "test_rows": int(len(data.y_test)),
        "best_pr_auc": float(pd.DataFrame(detection_rows)["pr_auc"].max()),
    }
    return {
        "summary": seed_summary,
        "detection": detection_rows,
        "federated": federated_rows,
        "privacy": privacy_rows,
        "ledger": ledger_rows,
        "localization": localization_rows,
        "runtime": runtime_rows,
        "scenario": scenario_frames,
        "client": client_frames,
        "manifest": data.manifest,
        "feature_names": feature_names,
    }


def run_benchmark(args: argparse.Namespace) -> Path:
    output_root = Path(args.run_dir) / args.output_tag
    output_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    write_json(output_root / "run_config.json", vars(args))

    all_detection = []
    all_federated = []
    all_privacy = []
    all_ledger = []
    all_localization = []
    all_runtime = []
    all_scenario = []
    all_client = []
    seed_summaries = []
    manifest = None
    feature_schema: list[str] = []

    for seed_pos, seed in enumerate(args.seeds, start=1):
        print(f"[{seed_pos}/{len(args.seeds)}] running seed {seed} for tier {args.tier}", flush=True)
        result = run_one_seed(args, seed, output_root)
        print(f"[{seed_pos}/{len(args.seeds)}] completed seed {seed}", flush=True)
        seed_summaries.append(result["summary"])
        all_detection.extend(result["detection"])
        all_federated.extend(result["federated"])
        all_privacy.extend(result["privacy"])
        all_ledger.extend(result["ledger"])
        all_localization.extend(result["localization"])
        all_runtime.extend(result["runtime"])
        all_scenario.extend(result["scenario"])
        all_client.extend(result["client"])
        manifest = result["manifest"]
        feature_schema = result["feature_names"]

    pd.DataFrame(all_detection).to_csv(output_root / "metrics_detection.csv", index=False, lineterminator="\n")
    pd.DataFrame(all_federated).to_csv(output_root / "metrics_federated.csv", index=False, lineterminator="\n")
    pd.DataFrame(all_privacy).to_csv(output_root / "metrics_privacy.csv", index=False, lineterminator="\n")
    pd.DataFrame(all_ledger).to_csv(output_root / "metrics_ledger.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        all_localization,
        columns=["seed", "method", "segment_f1_proxy", "segment_precision_proxy", "partial_segment_n"],
    ).to_csv(output_root / "metrics_localization.csv", index=False, lineterminator="\n")
    pd.DataFrame(all_runtime).to_csv(output_root / "runtime_profile.csv", index=False, lineterminator="\n")
    if all_scenario:
        pd.concat(all_scenario, ignore_index=True).to_csv(
            output_root / "metrics_scenario.csv", index=False, lineterminator="\n"
        )
    if all_client:
        pd.concat(all_client, ignore_index=True).to_csv(
            output_root / "metrics_client.csv", index=False, lineterminator="\n"
        )
    if manifest is not None:
        write_json(output_root / "dataset_manifest.json", manifest)
        write_json(output_root / "client_split_manifest.json", {"clients": args.clients, "noniid_alpha": args.noniid_alpha, "seeds": args.seeds})
        write_json(
            output_root / "transformation_manifest.json",
            {
                "transformations": manifest.get("transformations", []),
                "source": manifest.get("generator", "unknown"),
                "feature_map": args.feature_map,
                "feature_policy": args.feature_policy,
                "expanded_feature_count": len(feature_schema),
                "feature_names": feature_schema,
            },
        )

    detection_df = pd.DataFrame(all_detection)
    summary = {
        "output_tag": args.output_tag,
        "tier": args.tier,
        "total_runtime_sec": time.perf_counter() - start,
        "seeds": args.seeds,
        "seed_summaries": seed_summaries,
        "best_method_by_pr_auc": detection_df.sort_values("pr_auc", ascending=False).iloc[0].to_dict() if len(detection_df) else {},
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return output_root


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_benchmark(args)


if __name__ == "__main__":
    main()
