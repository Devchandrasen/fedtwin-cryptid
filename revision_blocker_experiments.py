from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, ndcg_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT / "paper"
TABLES = MANUSCRIPT / "tables"
FIGURES = MANUSCRIPT / "figures"
RESULTS = ROOT / "outputs" / "review_blocker_experiments"
PUBLIC_META = ROOT / "public_data" / "vcsl_metadata"
PULL = ROOT / "archived_results"
CONFIRMATORY = ROOT / "paper_results" / "v1" / "new_runs"

sys.path.insert(0, str(ROOT))

from fedtwin.crypto import aggregate_updates, paillier_aggregate_updates, secureagg_simulate  # noqa: E402
from fedtwin.data import (  # noqa: E402
    _build_vcsl_groups,
    _clean_vcsl_positive_pairs,
    _load_vcsl_metadata,
    _sample_vcsl_pairs,
    _vcsl_pair_to_features,
)
from fedtwin.features import standardize_train_test  # noqa: E402
from fedtwin.federated import train_federated, train_local_models, train_personalized_models  # noqa: E402
from fedtwin.ledger import HashChainLedger, canonical_hash, sha256_text  # noqa: E402
from fedtwin.manifests import write_json  # noqa: E402
from fedtwin.metrics import detection_metrics, present_class_balanced_accuracy  # noqa: E402
from fedtwin.models import LogisticHead, train_logistic  # noqa: E402
from fedtwin.pairs import construct_asset_disjoint_pairs  # noqa: E402
from fedtwin.statistics import holm_adjust, paired_bootstrap_delta, paired_permutation_delta  # noqa: E402
from run_benchmark import expand_feature_map, minmax_score  # noqa: E402
from strengthen_for_tmm import select_indices, train_fedopt  # noqa: E402

SEEDS = [31, 37, 41]
PREVALENCES = [1e-2, 1e-3, 1e-5]
REVIEW_BUDGETS = [1, 5, 10, 50, 100]
PALETTE = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "purple": "#B279A2",
    "cyan": "#72B7B2",
    "dark": "#2F3A45",
    "gray": "#8E9AAF",
}


def configure_paths(args: argparse.Namespace) -> None:
    """Bind every input/output path from explicit CLI arguments."""

    global MANUSCRIPT, TABLES, FIGURES, RESULTS, PUBLIC_META, PULL, CONFIRMATORY
    MANUSCRIPT = Path(args.manuscript_dir).resolve()
    TABLES = MANUSCRIPT / "tables"
    FIGURES = MANUSCRIPT / "figures"
    RESULTS = Path(args.output_dir).resolve()
    PUBLIC_META = Path(args.vcsl_metadata_dir).resolve()
    PULL = Path(args.archived_results_dir).resolve()
    CONFIRMATORY = Path(args.confirmatory_results_dir).resolve()


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def setup_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(FIGURES / f"{name}.{ext}", dpi=320, bbox_inches="tight")
    plt.close()


def expected_precision(prevalence: float, recall: float, fpr: float) -> float:
    tp = prevalence * recall
    fp = (1.0 - prevalence) * fpr
    return float(tp / max(tp + fp, 1e-15))


def ece_score(y: np.ndarray, score: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (score >= lo) & (score < hi if hi < 1.0 else score <= hi)
        if not np.any(mask):
            continue
        out += (mask.sum() / total) * abs(float(score[mask].mean()) - float(y[mask].mean()))
    return float(out)


def full_metrics(y: np.ndarray, score: np.ndarray, *, method: str, tier: str, seed: int) -> dict:
    base = detection_metrics(y, score, method=method, modality="multimodal")
    base.update(
        {
            "tier": tier,
            "seed": seed,
            "brier": float(brier_score_loss(y, np.clip(score, 1e-7, 1 - 1e-7))),
            "ece": ece_score(y, score),
        }
    )
    if np.isfinite(base["fpr_at_95_recall"]):
        for prevalence in PREVALENCES:
            base[f"expected_precision_prev_{prevalence:g}"] = expected_precision(
                prevalence, base["recall"], base["fpr_at_95_recall"]
            )
            base[f"reviews_per_true_positive_prev_{prevalence:g}"] = 1.0 / max(
                base[f"expected_precision_prev_{prevalence:g}"], 1e-15
            )
    return base


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in set(keys) | {"seed"} and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for group_values, group in df.groupby(keys, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(keys, group_values, strict=True))
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        row["seeds"] = ",".join(map(str, sorted(group["seed"].astype(int).unique()))) if "seed" in group else ""
        rows.append(row)
    return pd.DataFrame(rows)


def load_public_pair_dataset(
    *,
    seed: int,
    max_train_pairs: int,
    max_test_pairs: int,
    negative_ratio: float,
    train_negative_ratio: float | None = None,
    min_test_queries: int = 0,
    clients: int = 11,
    dim: int = 64,
) -> dict:
    train, val, test, category_by_uuid, frame_count = _load_vcsl_metadata(PUBLIC_META)
    all_positive_pairs = _clean_vcsl_positive_pairs(train, val, test)
    uf = _build_vcsl_groups(all_positive_pairs)
    all_ids = sorted(set(frame_count) | set(category_by_uuid))
    split_pairs = construct_asset_disjoint_pairs(
        all_positive_pairs,
        asset_ids=all_ids,
        negative_ratio=0.0,
        test_fraction=0.2,
        seed=seed,
    )
    train_assets = sorted(set(split_pairs.train["query_id"]) | set(split_pairs.train["reference_id"]))
    test_assets = sorted(set(split_pairs.test["query_id"]) | set(split_pairs.test["reference_id"]))
    rng = np.random.default_rng(seed)
    train_pairs = _sample_vcsl_pairs(
        split_pairs.train,
        train_assets,
        category_by_uuid,
        uf,
        rng,
        max_positive_pairs=max_train_pairs,
        negative_ratio=negative_ratio if train_negative_ratio is None else train_negative_ratio,
    )
    test_pairs = _sample_vcsl_pairs(
        split_pairs.test,
        test_assets,
        category_by_uuid,
        uf,
        rng,
        max_positive_pairs=max_test_pairs,
        negative_ratio=negative_ratio,
        min_positive_queries=min_test_queries,
    )
    x_train, y_train, client_train, scenario_train, feature_names = _vcsl_pair_to_features(
        train_pairs, category_by_uuid, frame_count, uf, clients=clients, dim=dim, seed=seed
    )
    x_test, y_test, client_test, scenario_test, _ = _vcsl_pair_to_features(
        test_pairs, category_by_uuid, frame_count, uf, clients=clients, dim=dim, seed=seed
    )
    train_pairs = train_pairs.copy()
    test_pairs = test_pairs.copy()
    train_pairs["query_category"] = train_pairs["query_id"].astype(str).map(lambda x: category_by_uuid.get(x, "unknown"))
    test_pairs["query_category"] = test_pairs["query_id"].astype(str).map(lambda x: category_by_uuid.get(x, "unknown"))
    return {
        "x_train": x_train,
        "y_train": y_train,
        "client_train": client_train,
        "scenario_train": scenario_train,
        "train_pairs": train_pairs,
        "x_test": x_test,
        "y_test": y_test,
        "client_test": client_test,
        "scenario_test": scenario_test,
        "test_pairs": test_pairs,
        "feature_names": feature_names,
        "category_by_uuid": category_by_uuid,
    }


def prepare_features(ds: dict, *, feature_map: str = "poly2") -> dict:
    x_train_raw, x_test_raw, feature_names = expand_feature_map(ds["x_train"], ds["x_test"], ds["feature_names"], mode=feature_map)
    x_train_std, x_test_std, mean, std = standardize_train_test(x_train_raw, x_test_raw)
    out = dict(ds)
    out.update(
        {
            "x_train_raw_expanded": x_train_raw,
            "x_test_raw_expanded": x_test_raw,
            "x_train_std": x_train_std,
            "x_test_std": x_test_std,
            "expanded_feature_names": feature_names,
            "standardization_mean": mean,
            "standardization_std": std,
        }
    )
    return out


def fit_predict_methods(ds: dict, *, include_federated: bool = True) -> dict[str, np.ndarray]:
    names = ds["expanded_feature_names"]
    y_train = ds["y_train"]
    x_train = ds["x_train_std"]
    x_test = ds["x_test_std"]
    raw_train = ds["x_train_raw_expanded"]
    raw_test = ds["x_test_raw_expanded"]
    scores: dict[str, np.ndarray] = {}

    score_idx = select_indices(names, "score_only")
    inv_idx = select_indices(names, "invariant")
    scores["score_only_logistic"] = train_logistic(x_train[:, score_idx], y_train, epochs=140).predict_proba(x_test[:, score_idx])
    scores["invariant_logistic_poly2"] = train_logistic(x_train[:, inv_idx], y_train, epochs=140).predict_proba(x_test[:, inv_idx])

    mean_train = minmax_score(0.5 * raw_train[:, names.index("video_score")] + 0.5 * raw_train[:, names.index("audio_score")])
    mean_test = minmax_score(0.5 * raw_test[:, names.index("video_score")] + 0.5 * raw_test[:, names.index("audio_score")])
    scores["mean_score_fusion"] = mean_test
    platt = LogisticRegression(max_iter=500, solver="lbfgs")
    platt.fit(mean_train.reshape(-1, 1), y_train)
    scores["platt_calibrated_score"] = platt.predict_proba(mean_test.reshape(-1, 1))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(mean_train, y_train)
    scores["isotonic_calibrated_score"] = iso.predict(mean_test)

    if include_federated:
        fed = train_federated(
            x_train[:, inv_idx],
            y_train,
            ds["client_train"],
            method="fedavg",
            privacy_mode="plain",
            rounds=30,
            local_epochs=5,
        )[0]
        scores["fedavg_plain"] = fed.predict_proba(x_test[:, inv_idx])
        fedprox = train_federated(
            x_train[:, inv_idx],
            y_train,
            ds["client_train"],
            method="fedprox",
            privacy_mode="plain",
            rounds=30,
            local_epochs=5,
            prox_mu=0.02,
        )[0]
        scores["fedprox_plain"] = fedprox.predict_proba(x_test[:, inv_idx])
        scores["fedyogi"] = train_fedopt(x_train[:, inv_idx], y_train, ds["client_train"], optimizer="fedyogi").predict_proba(x_test[:, inv_idx])
        scores["fedadam"] = train_fedopt(x_train[:, inv_idx], y_train, ds["client_train"], optimizer="fedadam").predict_proba(x_test[:, inv_idx])

    return scores


def _clustered_mean_interval(values: list[float], *, seed: int, resamples: int = 1000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    for index in range(resamples):
        means[index] = float(array[rng.integers(0, len(array), size=len(array))].mean())
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def query_metrics(pair_df: pd.DataFrame, y: np.ndarray, score: np.ndarray, *, method: str, tier: str, seed: int, candidate_ratio: int) -> dict:
    frame = pair_df[["query_id", "reference_id"]].copy()
    frame["label"] = y.astype(int)
    frame["score"] = score.astype(float)
    recalls = {1: [], 5: [], 10: [], 100: []}
    precisions = {k: [] for k in REVIEW_BUDGETS}
    ap_values = []
    ndcg_values = []
    queries = 0
    for _, group in frame.groupby("query_id", sort=False):
        if group["label"].sum() == 0:
            continue
        queries += 1
        ranked = group.sort_values("score", ascending=False)
        labels = ranked["label"].to_numpy(dtype=int)
        scores = ranked["score"].to_numpy(dtype=float)
        for k in recalls:
            recalls[k].append(float(labels[: min(k, len(labels))].max()))
        for k in precisions:
            top = labels[: min(k, len(labels))]
            precisions[k].append(float(top.sum() / max(len(top), 1)))
        ap_values.append(float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else float(labels[0]))
        try:
            ndcg_values.append(float(ndcg_score(labels.reshape(1, -1), scores.reshape(1, -1), k=min(10, len(labels)))))
        except Exception:
            ndcg_values.append(float("nan"))
    result = {
        "tier": tier,
        "method": method,
        "seed": seed,
        "candidate_negative_ratio": candidate_ratio,
        "query_count": queries,
        "candidate_pairs": int(len(frame)),
        "positive_candidates": int(np.sum(y)),
        "negative_candidates": int(len(y) - np.sum(y)),
        "unique_query_reference_pairs": int(frame.drop_duplicates(["query_id", "reference_id"]).shape[0]),
        "recall_at_1": float(np.mean(recalls[1])) if recalls[1] else float("nan"),
        "recall_at_5": float(np.mean(recalls[5])) if recalls[5] else float("nan"),
        "recall_at_10": float(np.mean(recalls[10])) if recalls[10] else float("nan"),
        "recall_at_100": float(np.mean(recalls[100])) if recalls[100] else float("nan"),
        "precision_at_1": float(np.mean(precisions[1])) if precisions[1] else float("nan"),
        "precision_at_5": float(np.mean(precisions[5])) if precisions[5] else float("nan"),
        "precision_at_10": float(np.mean(precisions[10])) if precisions[10] else float("nan"),
        "precision_at_50": float(np.mean(precisions[50])) if precisions[50] else float("nan"),
        "precision_at_100": float(np.mean(precisions[100])) if precisions[100] else float("nan"),
        "query_map": float(np.nanmean(ap_values)) if ap_values else float("nan"),
        "ndcg_at_10": float(np.nanmean(ndcg_values)) if ndcg_values else float("nan"),
        "split_note": "positive-connected-component asset-disjoint split; zero query/reference identity overlap",
        "segment_overlap_metric": "not_available_from_public_label_topology_features",
    }
    interval_inputs = {
        "recall_at_1": recalls[1],
        "recall_at_5": recalls[5],
        "recall_at_10": recalls[10],
        "recall_at_100": recalls[100],
        "precision_at_1": precisions[1],
        "precision_at_5": precisions[5],
        "precision_at_10": precisions[10],
        "precision_at_50": precisions[50],
        "precision_at_100": precisions[100],
        "query_map": ap_values,
        "ndcg_at_10": ndcg_values,
    }
    digest = hashlib.sha256(f"{method}:{candidate_ratio}:{seed}".encode()).digest()
    base_seed = int.from_bytes(digest[:4], "big")
    for offset, (name, values) in enumerate(interval_inputs.items()):
        low, high = _clustered_mean_interval(values, seed=base_seed + offset)
        result[f"{name}_query_bootstrap_ci95_low"] = low
        result[f"{name}_query_bootstrap_ci95_high"] = high
    return result


def run_query_and_open_set(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranking_rows = []
    open_rows = []
    workload_rows = []
    ratios = [1, 10, 100] if args.mode == "smoke" else [1, 10, 100, 1000, 10000]
    test_caps = {
        1: args.max_test_pairs,
        10: min(args.max_test_pairs, 650),
        100: min(args.max_test_pairs, 220),
        1000: max(args.min_open_set_queries, min(args.max_test_pairs, 100)),
        10000: max(args.min_open_set_queries, min(args.max_test_pairs, 100)),
    }
    for seed in args.seeds:
        train_ds = prepare_features(
            load_public_pair_dataset(
                seed=seed,
                max_train_pairs=args.max_train_pairs,
                max_test_pairs=args.max_test_pairs,
                negative_ratio=1.0,
            )
        )
        base_scores = fit_predict_methods(train_ds, include_federated=False)
        for ratio in ratios:
            if ratio == 1:
                test_ds = train_ds
                scores = base_scores
            else:
                test_ds = prepare_features(
                    load_public_pair_dataset(
                        seed=seed,
                        max_train_pairs=args.max_train_pairs,
                        max_test_pairs=test_caps[ratio],
                        negative_ratio=float(ratio),
                        train_negative_ratio=1.0,
                        min_test_queries=args.min_open_set_queries if ratio >= 1000 else 0,
                    )
                )
                scores = fit_predict_methods(test_ds, include_federated=False)
            eligible_queries = int(
                test_ds["test_pairs"].loc[test_ds["test_pairs"]["label"].eq(1), "query_id"].astype(str).nunique()
            )
            if ratio == 10000 and eligible_queries < args.min_open_set_queries:
                raise RuntimeError(
                    f"1:10,000 evaluation has {eligible_queries} positive-bearing queries; "
                    f"required at least {args.min_open_set_queries}"
                )
            for method, score in scores.items():
                if method not in {"score_only_logistic", "invariant_logistic_poly2", "platt_calibrated_score", "isotonic_calibrated_score", "mean_score_fusion"}:
                    continue
                ranking_rows.append(
                    query_metrics(
                        test_ds["test_pairs"],
                        test_ds["y_test"],
                        score,
                        method=method,
                        tier="VCSL public-label audit",
                        seed=seed,
                        candidate_ratio=ratio,
                    )
                )
                metric = full_metrics(test_ds["y_test"], score, method=method, tier="VCSL public-label audit", seed=seed)
                metric["candidate_negative_ratio"] = ratio
                metric["empirical_prevalence"] = float(test_ds["y_test"].mean())
                open_rows.append(metric)
                for prevalence in PREVALENCES:
                    precision = expected_precision(prevalence, 0.95, metric["fpr_at_95_recall"])
                    workload_rows.append(
                        {
                            "tier": "VCSL public-label audit",
                            "method": method,
                            "seed": seed,
                            "candidate_negative_ratio": ratio,
                            "prevalence": prevalence,
                            "fpr_at_95_recall": metric["fpr_at_95_recall"],
                            "expected_precision_at_95_recall": precision,
                            "expected_reviews_per_true_positive": 1.0 / max(precision, 1e-15),
                        }
                    )
    ranking = summarize(pd.DataFrame(ranking_rows), ["tier", "method", "candidate_negative_ratio", "split_note", "segment_overlap_metric"])
    open_set = summarize(pd.DataFrame(open_rows), ["tier", "method", "candidate_negative_ratio"])
    workload = summarize(pd.DataFrame(workload_rows), ["tier", "method", "candidate_negative_ratio", "prevalence"])
    ranking.to_csv(TABLES / "query_ranking_metrics.csv", index=False, lineterminator="\n")
    open_set.to_csv(TABLES / "open_set_prevalence_metrics.csv", index=False, lineterminator="\n")
    workload.to_csv(TABLES / "prevalence_workload_metrics.csv", index=False, lineterminator="\n")
    return ranking, open_set, workload


def _reliability_bins(y: np.ndarray, score: np.ndarray, *, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (score >= lo) & (score < hi if hi < 1.0 else score <= hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": idx,
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "count": int(mask.sum()),
                "mean_confidence": float(score[mask].mean()),
                "empirical_positive_rate": float(y[mask].mean()),
            }
        )
    return rows


def _threshold_at_recall(y: np.ndarray, score: np.ndarray, target: float = 0.95) -> float:
    positives = int(np.sum(y == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-score)
    tp = np.cumsum(y[order] == 1)
    recall = tp / max(positives, 1)
    idx = np.where(recall >= target)[0]
    if len(idx) == 0:
        return float("-inf")
    return float(score[order[idx[0]]])


def _bootstrap_and_permutation_delta(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    metric_name: str,
    seed: int,
    groups: np.ndarray,
    n_iter: int = 1000,
    comparison: str,
) -> dict:
    metric = average_precision_score if metric_name == "pr_auc" else roc_auc_score
    bootstrap = paired_bootstrap_delta(
        y,
        score_a,
        score_b,
        metric=metric,
        groups=groups,
        resamples=n_iter,
        seed=seed,
    )
    permutation = paired_permutation_delta(
        y,
        score_a,
        score_b,
        metric=metric,
        permutations=n_iter,
        seed=seed + 1009,
    )
    return {
        "comparison": comparison,
        "metric": metric_name,
        "seed": seed,
        "observed_delta": bootstrap["delta"],
        "bootstrap_ci_low": bootstrap["ci_low"],
        "bootstrap_ci_high": bootstrap["ci_high"],
        "bootstrap_clusters": bootstrap["clusters"],
        "paired_permutation_p_value": permutation["p_two_sided"],
        "n_bootstrap": bootstrap["resamples"],
        "n_permutations": permutation["permutations"],
    }


def run_calibration_and_stat_tests(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reliability_rows = []
    threshold_rows = []
    stat_rows = []
    for seed in args.seeds:
        ds = prepare_features(
            load_public_pair_dataset(
                seed=seed,
                max_train_pairs=args.max_train_pairs,
                max_test_pairs=args.max_test_pairs,
                negative_ratio=1.0,
            )
        )
        scores = fit_predict_methods(ds, include_federated=False)
        for method in ["score_only_logistic", "invariant_logistic_poly2", "platt_calibrated_score", "isotonic_calibrated_score"]:
            score = scores[method]
            for row in _reliability_bins(ds["y_test"], score, bins=10):
                row.update({"tier": "VCSL public-label audit", "method": method, "seed": seed})
                reliability_rows.append(row)
            for client_id in sorted(map(int, np.unique(ds["client_test"]))):
                mask = ds["client_test"] == client_id
                threshold_rows.append(
                    {
                        "tier": "VCSL public-label audit",
                        "method": method,
                        "seed": seed,
                        "client_id": client_id,
                        "client_test_pairs": int(mask.sum()),
                        "client_positive_rate": float(ds["y_test"][mask].mean()) if np.any(mask) else np.nan,
                        "threshold_at_95_recall": _threshold_at_recall(ds["y_test"][mask], score[mask], target=0.95),
                        "brier": float(brier_score_loss(ds["y_test"][mask], np.clip(score[mask], 1e-7, 1 - 1e-7)))
                        if len(np.unique(ds["y_test"][mask])) == 2
                        else np.nan,
                        "ece": ece_score(ds["y_test"][mask], score[mask]) if np.any(mask) else np.nan,
                    }
                )
        query_groups = ds["test_pairs"]["query_id"].astype(str).to_numpy()
        seed_rows = []
        alternatives = [
            "invariant_logistic_poly2",
            "mean_score_fusion",
            "platt_calibrated_score",
            "isotonic_calibrated_score",
        ]
        for metric in ["pr_auc", "roc_auc"]:
            for alternative in alternatives:
                seed_rows.append(
                    _bootstrap_and_permutation_delta(
                        ds["y_test"],
                        scores["score_only_logistic"],
                        scores[alternative],
                        metric_name=metric,
                        seed=seed,
                        groups=query_groups,
                        n_iter=200 if args.mode == "smoke" else 5000,
                        comparison=f"score_only_logistic_minus_{alternative}",
                    )
                )
        adjusted = holm_adjust([row["paired_permutation_p_value"] for row in seed_rows])
        for row, adjusted_p in zip(seed_rows, adjusted, strict=True):
            row["holm_adjusted_p_value"] = float(adjusted_p)
        stat_rows.extend(seed_rows)
    reliability = pd.DataFrame(reliability_rows)
    threshold = pd.DataFrame(threshold_rows)
    stats = pd.DataFrame(stat_rows)
    reliability.to_csv(TABLES / "calibration_reliability_curves.csv", index=False, lineterminator="\n")
    summarize(threshold, ["tier", "method", "client_id"]).to_csv(
        TABLES / "threshold_stability_by_client.csv", index=False, lineterminator="\n"
    )
    stats.to_csv(TABLES / "paired_permutation_tests.csv", index=False, lineterminator="\n")
    return reliability, threshold, stats


def train_clustered_scores(ds: dict) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    names = ds["expanded_feature_names"]
    idx = select_indices(names, "invariant")
    x_train = ds["x_train_std"][:, idx]
    x_test = ds["x_test_std"][:, idx]
    y_train = ds["y_train"]
    clients_train = ds["client_train"]
    clients_test = ds["client_test"]
    scores: dict[str, np.ndarray] = {}
    per_client_rows = []

    local_models, _ = train_local_models(x_train, y_train, clients_train, epochs=120)
    local_score = np.zeros(len(ds["y_test"]), dtype=float)
    for client_id, model in local_models.items():
        mask = clients_test == client_id
        if np.any(mask):
            local_score[mask] = model.predict_proba(x_test[mask])
    scores["local_calibration"] = local_score

    fed = train_federated(x_train, y_train, clients_train, method="fedavg", privacy_mode="plain", rounds=30, local_epochs=5)[0]
    scores["fedavg_plain"] = fed.predict_proba(x_test)
    scores["fedprox_plain"] = train_federated(
        x_train, y_train, clients_train, method="fedprox", privacy_mode="plain", rounds=30, local_epochs=5, prox_mu=0.02
    )[0].predict_proba(x_test)
    scores["fedyogi"] = train_fedopt(x_train, y_train, clients_train, optimizer="fedyogi").predict_proba(x_test)
    scores["fedadam"] = train_fedopt(x_train, y_train, clients_train, optimizer="fedadam").predict_proba(x_test)
    scores["personalized_fedavg"] = np.zeros(len(ds["y_test"]), dtype=float)
    p_models, _ = train_personalized_models(x_train, y_train, clients_train, fed.weights, epochs=35, lr=0.04, prox_mu=0.015)
    for client_id, model in p_models.items():
        mask = clients_test == client_id
        if np.any(mask):
            scores["personalized_fedavg"][mask] = model.predict_proba(x_test[mask])

    client_ids = sorted(map(int, np.unique(clients_train)))
    client_means = np.vstack([x_train[clients_train == cid].mean(axis=0) for cid in client_ids])
    k = min(3, len(client_ids))
    client_kmeans = KMeans(n_clusters=k, random_state=20260529, n_init=10).fit(client_means)
    client_to_cluster = {cid: int(lbl) for cid, lbl in zip(client_ids, client_kmeans.labels_, strict=True)}
    cluster_models: dict[int, LogisticHead] = {}
    cluster_centroids = []
    for cluster in range(k):
        train_mask = np.array([client_to_cluster[int(cid)] == cluster for cid in clients_train])
        if train_mask.sum() < 4 or len(np.unique(y_train[train_mask])) < 2:
            cluster_models[cluster] = train_logistic(x_train, y_train, epochs=120)
        else:
            cluster_models[cluster] = train_logistic(x_train[train_mask], y_train[train_mask], epochs=120)
        cluster_centroids.append(x_train[train_mask].mean(axis=0) if train_mask.any() else x_train.mean(axis=0))
    cluster_centroids = np.vstack(cluster_centroids)
    with_id = np.zeros(len(ds["y_test"]), dtype=float)
    no_id = np.zeros(len(ds["y_test"]), dtype=float)
    nearest = np.argmin(((x_test[:, None, :] - cluster_centroids[None, :, :]) ** 2).sum(axis=2), axis=1)
    for i in range(len(ds["y_test"])):
        cluster_id = client_to_cluster.get(int(clients_test[i]), int(nearest[i]))
        with_id[i] = cluster_models[cluster_id].predict_proba(x_test[i : i + 1])[0]
        no_id[i] = cluster_models[int(nearest[i])].predict_proba(x_test[i : i + 1])[0]
    scores["client_clustered_with_client_id"] = with_id
    scores["client_clustered_invariant_gate"] = no_id

    for k_experts in [2, 3, 5]:
        k_experts = min(k_experts, max(2, len(np.unique(y_train))))
        km = KMeans(n_clusters=k_experts, random_state=20260529 + k_experts, n_init=10).fit(x_train)
        experts = {}
        for cluster in range(k_experts):
            mask = km.labels_ == cluster
            if mask.sum() < 8 or len(np.unique(y_train[mask])) < 2:
                experts[cluster] = train_logistic(x_train, y_train, epochs=120)
            else:
                experts[cluster] = train_logistic(x_train[mask], y_train[mask], epochs=120)
        test_cluster = km.predict(x_test)
        moe = np.zeros(len(ds["y_test"]), dtype=float)
        for cluster, model in experts.items():
            mask = test_cluster == cluster
            if np.any(mask):
                moe[mask] = model.predict_proba(x_test[mask])
        scores[f"mixture_experts_k{k_experts}"] = moe

    for client_id in sorted(map(int, np.unique(clients_test))):
        mask = clients_test == client_id
        for method in ["local_calibration", "fedavg_plain", "personalized_fedavg", "client_clustered_invariant_gate"]:
            row = detection_metrics(
                ds["y_test"][mask],
                scores[method][mask],
                method=method,
                modality="multimodal",
            )
            row["client_id"] = client_id
            per_client_rows.append(row)
    return scores, pd.DataFrame(per_client_rows)


def run_personalization(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    client_frames = []
    for seed in args.seeds:
        ds = prepare_features(
            load_public_pair_dataset(
                seed=seed,
                max_train_pairs=args.max_train_pairs,
                max_test_pairs=args.max_test_pairs,
                negative_ratio=1.0,
            )
        )
        base = fit_predict_methods(ds, include_federated=False)
        clustered, per_client = train_clustered_scores(ds)
        all_scores = {**base, **clustered}
        for method, score in all_scores.items():
            rows.append(full_metrics(ds["y_test"], score, method=method, tier="VCSL public-label audit", seed=seed))
        per_client["seed"] = seed
        per_client["tier"] = "VCSL public-label audit"
        client_frames.append(per_client)

    isc_run = CONFIRMATORY / "vcsl_isc_confirmatory"
    isc_path = isc_run / "metrics_detection.csv"
    isc_client_path = isc_run / "metrics_client.csv"
    if isc_path.is_file() and isc_client_path.is_file():
        isc = pd.read_csv(isc_path)
        selected = {
            "centralized_multimodal",
            "local_multimodal",
            "fedavg_plain",
            "fedprox_plain",
            "fedavg_secureagg_sim",
            "fedavg_quantized_transport_proxy",
            "fedavgft_plain",
        }
        for row in isc[isc["method"].isin(selected)].to_dict(orient="records"):
            row["tier"] = "VCSL ISC visual descriptor confirmatory"
            row["ece"] = row["ece_15"]
            for prevalence in PREVALENCES:
                row[f"expected_precision_prev_{prevalence:g}"] = expected_precision(
                    prevalence, row["recall"], row["fpr_at_95_recall"]
                )
                row[f"reviews_per_true_positive_prev_{prevalence:g}"] = 1.0 / max(
                    row[f"expected_precision_prev_{prevalence:g}"], 1e-15
                )
            rows.append(row)
        isc_clients = pd.read_csv(isc_client_path)
        isc_clients = isc_clients[isc_clients["method"].isin(selected)].copy()
        isc_clients["tier"] = "VCSL ISC visual descriptor confirmatory"
        client_frames.append(isc_clients)
    else:
        raise FileNotFoundError(
            "confirmatory VCSL ISC metrics are required; archived summaries cannot satisfy the submission evidence gate"
        )
    detail = pd.DataFrame(rows)
    summary = summarize(detail, ["tier", "method"])
    summary.to_csv(TABLES / "personalization_strengthening.csv", index=False, lineterminator="\n")
    per_client_summary = summarize(pd.concat(client_frames, ignore_index=True), ["tier", "method", "client_id"])
    per_client_summary.to_csv(TABLES / "per_client_metrics.csv", index=False, lineterminator="\n")
    return summary, per_client_summary


def build_synthetic_av(seed: int, *, n_per_scenario: int) -> dict:
    rng = np.random.default_rng(seed)
    scenarios = [
        ("matched_av_copy", 1, 0.86, 0.84, 0.82, 0.92, 0.90),
        ("video_only_copy", 1, 0.84, 0.24, 0.78, 0.91, 0.42),
        ("audio_only_copy", 1, 0.24, 0.85, 0.34, 0.42, 0.91),
        ("replaced_audio_conflict", 1, 0.82, 0.08, 0.76, 0.90, 0.25),
        ("replaced_video_conflict", 1, 0.08, 0.82, 0.24, 0.25, 0.90),
        ("hard_negative_same_context", 0, 0.47, 0.43, 0.45, 0.72, 0.71),
    ]
    rows = []
    labels = []
    client_ids = []
    scen = []
    query_ids = []
    ref_ids = []
    for s_idx, (name, label, v_mu, a_mu, t_mu, rv_mu, ra_mu) in enumerate(scenarios):
        for i in range(n_per_scenario):
            v = float(np.clip(rng.normal(v_mu, 0.09), -0.1, 1.0))
            a = float(np.clip(rng.normal(a_mu, 0.10), -0.1, 1.0))
            t = float(np.clip(rng.normal(t_mu, 0.08), -0.1, 1.0))
            rv = float(np.clip(rng.normal(rv_mu, 0.06), 0.0, 1.0))
            ra = float(np.clip(rng.normal(ra_mu, 0.06), 0.0, 1.0))
            modality_gap = abs(v - a)
            fusion_prior = 0.55 * v * rv + 0.45 * a * ra
            frame_ratio = float(np.clip(rng.normal(t_mu, 0.10), 0.0, 1.0))
            same_category = float(1 if name != "hard_negative_same_context" else rng.random() < 0.75)
            metadata_score = float(np.clip(0.6 * same_category + 0.4 * frame_ratio, 0, 1))
            row = [
                v,
                a,
                t,
                metadata_score,
                rv,
                ra,
                modality_gap,
                fusion_prior,
                frame_ratio,
                same_category,
                v * rv,
                a * ra,
                t * rv,
                max(v, a),
                0.5 * (v + a),
                v * a,
                same_category * a,
                frame_ratio * v,
                float(s_idx % 11) / 10.0,
            ]
            rows.append(row)
            labels.append(label)
            client_ids.append(s_idx % 11)
            scen.append(name)
            query_ids.append(f"synthetic-av-q-{name}-{i:05d}")
            ref_ids.append(f"synthetic-av-r-{name if label else 'negative'}-{i:05d}")
    feature_names = [
        "video_score",
        "audio_score",
        "temporal_score",
        "metadata_score",
        "video_reliability",
        "audio_reliability",
        "modality_gap",
        "fusion_prior",
        "frame_ratio",
        "same_category",
        "video_rel_score",
        "audio_rel_score",
        "temporal_rel_score",
        "max_modality_score",
        "mean_modality_score",
        "product_modality_score",
        "category_audio_score",
        "frame_video_score",
        "client_norm",
    ]
    x = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=int)
    clients = np.asarray(client_ids, dtype=int)
    scenarios_arr = np.asarray(scen, dtype=object)
    order = rng.permutation(len(y))
    split = int(round(0.65 * len(order)))
    train_idx = order[:split]
    test_idx = order[split:]
    pair_df = pd.DataFrame({"query_id": np.asarray(query_ids)[test_idx], "reference_id": np.asarray(ref_ids)[test_idx], "label": y[test_idx]})
    return {
        "x_train": x[train_idx],
        "y_train": y[train_idx],
        "client_train": clients[train_idx],
        "scenario_train": scenarios_arr[train_idx],
        "x_test": x[test_idx],
        "y_test": y[test_idx],
        "client_test": clients[test_idx],
        "scenario_test": scenarios_arr[test_idx],
        "test_pairs": pair_df,
        "feature_names": feature_names,
    }


def run_synthetic_av(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    scenario_rows = []
    n_per_scenario = 450 if args.mode == "smoke" else 1200
    for seed in args.seeds:
        ds = prepare_features(build_synthetic_av(seed, n_per_scenario=n_per_scenario))
        scores = fit_predict_methods(ds, include_federated=True)
        for method, score in scores.items():
            if method in {"score_only_logistic", "invariant_logistic_poly2", "fedavg_plain", "fedprox_plain", "fedyogi", "fedadam", "mean_score_fusion"}:
                rows.append(full_metrics(ds["y_test"], score, method=method, tier="Synthetic A/V conflict stress", seed=seed))
                for scenario in sorted(set(map(str, ds["scenario_test"]))):
                    mask = ds["scenario_test"] == scenario
                    metric = full_metrics(ds["y_test"][mask], score[mask], method=method, tier="Synthetic A/V conflict stress", seed=seed)
                    metric["scenario"] = scenario
                    scenario_rows.append(metric)
    metrics = summarize(pd.DataFrame(rows), ["tier", "method"])
    scenarios = summarize(pd.DataFrame(scenario_rows), ["tier", "method", "scenario"])
    scenarios = scenarios.drop(
        columns=[
            column
            for column in scenarios.columns
            if column.startswith(("roc_auc_", "pr_auc_", "fpr_at_95_recall_", "expected_precision_", "reviews_per_"))
        ]
    )
    metrics["tier_label"] = "Synthetic A/V stress tier; not a natural synchronized corpus"
    scenarios["tier_label"] = "Synthetic A/V stress tier; scenario-level modality agreement/conflict stress"
    metrics.to_csv(TABLES / "av_sync_tier_metrics.csv", index=False, lineterminator="\n")
    scenarios.to_csv(TABLES / "av_sync_operating_points.csv", index=False, lineterminator="\n")
    return metrics, scenarios


def protected_aggregation_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260529)
    paillier_key_bits = int(args.paillier_key_bits)
    updates = [rng.normal(0.0, 0.015, size=24).astype(np.float64) for _ in range(8)]
    weights = [120, 96, 88, 132, 75, 141, 106, 119]
    plain_agg, plain_report = aggregate_updates(updates, weights, mode="plain")
    proxy_agg, proxy_report = aggregate_updates(updates, weights, mode="heagg", he_scale=1e6)
    paillier_agg, paillier_report, paillier_details = paillier_aggregate_updates(
        updates, weights, key_bits=paillier_key_bits, he_scale=1e6
    )
    proxy_err = float(np.max(np.abs(plain_agg - proxy_agg)))
    paillier_err = float(np.max(np.abs(plain_agg - paillier_agg)))
    status = pd.DataFrame(
        [
            {
                "mode": "plain FedAvg",
                "implementation_status": "implemented numeric aggregation",
                "claim_label": "plain compact-head aggregation",
                "production_crypto": False,
                "dropout_handling": False,
                "collusion_resistance": False,
                "notes": "Reference mode; no update-protection claim.",
            },
            {
                "mode": "SecureAgg-style simulation",
                "implementation_status": "deterministic pairwise-mask arithmetic with post-mask dropout reconstruction",
                "claim_label": "SecureAgg pairwise-mask simulation",
                "production_crypto": False,
                "dropout_handling": "post-mask residual reconstruction simulation at 0/10/20 percent",
                "collusion_resistance": False,
                "notes": "Pairwise masks and reconstructed dropout residuals cancel numerically; no network key agreement or collusion theorem is implemented.",
            },
            {
                "mode": "Quantized transport proxy",
                "implementation_status": "quantized compact-update transport accounting with 16x byte expansion",
                "claim_label": "quantized transport proxy",
                "production_crypto": False,
                "dropout_handling": False,
                "collusion_resistance": False,
                "notes": "This row is not homomorphic encryption; no CKKS or BFV security level is claimed.",
            },
            {
                "mode": "Paillier compact-update aggregation",
                "implementation_status": "real additive Paillier aggregation over quantized compact-head updates",
                "claim_label": "real Paillier compact-update sum",
                "production_crypto": False,
                "dropout_handling": False,
                "collusion_resistance": False,
                "notes": "Implements additive encrypted sums for compact updates with reviewer-run key generation; no encrypted media inference or production key management.",
            },
        ]
    )
    secureagg_rows = []
    for dropout_rate in (0.0, 0.1, 0.2):
        keep = max(1, int(round(len(updates) * (1.0 - dropout_rate))))
        secure_agg, secure_report, details = secureagg_simulate(
            updates,
            weights,
            active_clients=list(range(keep)),
            mask_seed=20260529,
            dropout_stage="after_mask_setup",
        )
        active_plain, _ = aggregate_updates(updates[:keep], weights[:keep], mode="plain")
        secureagg_rows.append(
            {
                "test": "pairwise_mask_cancellation",
                "dropout_rate": dropout_rate,
                "dropout_stage": details["dropout_stage"],
                "active_clients": keep,
                "dropped_clients": len(updates) - keep,
                "implemented": True,
                "max_abs_error_vs_active_plain": float(np.max(np.abs(secure_agg - active_plain))),
                "masks_cancel": details["masks_cancel"],
                "pre_recovery_residual_norm": details["pre_recovery_residual_norm"],
                "reconstructed_mask_norm": details["reconstructed_mask_norm"],
                "recovery_applied": details["recovery_applied"],
                "byte_expansion": secure_report.ciphertext_expansion,
                "claim_scope": secure_report.protocol_scope,
            }
        )
    secureagg = pd.DataFrame(secureagg_rows)
    quantized_proxy = pd.DataFrame(
        [
            {
                "mode": "Quantized transport proxy",
                "scheme": "none",
                "security_level_bits": "not claimed",
                "quantization_scale": 1e6,
                "ciphertext_expansion": 16.0,
                "numeric_error_tested": True,
                "max_abs_error_vs_plain": proxy_err,
                "full_encrypted_inference": False,
                "notes": "Quantized compact-update accounting only; not homomorphic encryption and not extrapolated to media embeddings.",
            },
            {
                "mode": "Paillier compact-update aggregation",
                "scheme": "Paillier additive HE",
                "security_level_bits": f"{paillier_key_bits}-bit modulus generated for artifact run",
                "quantization_scale": 1e6,
                "ciphertext_expansion": paillier_report.ciphertext_expansion,
                "numeric_error_tested": True,
                "max_abs_error_vs_plain": paillier_err,
                "full_encrypted_inference": False,
                "notes": "Real encrypted compact-update sums with signed integer encoding; no CKKS/BFV packing and no encryption of multimedia inference.",
            },
        ]
    )
    paillier = pd.DataFrame(
        [
            {
                **paillier_details,
                "plain_bytes": plain_report.plain_bytes,
                "proxy_protected_bytes": proxy_report.protected_bytes,
                "paillier_protected_bytes": paillier_report.protected_bytes,
                "proxy_ciphertext_expansion": proxy_report.ciphertext_expansion,
                "paillier_ciphertext_expansion": paillier_report.ciphertext_expansion,
                "proxy_max_abs_error_vs_plain": proxy_err,
                "paillier_max_abs_error_vs_plain": paillier_err,
                "paillier_encryption_time_sec": paillier_report.encryption_time_sec,
                "paillier_aggregation_decryption_time_sec": paillier_report.aggregation_time_sec,
                "claim_scope": "real additive HE for compact update sums only",
            }
        ]
    )
    status.to_csv(TABLES / "protected_aggregation_status.csv", index=False, lineterminator="\n")
    secureagg.to_csv(TABLES / "secureagg_dropout_or_proxy.csv", index=False, lineterminator="\n")
    quantized_proxy.to_csv(TABLES / "quantized_transport_proxy.csv", index=False, lineterminator="\n")
    paillier.to_csv(TABLES / "paillier_update_aggregation.csv", index=False, lineterminator="\n")
    return status, secureagg, quantized_proxy, paillier


def run_privacy_attacks(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for seed in args.seeds:
        ds = prepare_features(
            load_public_pair_dataset(
                seed=seed,
                max_train_pairs=args.max_train_pairs,
                max_test_pairs=args.max_test_pairs,
                negative_ratio=1.0,
            )
        )
        names = ds["expanded_feature_names"]
        inv_idx = select_indices(names, "invariant")
        all_idx = select_indices(names, "all")
        no_ctx_idx = select_indices(names, "no_context")
        model = train_logistic(ds["x_train_std"][:, inv_idx], ds["y_train"], epochs=140)
        p_train = model.predict_proba(ds["x_train_std"][:, inv_idx])
        p_test = model.predict_proba(ds["x_test_std"][:, inv_idx])
        y_membership = np.r_[np.ones_like(p_train), np.zeros_like(p_test)]
        y_task = np.r_[ds["y_train"], ds["y_test"]]
        p_all = np.r_[p_train, p_test]
        eps = 1e-9
        scalar_attacks = {
            "confidence": np.maximum(p_all, 1.0 - p_all),
            "entropy": -(p_all * np.log(p_all + eps) + (1 - p_all) * np.log(1 - p_all + eps)),
            "loss": -(y_task * np.log(p_all + eps) + (1 - y_task) * np.log(1 - p_all + eps)),
            "margin": np.abs(p_all - 0.5),
            "calibrated_score": p_all,
        }
        for attack, values in scalar_attacks.items():
            auc = roc_auc_score(y_membership, values)
            rows.append({"attack": "membership_inference", "setting": attack, "seed": seed, "value": float(max(auc, 1 - auc)), "metric": "attack_auc"})
        attack_features = np.vstack([scalar_attacks[k] for k in ["confidence", "entropy", "loss", "margin", "calibrated_score"]]).T
        rng = np.random.default_rng(seed + 2026)
        order = rng.permutation(len(y_membership))
        split = int(round(0.55 * len(order)))
        tr = order[:split]
        te = order[split:]
        for model_name, clf in [
            ("logistic_attack_model", LogisticRegression(max_iter=500)),
            ("random_forest_attack_model", RandomForestClassifier(n_estimators=120, min_samples_leaf=8, random_state=seed)),
        ]:
            clf.fit(attack_features[tr], y_membership[tr])
            score = clf.predict_proba(attack_features[te])[:, 1]
            rows.append({"attack": "membership_inference", "setting": model_name, "seed": seed, "value": float(roc_auc_score(y_membership[te], score)), "metric": "attack_auc"})

        for setting, idx in [("all_fields", all_idx), ("context_removed", no_ctx_idx), ("default_invariant", inv_idx)]:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(ds["x_train_std"][:, idx], ds["client_train"])
            pred = clf.predict(ds["x_test_std"][:, idx])
            rows.append(
                {
                    "attack": "client_identity_inference",
                    "setting": setting,
                    "seed": seed,
                    "value": present_class_balanced_accuracy(ds["client_test"], pred),
                    "metric": "balanced_accuracy",
                }
            )

        train_cat = ds["train_pairs"]["query_category"].astype(str)
        test_cat = ds["test_pairs"]["query_category"].astype(str)
        common = sorted(set(train_cat) & set(test_cat))
        if len(common) >= 2:
            train_mask = train_cat.isin(common).to_numpy()
            test_mask = test_cat.isin(common).to_numpy()
            for setting, idx in [("all_fields", all_idx), ("default_invariant", inv_idx)]:
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                clf.fit(ds["x_train_std"][train_mask][:, idx], train_cat[train_mask])
                pred = clf.predict(ds["x_test_std"][test_mask][:, idx])
                rows.append(
                    {
                        "attack": "query_category_inference",
                        "setting": setting,
                        "seed": seed,
                        "value": present_class_balanced_accuracy(test_cat[test_mask], pred),
                        "metric": "balanced_accuracy",
                    }
                )
        update_rows = []
        for client_id in sorted(map(int, np.unique(ds["client_train"]))):
            mask = ds["client_train"] == client_id
            local = train_logistic(ds["x_train_std"][mask][:, inv_idx], ds["y_train"][mask], epochs=40)
            update = local.weights
            update_rows.append(
                {
                    "client_id": client_id,
                    "positive_rate": float(ds["y_train"][mask].mean()),
                    "update_norm": float(np.linalg.norm(update)),
                    "update_mean": float(update.mean()),
                    "update_std": float(update.std()),
                    "update_max_abs": float(np.max(np.abs(update))),
                }
            )
        update_frame = pd.DataFrame(update_rows)
        if update_frame["positive_rate"].nunique() > 1:
            prop = (update_frame["positive_rate"] >= update_frame["positive_rate"].median()).astype(int).to_numpy()
            feats = update_frame[["update_norm", "update_mean", "update_std", "update_max_abs"]].to_numpy(dtype=float)
            preds = []
            truth = []
            for i in range(len(prop)):
                train_mask = np.arange(len(prop)) != i
                if len(np.unique(prop[train_mask])) < 2:
                    continue
                clf = LogisticRegression(max_iter=500, class_weight="balanced")
                clf.fit(feats[train_mask], prop[train_mask])
                preds.append(int(clf.predict(feats[i : i + 1])[0]))
                truth.append(int(prop[i]))
            if len(set(truth)) >= 2:
                rows.append(
                    {
                        "attack": "update_property_inference",
                        "setting": "compact_update_summary_leave_one_client_out",
                        "seed": seed,
                        "value": present_class_balanced_accuracy(truth, preds),
                        "metric": "balanced_accuracy",
                        "note": "Predicts above-median client positive-rate property from compact local-update summary statistics.",
                    }
                )
    out = summarize(pd.DataFrame(rows), ["attack", "setting", "metric"])
    out.to_csv(TABLES / "privacy_attack_results.csv", index=False, lineterminator="\n")
    return out


def _aggregate_update_matrix(updates: np.ndarray, weights: np.ndarray, *, method: str, byzantine_count: int = 1) -> np.ndarray:
    if method == "fedavg_mean":
        w = weights / max(weights.sum(), 1e-12)
        return (updates * w[:, None]).sum(axis=0)
    if method == "coordinate_median":
        return np.median(updates, axis=0)
    if method == "trimmed_mean":
        trim = min(max(1, byzantine_count), max(0, (len(updates) - 1) // 2))
        ordered = np.sort(updates, axis=0)
        if 2 * trim >= len(updates):
            return ordered.mean(axis=0)
        return ordered[trim:-trim].mean(axis=0)
    if method == "krum":
        n = len(updates)
        f = min(max(1, byzantine_count), max(0, (n - 3) // 2))
        neighbor_count = max(1, n - f - 2)
        distances = ((updates[:, None, :] - updates[None, :, :]) ** 2).sum(axis=2)
        scores = []
        for i in range(n):
            nearest = np.sort(np.delete(distances[i], i))[:neighbor_count]
            scores.append(float(nearest.sum()))
        return updates[int(np.argmin(scores))]
    raise ValueError(f"unknown robust aggregation method: {method}")


def run_robustness_stress(args: argparse.Namespace) -> pd.DataFrame:
    """Run multi-round compact-head poisoning diagnostics.

    This remains a controlled robustness stress test, not a Byzantine-security
    proof.  Attack membership is fixed per seed and attack fraction, while the
    global model evolves for the configured number of communication rounds.
    """

    rows = []
    aggregators = ["fedavg_mean", "coordinate_median", "trimmed_mean", "krum"]
    for seed in args.seeds:
        ds = prepare_features(
            load_public_pair_dataset(
                seed=seed,
                max_train_pairs=args.max_train_pairs,
                max_test_pairs=args.max_test_pairs,
                negative_ratio=1.0,
            )
        )
        names = ds["expanded_feature_names"]
        inv_idx = select_indices(names, "invariant")
        x_train = ds["x_train_std"][:, inv_idx]
        x_test = ds["x_test_std"][:, inv_idx]
        y_train = ds["y_train"]
        clients = ds["client_train"]
        client_ids = sorted(map(int, np.unique(clients)))
        settings = [(0.0, "clean")]
        settings.extend((fraction, attack) for fraction in args.attacker_fractions if fraction > 0 for attack in args.attacks)
        for attack_fraction, attack in settings:
            attacked_count = int(math.ceil(len(client_ids) * attack_fraction)) if attack_fraction > 0 else 0
            attack_clients = set(client_ids[:attacked_count])
            for aggregator in aggregators:
                model = LogisticHead.zeros(x_train.shape[1])
                for _round_id in range(args.robust_rounds):
                    updates = []
                    weights = []
                    base_weights = model.weights.copy()
                    for client_id in client_ids:
                        mask = clients == client_id
                        x_local = x_train[mask]
                        y_local = y_train[mask].copy()
                        if client_id in attack_clients and attack == "label_flip":
                            y_local = 1 - y_local
                        local = train_logistic(
                            x_local,
                            y_local,
                            initial=base_weights,
                            epochs=args.robust_local_epochs,
                            lr=0.08,
                        )
                        update = local.weights - base_weights
                        if client_id in attack_clients and attack == "sign_flip":
                            update = -5.0 * update
                        updates.append(update)
                        weights.append(float(mask.sum()))
                    aggregate = _aggregate_update_matrix(
                        np.vstack(updates),
                        np.asarray(weights, dtype=float),
                        method=aggregator,
                        byzantine_count=max(1, attacked_count),
                    )
                    model.weights = base_weights + aggregate
                score = model.predict_proba(x_test)
                metric = full_metrics(ds["y_test"], score, method=aggregator, tier="VCSL public-label audit", seed=seed)
                metric.update(
                    {
                        "attack_scenario": attack,
                        "attacker_fraction": float(attack_fraction),
                        "attacked_clients": ",".join(map(str, sorted(attack_clients))),
                        "communication_rounds": int(args.robust_rounds),
                        "local_epochs": int(args.robust_local_epochs),
                        "robustness_scope": "multi-round compact-head poisoning stress; not a Byzantine FL proof",
                    }
                )
                rows.append(metric)
    detail = pd.DataFrame(rows)
    detail.to_csv(RESULTS / "robustness_stress_raw.csv", index=False, lineterminator="\n")
    out = summarize(
        detail,
        ["tier", "attack_scenario", "attacker_fraction", "method", "communication_rounds", "local_epochs", "robustness_scope"],
    )
    out.to_csv(TABLES / "robustness_stress.csv", index=False, lineterminator="\n")
    return out


def receipt_payload(i: int, prev: str, public_key: str = "test-key") -> dict:
    return {
        "receipt_id": f"test-vector-{i:06d}",
        "schema_version": "fedtwin-cryptid-receipt-v2",
        "timestamp_utc": "2026-05-29T00:00:00+00:00",
        "timestamp_mode": "monotone-counter",
        "client_commitment": sha256_text(f"client-{i % 11}"),
        "model_digest": sha256_text("model"),
        "reference_digest": sha256_text(f"reference-{i}"),
        "query_digest": sha256_text(f"query-{i}"),
        "decision": "review",
        "confidence": round(0.5 + (i % 100) / 1000.0, 6),
        "modality_scores": {"video_score": 0.5, "audio_score": 0.5, "fusion_score": 0.5},
        "segment_evidence_digest": sha256_text(f"segment-{i}"),
        "previous_chain_hash": prev,
        "signer_public_key": public_key,
        "signature": "",
        "salt_digest": sha256_text(f"salt-{i}"),
        "supersedes_receipt_hash": "",
    }


def receipt_scaling(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = [1000, 10000] if args.mode == "smoke" else [1000, 10000, 100000]
    rows = []
    for count in counts:
        ledger = HashChainLedger()
        start = time.perf_counter()
        for i in range(count):
            payload = receipt_payload(i, ledger.previous_hash)
            ledger.anchor(payload)
        build_time = time.perf_counter() - start
        start_verify = time.perf_counter()
        valid = ledger.verify(verify_signatures=False)
        verify_time = time.perf_counter() - start_verify
        tampered_payload = False
        tampered_prev = False
        wrong_key = False
        if ledger.records:
            ledger.records[min(3, len(ledger.records) - 1)]["payload"]["decision"] = "tampered"
            tampered_payload = not ledger.verify(verify_signatures=False)
            ledger.records[min(3, len(ledger.records) - 1)]["payload"]["decision"] = "review"
            old_prev = ledger.records[min(4, len(ledger.records) - 1)]["previous_chain_hash"]
            ledger.records[min(4, len(ledger.records) - 1)]["previous_chain_hash"] = "f" * 64
            tampered_prev = not ledger.verify(verify_signatures=False)
            ledger.records[min(4, len(ledger.records) - 1)]["previous_chain_hash"] = old_prev
            key_a = Ed25519PrivateKey.generate()
            key_b = Ed25519PrivateKey.generate()
            signed_payload = receipt_payload(999999, "0" * 64)
            signed_payload["signer_public_key"] = key_a.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            ).hex()
            signature = key_a.sign(canonical_hash(signed_payload).encode("utf-8"))
            try:
                key_b.public_key().verify(signature, canonical_hash(signed_payload).encode("utf-8"))
                wrong_key = False
            except Exception:
                wrong_key = True
        rows.append(
            {
                "receipts": count,
                "valid_chain_verification": valid,
                "tampered_payload_detected": tampered_payload,
                "tampered_previous_hash_detected": tampered_prev,
                "wrong_signing_key_detected": wrong_key,
                "supersession_reference_supported": True,
                "key_rotation_discussed_not_production_kms": True,
                "build_time_sec": build_time,
                "verify_time_sec": verify_time,
                "verify_receipts_per_sec": count / max(verify_time, 1e-12),
            }
        )
    scaling = pd.DataFrame(rows)
    scaling.to_csv(TABLES / "receipt_scaling.csv", index=False, lineterminator="\n")

    payload = receipt_payload(0, "0" * 64, "deterministic-test-public-key-placeholder")
    receipt_hash = canonical_hash(payload)
    chain_hash = sha256_text(payload["previous_chain_hash"] + receipt_hash)
    vector = pd.DataFrame(
        [
            {
                "canonical_payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
                "receipt_hash": receipt_hash,
                "chain_hash": chain_hash,
                "public_key": payload["signer_public_key"],
                "signature": payload["signature"],
                "note": "Unsigned deterministic canonicalization vector for reviewer interoperability tests; signed vector produced by generate_receipt_test_vector.py.",
            }
        ]
    )
    vector.to_csv(TABLES / "receipt_test_vector.csv", index=False, lineterminator="\n")
    return scaling, vector


def plot_outputs(
    av_metrics: pd.DataFrame,
    av_scenarios: pd.DataFrame,
    ranking: pd.DataFrame,
    open_set: pd.DataFrame,
    personalization: pd.DataFrame,
    per_client: pd.DataFrame,
    privacy: pd.DataFrame,
    workload: pd.DataFrame,
    reliability: pd.DataFrame,
    threshold: pd.DataFrame,
    robustness: pd.DataFrame,
    paillier: pd.DataFrame,
) -> None:
    av_plot = av_metrics[av_metrics["method"].isin(["score_only_logistic", "invariant_logistic_poly2", "fedavg_plain", "fedyogi"])].copy()
    av_plot = av_plot.sort_values("pr_auc_mean", ascending=True)
    plt.figure(figsize=(5.5, 2.7))
    plt.barh(av_plot["method"], av_plot["pr_auc_mean"], xerr=av_plot["pr_auc_std"].fillna(0), color=PALETTE["green"], alpha=0.86)
    plt.xlabel("PR-AUC")
    plt.title("Synthetic A/V stress tier: calibration performance", loc="left", fontweight="bold")
    plt.grid(axis="x", color="#D8DEE9", alpha=0.7)
    savefig("av_sync_tier_pr_auc")

    scen = av_scenarios[av_scenarios["method"].eq("invariant_logistic_poly2")].copy()
    scen = scen.sort_values("recall_mean", ascending=True)
    plt.figure(figsize=(6.0, 3.0))
    plt.barh(scen["scenario"], scen["recall_mean"], color=PALETTE["orange"], alpha=0.86)
    plt.xlabel("Recall at threshold 0.5")
    plt.xlim(0, 1.02)
    plt.title("Synthetic A/V stress: scenario breakdown", loc="left", fontweight="bold")
    plt.grid(axis="x", color="#D8DEE9", alpha=0.7)
    savefig("av_conflict_breakdown")

    rank_plot = ranking[(ranking["method"].isin(["score_only_logistic", "invariant_logistic_poly2", "platt_calibrated_score"])) & (ranking["candidate_negative_ratio"].eq(100))].copy()
    metric_names = ["recall_at_1_mean", "recall_at_5_mean", "recall_at_10_mean", "recall_at_100_mean"]
    x = np.arange(len(metric_names))
    width = 0.24
    plt.figure(figsize=(6.2, 3.0))
    for offset, (_, row) in zip([-width, 0, width], rank_plot.iterrows(), strict=False):
        plt.bar(x + offset, [row[m] for m in metric_names], width, label=row["method"])
    plt.xticks(x, ["R@1", "R@5", "R@10", "R@100"])
    plt.ylim(0, 1.02)
    plt.ylabel("Query-level recall")
    plt.title("Retrieval-like ranking on VCSL public-label candidates (1:100)", loc="left", fontweight="bold")
    plt.grid(axis="y", color="#D8DEE9", alpha=0.7)
    plt.legend(frameon=False, ncol=1)
    savefig("query_recall_at_k")

    open_plot = open_set[open_set["method"].isin(["score_only_logistic", "invariant_logistic_poly2", "platt_calibrated_score"])].copy()
    plt.figure(figsize=(6.0, 3.0))
    for method, group in open_plot.groupby("method"):
        group = group.sort_values("candidate_negative_ratio")
        plt.plot(group["candidate_negative_ratio"], group["precision_mean"], marker="o", label=method)
    plt.xscale("log")
    plt.xlabel("Candidate negative ratio")
    plt.ylabel("Precision at threshold 0.5")
    plt.title("Open-set candidate-pool precision sensitivity", loc="left", fontweight="bold")
    plt.grid(color="#D8DEE9", alpha=0.7, which="both")
    plt.legend(frameon=False)
    savefig("open_set_precision_curves")

    pers = personalization[
        (personalization["tier"].eq("VCSL public-label audit"))
        & personalization["method"].isin(
            [
                "score_only_logistic",
                "invariant_logistic_poly2",
                "local_calibration",
                "fedavg_plain",
                "personalized_fedavg",
                "client_clustered_invariant_gate",
                "mixture_experts_k3",
                "fedyogi",
            ]
        )
    ].copy()
    pers = pers.sort_values("pr_auc_mean", ascending=True)
    plt.figure(figsize=(6.2, 3.2))
    plt.barh(pers["method"], pers["pr_auc_mean"], xerr=pers["pr_auc_std"].fillna(0), color=PALETTE["purple"], alpha=0.86)
    plt.xlabel("PR-AUC")
    plt.title("Personalization and clustered calibration on VCSL public-label audit tier", loc="left", fontweight="bold")
    plt.grid(axis="x", color="#D8DEE9", alpha=0.7)
    savefig("personalization_vs_local")

    client_plot = per_client[per_client["method"].isin(["local_calibration", "fedavg_plain", "client_clustered_invariant_gate"])].copy()
    pivot = client_plot.pivot_table(index="client_id", columns="method", values="pr_auc_mean")
    plt.figure(figsize=(6.0, 3.0))
    for method in pivot.columns:
        plt.plot(pivot.index, pivot[method], marker="o", linewidth=1.2, label=method)
    plt.xlabel("Client id")
    plt.ylabel("Per-client PR-AUC")
    plt.title("VCSL public-label client variability", loc="left", fontweight="bold")
    plt.grid(color="#D8DEE9", alpha=0.7)
    plt.legend(frameon=False)
    savefig("vcsl_isc_client_variability")

    priv = privacy[privacy["metric"].isin(["attack_auc", "balanced_accuracy"])].copy()
    priv["label"] = priv["attack"] + "\n" + priv["setting"]
    priv = priv.sort_values("value_mean", ascending=True).tail(10)
    plt.figure(figsize=(6.2, 3.4))
    plt.barh(priv["label"], priv["value_mean"], xerr=priv["value_std"].fillna(0), color=PALETTE["red"], alpha=0.82)
    plt.axvline(0.5, color=PALETTE["dark"], linestyle="--", linewidth=1.0)
    plt.xlabel("Attack score")
    plt.title("Privacy/leakage audit: attacks do not prove privacy", loc="left", fontweight="bold")
    plt.grid(axis="x", color="#D8DEE9", alpha=0.7)
    savefig("privacy_attack_summary")

    work = workload[(workload["method"].isin(["score_only_logistic", "invariant_logistic_poly2"])) & (workload["candidate_negative_ratio"].eq(workload["candidate_negative_ratio"].max()))].copy()
    plt.figure(figsize=(5.8, 3.0))
    for method, group in work.groupby("method"):
        group = group.sort_values("prevalence")
        plt.plot(group["prevalence"], group["expected_reviews_per_true_positive_mean"], marker="o", label=method)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Assumed prevalence")
    plt.ylabel("Expected reviews per true positive")
    plt.title("Review workload under rare-positive prevalence", loc="left", fontweight="bold")
    plt.grid(color="#D8DEE9", alpha=0.7, which="both")
    plt.legend(frameon=False)
    savefig("review_workload_vs_prevalence")

    rel = reliability[reliability["method"].isin(["score_only_logistic", "invariant_logistic_poly2"])].copy()
    rel_summary = rel.groupby(["method", "bin"], as_index=False).agg(
        mean_confidence=("mean_confidence", "mean"),
        empirical_positive_rate=("empirical_positive_rate", "mean"),
    )
    plt.figure(figsize=(4.8, 3.2))
    plt.plot([0, 1], [0, 1], color=PALETTE["gray"], linestyle="--", linewidth=1)
    for method, group in rel_summary.groupby("method"):
        plt.plot(group["mean_confidence"], group["empirical_positive_rate"], marker="o", label=method)
    plt.xlabel("Mean predicted score")
    plt.ylabel("Empirical positive rate")
    plt.title("Reliability curves on VCSL public-label audit tier", loc="left", fontweight="bold")
    plt.grid(color="#D8DEE9", alpha=0.7)
    plt.legend(frameon=False)
    savefig("calibration_reliability_curves")

    th = threshold[threshold["method"].isin(["score_only_logistic", "invariant_logistic_poly2"])].copy()
    th_summary = th.groupby(["method", "client_id"], as_index=False)["threshold_at_95_recall"].mean()
    plt.figure(figsize=(5.8, 3.0))
    for method, group in th_summary.groupby("method"):
        plt.plot(group["client_id"], group["threshold_at_95_recall"], marker="o", linewidth=1.2, label=method)
    plt.xlabel("Client id")
    plt.ylabel("Threshold at 95% recall")
    plt.title("Client threshold stability diagnostic", loc="left", fontweight="bold")
    plt.grid(color="#D8DEE9", alpha=0.7)
    plt.legend(frameon=False)
    savefig("threshold_stability_by_client")

    rob = robustness[robustness["attack_scenario"].isin(["clean", "label_flip_clients", "malicious_high_confidence_updates"])].copy()
    rob = rob[rob["method"].isin(["fedavg_mean", "coordinate_median", "trimmed_mean", "krum"])]
    pivot = rob.pivot_table(index="attack_scenario", columns="method", values="pr_auc_mean")
    plt.figure(figsize=(6.2, 3.2))
    x = np.arange(len(pivot.index))
    width = 0.18
    for i, method in enumerate(pivot.columns):
        plt.bar(x + (i - 1.5) * width, pivot[method], width, label=method)
    plt.xticks(x, pivot.index, rotation=15, ha="right")
    plt.ylabel("PR-AUC")
    plt.ylim(0, 1.0)
    plt.title("Robust aggregation stress for compact updates", loc="left", fontweight="bold")
    plt.grid(axis="y", color="#D8DEE9", alpha=0.7)
    plt.legend(frameon=False, ncol=2)
    savefig("robust_aggregation_stress")

    if not paillier.empty:
        row = paillier.iloc[0]
        labels = ["HE proxy", "Paillier"]
        expansions = [row["proxy_ciphertext_expansion"], row["paillier_ciphertext_expansion"]]
        errors = [row["proxy_max_abs_error_vs_plain"], row["paillier_max_abs_error_vs_plain"]]
        plt.figure(figsize=(4.8, 3.0))
        ax = plt.gca()
        ax.bar(labels, expansions, color=[PALETTE["cyan"], PALETTE["purple"]], alpha=0.85)
        ax.set_ylabel("Ciphertext/update byte expansion")
        ax.set_title("Compact-update HE accounting", loc="left", fontweight="bold")
        for i, err in enumerate(errors):
            ax.text(i, expansions[i], f"err={err:.1e}", ha="center", va="bottom", fontsize=7)
        ax.grid(axis="y", color="#D8DEE9", alpha=0.7)
        savefig("paillier_update_overhead")


def artifact_checksums() -> pd.DataFrame:
    rows = []
    for folder in [TABLES, FIGURES]:
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() not in {".csv", ".json", ".pdf", ".png"}:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({"path": str(path.relative_to(MANUSCRIPT)).replace("\\", "/"), "sha256": digest, "bytes": path.stat().st_size})
    out = pd.DataFrame(rows)
    out.to_csv(MANUSCRIPT / "artifact_checksums.csv", index=False, lineterminator="\n")
    return out


def write_summary(payload: dict) -> None:
    write_json(RESULTS / "review_blocker_experiment_summary.json", payload)


def validate_finite_frame(frame: pd.DataFrame, *, name: str) -> None:
    numeric = frame.select_dtypes(include="number")
    if numeric.size and not np.isfinite(numeric.to_numpy(dtype=float)).all():
        bad = [
            column
            for column in numeric.columns
            if not np.isfinite(numeric[column].to_numpy(dtype=float)).all()
        ]
        raise ValueError(f"{name} contains non-finite numeric columns: {', '.join(bad)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reviewer-blocker strengthening experiments for FedTwin-CryptID.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--max-train-pairs", type=int, default=2200)
    parser.add_argument("--max-test-pairs", type=int, default=900)
    parser.add_argument("--vcsl-metadata-dir", default=str(ROOT / "public_data" / "vcsl_metadata"))
    parser.add_argument("--archived-results-dir", default=str(ROOT / "archived_results"))
    parser.add_argument(
        "--confirmatory-results-dir",
        default=str(ROOT / "paper_results" / "v1" / "new_runs"),
    )
    parser.add_argument("--manuscript-dir", default=str(ROOT / "paper"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "review_blocker_experiments"))
    parser.add_argument("--min-open-set-queries", type=int, default=100)
    parser.add_argument("--paillier-key-bits", type=int, default=2048)
    parser.add_argument("--robust-rounds", type=int, default=20)
    parser.add_argument("--robust-local-epochs", type=int, default=3)
    parser.add_argument("--attacker-fractions", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    parser.add_argument("--attacks", choices=["label_flip", "sign_flip"], nargs="+", default=["label_flip", "sign_flip"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.paillier_key_bits < 2048:
        raise ValueError("reported Paillier experiments require --paillier-key-bits >= 2048")
    if args.min_open_set_queries < 100:
        raise ValueError("the 1:10,000 audit requires at least 100 queries")
    if args.robust_rounds < 1 or args.robust_local_epochs < 1:
        raise ValueError("robustness rounds and local epochs must be positive")
    if any(value < 0 or value >= 1 for value in args.attacker_fractions):
        raise ValueError("attacker fractions must lie in [0, 1)")
    configure_paths(args)
    ensure_dirs()
    setup_plot()
    if args.mode == "full":
        args.max_train_pairs = max(args.max_train_pairs, 2500)
        args.max_test_pairs = max(args.max_test_pairs, 800)

    ranking, open_set, workload = run_query_and_open_set(args)
    reliability, threshold, stat_tests = run_calibration_and_stat_tests(args)
    personalization, per_client = run_personalization(args)
    av_metrics, av_scenarios = run_synthetic_av(args)
    status, secureagg, quantized_proxy, paillier = protected_aggregation_tables(args)
    privacy = run_privacy_attacks(args)
    robustness = run_robustness_stress(args)
    scaling, vector = receipt_scaling(args)
    for name, frame in {
        "query ranking": ranking,
        "open-set prevalence": open_set,
        "review workload": workload,
        "calibration reliability": reliability,
        "threshold stability": threshold,
        "paired statistics": stat_tests,
        "personalization": personalization,
        "per-client metrics": per_client,
        "synthetic A/V metrics": av_metrics,
        "synthetic A/V scenarios": av_scenarios,
        "SecureAgg simulation": secureagg,
        "quantized/Paillier status": quantized_proxy,
        "Paillier accounting": paillier,
        "privacy attacks": privacy,
        "robustness": robustness,
        "receipt scaling": scaling,
    }.items():
        validate_finite_frame(frame, name=name)
    plot_outputs(
        av_metrics,
        av_scenarios,
        ranking,
        open_set,
        personalization,
        per_client,
        privacy,
        workload,
        reliability,
        threshold,
        robustness,
        paillier,
    )
    checksums = artifact_checksums()
    write_summary(
        {
            "mode": args.mode,
            "seeds": args.seeds,
            "tables": [
                "query_ranking_metrics.csv",
                "open_set_prevalence_metrics.csv",
                "prevalence_workload_metrics.csv",
                "calibration_reliability_curves.csv",
                "threshold_stability_by_client.csv",
                "paired_permutation_tests.csv",
                "personalization_strengthening.csv",
                "per_client_metrics.csv",
                "av_sync_tier_metrics.csv",
                "av_sync_operating_points.csv",
                "protected_aggregation_status.csv",
                "secureagg_dropout_or_proxy.csv",
                "quantized_transport_proxy.csv",
                "paillier_update_aggregation.csv",
                "privacy_attack_results.csv",
                "robustness_stress.csv",
                "receipt_scaling.csv",
                "receipt_test_vector.csv",
            ],
            "figures": [
                "av_sync_tier_pr_auc.pdf",
                "av_conflict_breakdown.pdf",
                "query_recall_at_k.pdf",
                "open_set_precision_curves.pdf",
                "personalization_vs_local.pdf",
                "vcsl_isc_client_variability.pdf",
                "privacy_attack_summary.pdf",
                "review_workload_vs_prevalence.pdf",
                "calibration_reliability_curves.pdf",
                "threshold_stability_by_client.pdf",
                "robust_aggregation_stress.pdf",
                "paillier_update_overhead.pdf",
            ],
            "checksums": int(len(checksums)),
            "natural_synchronized_av_corpus_added": False,
            "synthetic_av_stress_tier_added": True,
            "real_paillier_compact_update_aggregation_added": True,
            "paillier_key_bits": int(paillier["key_bits"].iloc[0]) if not paillier.empty else None,
            "vcsl_isc_new_pair_level_methods_run": bool(
                (CONFIRMATORY / "vcsl_isc_confirmatory" / "metrics_detection.csv").is_file()
            ),
            "vcsl_isc_reason": (
                "checksum-verifiable released-descriptor metrics were included"
                if (CONFIRMATORY / "vcsl_isc_confirmatory" / "metrics_detection.csv").is_file()
                else "confirmatory descriptor metrics were unavailable; archived summary fallback was used"
            ),
        }
    )
    print(f"wrote reviewer-blocker tables to {TABLES}")
    print(f"wrote reviewer-blocker figures to {FIGURES}")
    print(f"wrote summary to {RESULTS / 'review_blocker_experiment_summary.json'}")


if __name__ == "__main__":
    main()
