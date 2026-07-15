from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT / "paper"
TABLES = MANUSCRIPT / "tables"
FIGURES = MANUSCRIPT / "figures"
PULL = ROOT / "archived_results"
OUT = ROOT / "outputs" / "tmm_revision_strengthening"
VCSL_METADATA = ROOT / "public_data" / "vcsl_metadata"

sys.path.insert(0, str(ROOT))

from fedtwin.data import generate_vcsl_public_benchmark  # noqa: E402
from fedtwin.features import standardize_train_test  # noqa: E402
from fedtwin.ledger import make_receipts  # noqa: E402
from fedtwin.metrics import detection_metrics  # noqa: E402
from fedtwin.models import LogisticHead, train_logistic  # noqa: E402
from fedtwin.statistics import holm_adjust  # noqa: E402
from run_benchmark import expand_feature_map, minmax_score, modality_indices  # noqa: E402

SEEDS = [31, 37, 41]
PALETTE = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "purple": "#B279A2",
    "cyan": "#72B7B2",
    "dark": "#2F3A45",
}


def configure_paths(args: argparse.Namespace) -> None:
    global MANUSCRIPT, TABLES, FIGURES, PULL, OUT, VCSL_METADATA
    MANUSCRIPT = Path(args.manuscript_dir).resolve()
    TABLES = MANUSCRIPT / "tables"
    FIGURES = MANUSCRIPT / "figures"
    PULL = Path(args.archived_results_dir).resolve()
    OUT = Path(args.output_dir).resolve()
    VCSL_METADATA = Path(args.vcsl_metadata_dir).resolve()


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


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


def atoms(name: str) -> list[str]:
    return name.replace("__sq", "").split("__x__")


def has_prefix(atom: str, prefixes: tuple[str, ...]) -> bool:
    return any(atom.startswith(prefix) for prefix in prefixes)


def keep_without(name: str, *, drop_exact: set[str] | None = None, drop_prefix: tuple[str, ...] = ()) -> bool:
    drop_exact = drop_exact or set()
    for atom in atoms(name):
        if atom in drop_exact or has_prefix(atom, drop_prefix):
            return False
    return True


def select_indices(feature_names: list[str], policy: str) -> list[int]:
    if policy == "all":
        return list(range(len(feature_names)))
    if policy == "invariant":
        return modality_indices(feature_names, "multimodal", "invariant")
    if policy == "linear_all":
        return [i for i, n in enumerate(feature_names) if "__" not in n]
    if policy == "visual_temporal":
        allowed = {
            "video_score",
            "temporal_score",
            "frame_ratio",
            "video_reliability",
            "video_rel_score",
            "temporal_rel_score",
            "frame_video_score",
            "max_modality_score",
        }
        return [i for i, n in enumerate(feature_names) if all(a in allowed for a in atoms(n))]
    if policy == "audio":
        allowed = {"audio_score", "audio_reliability", "audio_rel_score", "category_audio_score", "max_modality_score"}
        return [i for i, n in enumerate(feature_names) if all(a in allowed for a in atoms(n))]
    if policy == "score_only":
        allowed = {"video_score", "audio_score", "temporal_score", "max_modality_score", "mean_modality_score"}
        return [i for i, n in enumerate(feature_names) if all(a in allowed for a in atoms(n))]
    if policy == "no_context":
        drops = {"metadata_score", "same_category", "category_audio_score", "client_norm"}
        return [i for i, n in enumerate(feature_names) if keep_without(n, drop_exact=drops, drop_prefix=("category_",))]
    if policy == "no_reliability":
        drops = {
            "video_reliability",
            "audio_reliability",
            "video_rel_score",
            "audio_rel_score",
            "temporal_rel_score",
            "fusion_prior",
        }
        return [i for i, n in enumerate(feature_names) if keep_without(n, drop_exact=drops)]
    if policy == "no_interactions":
        return [i for i, n in enumerate(feature_names) if "__" not in n]
    raise ValueError(f"unknown policy {policy}")


def fit_head(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, epochs: int = 140) -> np.ndarray:
    model = train_logistic(x_train, y_train, epochs=epochs, lr=0.08, l2=1e-4)
    return model.predict_proba(x_test)


def train_fedopt(
    x: np.ndarray,
    y: np.ndarray,
    clients: np.ndarray,
    *,
    optimizer: str,
    rounds: int = 30,
    local_epochs: int = 5,
    client_lr: float = 0.08,
    server_lr: float = 0.65,
    beta1: float = 0.9,
    beta2: float = 0.99,
    tau: float = 1e-6,
) -> LogisticHead:
    client_ids = sorted(set(map(int, clients)))
    global_model = LogisticHead.zeros(x.shape[1])
    m = np.zeros_like(global_model.weights)
    v = np.zeros_like(global_model.weights)
    for _ in range(rounds):
        base_w = global_model.weights.copy()
        updates = []
        weights = []
        for client_id in client_ids:
            idx = clients == client_id
            local = train_logistic(x[idx], y[idx], initial=base_w, epochs=local_epochs, lr=client_lr)
            updates.append(local.weights - base_w)
            weights.append(float(idx.sum()))
        w = np.asarray(weights, dtype=float)
        w = w / max(w.sum(), 1e-12)
        delta = np.zeros_like(base_w)
        for wi, update in zip(w, updates, strict=True):
            delta += wi * update
        m = beta1 * m + (1.0 - beta1) * delta
        if optimizer == "fedadam":
            v = beta2 * v + (1.0 - beta2) * (delta * delta)
        elif optimizer == "fedyogi":
            v = v - (1.0 - beta2) * np.sign(v - delta * delta) * (delta * delta)
            v = np.maximum(v, 0.0)
        else:
            raise ValueError(optimizer)
        global_model.weights = base_w + server_lr * m / (np.sqrt(v) + tau)
    return global_model


def summarize_metric(rows: list[dict], y: np.ndarray, score: np.ndarray, *, method: str, seed: int, group: str) -> None:
    row = detection_metrics(y, score, method=method, modality="multimodal")
    row["seed"] = seed
    row["group"] = group
    rows.append(row)


def precision_from_recall_fpr(prevalence: float, recall: float, fpr: float) -> float:
    tp = prevalence * recall
    fp = (1.0 - prevalence) * fpr
    return float(tp / max(tp + fp, 1e-15))


def fpr_at_recall(y_true: np.ndarray, score: np.ndarray, target_recall: float = 0.95) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    recall = tp / positives
    fpr = fp / negatives
    feasible = fpr[recall >= target_recall]
    return float(feasible.min()) if feasible.size else 1.0


def paired_bootstrap_table(paired_cases: list[dict], n_boot: int = 5000) -> pd.DataFrame:
    comparisons = [
        ("score_only", "default_invariant"),
        ("default_invariant", "mean_score_fusion"),
        ("default_invariant", "platt_calibrated_score"),
        ("default_invariant", "fedyogi"),
        ("default_invariant", "all_fields"),
    ]
    y = np.concatenate([case["y"] for case in paired_cases])
    scores = {name: np.concatenate([case["scores"][name] for case in paired_cases]) for name in paired_cases[0]["scores"]}
    rng = np.random.default_rng(20260530)
    rows = []
    n = len(y)
    for left, right in comparisons:
        left_score = scores[left]
        right_score = scores[right]
        base_left_ap = average_precision_score(y, left_score)
        base_right_ap = average_precision_score(y, right_score)
        base_left_fpr = fpr_at_recall(y, left_score)
        base_right_fpr = fpr_at_recall(y, right_score)
        ap_deltas = []
        fpr_deltas = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            # Degenerate resamples are extremely unlikely, but skip them defensively.
            if len(np.unique(y[idx])) < 2:
                continue
            ap_deltas.append(average_precision_score(y[idx], left_score[idx]) - average_precision_score(y[idx], right_score[idx]))
            fpr_deltas.append(fpr_at_recall(y[idx], left_score[idx]) - fpr_at_recall(y[idx], right_score[idx]))
        ap_arr = np.asarray(ap_deltas, dtype=float)
        fpr_arr = np.asarray(fpr_deltas, dtype=float)
        lower_tail = int(np.count_nonzero(ap_arr <= 0.0))
        upper_tail = int(np.count_nonzero(ap_arr >= 0.0))
        p_two_sided = min(1.0, 2.0 * (min(lower_tail, upper_tail) + 1) / (len(ap_arr) + 1))
        rows.append(
            {
                "left_method": left,
                "right_method": right,
                "delta_pr_auc": base_left_ap - base_right_ap,
                "delta_pr_auc_ci_low": float(np.quantile(ap_arr, 0.025)),
                "delta_pr_auc_ci_high": float(np.quantile(ap_arr, 0.975)),
                "delta_pr_auc_boot_p_two_sided": float(p_two_sided),
                "delta_fpr_at_95_recall": base_left_fpr - base_right_fpr,
                "delta_fpr95_ci_low": float(np.quantile(fpr_arr, 0.025)),
                "delta_fpr95_ci_high": float(np.quantile(fpr_arr, 0.975)),
                "pooled_pairs": n,
                "bootstrap_resamples": len(ap_arr),
            }
        )
    frame = pd.DataFrame(rows)
    frame["holm_adjusted_p_value"] = holm_adjust(frame["delta_pr_auc_boot_p_two_sided"].to_numpy())
    return frame


def load_detection_summary(run: str) -> pd.DataFrame:
    return pd.read_csv(PULL / run / "summary_artifacts" / "detection_summary.csv")


def build_feature_schema() -> pd.DataFrame:
    rows = [
        ("video_score", "visual evidence", "cosine or descriptor-match score from the upstream visual branch", 1, "yes", "may reveal visual similarity distribution"),
        ("audio_score", "audio evidence", "cosine or fingerprint/descriptor score from the upstream audio branch", 1, "yes", "may reveal audio similarity distribution"),
        ("temporal_score", "temporal evidence", "temporal/frame-consistency score or audio transform consistency", 1, "yes", "may reveal copy duration or segment structure"),
        ("metadata_score", "permitted metadata", "same category or allowed benchmark metadata score", 1, "optional", "can leak domain/category information"),
        ("video_reliability", "reliability", "visual branch confidence from frame ratio or transform reliability", 1, "yes", "can reveal transformation quality"),
        ("audio_reliability", "reliability", "audio branch confidence from transform reliability", 1, "yes", "can reveal modality degradation"),
        ("modality_gap", "cross-modal agreement", "absolute difference between visual and audio evidence", 1, "yes", "can reveal modality disagreement"),
        ("fusion_prior", "cross-modal agreement", "weighted reliability-aware visual/audio score", 1, "yes", "derived score leakage only"),
        ("frame_ratio", "temporal evidence", "min query/reference frames divided by max frames", 1, "yes", "can reveal approximate clip-length relation"),
        ("same_category", "permitted metadata", "indicator for same VCSL category or FMA genre", 1, "optional", "can leak category/domain membership"),
        ("derived products", "derived evidence", "video/audio reliability products, max/mean/product scores", 8, "yes", "derived from evidence scores"),
        ("category_*", "client/domain context", "one-hot public category or genre used only in all-feature audit runs", "variable", "optional", "high client/domain leakage risk"),
        ("client_norm", "client context", "normalized client id used for leakage audit and non-IID analysis", 1, "no for default model", "direct client identity leakage"),
        ("poly2 interactions", "feature map", "squares and pairwise products among selected evidence roots", "variable", "yes if roots allowed", "inherits source feature leakage"),
    ]
    df = pd.DataFrame(rows, columns=["feature_or_family", "group", "definition", "dimension", "online_available", "leakage_note"])
    df.to_csv(TABLES / "feature_schema.csv", index=False)
    return df


def build_hyperparameter_table() -> pd.DataFrame:
    rows = [
        ("fusion head", "model", "logistic regression head with bias"),
        ("feature map", "default", "poly2 on selected invariant score, reliability-weighted score, and agreement roots"),
        ("feature policy", "main results", "invariant multimodal features; category/client fields excluded from default detector"),
        ("standardization", "fit", "mean/std fit on train split and applied to test split"),
        ("centralized epochs", "value", "140 full-batch gradient epochs"),
        ("local epochs", "value", "8 for main HPC runs, 5 for FedOpt strengthening run"),
        ("rounds", "value", "30--40 communication rounds depending on tier"),
        ("learning rate", "value", "0.08 client/head step size"),
        ("regularization", "value", "L2=1e-4, bias unregularized"),
        ("FedProx mu", "value", "0.02 in main experiments"),
        ("FedAdam beta1/beta2/tau", "value", "0.9 / 0.99 / 1e-6 in strengthening run"),
        ("FedYogi beta1/beta2/tau", "value", "0.9 / 0.99 / 1e-6 in strengthening run"),
        ("FedOpt server learning rate", "value", "0.65 in strengthening run"),
        ("HE proxy scale", "value", "1e6 integer quantization scale; 16x packed-update expansion reported"),
        ("decision threshold", "value", "0.5 for default precision/recall; FPR@95R by threshold sweep"),
    ]
    df = pd.DataFrame(rows, columns=["item", "field", "value"])
    df.to_csv(TABLES / "model_hyperparameters.csv", index=False)
    return df


def run_vcsl_strengthening() -> dict[str, pd.DataFrame]:
    baseline_rows: list[dict] = []
    ablation_rows: list[dict] = []
    leakage_rows: list[dict] = []
    signature_rows: list[dict] = []
    paired_cases: list[dict] = []
    dimensionality_rows: list[dict] = []

    for seed in SEEDS:
        data = generate_vcsl_public_benchmark(
            metadata_dir=VCSL_METADATA,
            clients=11,
            dim=64,
            max_train_pairs=15000,
            max_test_pairs=7500,
            negative_ratio=1.0,
            seed=seed,
        )
        x_train_raw, x_test_raw, feature_names = expand_feature_map(data.x_train, data.x_test, data.feature_names, mode="poly2")
        x_train_std, x_test_std, _, _ = standardize_train_test(x_train_raw, x_test_raw)
        if seed == SEEDS[0]:
            for policy, label in [
                ("invariant", "default_invariant"),
                ("all", "all_fields"),
                ("score_only", "score_only"),
                ("visual_temporal", "visual_temporal"),
                ("audio", "audio_only"),
                ("no_context", "all_fields_minus_context"),
                ("no_reliability", "no_reliability"),
                ("no_interactions", "linear_no_poly2"),
            ]:
                idx = select_indices(feature_names, policy)
                dimensionality_rows.append(
                    {
                        "policy": label,
                        "feature_dimension_after_map": len(idx),
                        "total_poly2_dimension": len(feature_names),
                        "source_root_dimension": len(data.feature_names),
                    }
                )

        video_score = minmax_score(0.65 * data.x_test[:, data.feature_names.index("video_score")] + 0.35 * data.x_test[:, data.feature_names.index("temporal_score")])
        audio_score = minmax_score(data.x_test[:, data.feature_names.index("audio_score")])
        early_fusion = minmax_score(0.5 * video_score + 0.5 * audio_score)
        summarize_metric(baseline_rows, data.y_test, video_score, method="video_similarity", seed=seed, group="fixed evidence")
        summarize_metric(baseline_rows, data.y_test, audio_score, method="audio_similarity", seed=seed, group="fixed evidence")
        summarize_metric(baseline_rows, data.y_test, early_fusion, method="mean_score_fusion", seed=seed, group="fixed evidence")

        train_video = minmax_score(0.65 * data.x_train[:, data.feature_names.index("video_score")] + 0.35 * data.x_train[:, data.feature_names.index("temporal_score")])
        train_audio = minmax_score(data.x_train[:, data.feature_names.index("audio_score")])
        train_fusion = minmax_score(0.5 * train_video + 0.5 * train_audio)

        platt = LogisticRegression(max_iter=500, solver="lbfgs")
        platt.fit(train_fusion.reshape(-1, 1), data.y_train)
        platt_score = platt.predict_proba(early_fusion.reshape(-1, 1))[:, 1]
        summarize_metric(baseline_rows, data.y_test, platt_score, method="platt_calibrated_score", seed=seed, group="calibration")

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(train_fusion, data.y_train)
        iso_score = np.asarray(iso.predict(early_fusion), dtype=float)
        summarize_metric(baseline_rows, data.y_test, iso_score, method="isotonic_calibrated_score", seed=seed, group="calibration")

        method_scores = {
            "mean_score_fusion": early_fusion,
            "platt_calibrated_score": platt_score,
            "isotonic_calibrated_score": iso_score,
        }

        invariant_idx = select_indices(feature_names, "invariant")
        invariant_score = fit_head(x_train_std[:, invariant_idx], data.y_train, x_test_std[:, invariant_idx])
        summarize_metric(baseline_rows, data.y_test, invariant_score, method="logistic_poly2_invariant", seed=seed, group="fusion head")
        method_scores["default_invariant"] = invariant_score

        all_idx = select_indices(feature_names, "all")
        all_score = fit_head(x_train_std[:, all_idx], data.y_train, x_test_std[:, all_idx])
        summarize_metric(baseline_rows, data.y_test, all_score, method="logistic_poly2_all_fields", seed=seed, group="fusion head")
        method_scores["all_fields"] = all_score

        hgb = HistGradientBoostingClassifier(max_iter=140, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1e-3, random_state=seed)
        hgb.fit(x_train_std[:, invariant_idx], data.y_train)
        hgb_score = hgb.predict_proba(x_test_std[:, invariant_idx])[:, 1]
        summarize_metric(baseline_rows, data.y_test, hgb_score, method="hist_gradient_boosting_invariant", seed=seed, group="fusion head")

        for opt in ("fedadam", "fedyogi"):
            fed_model = train_fedopt(x_train_std[:, invariant_idx], data.y_train, data.client_train, optimizer=opt)
            fed_score = fed_model.predict_proba(x_test_std[:, invariant_idx])
            summarize_metric(baseline_rows, data.y_test, fed_score, method=opt, seed=seed, group="fedopt")
            method_scores[opt] = fed_score

        for policy, label in [
            ("invariant", "default invariant"),
            ("all", "all fields"),
            ("linear_all", "linear all fields"),
            ("score_only", "score only"),
            ("visual_temporal", "visual+temporal"),
            ("audio", "audio only"),
            ("no_context", "all fields minus context"),
            ("no_reliability", "remove reliability"),
            ("no_interactions", "remove poly2 interactions"),
        ]:
            idx = select_indices(feature_names, policy)
            score = fit_head(x_train_std[:, idx], data.y_train, x_test_std[:, idx])
            summarize_metric(ablation_rows, data.y_test, score, method=label, seed=seed, group="ablation")
            if label == "score only":
                method_scores["score_only"] = score
                summarize_metric(baseline_rows, data.y_test, score, method="logistic_score_only", seed=seed, group="calibration")

        full_model = train_logistic(x_train_std[:, invariant_idx], data.y_train, epochs=140, lr=0.08, l2=1e-4)
        train_prob = full_model.predict_proba(x_train_std[:, invariant_idx])
        test_prob = full_model.predict_proba(x_test_std[:, invariant_idx])
        attack_score = np.concatenate([np.maximum(train_prob, 1 - train_prob), np.maximum(test_prob, 1 - test_prob)])
        attack_y = np.concatenate([np.ones_like(train_prob), np.zeros_like(test_prob)])
        membership_auc = roc_auc_score(attack_y, attack_score)
        leakage_rows.append({"seed": seed, "attack": "confidence_membership_auc", "setting": "default invariant head", "value": membership_auc})

        for policy, label in [("all", "all fields"), ("no_context", "context removed"), ("invariant", "default invariant")]:
            idx = select_indices(feature_names, policy)
            scaler = StandardScaler()
            xtr = scaler.fit_transform(x_train_raw[:, idx])
            xte = scaler.transform(x_test_raw[:, idx])
            clf = LogisticRegression(max_iter=600, solver="lbfgs")
            clf.fit(xtr, data.client_train)
            pred = clf.predict(xte)
            leakage_rows.append(
                {
                    "seed": seed,
                    "attack": "client_identity_balanced_accuracy",
                    "setting": label,
                    "value": balanced_accuracy_score(data.client_test, pred),
                }
            )

        receipts, summary = make_receipts(
            scores=invariant_score,
            y=data.y_test,
            clients=data.client_test,
            model_digest=f"vcsl-public-strengthening-{seed}",
            limit=1000,
            threshold=0.5,
            sign_receipts=True,
        )
        summary["seed"] = seed
        summary["example_schema_version"] = receipts[0].schema_version if receipts else ""
        summary["example_signature_len"] = len(receipts[0].signature) if receipts else 0
        signature_rows.append(summary)
        paired_cases.append({"seed": seed, "y": data.y_test.copy(), "scores": method_scores})

    baseline = pd.DataFrame(baseline_rows)
    ablation = pd.DataFrame(ablation_rows)
    leakage = pd.DataFrame(leakage_rows)
    receipts = pd.DataFrame(signature_rows)
    paired = paired_bootstrap_table(paired_cases)
    dimensionality = pd.DataFrame(dimensionality_rows)

    baseline.to_csv(OUT / "fixed_evidence_baselines_raw.csv", index=False)
    ablation.to_csv(OUT / "feature_ablation_raw.csv", index=False)
    leakage.to_csv(OUT / "privacy_leakage_raw.csv", index=False)
    receipts.to_csv(OUT / "receipt_v2_raw.csv", index=False)
    paired.to_csv(OUT / "paired_bootstrap_raw.csv", index=False)
    dimensionality.to_csv(TABLES / "model_dimensionality.csv", index=False)

    return {
        "baseline": baseline,
        "ablation": ablation,
        "leakage": leakage,
        "receipts": receipts,
        "paired": paired,
        "dimensionality": dimensionality,
    }


def summarize_strengthening(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for key, df in raw.items():
        if key in {"paired", "dimensionality"}:
            out[key] = df
            continue
        numeric_cols = [c for c in df.columns if c not in {"method", "group", "attack", "setting", "ledger_type", "schema_version", "example_schema_version"} and pd.api.types.is_numeric_dtype(df[c])]
        group_cols = [c for c in ["method", "group", "attack", "setting", "ledger_type", "schema_version", "example_schema_version"] if c in df.columns]
        summary = df.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
        summary.columns = ["_".join([x for x in col if x]).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
        out[key] = summary
    out["baseline"].to_csv(TABLES / "fixed_evidence_baselines.csv", index=False)
    out["ablation"].to_csv(TABLES / "feature_ablation.csv", index=False)
    out["leakage"].to_csv(TABLES / "privacy_leakage.csv", index=False)
    out["receipts"].to_csv(TABLES / "receipt_v2_summary.csv", index=False)
    out["paired"].to_csv(TABLES / "paired_bootstrap_tests.csv", index=False)
    out["dimensionality"].to_csv(TABLES / "model_dimensionality.csv", index=False)
    return out


def build_prevalence_table() -> pd.DataFrame:
    runs = {
        "VCSL public labels": "hpc_vcsl_public_label_run",
        "VCSL ISC visual": "hpc_vcsl_isc_feature_run_fedavgft",
        "FMA audio 5 s": "hpc_fma_audio_shortclip_stress",
    }
    methods = ["centralized_multimodal", "fedavg_secureagg", "fedavgft_plain", "local_multimodal", "video_similarity", "audio_similarity"]
    prevalences = [1e-2, 1e-3, 1e-5]
    rows = []
    for tier, run in runs.items():
        df = load_detection_summary(run)
        for _, row in df[df["method"].isin(methods)].iterrows():
            fpr = float(row["fpr_at_95_recall_mean"])
            for prevalence in prevalences:
                rows.append(
                    {
                        "tier": tier,
                        "method": row["method"],
                        "recall_target": 0.95,
                        "fpr_at_95_recall": fpr,
                        "prevalence": prevalence,
                        "expected_precision_at_95_recall": precision_from_recall_fpr(prevalence, 0.95, fpr),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "realistic_prevalence_precision.csv", index=False)
    return df


def plot_strengthening(summary: dict[str, pd.DataFrame], prevalence: pd.DataFrame) -> None:
    baseline = summary["baseline"].sort_values("pr_auc_mean", ascending=False)
    top = baseline[["method", "group", "pr_auc_mean", "pr_auc_std", "fpr_at_95_recall_mean"]].head(9)
    plt.figure(figsize=(7.1, 3.2))
    colors = [PALETTE["green"] if "logistic" in m or "boosting" in m else PALETTE["blue"] for m in top["method"]]
    plt.barh(top["method"], top["pr_auc_mean"], xerr=top["pr_auc_std"].fillna(0.0), color=colors, alpha=0.9)
    plt.gca().invert_yaxis()
    plt.xlim(0.80, 0.94)
    plt.xlabel("PR-AUC")
    plt.title("Fixed-evidence calibration baselines on VCSL public-label audit subset", loc="left", fontweight="bold")
    plt.grid(axis="x", color="#D8DEE9", linewidth=0.7, alpha=0.7)
    savefig("fig09_fixed_evidence_baselines")

    ablation = summary["ablation"].copy()
    order = [
        "default invariant",
        "all fields",
        "all fields minus context",
        "remove reliability",
        "score only",
        "visual+temporal",
        "audio only",
        "remove poly2 interactions",
    ]
    ablation["method"] = pd.Categorical(ablation["method"], categories=order, ordered=True)
    ablation = ablation.sort_values("method")
    plt.figure(figsize=(7.1, 3.35))
    plt.barh(ablation["method"].astype(str), ablation["pr_auc_mean"], xerr=ablation["pr_auc_std"].fillna(0.0), color=PALETTE["purple"], alpha=0.85)
    plt.gca().invert_yaxis()
    plt.xlim(0.68, 0.94)
    plt.xlabel("PR-AUC")
    plt.title("Feature ablation on VCSL public-label audit subset", loc="left", fontweight="bold")
    plt.grid(axis="x", color="#D8DEE9", linewidth=0.7, alpha=0.7)
    savefig("fig10_feature_ablation")

    leakage = summary["leakage"].copy()
    client = leakage[leakage["attack"].eq("client_identity_balanced_accuracy")].sort_values("value_mean", ascending=True)
    membership = leakage[leakage["attack"].eq("confidence_membership_auc")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].barh(client["setting"], client["value_mean"], xerr=client["value_std"].fillna(0.0), color=PALETTE["red"], alpha=0.82)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Balanced accuracy")
    axes[0].set_title("Client-context leakage audit", loc="left", fontweight="bold")
    axes[0].grid(axis="x", color="#D8DEE9", linewidth=0.7, alpha=0.7)
    axes[1].bar(["membership\nconfidence"], membership["value_mean"], yerr=membership["value_std"].fillna(0.0), color=PALETTE["cyan"], alpha=0.9)
    axes[1].axhline(0.5, color="#374151", linestyle="--", linewidth=1.0)
    axes[1].set_ylim(0.45, 0.65)
    axes[1].set_ylabel("Attack AUC")
    axes[1].set_title("Membership-leakage sanity check", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#D8DEE9", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    savefig("fig11_privacy_leakage_audit")

    plot = prevalence[prevalence["method"].isin(["fedavg_secureagg", "fedavgft_plain", "local_multimodal", "video_similarity", "audio_similarity"])].copy()
    plot = plot[plot["prevalence"].isin([1e-2, 1e-3, 1e-5])]
    pivot = plot.groupby(["tier", "prevalence"], as_index=False)["expected_precision_at_95_recall"].max()
    pivot["label"] = pivot["prevalence"].map(lambda p: f"1e{int(math.log10(p))}")
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    tiers = list(pivot["tier"].unique())
    x = np.arange(len(tiers))
    width = 0.24
    for offset, prevalence_value, color in [(-width, 1e-2, PALETTE["green"]), (0.0, 1e-3, PALETTE["orange"]), (width, 1e-5, PALETTE["red"])]:
        vals = []
        for tier in tiers:
            hit = pivot[(pivot["tier"].eq(tier)) & (pivot["prevalence"].eq(prevalence_value))]
            vals.append(float(hit["expected_precision_at_95_recall"].iloc[0]) if len(hit) else np.nan)
        ax.bar(x + offset, vals, width, label=f"prevalence {prevalence_value:g}", color=color, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, rotation=18, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.0)
    ax.set_ylabel("Expected precision at 95% recall")
    ax.set_title("High-recall precision under realistic rare-positive prevalence", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.7, alpha=0.7, which="both")
    ax.legend(frameon=False, ncol=3, loc="lower left")
    savefig("fig12_prevalence_precision")


def write_summary_json(summary: dict[str, pd.DataFrame], prevalence: pd.DataFrame) -> None:
    payload = {
        "generated_tables": sorted(p.name for p in TABLES.glob("*.csv")),
        "generated_figures": sorted(p.name for p in FIGURES.glob("fig09_*.pdf")) + sorted(p.name for p in FIGURES.glob("fig10_*.pdf")) + sorted(p.name for p in FIGURES.glob("fig11_*.pdf")) + sorted(p.name for p in FIGURES.glob("fig12_*.pdf")),
        "best_fixed_evidence_baseline": summary["baseline"].sort_values("pr_auc_mean", ascending=False).iloc[0].to_dict(),
        "membership_attack_auc": summary["leakage"][summary["leakage"]["attack"].eq("confidence_membership_auc")].iloc[0].to_dict(),
        "paired_bootstrap_tests": summary["paired"].to_dict(orient="records"),
        "prevalence_rows": int(len(prevalence)),
    }
    (OUT / "strengthening_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate fixed-evidence strengthening tables and figures.")
    parser.add_argument("--vcsl-metadata-dir", default=str(ROOT / "public_data" / "vcsl_metadata"))
    parser.add_argument("--archived-results-dir", default=str(ROOT / "archived_results"))
    parser.add_argument("--manuscript-dir", default=str(ROOT / "paper"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "tmm_revision_strengthening"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_paths(args)
    ensure_dirs()
    setup_plot()
    build_feature_schema()
    build_hyperparameter_table()
    raw = run_vcsl_strengthening()
    summary = summarize_strengthening(raw)
    prevalence = build_prevalence_table()
    plot_strengthening(summary, prevalence)
    write_summary_json(summary, prevalence)
    print(f"wrote strengthening artifacts to {OUT}")
    print(f"updated manuscript tables in {TABLES}")
    print(f"updated manuscript figures in {FIGURES}")


if __name__ == "__main__":
    main()
