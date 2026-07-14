from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FedTwin-CryptID results.")
    parser.add_argument("run_dir", help="Run output directory.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def mean_std(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)[value_cols]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std().fillna(0.0).add_suffix("_std")
    return pd.concat([mean, std], axis=1).reset_index()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "summary_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    detection = pd.read_csv(run_dir / "metrics_detection.csv")
    privacy = pd.read_csv(run_dir / "metrics_privacy.csv") if (run_dir / "metrics_privacy.csv").exists() else pd.DataFrame()
    ledger = pd.read_csv(run_dir / "metrics_ledger.csv") if (run_dir / "metrics_ledger.csv").exists() else pd.DataFrame()

    det_summary = mean_std(
        detection,
        ["method", "modality", "privacy_mode"],
        ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "fpr_at_95_recall"],
    )
    det_summary.to_csv(out_dir / "detection_summary.csv", index=False)
    if not privacy.empty:
        priv_summary = mean_std(
            privacy,
            ["method", "privacy_mode"],
            ["mean_encryption_time_sec", "mean_aggregation_time_sec", "ciphertext_expansion"],
        )
        priv_summary.to_csv(out_dir / "privacy_summary.csv", index=False)
    if not ledger.empty:
        ledger_summary = mean_std(
            ledger,
            ["method", "ledger_type"],
            ["receipts", "verification_success", "mean_receipt_latency_sec", "ledger_bytes"],
        )
        ledger_summary.to_csv(out_dir / "ledger_summary.csv", index=False)

    top = det_summary.sort_values("pr_auc_mean", ascending=False).head(12)
    plt.figure(figsize=(10, 5))
    plt.barh(top["method"], top["pr_auc_mean"])
    plt.xlabel("Mean PR-AUC")
    plt.title("FedTwin-CryptID Detection Performance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "detection_pr_auc.png", dpi=180)
    plt.close()

    if not privacy.empty:
        p = privacy.groupby("method", dropna=False)["ciphertext_expansion"].mean().sort_values()
        plt.figure(figsize=(8, 4))
        plt.barh(p.index, p.values)
        plt.xlabel("Ciphertext / plain update bytes")
        plt.title("Privacy Mode Communication Expansion")
        plt.tight_layout()
        plt.savefig(out_dir / "privacy_expansion.png", dpi=180)
        plt.close()

    summary = {
        "best_detection_method": top.iloc[0].to_dict() if len(top) else {},
        "summary_dir": str(out_dir),
    }
    (out_dir / "summary_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
