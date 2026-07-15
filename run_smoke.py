"""Backward-compatible wrapper for the portable smoke command."""

from __future__ import annotations

import argparse
from pathlib import Path

from fedtwin.cli import main as cli_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic FedTwin-CryptID smoke benchmark.")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--assets", type=int, default=80)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output-tag", default="smoke")
    parser.add_argument("--run-dir", default="outputs")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--difficulty", type=float, default=1.0, help="Retained for CLI compatibility.")
    args = parser.parse_args()
    output_dir = Path(args.run_dir) / args.output_tag
    cli_main(
        [
            "smoke",
            "--output-dir",
            str(output_dir),
            "--clients",
            str(args.clients),
            "--assets",
            str(args.assets),
            "--queries",
            str(args.queries),
            "--seed",
            str(args.seed),
            "--rounds",
            str(args.rounds),
            "--local-epochs",
            str(args.local_epochs),
        ]
    )


if __name__ == "__main__":
    main()
