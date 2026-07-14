from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FedTwin-CryptID local smoke benchmark.")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--assets", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-tag", default="smoke")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=4)
    parser.add_argument("--difficulty", type=float, default=1.0)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "run_benchmark.py",
        "--tier",
        "tier0",
        "--clients",
        str(args.clients),
        "--assets",
        str(args.assets),
        "--queries",
        str(args.queries),
        "--seeds",
        str(args.seed),
        "--rounds",
        str(args.rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--output-tag",
        args.output_tag,
        "--difficulty",
        str(args.difficulty),
    ]
    subprocess.run(cmd, cwd=Path(__file__).parent, check=True)


if __name__ == "__main__":
    main()
