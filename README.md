# FedTwin-CryptID

FedTwin-CryptID is a research implementation for **raw-media non-export fixed-evidence calibration and media-free audit logging** in multimedia copy-review workflows. It trains compact calibration heads over precomputed query-reference evidence while raw media and raw embeddings remain at their source.

This repository contains implementation code only. It intentionally excludes the manuscript, copyrighted media, raw embeddings, private labels, downloaded descriptor archives, and generated experiment results.

## Scope

The implementation provides:

- synthetic, VCSL metadata, VCSL ISC descriptor, and FMA audio benchmark adapters;
- centralized, local, FedAvg, FedProx, personalized, and FedOpt-style compact-head training;
- score-only and leakage-aware fixed-evidence calibration policies;
- SecureAgg-style aggregate-only simulation and an HE packed-update proxy for protected-update accounting;
- an experimental Paillier additive aggregation path for compact model updates;
- signed, salted, hash-chain receipt generation and verification;
- retrieval-like ranking, calibration, leakage, robustness, prevalence, and audit-scaling experiments.

The code does **not** implement a new multimedia retrieval/localization model, a complete secure-aggregation deployment, encryption of multimedia inference, automated enforcement, ownership adjudication, or legally sufficient evidence. The Paillier path is a compact-update research prototype without production key management.

## Quick start

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python run_smoke.py --assets 200 --queries 240 --rounds 2 --local-epochs 1 --output-tag github_smoke
python -m pytest -q
```

Linux/macOS activation uses `source .venv/bin/activate`; the remaining commands are unchanged.

Smoke outputs are written under `outputs/` and are ignored by Git.

## Main commands

Run the synthetic benchmark:

```powershell
python run_benchmark.py --tier tier0 --clients 5 --assets 1000 --queries 1000 --seeds 31 37 41 --rounds 20 --local-epochs 5 --output-tag tier0_full
```

Run the reviewer-blocker audit suite in bounded mode:

```powershell
python revision_blocker_experiments.py --mode smoke --seeds 31 37 41 --max-train-pairs 2200 --max-test-pairs 900
```

Generate a canonical signed-receipt test vector:

```powershell
python generate_receipt_test_vector.py
```

Summarize a benchmark run:

```powershell
python summarize_results.py outputs\tier0_full
```

See [HPC_README.md](HPC_README.md) and [hpc_run_full.slurm](hpc_run_full.slurm) for cluster execution.

## Public-data tiers

Public datasets are not redistributed. Place locally obtained files under the paths described in [data/README.md](data/README.md), or pass the corresponding command-line path options to `run_benchmark.py`.

Available tiers are:

- `tier0`: fully synthetic smoke and correctness testing;
- `tier1`: synthetic feature-level stress testing;
- `vcsl_public`: VCSL public-label/metadata calibration;
- `vcsl_isc`: VCSL with released ISC frame descriptors;
- `fma_audio`: FMA audio transformation evidence.

These tiers validate fixed-evidence calibration. Separate visual and audio tiers must not be described as a natural synchronized audio-video corpus.

## Repository layout

```text
fedtwin/                         Core calibration, FL, metrics, crypto, and receipt modules
tests/                           Aggregation and receipt integrity tests
run_benchmark.py                 Main benchmark runner
run_smoke.py                     Fast synthetic smoke run
revision_blocker_experiments.py  Ranking, A/V stress, robustness, and audit experiments
strengthen_for_tmm.py            Calibration, ablation, and leakage analyses
summarize_results.py             Result summarizer
generate_receipt_test_vector.py  Canonical signed-receipt test vector
hpc_run_full.slurm               HPC launcher
```

## Scientific safeguards

- Keep score-only logistic calibration as a primary baseline.
- Report weak or negative results rather than hiding them.
- Treat high-recall visual operating points as review-triage evidence, not automated enforcement readiness.
- Label `secureagg` results as SecureAgg-style simulation unless a complete protocol is substituted.
- Label `heagg` results as an HE packed-update proxy.
- Treat receipts as evidence of commitment consistency and ordering under signing keys, not correctness, ownership, or infringement.

## License

Code is released under the MIT License. Dataset licenses and upstream descriptor licenses remain with their respective owners.
