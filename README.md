# FedTwin-CryptID

FedTwin-CryptID is a research implementation for **raw-media non-export fixed-evidence calibration and media-free audit logging** in multimedia copy-review workflows. It trains compact calibration heads over precomputed query-reference evidence while raw media and raw embeddings remain at their source.

The repository is a versioned research package: implementation, tests, manuscript sources, and permitted derived evidence are included. Copyrighted media, raw embeddings, private labels, and downloaded descriptor archives are never redistributed.

## Scope

The implementation provides:

- synthetic, VCSL metadata, VCSL ISC descriptor, and FMA audio benchmark adapters;
- centralized, local, FedAvg, FedProx, personalized, and FedOpt-style compact-head training;
- score-only and leakage-aware fixed-evidence calibration policies;
- protocol-faithful SecureAgg simulation with dropout diagnostics and a separately named quantized transport proxy;
- real Paillier additive aggregation for compact model updates with reported 2048-bit keys;
- signed, salted, hash-chain receipt generation and verification;
- retrieval-like ranking, calibration, leakage, robustness, prevalence, and audit-scaling experiments.

The code does **not** implement a new multimedia retrieval/localization model, production SecureAgg, encrypted multimedia inference, differential privacy, synchronized audio-video learning, automated enforcement, ownership adjudication, or legally sufficient evidence. The Paillier path is a compact-update research prototype without production key management.

## Quick start

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m fedtwin.cli smoke --output-dir outputs\smoke
python -m pytest --cov=fedtwin --cov-report=term-missing -q
```

Linux/macOS activation uses `source .venv/bin/activate`; the remaining commands are unchanged.

Smoke outputs are written under `outputs/` and are ignored by Git. The same command is exercised on Python 3.10 and 3.12 in continuous integration.

## Reproducibility commands

Run the synthetic smoke workflow:

```powershell
python -m fedtwin.cli smoke --output-dir outputs\smoke
```

Run a validated experiment configuration. Dataset roots must be supplied explicitly; commands do not search private parent folders:

```powershell
python -m fedtwin.cli reproduce --config configs\synthetic.yaml --output-dir outputs\synthetic-rerun
python -m fedtwin.cli reproduce --config configs\paper.yaml --data-root D:\datasets --output-dir outputs\paper-rerun
python -m fedtwin.cli reproduce --config configs\paper_vcsl_isc.yaml --data-root D:\datasets --output-dir outputs\vcsl-isc-rerun
python -m fedtwin.cli reproduce --config configs\paper_fma_audio_20s.yaml --data-root D:\datasets --output-dir outputs\fma-20s-rerun
python -m fedtwin.cli reproduce --config configs\paper_fma_audio_5s.yaml --data-root D:\datasets --output-dir outputs\fma-5s-rerun
```

The synthetic configuration is a no-data portability check. The four paper configurations are the frozen VCSL public-label, VCSL ISC visual-descriptor, FMA 20-s audio, and FMA 5-s audio protocols. Each fails with an explicit missing-file message until its licensed inputs are staged under the supplied data root.

Regenerate the confirmatory public-label feature-policy analysis without any
archived-result dependency:

```powershell
python strengthen_for_tmm.py --scope feature-policy --vcsl-metadata-dir D:\datasets\public_data\vcsl_metadata --manuscript-dir outputs\feature-policy-paper --output-dir outputs\feature-policy-run
```

Pinned upstream commits, archive links, license boundaries, and expected hashes are listed in [`docs/DATA_ACQUISITION.md`](docs/DATA_ACQUISITION.md).

Regenerate released tables and figures, then verify every released hash and claim mapping:

```powershell
python -m fedtwin.cli regenerate-paper --results-dir paper_results\v1 --output-dir outputs\paper-assets
python -m fedtwin.cli verify-artifact --manifest paper_results\v1\manifest.json
```

The backward-compatible runners remain available:

```powershell
python run_benchmark.py --tier tier0 --clients 5 --assets 1000 --queries 1000 --seeds 31 37 41 --rounds 20 --local-epochs 5 --output-tag tier0_full
python revision_blocker_experiments.py --help
python strengthen_for_tmm.py --help
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the result contract and table/figure map, and [HPC_README.md](HPC_README.md) for cluster execution.

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
configs/paper.yaml               Validated paper experiment configuration
paper_results/v1/                Permitted frozen evidence, claims, and hashes
paper/                           IEEE manuscript and supplement sources
tests/                           Unit and integration tests
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
- Label `secureagg_sim` results as SecureAgg simulation unless a production protocol is substituted.
- Label `quantized_transport_proxy` as a numeric/transport proxy, never as homomorphic encryption.
- Treat receipts as evidence of commitment consistency and ordering under signing keys, not correctness, ownership, or infringement.

## Release status

Version `1.0.0` is the TMM artifact candidate. A Zenodo DOI and final manuscript author metadata are intentionally absent until the public release is archived and every author supplies an ORCID and approves the submission. See [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md).

## License

Code is released under the MIT License. Dataset licenses and upstream descriptor licenses remain with their respective owners.
