# FedTwin-CryptID TMM revision implementation status

Date: 2026-07-15

## Implemented and verified

- Portable `fedtwin` CLI entry points for smoke execution, paper reproduction, paper-asset regeneration, and artifact verification.
- Validated experiment configuration, explicit dataset adapters, deterministic asset-disjoint pair construction, leakage checks, strict manifests, and dependency locking.
- Accurately separated plain FedAvg/FedProx, SecureAgg arithmetic simulation, quantized transport proxy, and real Paillier compact-update aggregation.
- Reported 2048-bit Paillier benchmark with numerical error, timing, byte expansion, and safe-range accounting; small keys remain test-only.
- Three-seed blocker experiment suite, 20-round robustness stress, dropout reconstruction checks, paired statistical tests, 100-query candidate-instance stress, and regenerated paper tables/figures.
- Checksum-verified asset-disjoint VCSL public-label rerun with 5,000 training pairs, 1,800 test pairs, and 151 evidence features.
- Versioned `paper_results/v1` artifact with configurations, permitted results, source provenance, claims mapping, figures, and 273-file hash manifest.
- Main manuscript and supplement revised around fixed-evidence calibration, protected compact-update accounting, and signed media-free audit receipts.
- Main PDF compiles to 10 pages, supplement to 2 pages, and the abstract contains 238 words.
- Test suite passes on Python 3.10 and 3.12: 31 tests; current Python 3.12 coverage is 85.98%.

## Deliberately retained as exploratory

- VCSL ISC descriptor, FMA audio, and legacy feature-ablation rows were not rerun because the licensed inputs and their hashes are unavailable in the staged workspace. They are visibly labeled exploratory archived evidence and are not represented as newly generated results.
- The 1:10,000 row is a 100-query candidate-instance stress with repeated references/noise realizations, not a 10,000-unique-reference benchmark.

## Mandatory blockers before submission

- Confirm the complete author list and order, affiliations, institutional emails, ORCIDs, ScholarOne metadata, contributions, and final approval.
- Confirm preprint/prior-publication disclosure and acceptance of possible mandatory TMM overlength charges.
- Merge the reviewed branch, publish GitHub release `v1.0.0`, archive that exact release with Zenodo, and insert the real DOI. No placeholder DOI is used.
- Reacquire licensed VCSL ISC/FMA inputs and rerun them through the new validated adapters if those tiers are to be promoted from exploratory to submission-grade evidence.

The implementation package is technically reproducible and reviewer-ready within these stated evidence boundaries, but the manuscript remains blocked from actual submission until the human metadata and approval gates are complete.
