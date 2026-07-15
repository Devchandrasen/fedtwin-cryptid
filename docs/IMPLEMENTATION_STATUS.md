# FedTwin-CryptID TMM revision implementation status

Date: 2026-07-15

## Implemented and verified

- Portable `fedtwin` CLI entry points for smoke execution, paper reproduction, paper-asset regeneration, and artifact verification.
- Validated experiment configuration, explicit dataset adapters, deterministic asset-disjoint pair construction, leakage checks, strict manifests, and dependency locking.
- Accurately separated plain FedAvg/FedProx, SecureAgg arithmetic simulation, quantized transport proxy, and real Paillier compact-update aggregation.
- Reported 2048-bit Paillier benchmark with numerical error, timing, byte expansion, and safe-range accounting; small keys remain test-only.
- Three-seed blocker experiment suite, 20-round robustness stress, dropout reconstruction checks, paired statistical tests, 100-query candidate-instance stress, and regenerated paper tables/figures.
- Checksum-verified, asset-disjoint VCSL public-label (5,000/1,800 pairs), released VCSL ISC visual-descriptor (6,000/3,000 pairs), FMA 20-s and 5-s audio-only (12,000/6,000 pairs), and VCSL feature-policy (30,000/15,000 pairs) reruns for each seed.
- Pinned FMA-small-compatible audio mirror with official FMA metadata, complete selected-file hashes, and 1,200 deterministically selected, successfully decoded tracks from 1,584 tracks extracted from the first three pinned shards. The mirror maintainer reports omitting six unreadable and 78 license-unclear files from the full redistribution pack.
- Versioned `paper_results/v1` artifact with configurations, permitted results, source provenance, claims mapping, figures, and a complete hash manifest.
- Main manuscript and supplement revised around fixed-evidence calibration, protected compact-update accounting, and signed media-free audit receipts.
- Test suite passes locally on Python 3.12: 38 tests and 86.10% package coverage. Python 3.10/3.12 matrix status remains a CI release gate.

## Deliberate interpretation limits

- The 1:10,000 row is a 100-query candidate-instance stress with repeated references/noise realizations, not a 10,000-unique-reference benchmark.
- VCSL visual and FMA audio are separate tiers and do not establish synchronized audiovisual performance.
- SecureAgg remains an arithmetic protocol simulation, quantization remains a transport proxy, and Paillier covers compact additive update sums only.

## Mandatory blockers before submission

- Confirm the complete author list and order, affiliations, institutional emails, ORCIDs, ScholarOne metadata, contributions, and final approval.
- Confirm preprint/prior-publication disclosure and acceptance of possible mandatory TMM overlength charges.
- Merge the reviewed branch, publish GitHub release `v1.0.0`, archive that exact release with Zenodo, and insert the real DOI. No placeholder DOI is used.

The implementation package is a technically reproducible submission candidate within these stated evidence boundaries, but actual submission remains blocked until PDF/source-package checks, public release archiving, and the human metadata/approval gates are complete.
