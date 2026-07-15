# FedTwin-CryptID TMM revision implementation status

Date: 2026-07-15

## Implemented and verified

- Portable `fedtwin` CLI entry points for smoke execution, paper reproduction, paper-asset regeneration, and artifact verification.
- Validated experiment configuration, explicit dataset adapters, deterministic asset-disjoint pair construction, leakage checks, strict manifests, and dependency locking.
- Accurately separated plain FedAvg/FedProx, SecureAgg arithmetic simulation, quantized transport proxy, and real Paillier compact-update aggregation.
- Reported three-seed 2048-bit Paillier benchmark with numerical error, timing, byte expansion, and safe-range accounting; small keys remain test-only.
- Three-seed blocker experiment suite, complete 20-round communication/convergence traces, 20-round robustness stress, dropout reconstruction checks, paired statistical tests, 100-query candidate-instance stress, and regenerated paper tables/figures.
- Checksum-verified, asset-disjoint VCSL public-label (5,000/1,800 pairs), released VCSL ISC visual-descriptor (6,000/3,000 pairs), FMA 20-s and 5-s audio-only (12,000/6,000 pairs), and VCSL feature-policy (30,000/15,000 pairs) reruns for each seed.
- Pinned FMA-small-compatible audio mirror with official FMA metadata, complete selected-file hashes, and 1,200 deterministically selected, successfully decoded tracks from 1,584 tracks extracted from the first three pinned shards. The mirror maintainer reports omitting six unreadable and 78 license-unclear files from the full redistribution pack.
- Versioned `paper_results/v1` artifact with configurations, permitted results, source provenance, claims mapping, figures, and a complete hash manifest.
- Main manuscript and supplement revised around fixed-evidence calibration, protected compact-update accounting, and signed media-free audit receipts.
- IEEE/SPS AI-use disclosure added; author attestation remains mandatory because the policy requires thorough human verification of AI-assisted material.
- Current Python 3.12 validation passes: Ruff, 40 tests, 85.80% branch-aware package coverage, installed-wheel CLI packaging, smoke execution, 362-file artifact verification, paper-asset regeneration, and clean PDF compilation. Final clean-clone Python 3.10/3.12 and remote GitHub Actions remain release gates until the branch is committed and pushed.

## Deliberate interpretation limits

- The 1:10,000 row is a 100-query candidate-instance stress with repeated references/noise realizations, not a 10,000-unique-reference benchmark.
- VCSL visual and FMA audio are separate tiers and do not establish synchronized audiovisual performance.
- SecureAgg remains an arithmetic protocol simulation, quantization remains a transport proxy, and Paillier covers compact additive update sums only.

## Mandatory blockers before submission

- Confirm the complete author list and order, affiliations, institutional emails, ORCIDs, ScholarOne metadata, contributions, and final approval.
- Confirm preprint/prior-publication disclosure and acceptance of possible mandatory TMM overlength charges.
- Merge the reviewed branch, publish GitHub release `v1.0.0`, archive that exact release with Zenodo, and insert the real DOI. No placeholder DOI is used.

The implementation package is a technically reproducible submission candidate within these stated evidence boundaries, but actual submission remains blocked until clean-clone/public-CI checks, final release archiving, and the human metadata/approval gates are complete.
