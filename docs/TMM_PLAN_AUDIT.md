# TMM submission-readiness audit

Date: 2026-07-15

This audit evaluates the approved FedTwin-CryptID revision plan. “Pass” means
the requirement is implemented and locally evidenced. “External gate” requires
a public service or final release. “Human gate” cannot be completed from source
code or public inference and blocks ScholarOne submission.

## Implementation and artifact

| Requirement | Status | Evidence |
|---|---|---|
| Portable smoke, reproduce, paper-regeneration, and artifact-verification CLI | Pass | `fedtwin/cli.py`, installed `fedtwin` entry point, CLI integration tests |
| Validated configuration with explicit roots and no implicit external writes | Pass | `fedtwin/config.py`, CLI path validation, smoke/reproduction tests |
| Asset-disjoint deterministic pair construction and leakage rejection | Pass | `fedtwin/pairs.py`, pair/leakage tests, per-run split manifests |
| Explicit VCSL metadata/descriptor, FMA audio, and synthetic adapters | Pass | `fedtwin/data_adapters.py` and adapter validation tests |
| Git/config/input/output/hardware/timing manifests | Pass | `fedtwin/manifests.py` and all released `run_manifest.json` files |
| Accurate plain, SecureAgg-simulation, quantized-proxy, and real-Paillier labels | Pass | `fedtwin/crypto.py`, manuscript limitations, protected-aggregation tables |
| Three-seed 2048-bit real Paillier evidence and safe-range accounting | Pass | `tables/paillier_update_aggregation.csv` for seeds 31/37/41 |
| Locked dependencies, strict schemas, acquisition documentation | Pass | `uv.lock`, package validation, `docs/DATA_ACQUISITION.md` |
| Versioned reviewer artifact without restricted raw media/private paths | Pass | `paper_results/v1`, manifest verifier, release checks |

## Experimental revision

| Requirement | Status | Evidence or boundary |
|---|---|---|
| Seeds 31/37/41, frozen asset-disjoint central/local/federated/personalized runs | Pass | Confirmatory VCSL and FMA `new_runs/*` manifests and metrics |
| PR-AUC, ROC-AUC, FPR@95 recall, Brier, ECE, budget recall, workload, and counts | Pass | Per-tier metrics, operating-point and summary tables |
| 1:100, 1:1,000, and 1:10,000 streaming candidate stress with at least 100 hardest queries | Pass with disclosed boundary | 100 positive-bearing queries at 1:10,000; candidate instances include repeated references/noise and are not a unique-reference benchmark |
| Query-clustered confidence intervals and paired/Holm comparisons | Pass | Ranking and paired bootstrap/permutation tables |
| Feature, heterogeneity, round, personalization, prevalence, threshold, and evidence-quality ablations | Pass | Released ablation tables; round source covers complete rounds 1--20 |
| 20-round robustness at 0/10/20 percent label-flip and sign-flip attackers | Pass | `tables/robustness_stress.csv` |
| SecureAgg simulation under 0/10/20 percent dropout with cancellation/recovery checks | Pass | `secureagg_dropout_or_proxy.csv`; explicitly not a production proof |
| Separate VCSL visual and FMA audio tiers with no synchronized-fusion/SOTA claim | Pass | Main paper scope, benchmark table, limitations |
| Newly reported results are rerun or checksum-verifiable | Pass | Per-run manifests, input records, artifact hash manifest |

## Manuscript and supplement

| Requirement | Status | Evidence |
|---|---|---|
| Fixed-evidence problem, objective, FedProx, threat model, receipts, flow, split policy, limitations | Pass | Main Sections III--VIII and supplement |
| Main paper at most 10 pages; supplement at most four | Pass locally | Compiled main is 10 pages; supplement is 2 pages |
| Abstract 150--250 words and bounded claims | Pass | 222-word abstract; explicit non-claims in abstract and limitations |
| Current related-work separation and April 2026 EDICS | Pass | Bibliography/positioning; workbook-verified EDICS recorded in checklist |
| Exact seeds, counts, statistics, software, and licensing limitations | Pass | Main experiment protocol, supplement, run manifests |
| Finite, regenerated figures without `nan`/`inf` or clipping | Pass locally | Asset regeneration, text scan, rendered-page inspection |
| AI-use disclosure and required human review | Human gate | Disclosure is drafted; every author must attest that the whole manuscript and evidence were reviewed |
| Final names, affiliations, emails, ORCIDs, order, and ScholarOne match | Human gate | Chandrasen Pandey / ORCID `0000-0002-7031-1619` / UPES are public candidates; final list and institutional email are not inferable |
| All-author approval, contributions, prior-publication disclosure, and charge approval | Human gate | Written approval required; a 10-page published paper may incur mandatory overlength charges above eight pages |

## Release and submission handoff

| Requirement | Status | Required action |
|---|---|---|
| Public branch and GitHub Actions matrix | External gate | Push the reviewed branch and require all checks to pass |
| GitHub release `v1.0.0` | External gate | Merge only after human metadata is final, then tag the verified commit |
| Zenodo archive and real DOI in all materials | External gate | Archive the exact release, insert the DOI, and rebuild once |
| Final source ZIP, PDFs, reviewer ZIP, checksums, cover letter, and disclosures | Pass locally, final rebuild pending DOI | Rebuild from the DOI-bearing commit and re-run all gates |

## Decision

The research code, evidence, and manuscript are a technically reproducible TMM
submission candidate within the stated limitations. They are **not yet ready
for ScholarOne submission**. Submission remains blocked by final author
metadata and ORCIDs, human review of AI-assisted material, all-author approval,
prior-publication and charge decisions, public CI, and the final GitHub/Zenodo
release DOI. No technical claim should be broadened while closing those gates.
