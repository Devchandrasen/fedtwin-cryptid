# v1.0.0 release and TMM submission checklist

## Automated gates

- [x] Clean-clone validation of code/evidence commit `7a12415` passes on Python 3.10 and 3.12: lint, tests, coverage, smoke reproduction, artifact verification, and paper regeneration.
- [ ] The same matrix passes on public GitHub Actions after the branch is pushed.
- [x] Package coverage is at least 85% (85.80% in the current Python 3.12 run).
- [x] `paper_results/v1/manifest.json` verifies and all active claims resolve.
- [x] Main paper compiles to at most 10 pages; supplement compiles to at most four pages before final metadata/DOI insertion.
- [x] No unresolved references, private paths, non-finite figure labels, clipping, or anonymous-review mismatch remains before final metadata insertion.

## Human gates — submission remains blocked until complete

- [ ] Confirm final author names and order.
- [ ] Add affiliation, email, and ORCID for every author; synchronize ScholarOne metadata.
- [ ] Obtain contribution and final-text approval from every author.
- [ ] Confirm acceptance of possible mandatory TMM page charges above eight published pages.
- [ ] Confirm prior-publication/preprint disclosure.
- [ ] Complete the IEEE/SPS AI-use human-review attestation in `paper/AI_USE_DISCLOSURE_REQUIRED.md`.
- [ ] Select April 2026 TMM-listed EDICS in ScholarOne: `MM-COM-SECU` (Multimedia security and watermarking) primary, `MM-MOD-DIST` (Distributed multimedia processing and Internet-of-Things) secondary, and `MM-DAT-SEAR` (Multimedia search and retrieval) tertiary.

## Publication handoff

- [ ] Merge the reviewed release branch.
- [ ] Tag and publish GitHub release `v1.0.0` from the verified commit.
- [ ] Deposit that exact release in Zenodo.
- [ ] Add the real Zenodo DOI to `CITATION.cff`, README, manuscript, and supplement; never use a placeholder DOI.
- [ ] Rebuild the source ZIP, PDFs, checksums, and submission checklist after the DOI-only change.
