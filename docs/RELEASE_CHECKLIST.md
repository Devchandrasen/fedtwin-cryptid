# v1.0.0 release and TMM submission checklist

## Automated gates

- [ ] Python 3.10 and 3.12 CI passes lint, tests, coverage, smoke reproduction, artifact verification, and paper regeneration.
- [ ] Package coverage is at least 85%.
- [ ] `paper_results/v1/manifest.json` verifies and all active claims resolve.
- [ ] Main paper compiles to at most 10 pages; supplement compiles to at most four pages.
- [ ] No unresolved references, private paths, non-finite figure labels, clipping, or anonymous-review mismatch remains.

## Human gates — submission remains blocked until complete

- [ ] Confirm final author names and order.
- [ ] Add affiliation, email, and ORCID for every author; synchronize ScholarOne metadata.
- [ ] Obtain contribution and final-text approval from every author.
- [ ] Confirm acceptance of possible mandatory TMM page charges above eight published pages.
- [ ] Confirm prior-publication/preprint disclosure.
- [ ] Select April 2026 TMM-listed EDICS in ScholarOne: `MM-COM-SECU` (Multimedia security and watermarking) primary, `MM-MOD-DIST` (Distributed multimedia processing and Internet-of-Things) secondary, and `MM-DAT-SEAR` (Multimedia search and retrieval) tertiary.

## Publication handoff

- [ ] Merge the reviewed release branch.
- [ ] Tag and publish GitHub release `v1.0.0` from the verified commit.
- [ ] Deposit that exact release in Zenodo.
- [ ] Add the real Zenodo DOI to `CITATION.cff`, README, manuscript, and supplement; never use a placeholder DOI.
- [ ] Rebuild the source ZIP, PDFs, checksums, and submission checklist after the DOI-only change.
