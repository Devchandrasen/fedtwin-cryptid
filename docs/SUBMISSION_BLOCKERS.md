# Submission blockers and evidence gaps

This package is a strong revision candidate, not authorization to submit. The following items cannot be resolved from the repository alone.

## Mandatory human metadata

- Final author list/order, affiliations, institutional emails, ORCIDs, and ScholarOne records are unconfirmed.
- Coauthor contribution/final-text approval and acceptance of possible mandatory overlength charges are unconfirmed.
- Prior-publication and preprint overlap has not been confirmed by all authors.
- The Zenodo DOI cannot be added until the reviewed GitHub release is published and deposited.

## Dataset-dependent reruns

- The VCSL public-label audit is newly regenerated with connected-component asset-disjoint splits.
- The released VCSL ISC and FMA rows are retained only as exploratory archived runs because their descriptor/media inputs and input hashes are not staged on this machine. Their recorded configurations and derived-output hashes are released, but they have not been rerun through the new asset-disjoint adapter in this revision.
- A complete submission-grade rerun of those tiers requires reacquiring the upstream data under its license, validating hashes, and rerunning seeds 31, 37, and 41.

## Candidate-pool interpretation

VCSL contains fewer than 10,001 unique reference assets. The 1:10,000 experiment is therefore a 100-positive-query candidate-instance stress with repeated reference assets/noise realizations, not a pool of 10,000 unique reference assets per query. The released table reports both total candidate instances and unique query-reference pairs. Do not describe this row as a 10,000-unique-reference retrieval benchmark. A stronger replacement requires a licensed external distractor corpus or a segment-window protocol with unique candidate identifiers.

## Claim boundary

Separate VCSL visual and FMA audio tiers do not establish synchronized audiovisual fusion. SecureAgg is an arithmetic protocol simulation, the quantized mode is not encryption, Paillier covers compact sums only, robustness is not a Byzantine proof, and no differential-privacy, ownership, infringement, or state-of-the-art localization claim is supported.
