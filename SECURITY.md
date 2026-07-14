# Security scope

FedTwin-CryptID is research software for fixed-evidence calibration and protected-update accounting. It has not received a production cryptographic audit.

In particular:

- `secureagg` is a SecureAgg-style aggregate-only simulation, not a deployed protocol with key agreement, dropout recovery, and collusion guarantees;
- `heagg` is an HE packed-update proxy for numeric and communication accounting;
- the Paillier implementation demonstrates additive compact-update aggregation but does not provide production key storage, rotation, access control, or side-channel hardening;
- signed receipt chains establish commitment consistency and ordering under the active signing keys, not media ownership, correctness, or infringement.

Do not use this repository to make automated enforcement decisions or to protect production secrets. Report implementation vulnerabilities privately through GitHub's security-advisory interface for this repository.
