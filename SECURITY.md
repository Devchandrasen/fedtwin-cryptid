# Security scope

FedTwin-CryptID is research software for fixed-evidence calibration and protected-update accounting. It has not received a production cryptographic audit.

In particular:

- `secureagg_sim` is a protocol-behavior simulation, not a deployed protocol with authenticated key agreement, production dropout recovery, and collusion guarantees;
- `quantized_transport_proxy` is numeric and communication accounting, not homomorphic encryption;
- the Paillier implementation demonstrates additive compact-update aggregation but does not provide production key storage, rotation, access control, or side-channel hardening;
- signed receipt chains establish commitment consistency and ordering under the active signing keys, not media ownership, correctness, or infringement.

Do not use this repository to make automated enforcement decisions or to protect production secrets. Report implementation vulnerabilities privately through GitHub's security-advisory interface for this repository.
