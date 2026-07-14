from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fedtwin.ledger import canonical_hash, keyed_commitment, sha256_text


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase_5_manuscript" / "tables" / "receipt_test_vector.json"


def main() -> None:
    secret = "fedtwin-test-secret"
    private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(sha256_text("fedtwin-test-key")))
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    previous_chain_hash = "0" * 64
    payload = {
        "receipt_id": "receipt-test-000001",
        "schema_version": "fedtwin-cryptid-receipt-v2",
        "timestamp_utc": "2026-05-29T00:00:00+00:00",
        "timestamp_mode": "utc",
        "client_commitment": keyed_commitment(secret, "client-7"),
        "salt_digest": sha256_text(secret),
        "model_digest": "model-sha256:example",
        "reference_digest": keyed_commitment(secret, "reference-example"),
        "query_digest": keyed_commitment(secret, "query-example"),
        "decision": "review",
        "confidence": 0.8125,
        "modality_scores": {
            "audio_reliability": 0.91,
            "audio_score": 0.74,
            "fusion_score": 0.8125,
            "video_reliability": 0.88,
            "video_score": 0.83,
        },
        "segment_evidence_digest": keyed_commitment(secret, "segment-example"),
        "previous_chain_hash": previous_chain_hash,
        "supersedes_receipt_hash": "",
        "signer_public_key": public_key,
    }
    payload_hash = canonical_hash(payload)
    signature = private_key.sign(payload_hash.encode("utf-8")).hex()
    signed_payload = dict(payload)
    signed_payload["signature"] = signature
    receipt_hash = canonical_hash(signed_payload)
    chain_hash = sha256_text(previous_chain_hash + receipt_hash)
    OUT.write_text(
        json.dumps(
            {
                "canonicalization": "json.dumps(sort_keys=True,separators=(',',':')) encoded as UTF-8",
                "hash": "SHA-256 hex",
                "signature": "Ed25519 over payload_hash UTF-8 hex string",
                "payload_hash": payload_hash,
                "receipt_hash": receipt_hash,
                "previous_chain_hash": previous_chain_hash,
                "chain_hash": chain_hash,
                "signed_payload": signed_payload,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
