"""Media-free evidence receipt and hash-chain utilities."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(payload: dict) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def keyed_commitment(secret: str, value: str) -> str:
    return hmac.new(secret.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass
class EvidenceReceipt:
    receipt_id: str
    schema_version: str
    timestamp_utc: str
    client_commitment: str
    model_digest: str
    reference_digest: str
    query_digest: str
    decision: str
    confidence: float
    modality_scores: dict
    segment_evidence_digest: str
    previous_chain_hash: str
    receipt_hash: str
    chain_hash: str
    signer_public_key: str = ""
    signature: str = ""
    salt_digest: str = ""
    timestamp_mode: str = "utc"
    supersedes_receipt_hash: str = ""


class HashChainLedger:
    def __init__(self) -> None:
        self.previous_hash = "0" * 64
        self.records: list[dict] = []

    def anchor(self, signed_payload: dict) -> tuple[str, str, float]:
        start = time.perf_counter()
        receipt_hash = canonical_hash(signed_payload)
        chain_hash = sha256_text(self.previous_hash + receipt_hash)
        self.records.append(
            {
                "previous_chain_hash": self.previous_hash,
                "receipt_hash": receipt_hash,
                "chain_hash": chain_hash,
                "payload": dict(signed_payload),
            }
        )
        self.previous_hash = chain_hash
        return receipt_hash, chain_hash, time.perf_counter() - start

    def verify(self, *, verify_signatures: bool = False) -> bool:
        prev = "0" * 64
        for item in self.records:
            payload = dict(item.get("payload", {}))
            if item.get("previous_chain_hash") != prev:
                return False
            if payload.get("previous_chain_hash") != prev:
                return False
            if canonical_hash(item.get("payload", {})) != item["receipt_hash"]:
                return False
            expected = sha256_text(prev + item["receipt_hash"])
            if expected != item["chain_hash"]:
                return False
            if verify_signatures:
                signature = payload.pop("signature", "")
                public_key_hex = payload.get("signer_public_key", "")
                if not signature or not public_key_hex:
                    return False
                try:
                    public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
                    public_key.verify(bytes.fromhex(signature), canonical_hash(payload).encode("utf-8"))
                except Exception:
                    return False
            prev = item["chain_hash"]
        return True


def make_receipts(
    *,
    scores: np.ndarray,
    y: np.ndarray,
    clients: np.ndarray,
    model_digest: str,
    limit: int = 250,
    threshold: float = 0.5,
    sign_receipts: bool = True,
) -> tuple[list[EvidenceReceipt], dict]:
    ledger = HashChainLedger()
    receipts: list[EvidenceReceipt] = []
    latencies = []
    private_key = Ed25519PrivateKey.generate() if sign_receipts else None
    public_key_hex = ""
    if private_key is not None:
        public_key_hex = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()
    selected = np.argsort(-scores)[: min(limit, len(scores))]
    for rank, idx in enumerate(selected):
        decision = "positive" if scores[idx] >= threshold else "review"
        previous = ledger.previous_hash
        salt = sha256_text(f"receipt-salt:{model_digest}:{rank}:{int(clients[idx])}")[:32]
        base = {
            "receipt_id": f"receipt-{rank:06d}",
            "schema_version": "fedtwin-cryptid-receipt-v2",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timestamp_mode": "utc",
            "client_commitment": keyed_commitment(salt, f"client-{int(clients[idx])}"),
            "salt_digest": sha256_text(salt),
            "model_digest": model_digest,
            "reference_digest": keyed_commitment(salt, f"reference-{idx}-{int(y[idx])}"),
            "query_digest": keyed_commitment(salt, f"query-{idx}-{float(scores[idx]):.8f}"),
            "decision": decision,
            "confidence": float(scores[idx]),
            "modality_scores": {
                "video_score": float(scores[idx]),
                "audio_score": float(scores[idx]),
                "fusion_score": float(scores[idx]),
                "video_reliability": 1.0,
                "audio_reliability": 1.0,
            },
            "segment_evidence_digest": keyed_commitment(salt, f"segment-{idx}-{decision}"),
            "previous_chain_hash": previous,
            "supersedes_receipt_hash": "",
            "signer_public_key": public_key_hex,
        }
        if private_key is not None:
            base["signature"] = private_key.sign(canonical_hash(base).encode("utf-8")).hex()
        else:
            base["signature"] = ""
        receipt_hash, chain_hash, latency = ledger.anchor(base)
        base["receipt_hash"] = receipt_hash
        base["chain_hash"] = chain_hash
        receipts.append(EvidenceReceipt(**base))
        latencies.append(latency)
    verification_success = bool(ledger.verify(verify_signatures=sign_receipts))
    tamper_detection_success = True
    if ledger.records:
        original = ledger.records[0]["payload"]["decision"]
        ledger.records[0]["payload"]["decision"] = "tampered"
        tamper_detection_success = not ledger.verify(verify_signatures=sign_receipts)
        ledger.records[0]["payload"]["decision"] = original
    summary = {
        "ledger_type": "hashchain",
        "schema_version": "fedtwin-cryptid-receipt-v2",
        "receipts": len(receipts),
        "verification_success": verification_success,
        "signature_verification_success": verification_success if sign_receipts else False,
        "tamper_detection_success": tamper_detection_success,
        "mean_receipt_latency_sec": float(np.mean(latencies)) if latencies else 0.0,
        "ledger_bytes": int(sum(len(json.dumps(asdict(r), sort_keys=True)) for r in receipts)),
    }
    return receipts, summary
