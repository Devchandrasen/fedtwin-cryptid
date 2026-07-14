"""Privacy-mode simulations and compact-update aggregation helpers.

The benchmark reports the cost of update protection without claiming a full
production cryptosystem. HE mode uses quantization plus configurable ciphertext
expansion to mimic small-head encrypted aggregation cost, which is sufficient
for comparing overhead in the Phase 4 feature-level benchmark.

The Paillier helper is a real additive homomorphic aggregation implementation
for compact integer model-update sums. It is intentionally scoped to small
calibration-head updates; it does not encrypt multimedia inference and it does
not include production key management.
"""

from __future__ import annotations

import math
import secrets
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class AggregationReport:
    mode: str
    encryption_time_sec: float
    aggregation_time_sec: float
    plain_bytes: int
    protected_bytes: int
    ciphertext_expansion: float


@dataclass(frozen=True)
class PaillierPublicKey:
    n: int
    g: int
    n_square: int
    key_bits: int


@dataclass(frozen=True)
class PaillierPrivateKey:
    public_key: PaillierPublicKey
    lam: int
    mu: int


def _is_probable_prime(n: int, rounds: int = 16) -> bool:
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _generate_prime(bits: int) -> int:
    while True:
        candidate = secrets.randbits(bits)
        candidate |= (1 << (bits - 1)) | 1
        if _is_probable_prime(candidate):
            return candidate


def generate_paillier_keypair(key_bits: int = 2048) -> tuple[PaillierPublicKey, PaillierPrivateKey]:
    """Generate a Paillier keypair using ``g=n+1``.

    The implementation is deliberately dependency-free for reviewer artifacts.
    It is suitable for compact-update experiments and test vectors, not for
    production key-management deployment.
    """

    if key_bits < 512:
        raise ValueError("Paillier key_bits must be at least 512 for artifact experiments")
    half = key_bits // 2
    while True:
        p = _generate_prime(half)
        q = _generate_prime(key_bits - half)
        if p == q:
            continue
        n = p * q
        if n.bit_length() < key_bits:
            continue
        lam = math.lcm(p - 1, q - 1)
        g = n + 1
        n_square = n * n
        l_value = (pow(g, lam, n_square) - 1) // n
        if math.gcd(l_value, n) == 1:
            mu = pow(l_value, -1, n)
            pub = PaillierPublicKey(n=n, g=g, n_square=n_square, key_bits=n.bit_length())
            priv = PaillierPrivateKey(public_key=pub, lam=lam, mu=mu)
            return pub, priv


def _encode_signed(value: int, n: int) -> int:
    if abs(value) >= n // 3:
        raise OverflowError("encoded Paillier integer is too large for signed range")
    return value % n


def _decode_signed(value: int, n: int) -> int:
    if value > n // 2:
        return value - n
    return value


def paillier_encrypt_int(value: int, public_key: PaillierPublicKey) -> int:
    encoded = _encode_signed(int(value), public_key.n)
    while True:
        r = secrets.randbelow(public_key.n - 1) + 1
        if math.gcd(r, public_key.n) == 1:
            break
    return (pow(public_key.g, encoded, public_key.n_square) * pow(r, public_key.n, public_key.n_square)) % public_key.n_square


def paillier_decrypt_int(ciphertext: int, private_key: PaillierPrivateKey) -> int:
    pub = private_key.public_key
    l_value = (pow(ciphertext, private_key.lam, pub.n_square) - 1) // pub.n
    return _decode_signed((l_value * private_key.mu) % pub.n, pub.n)


def paillier_aggregate_updates(
    updates: list[np.ndarray],
    weights: list[float],
    *,
    key_bits: int = 2048,
    he_scale: float = 1e6,
) -> tuple[np.ndarray, AggregationReport, dict]:
    """Aggregate compact updates with real Paillier additive homomorphism.

    Updates are quantized, encrypted elementwise, combined as weighted integer
    sums, decrypted, and rescaled. Weights are rounded to nonnegative integer
    client sample counts, matching FedAvg-style weighting in the benchmark.
    """

    if not updates:
        raise ValueError("updates must be non-empty")
    start_encrypt = time.perf_counter()
    public_key, private_key = generate_paillier_keypair(key_bits)
    plain_bytes = int(sum(update.nbytes for update in updates))
    int_weights = np.asarray(np.rint(weights), dtype=np.int64)
    if np.any(int_weights < 0) or int_weights.sum() <= 0:
        raise ValueError("Paillier aggregation requires positive integer-like weights")
    quantized = [np.rint(update * he_scale).astype(np.int64) for update in updates]
    encrypted = [[paillier_encrypt_int(int(v), public_key) for v in update] for update in quantized]
    encryption_time = time.perf_counter() - start_encrypt

    start_agg = time.perf_counter()
    aggregated_ciphertexts = []
    for dim in range(len(quantized[0])):
        c = 1
        for client_ciphertexts, weight in zip(encrypted, int_weights):
            c = (c * pow(client_ciphertexts[dim], int(weight), public_key.n_square)) % public_key.n_square
        aggregated_ciphertexts.append(c)
    decrypted = np.asarray([paillier_decrypt_int(c, private_key) for c in aggregated_ciphertexts], dtype=np.float64)
    aggregate = decrypted / float(int_weights.sum()) / he_scale
    aggregation_time = time.perf_counter() - start_agg

    ciphertext_bytes = math.ceil(public_key.n_square.bit_length() / 8) * len(quantized[0]) * len(quantized)
    report = AggregationReport(
        mode="paillier",
        encryption_time_sec=float(encryption_time),
        aggregation_time_sec=float(aggregation_time),
        plain_bytes=plain_bytes,
        protected_bytes=int(ciphertext_bytes),
        ciphertext_expansion=float(ciphertext_bytes / max(plain_bytes, 1)),
    )
    details = {
        "scheme": "Paillier",
        "key_bits": int(public_key.key_bits),
        "n_square_bits": int(public_key.n_square.bit_length()),
        "clients": int(len(updates)),
        "update_dimension": int(len(quantized[0])),
        "quantization_scale": float(he_scale),
        "integer_weight_sum": int(int_weights.sum()),
    }
    return aggregate, report, details


def aggregate_updates(
    updates: list[np.ndarray],
    weights: list[float],
    *,
    mode: str,
    he_scale: float = 1e6,
) -> tuple[np.ndarray, AggregationReport]:
    start_encrypt = time.perf_counter()
    plain_bytes = int(sum(update.nbytes for update in updates))

    if mode == "plain":
        protected = updates
        protected_bytes = plain_bytes
    elif mode == "secureagg":
        # Masking simulation: individual masks cancel in aggregate.
        protected = updates
        protected_bytes = plain_bytes * 2
    elif mode == "heagg":
        protected = [np.round(update * he_scale).astype(np.int64) for update in updates]
        # Conservative expansion for packed approximate HE ciphertexts.
        protected_bytes = plain_bytes * 16
    elif mode == "paillier":
        aggregate, report, _ = paillier_aggregate_updates(updates, weights, key_bits=1024, he_scale=he_scale)
        return aggregate, report
    else:
        raise ValueError(f"unknown aggregation mode: {mode}")

    encryption_time = time.perf_counter() - start_encrypt
    start_agg = time.perf_counter()
    weight_arr = np.asarray(weights, dtype=float)
    weight_arr = weight_arr / max(weight_arr.sum(), 1e-12)

    if mode == "heagg":
        agg_int = np.zeros_like(protected[0], dtype=np.float64)
        for w, update in zip(weight_arr, protected):
            agg_int += w * update.astype(np.float64)
        aggregate = agg_int / he_scale
    else:
        aggregate = np.zeros_like(updates[0], dtype=float)
        for w, update in zip(weight_arr, protected):
            aggregate += w * update

    aggregation_time = time.perf_counter() - start_agg
    report = AggregationReport(
        mode=mode,
        encryption_time_sec=float(encryption_time),
        aggregation_time_sec=float(aggregation_time),
        plain_bytes=plain_bytes,
        protected_bytes=int(protected_bytes),
        ciphertext_expansion=float(protected_bytes / max(plain_bytes, 1)),
    )
    return aggregate, report
