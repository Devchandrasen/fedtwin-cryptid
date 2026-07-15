"""Protected compact-update aggregation helpers.

The SecureAgg path is a deterministic, protocol-faithful arithmetic simulator:
pairwise masks hide individual weighted updates and cancel in the aggregate.
It is not a network deployment, collusion proof, or production dropout-recovery
implementation.  The quantized transport mode is an accounting proxy and is
never labelled as homomorphic encryption.

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
    protocol_scope: str = ""
    active_clients: int = 0
    dropped_clients: int = 0
    max_abs_error_vs_plain: float = 0.0


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

    update_arrays, weight_values = _validate_updates(updates, weights)
    start_encrypt = time.perf_counter()
    public_key, private_key = generate_paillier_keypair(key_bits)
    plain_bytes = int(sum(update.nbytes for update in update_arrays))
    int_weights = np.asarray(np.rint(weight_values), dtype=np.int64)
    if np.any(int_weights < 0) or int_weights.sum() <= 0:
        raise ValueError("Paillier aggregation requires positive integer-like weights")
    if not np.isfinite(he_scale) or he_scale <= 0:
        raise ValueError("he_scale must be a positive finite value")
    quantized = [np.rint(update * he_scale).astype(np.int64) for update in update_arrays]
    encrypted = [[paillier_encrypt_int(int(v), public_key) for v in update] for update in quantized]
    encryption_time = time.perf_counter() - start_encrypt

    start_agg = time.perf_counter()
    aggregated_ciphertexts = []
    for dim in range(len(quantized[0])):
        c = 1
        for client_ciphertexts, weight in zip(encrypted, int_weights, strict=True):
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
        protocol_scope="real additive Paillier aggregation for compact quantized update sums only",
        active_clients=len(update_arrays),
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


def _validate_updates(
    updates: list[np.ndarray], weights: list[float]
) -> tuple[list[np.ndarray], np.ndarray]:
    if not updates:
        raise ValueError("updates must be non-empty")
    if len(updates) != len(weights):
        raise ValueError("updates and weights must have the same length")
    arrays = [np.asarray(update, dtype=float) for update in updates]
    shape = arrays[0].shape
    if not shape or any(array.shape != shape for array in arrays):
        raise ValueError("all updates must have the same non-scalar shape")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("updates must contain only finite values")
    values = np.asarray(weights, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, and have positive sum")
    return arrays, values


def secureagg_simulate(
    updates: list[np.ndarray],
    weights: list[float],
    *,
    active_clients: list[int] | None = None,
    mask_seed: int = 0,
    dropout_stage: str = "before_mask_setup",
) -> tuple[np.ndarray, AggregationReport, dict[str, object]]:
    """Simulate aggregate-only visibility with cancelling pairwise masks.

    ``after_mask_setup`` reconstructs the residual pairwise masks associated
    with dropped clients and removes them from the active aggregate. This is an
    arithmetic protocol-behavior check, not production recovery or a security
    proof.
    """

    arrays, weight_values = _validate_updates(updates, weights)
    if dropout_stage not in {"before_mask_setup", "after_mask_setup"}:
        raise ValueError("dropout_stage must be before_mask_setup or after_mask_setup")
    total_clients = len(arrays)
    if active_clients is None:
        active = list(range(total_clients))
    else:
        active = sorted(set(map(int, active_clients)))
        if not active or active[0] < 0 or active[-1] >= total_clients:
            raise ValueError("active_clients must select at least one valid client")
    active_updates = [arrays[index] for index in active]
    active_weights = weight_values[active]
    normalized = active_weights / active_weights.sum()
    weighted = [weight * update for weight, update in zip(normalized, active_updates, strict=True)]
    plain = np.sum(weighted, axis=0)

    start_encrypt = time.perf_counter()
    masked = [update.copy() for update in weighted]
    active_position = {client: position for position, client in enumerate(active)}
    residual_mask = np.zeros_like(plain)
    mask_clients = active if dropout_stage == "before_mask_setup" else list(range(total_clients))
    for left_pos in range(len(mask_clients)):
        for right_pos in range(left_pos + 1, len(mask_clients)):
            left_client = mask_clients[left_pos]
            right_client = mask_clients[right_pos]
            pair_seed = np.random.SeedSequence([int(mask_seed), left_client, right_client])
            mask = np.random.default_rng(pair_seed).normal(0.0, 1.0, size=plain.shape)
            if left_client in active_position:
                masked[active_position[left_client]] += mask
                if right_client not in active_position:
                    residual_mask += mask
            if right_client in active_position:
                masked[active_position[right_client]] -= mask
                if left_client not in active_position:
                    residual_mask -= mask
    encryption_time = time.perf_counter() - start_encrypt

    start_aggregate = time.perf_counter()
    masked_aggregate = np.sum(masked, axis=0)
    aggregate = masked_aggregate - residual_mask
    aggregation_time = time.perf_counter() - start_aggregate
    max_error = float(np.max(np.abs(aggregate - plain)))
    plain_bytes = int(sum(array.nbytes for array in active_updates))
    report = AggregationReport(
        mode="secureagg_sim",
        encryption_time_sec=float(encryption_time),
        aggregation_time_sec=float(aggregation_time),
        plain_bytes=plain_bytes,
        protected_bytes=plain_bytes * 2,
        ciphertext_expansion=2.0,
        protocol_scope="pairwise-mask and dropout-reconstruction arithmetic simulation; no network or collusion proof",
        active_clients=len(active),
        dropped_clients=total_clients - len(active),
        max_abs_error_vs_plain=max_error,
    )
    details: dict[str, object] = {
        "active_clients": active,
        "dropped_clients": sorted(set(range(total_clients)) - set(active)),
        "dropout_stage": dropout_stage,
        "pre_recovery_residual_norm": float(np.linalg.norm(masked_aggregate - plain)),
        "reconstructed_mask_norm": float(np.linalg.norm(residual_mask)),
        "recovery_applied": bool(dropout_stage == "after_mask_setup" and len(active) < total_clients),
        "masks_cancel": bool(max_error <= 1e-10),
    }
    return aggregate, report, details


def aggregate_updates(
    updates: list[np.ndarray],
    weights: list[float],
    *,
    mode: str,
    he_scale: float = 1e6,
) -> tuple[np.ndarray, AggregationReport]:
    updates, weight_values = _validate_updates(updates, weights)
    aliases = {
        "secureagg": "secureagg_sim",
        "heagg": "quantized_transport_proxy",
    }
    mode = aliases.get(mode, mode)
    if mode == "secureagg_sim":
        aggregate, report, _ = secureagg_simulate(updates, list(weight_values), mask_seed=0)
        return aggregate, report
    if mode == "paillier":
        aggregate, report, _ = paillier_aggregate_updates(updates, list(weight_values), key_bits=2048, he_scale=he_scale)
        return aggregate, report

    start_encrypt = time.perf_counter()
    plain_bytes = int(sum(update.nbytes for update in updates))

    if mode == "plain":
        protected = updates
        protected_bytes = plain_bytes
    elif mode == "quantized_transport_proxy":
        protected = [np.round(update * he_scale).astype(np.int64) for update in updates]
        # Conservative byte expansion for packed protected transport accounting.
        protected_bytes = plain_bytes * 16
    else:
        raise ValueError(f"unknown aggregation mode: {mode}")

    encryption_time = time.perf_counter() - start_encrypt
    start_agg = time.perf_counter()
    weight_arr = np.asarray(weight_values, dtype=float)
    weight_arr = weight_arr / max(weight_arr.sum(), 1e-12)

    if mode == "quantized_transport_proxy":
        agg_int = np.zeros_like(protected[0], dtype=np.float64)
        for w, update in zip(weight_arr, protected, strict=True):
            agg_int += w * update.astype(np.float64)
        aggregate = agg_int / he_scale
    else:
        aggregate = np.zeros_like(updates[0], dtype=float)
        for w, update in zip(weight_arr, protected, strict=True):
            aggregate += w * update

    aggregation_time = time.perf_counter() - start_agg
    report = AggregationReport(
        mode=mode,
        encryption_time_sec=float(encryption_time),
        aggregation_time_sec=float(aggregation_time),
        plain_bytes=plain_bytes,
        protected_bytes=int(protected_bytes),
        ciphertext_expansion=float(protected_bytes / max(plain_bytes, 1)),
        protocol_scope=(
            "plain weighted averaging"
            if mode == "plain"
            else "quantized transport accounting proxy; not homomorphic encryption"
        ),
        active_clients=len(updates),
    )
    return aggregate, report
