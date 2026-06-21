"""Secure aggregation via Shamir secret sharing (privacy of FL updates).

Threat addressed: an *honest-but-curious* aggregator must not see any individual
client's model update (mitigates model-inversion / membership-inference on a
single contribution). Shamir ``(t, n)`` sharing is additively homomorphic, so
the aggregator can sum the i-th share of every client to obtain the i-th share
of the *sum* without ever reconstructing an individual update; any ``t`` summed
shares reconstruct the aggregate.

Pure-Python integer arithmetic over the Mersenne prime ``2**127 - 1`` — no
torch, no numpy required for the core. This is privacy, not poisoning-robustness;
combine with :mod:`horizon_ric.federated.robust` for Byzantine resistance, and
note that an *actively malicious* server still needs verifiable secret sharing
(Feldman/Pedersen) which is out of scope here.
"""

from __future__ import annotations

import secrets
from typing import List, Sequence, Tuple

PRIME: int = (1 << 127) - 1
QUANT_SCALE: int = 1_000_000  # 6 decimal digits of recoverable precision


def _mod_inverse(a: int, p: int = PRIME) -> int:
    return pow(a % p, p - 2, p)


class ShamirSecretSharing:
    """(t, n) Shamir secret sharing over GF(2**127 - 1)."""

    def __init__(self, prime: int = PRIME) -> None:
        if prime <= 1:
            raise ValueError("prime must be > 1")
        self.prime = prime

    def split(self, secret: int, n_shares: int, threshold: int) -> List[Tuple[int, int]]:
        if not (0 <= secret < self.prime):
            raise ValueError(f"secret must be in [0, {self.prime})")
        if threshold < 2:
            raise ValueError("threshold must be ≥ 2")
        if n_shares < threshold:
            raise ValueError("n_shares must be ≥ threshold")
        coeffs = [secret] + [secrets.randbelow(self.prime) for _ in range(threshold - 1)]
        shares: List[Tuple[int, int]] = []
        for x in range(1, n_shares + 1):  # x = 0 would leak the secret
            y, xp = 0, 1
            for c in coeffs:
                y = (y + c * xp) % self.prime
                xp = (xp * x) % self.prime
            shares.append((x, y))
        return shares

    def reconstruct(self, shares: Sequence[Tuple[int, int]]) -> int:
        if len(shares) < 2:
            raise ValueError("need ≥ 2 shares to reconstruct")
        xs = [x for x, _ in shares]
        if len(set(xs)) != len(xs):
            raise ValueError("share x-coordinates must be distinct")
        p = self.prime
        secret = 0
        for j, (xj, yj) in enumerate(shares):
            num, den = 1, 1
            for m, (xm, _) in enumerate(shares):
                if m == j:
                    continue
                num = (num * (-xm)) % p
                den = (den * (xj - xm)) % p
            lj = (num * _mod_inverse(den, p)) % p
            secret = (secret + yj * lj) % p
        return secret


def _quantise(x: float) -> int:
    return int(round(x * QUANT_SCALE)) % PRIME


def _dequantise(q: int, n_clients: int) -> float:
    if q > PRIME // 2:
        q = q - PRIME
    return float(q) / float(QUANT_SCALE * n_clients)


def secure_mean(
    client_vectors: Sequence[Sequence[float]],
    *,
    n_shares: int = 5,
    threshold: int = 3,
) -> list[float]:
    """Privacy-preserving element-wise mean of equal-length client vectors.

    Each client splits each quantised element into shares; the server sums
    shares per holder; ``threshold`` holders reconstruct the summed secret; the
    result is dequantised to the mean. Equivalent to the plain mean within the
    quantisation precision, but the server never sees an individual vector.
    """
    sss = ShamirSecretSharing()
    n_clients = len(client_vectors)
    if n_clients == 0:
        return []
    length = len(client_vectors[0])
    if any(len(v) != length for v in client_vectors):
        raise ValueError("all client vectors must have the same length")

    # summed_shares[holder][element] = (x, sum_y mod p)
    summed: list[list[Tuple[int, int]]] = [
        [(i + 1, 0) for _ in range(length)] for i in range(n_shares)
    ]
    for vec in client_vectors:
        for elem_idx, value in enumerate(vec):
            q = _quantise(float(value))
            shares = sss.split(q, n_shares, threshold)
            for holder_idx, (x, y) in enumerate(shares):
                sx, sy = summed[holder_idx][elem_idx]
                summed[holder_idx][elem_idx] = (sx, (sy + y) % PRIME)

    out: list[float] = []
    for elem_idx in range(length):
        pts = [summed[h][elem_idx] for h in range(threshold)]
        secret = sss.reconstruct(pts)
        out.append(_dequantise(secret, n_clients))
    return out


__all__ = [
    "PRIME",
    "QUANT_SCALE",
    "ShamirSecretSharing",
    "secure_mean",
]
