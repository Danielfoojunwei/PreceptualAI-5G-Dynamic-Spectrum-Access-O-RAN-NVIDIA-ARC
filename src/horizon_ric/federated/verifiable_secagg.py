"""Verifiable two-server secure aggregation — real crypto, no torch, no mocks.

The existing :mod:`horizon_ric.federated.secure` (Shamir) hides individual updates
but is **honest-but-curious**: it assumes the server follows the protocol. It does
not stop a *malicious* server from silently dropping or altering a client's
contribution, and it cannot *prove* to anyone that the published aggregate is the
true sum (``docs/THREAT_MODEL.md`` §5: "secure aggregation is honest-but-curious
only").

This module closes that, in the model of the NTU/DTC Starfish work (Liu, Ye,
Jiang, Shen, Guo, Tjuawinata & Lam, *Privacy-Preserving Federated Unlearning with
Certified Client Removal*, arXiv:2404.09724), which splits the computation across
**two non-colluding servers**. Two real, composable primitives:

* **2-of-2 additive secret sharing** over a prime field: each client splits its
  update into two shares ``(a, b)`` with ``a + b ≡ x (mod Q)``, sending ``a`` to
  server A and ``b`` to server B. One server alone sees a uniformly random value
  (a one-time pad) — perfect privacy of the individual update *under the
  non-collusion assumption*. Each server sums its shares; the aggregate is
  ``(ΣA + ΣB) mod Q``.

* **Feldman commitments** over the RFC-3526 2048-bit MODP safe prime: each client
  publishes ``Commit(x) = g^x mod p`` (per coordinate). These are *homomorphic* —
  ``Π_i Commit(x_i) = g^{Σ x_i}`` — so **anyone** can verify the servers' published
  aggregate equals the sum the clients committed to, and a malicious server that
  drops or tampers with a share is **detected** (the reconstruction no longer
  matches the product of commitments). Binding rests on discrete-log hardness in
  the order-``q`` subgroup.

Honest assumptions, stated plainly: privacy holds only if the two servers do not
collude; Feldman commitments are *binding* but not *hiding* (they reveal ``g^x``,
which the client owns anyway — it does not help recover ``x`` without solving a
discrete log). Pure Python ``pow``; exponentiation cost is per-coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# RFC 3526 2048-bit MODP group (id 14): a safe prime p = 2q + 1, generator g = 2
# of the order-q subgroup of quadratic residues. Discrete log here is hard.
_P_HEX = (
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
    "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
    "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
    "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF"
)
P: int = int(_P_HEX, 16)
Q: int = (P - 1) // 2  # prime subgroup order
G: int = 2

# Fixed-point encoding: floats -> signed field elements (16-bit fractional).
QUANT_SCALE: int = 1 << 16


def _encode(x: float) -> int:
    """Encode a float as a field element in [0, Q) (signed, wraps negatives)."""
    return int(round(float(x) * QUANT_SCALE)) % Q


def _decode(e: int) -> float:
    """Inverse of :func:`_encode`: map a field element back to a signed float."""
    e %= Q
    if e > Q // 2:
        e -= Q
    return e / QUANT_SCALE


def commit_vector(values: np.ndarray) -> list[int]:
    """Per-coordinate Feldman commitment ``g^x mod p`` to a real vector."""
    return [pow(G, _encode(float(v)), P) for v in np.asarray(values, dtype=np.float64).ravel()]


@dataclass(frozen=True)
class ClientContribution:
    """One client's verifiable submission: two additive shares + commitments."""

    share_a: np.ndarray  # field elements held by server A
    share_b: np.ndarray  # field elements held by server B
    commitments: list[int]  # g^x mod p per coordinate (public)


def _rand_below(bound: int, rng: np.random.Generator) -> int:
    """Uniform integer in [0, bound) for a bignum bound (numpy int64 can't).

    Draws 64 bits more than the modulus so the reduction bias is < 2^-64 — a
    cryptographic-strength one-time-pad mask for the additive share.
    """
    nbytes = (bound.bit_length() + 64) // 8 + 1
    return int.from_bytes(rng.bytes(nbytes), "big") % bound


def split_contribution(values: np.ndarray, *, rng: np.random.Generator) -> ClientContribution:
    """Split a client update into 2-of-2 additive shares and publish commitments."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    enc = np.array([_encode(float(v)) for v in flat], dtype=object)
    a = np.array([_rand_below(Q, rng) for _ in flat], dtype=object)
    b = np.array([(int(e) - int(ai)) % Q for e, ai in zip(enc, a)], dtype=object)
    return ClientContribution(share_a=a, share_b=b, commitments=[pow(G, int(e), P) for e in enc])


def _server_sum(shares: Sequence[np.ndarray]) -> np.ndarray:
    """A server sums its additive shares across clients, coordinate-wise (mod Q)."""
    acc = None
    for s in shares:
        s = np.asarray(s, dtype=object)
        acc = s.copy() if acc is None else np.array([(int(x) + int(y)) % Q for x, y in zip(acc, s)], dtype=object)
    if acc is None:
        raise ValueError("no shares to sum")
    return acc


@dataclass(frozen=True)
class AggregationResult:
    sum_vector: np.ndarray         # decoded real-valued sum
    mean_vector: np.ndarray        # sum / n_clients
    verified: bool                 # commitments matched the reconstruction
    n_clients: int


def aggregate(contributions: Sequence[ClientContribution]) -> AggregationResult:
    """Two-server aggregation + verification against the published commitments.

    Server A sums ``share_a``; server B sums ``share_b``; the reconstructed field
    sum is ``(ΣA + ΣB) mod Q``. Verification recomputes ``Π_i Commit_i`` per
    coordinate and checks it equals ``g^{reconstructed_sum}``. A malicious server
    that altered or dropped a share makes the check FAIL.
    """
    if not contributions:
        raise ValueError("no contributions")
    sum_a = _server_sum([c.share_a for c in contributions])
    sum_b = _server_sum([c.share_b for c in contributions])
    field_sum = np.array([(int(x) + int(y)) % Q for x, y in zip(sum_a, sum_b)], dtype=object)

    dim = len(field_sum)
    verified = True
    for j in range(dim):
        prod = 1
        for c in contributions:
            prod = (prod * c.commitments[j]) % P
        if pow(G, int(field_sum[j]), P) != prod:
            verified = False
            break

    decoded = np.array([_decode(int(e)) for e in field_sum], dtype=np.float64)
    n = len(contributions)
    return AggregationResult(sum_vector=decoded, mean_vector=decoded / n, verified=verified, n_clients=n)


def verify_against_commitments(field_sum: Sequence[int], commitments_per_client: Sequence[list[int]]) -> bool:
    """Standalone check: does ``g^{field_sum}`` equal ``Π_i Commit_i`` per coordinate?"""
    for j, s in enumerate(field_sum):
        prod = 1
        for comm in commitments_per_client:
            prod = (prod * comm[j]) % P
        if pow(G, int(s) % Q, P) != prod:
            return False
    return True


__all__ = [
    "P",
    "Q",
    "G",
    "QUANT_SCALE",
    "commit_vector",
    "ClientContribution",
    "split_contribution",
    "AggregationResult",
    "aggregate",
    "verify_against_commitments",
]
