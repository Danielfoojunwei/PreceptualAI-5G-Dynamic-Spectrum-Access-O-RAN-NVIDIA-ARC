"""Shamir secret-sharing + SecureFedAvg tests (Row 15 of GAPS_TO_PILOT.md).

Verifies that the new ``secure_aggregation`` module delivers:
    1. Correct round-trip Shamir split/reconstruct.
    2. Threshold-security: < t shares → no recovery (info-theoretic).
    3. SecureFedAvg numerically matches plain FedAvg within 1e-4.
    4. A single share is statistically uncorrelated with its secret
       (chi-square uniformity over the field's low bits).
"""

from __future__ import annotations

import random
import secrets

import pytest
import torch

from horizon_ric.federated import (
    ClientUpdate,
    FedAvg,
    SecureFedAvg,
    ShamirSecretSharing,
)
from horizon_ric.federated.secure_aggregation import PRIME


# ─── 1. Round-trip ──────────────────────────────────────────────────────────
def test_shamir_split_reconstruct_roundtrip() -> None:
    sss = ShamirSecretSharing()
    rng = random.Random(0xC0FFEE)
    for _ in range(20):
        secret = rng.randrange(PRIME)
        shares = sss.split(secret, n_shares=5, threshold=3)
        # Any subset of size ≥ threshold works.
        for subset_idx in [(0, 1, 2), (1, 3, 4), (0, 2, 4), (0, 1, 2, 3, 4)]:
            subset = [shares[i] for i in subset_idx]
            assert sss.reconstruct(subset) == secret


# ─── 2. Threshold security ──────────────────────────────────────────────────
def test_shamir_threshold_security() -> None:
    """With t=3, any 2 shares must leave the secret information-theoretically
    hidden — the polynomial of degree 2 has one free coefficient even after
    fixing two points, so reconstruct(0) is uniform over GF(p) when we
    naively fit a degree-1 line through the 2 points.

    We *do not* call ``reconstruct`` with too few shares (it raises). Instead
    we show that *guessing* the secret from any 2 shares is no better than
    chance: across many trials with the SAME secret, the degree-1
    line-fit-at-zero distribution is uniform, not concentrated on the secret.
    """
    sss = ShamirSecretSharing()
    secret = 42_000_000
    n_trials = 400
    line_fits: list[int] = []
    for _ in range(n_trials):
        shares = sss.split(secret, n_shares=5, threshold=3)
        # Take any 2 shares; fit degree-1 line through them; evaluate at x=0.
        (x1, y1), (x2, y2) = shares[0], shares[1]
        # Lagrange at 0 for two points:
        # L(0) = y1 * (-x2)/(x1-x2) + y2 * (-x1)/(x2-x1)
        from horizon_ric.federated.secure_aggregation import _mod_inverse
        p = PRIME
        l1 = ((-x2) % p * _mod_inverse((x1 - x2) % p, p)) % p
        l2 = ((-x1) % p * _mod_inverse((x2 - x1) % p, p)) % p
        guess = (y1 * l1 + y2 * l2) % p
        line_fits.append(guess)

    # The guesses must NOT collapse to the true secret.
    matches = sum(1 for g in line_fits if g == secret)
    assert matches <= 1, (
        f"with t=3, 2 shares should not reveal the secret; got {matches}/{n_trials} hits"
    )

    # Sanity: also verify reconstruct() refuses below threshold by raising
    # ValueError when we feed only 1 share.
    with pytest.raises(ValueError):
        sss.reconstruct(shares[:1])


# ─── 3. SecureFedAvg ≈ plain FedAvg ─────────────────────────────────────────
def _uniform_clients(n_clients: int, n_elems: int, seed: int) -> list[ClientUpdate]:
    g = torch.Generator().manual_seed(seed)
    return [
        ClientUpdate(
            client_id=f"c{i}",
            state_dict={
                "w": torch.randn(n_elems, generator=g, dtype=torch.float32),
            },
            sample_count=100,  # equal weighting
        )
        for i in range(n_clients)
    ]


def test_secure_fedavg_matches_plaintext() -> None:
    clients = _uniform_clients(n_clients=8, n_elems=32, seed=7)
    plain = FedAvg().aggregate(clients)
    secure = SecureFedAvg(n_shares=5, threshold=3).aggregate(clients)
    delta = (plain["w"].to(torch.float64) - secure["w"].to(torch.float64)).abs().max().item()
    assert delta < 1e-4, f"max abs delta {delta} exceeds 1e-4"


# ─── 4. Per-share leakage / chi-square ──────────────────────────────────────
def test_secure_fedavg_per_share_does_not_leak() -> None:
    """One share, taken alone, must look uniform on GF(p).

    We bin the *low 8 bits* of share-y across many independent splits of the
    same secret and run a chi-square goodness-of-fit against uniform.
    """
    sss = ShamirSecretSharing()
    secret = secrets.randbelow(PRIME)
    n_trials = 4096
    # Bin low 8 bits → 256 buckets → expected count = n_trials/256 = 16.
    buckets = [0] * 256
    for _ in range(n_trials):
        shares = sss.split(secret, n_shares=5, threshold=3)
        # Inspect share #0 only.
        _, y = shares[0]
        buckets[y & 0xFF] += 1

    expected = n_trials / 256.0
    chi2 = sum((c - expected) ** 2 / expected for c in buckets)
    # With 255 d.f., the 0.001 critical value is ≈ 330; observed chi2
    # should be well below that for a CSPRNG-driven uniform distribution.
    assert chi2 < 330.0, (
        f"chi2 {chi2:.1f} > 330 (255 d.f., α=0.001) — share statistically "
        f"correlated with secret?"
    )


# ─── Bonus: 100 × 100 benchmark used for the report ─────────────────────────
def test_secure_fedavg_benchmark_100x100() -> None:
    """Sanity-check the 100-client × 100-element delta the report cites."""
    clients = _uniform_clients(n_clients=100, n_elems=100, seed=13)
    plain = FedAvg().aggregate(clients)
    secure = SecureFedAvg(n_shares=5, threshold=3).aggregate(clients)
    delta = (plain["w"].to(torch.float64) - secure["w"].to(torch.float64)).abs()
    assert delta.max().item() < 1e-4
