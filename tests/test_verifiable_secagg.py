"""Verifiable two-server secure aggregation — real tests (no mocks, no torch).

Exercises the real crypto in :mod:`horizon_ric.federated.verifiable_secagg`:
that 2-of-2 additive shares reconstruct the true sum/mean within the fixed-point
tolerance, that the encode/decode is a faithful round-trip, that one server's
share is a one-time pad (masked), that an honest aggregate verifies, that
tampering a share or dropping a client is detected against the Feldman
commitments, and that the commitments are homomorphic. Dims are small so the
per-coordinate modular exponentiations stay fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from horizon_ric.federated import verifiable_secagg as V
from horizon_ric.federated.dp import DPConfig

TOL = 1.0 / V.QUANT_SCALE  # one fixed-point quantum


def _split_all(vecs, rng):
    return [V.split_contribution(v, rng=rng) for v in vecs]


def test_additive_shares_reconstruct_true_sum_and_mean():
    rng = np.random.default_rng(0)
    n, d = 5, 8
    vecs = [rng.normal(size=d) for _ in range(n)]
    res = V.aggregate(_split_all(vecs, rng))

    plaintext_sum = np.sum(vecs, axis=0)
    plaintext_mean = plaintext_sum / n
    # Error is pure 16-bit fixed-point round-off, bounded by d quanta in the worst
    # case; a small multiple of TOL is a faithful, honest tolerance.
    assert np.max(np.abs(res.sum_vector - plaintext_sum)) < n * TOL
    assert np.max(np.abs(res.mean_vector - plaintext_mean)) < TOL
    assert res.n_clients == n


def test_encode_decode_round_trips_signed_float():
    for x in (0.0, 1.5, -3.14159, 12.3456, -0.0001, 250.0, -250.0):
        assert abs(V._decode(V._encode(x)) - x) <= TOL


def test_share_a_is_masked_one_time_pad():
    rng = np.random.default_rng(1)
    vals = rng.normal(size=32)
    enc = [V._encode(float(v)) for v in vals]
    contrib = V.split_contribution(vals, rng=rng)
    # share_a is a fresh uniform field element, never the encoded secret itself.
    for ai, ei in zip(contrib.share_a, enc):
        assert int(ai) != int(ei)
    # and it is in-range for the field
    assert all(0 <= int(ai) < V.Q for ai in contrib.share_a)
    # the two shares still reconstruct the encoding (a + b == x mod Q)
    for ai, bi, ei in zip(contrib.share_a, contrib.share_b, enc):
        assert (int(ai) + int(bi)) % V.Q == int(ei) % V.Q


def test_honest_aggregate_is_verified():
    rng = np.random.default_rng(2)
    vecs = [rng.normal(size=6) for _ in range(4)]
    res = V.aggregate(_split_all(vecs, rng))
    assert res.verified is True


def test_tampering_one_share_element_flips_verified_false():
    rng = np.random.default_rng(3)
    vecs = [rng.normal(size=6) for _ in range(4)]
    contribs = _split_all(vecs, rng)
    assert V.aggregate(contribs).verified is True

    bad_a = contribs[1].share_a.copy()
    bad_a[2] = (int(bad_a[2]) + 1) % V.Q  # flip a single field element
    contribs[1] = V.ClientContribution(
        share_a=bad_a,
        share_b=contribs[1].share_b,
        commitments=contribs[1].commitments,
    )
    assert V.aggregate(contribs).verified is False


def test_dropping_a_client_fails_verification():
    rng = np.random.default_rng(4)
    vecs = [rng.normal(size=6) for _ in range(4)]
    contribs = _split_all(vecs, rng)

    # malicious server sums only 3 of 4 clients' shares, but the public commitment
    # set still lists all 4 → the reconstruction can't match Π Commit.
    kept = contribs[:-1]
    sum_a = V._server_sum([c.share_a for c in kept])
    sum_b = V._server_sum([c.share_b for c in kept])
    field_sum = [(int(x) + int(y)) % V.Q for x, y in zip(sum_a, sum_b)]
    all_commitments = [c.commitments for c in contribs]
    assert V.verify_against_commitments(field_sum, all_commitments) is False

    # sanity: summing ALL clients verifies
    sum_a_full = V._server_sum([c.share_a for c in contribs])
    sum_b_full = V._server_sum([c.share_b for c in contribs])
    field_sum_full = [(int(x) + int(y)) % V.Q for x, y in zip(sum_a_full, sum_b_full)]
    assert V.verify_against_commitments(field_sum_full, all_commitments) is True


def test_commitment_homomorphism_holds():
    # Commit(a) * Commit(b) == g^{enc(a) + enc(b)} mod P, the property that makes
    # the public aggregate check possible.
    for a, b in ((1.5, -2.25), (0.0, 3.3), (-7.7, 7.7), (10.0, 0.5)):
        ca = V.commit_vector([a])[0]
        cb = V.commit_vector([b])[0]
        exponent = (V._encode(a) + V._encode(b)) % V.Q
        assert pow(V.G, exponent, V.P) == (ca * cb) % V.P


def test_client_side_dp_limits_colluding_servers_to_noised_release():
    raw = np.array([0.5, -1.0, 2.0, 0.25])
    config = DPConfig(clip_norm=3.0, noise_multiplier=4.0)
    seed = 73

    expected_rng = np.random.default_rng(seed)
    expected_release = raw + expected_rng.normal(
        0.0,
        config.noise_multiplier * 2.0 * config.clip_norm,
        size=raw.shape,
    )
    contribution = V.split_private_contribution(
        raw,
        dp_config=config,
        delta=1e-5,
        rng=np.random.default_rng(seed),
    )
    colluding_view = V.reconstruct_contribution(contribution)

    assert np.allclose(
        colluding_view,
        expected_release,
        atol=1.0 / V.QUANT_SCALE,
    )
    assert not np.allclose(colluding_view, raw)
    assert contribution.local_dp is not None
    assert contribution.local_dp.epsilon == pytest.approx(1.2675, abs=0.01)
    assert contribution.local_dp.adjacency == "replace_one_client"


def test_locally_private_contributions_still_verify_when_aggregated():
    rng = np.random.default_rng(99)
    config = DPConfig(clip_norm=2.0, noise_multiplier=2.0)
    contributions = [
        V.split_private_contribution(
            rng.normal(size=3),
            dp_config=config,
            delta=1e-5,
            rng=rng,
        )
        for _ in range(4)
    ]
    result = V.aggregate(contributions)
    assert result.verified
    assert result.n_clients == 4


def test_local_dp_rejects_zero_noise():
    with pytest.raises(ValueError, match="noise_multiplier"):
        V.split_private_contribution(
            np.ones(2),
            dp_config=DPConfig(clip_norm=1.0, noise_multiplier=0.0),
            delta=1e-5,
            rng=np.random.default_rng(1),
        )
