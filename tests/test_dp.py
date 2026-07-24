"""Differential privacy for federated DSA — real tests (no mocks, no torch).

Exercises the real ``horizon_ric.federated.dp`` module: that L2 clipping caps the
sensitivity, that the Rényi-DP accountant composes (epsilon grows with rounds and
shrinks with more noise), that the RDP→(ε,δ) conversion is well-formed, that
DP-FedAvg perturbs the mean and charges the accountant exactly once, and that the
membership-inference distance signal it defends against is real. Configs are
deliberately small so the suite stays fast.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from horizon_ric.federated import dp
from horizon_ric.spectrum.dsa_env import DSAConfig, DSAEnv
from horizon_ric.spectrum.federated_q import QLearnConfig, train_local_q


# ── L2 clipping (DP sensitivity bound) ──────────────────────────────────────
def test_l2_clip_caps_large_vector_and_leaves_small_one_unchanged():
    big = np.array([3.0, 4.0])  # norm 5
    clipped = dp.l2_clip(big, 1.0)
    assert np.linalg.norm(clipped) == pytest.approx(1.0, abs=1e-12)
    # direction preserved
    assert np.allclose(clipped, big / 5.0)

    small = np.array([0.1, -0.2, 0.05])
    assert np.linalg.norm(small) < 1.0
    out = dp.l2_clip(small, 1.0)
    assert np.allclose(out, small)
    # returns a copy, not a view into the input
    assert out is not small


def test_l2_clip_rejects_nonpositive_norm():
    with pytest.raises(ValueError):
        dp.l2_clip(np.ones(3), 0.0)


# ── Rényi-DP accountant composition ─────────────────────────────────────────
def test_epsilon_increases_with_more_steps():
    acc = dp.RDPAccountant()
    acc.step(1.0)
    eps1 = acc.get_epsilon(1e-5)
    for _ in range(9):
        acc.step(1.0)
    eps10 = acc.get_epsilon(1e-5)
    assert acc.steps == 10
    assert eps10 > eps1


def test_epsilon_lower_for_larger_noise_multiplier():
    # More noise => smaller epsilon => more privacy (same number of rounds).
    acc_lo = dp.RDPAccountant()
    acc_hi = dp.RDPAccountant()
    for _ in range(10):
        acc_lo.step(1.0)
        acc_hi.step(4.0)
    eps_lo = acc_lo.get_epsilon(1e-5)
    eps_hi = acc_hi.get_epsilon(1e-5)
    assert eps_hi < eps_lo
    # the module's documented anchors: z=1 over 10 rounds ~20.2, z=4 ~4.1
    assert eps_lo == pytest.approx(20.2, abs=1.5)
    assert eps_hi == pytest.approx(4.1, abs=1.0)


def test_get_epsilon_rejects_delta_out_of_range():
    acc = dp.RDPAccountant()
    acc.step(1.0)
    with pytest.raises(ValueError):
        acc.get_epsilon(1.0)  # delta >= 1 is invalid
    with pytest.raises(ValueError):
        acc.get_epsilon(0.0)  # delta <= 0 is invalid


def test_rdp_to_dp_epsilon_returns_finite_eps_and_valid_order():
    acc = dp.RDPAccountant()
    for _ in range(5):
        acc.step(2.0)
    eps, order = acc.get_epsilon_and_order(1e-5)
    assert math.isfinite(eps)
    assert eps > 0.0
    assert order > 1.0  # a valid Rényi order is strictly > 1
    assert order in dp.DEFAULT_ORDERS


# ── DP-FedAvg mechanism ─────────────────────────────────────────────────────
def test_dp_fedavg_output_shape_matches_flattened_update():
    rng = np.random.default_rng(0)
    updates = [np.ones((4, 3)) * (i + 1) for i in range(5)]
    cfg = dp.DPConfig(clip_norm=10.0, noise_multiplier=1.0)
    out = dp.dp_fedavg(updates, cfg, rng=rng)
    # dp_fedavg works on flattened updates; output is the flattened aggregate.
    assert out.shape == (12,)


def test_dp_fedavg_large_noise_moves_far_from_true_mean():
    updates = [np.full(50, 1.0), np.full(50, 1.0)]  # true mean is all-ones (norm < clip)
    true_mean = np.mean(np.stack(updates), axis=0)
    rng = np.random.default_rng(7)
    cfg = dp.DPConfig(clip_norm=10.0, noise_multiplier=50.0)  # huge noise
    out = dp.dp_fedavg(updates, cfg, rng=rng)
    dist = float(np.linalg.norm(out - true_mean))
    # with z=50 the Gaussian noise dwarfs the signal => output far from the mean
    assert dist > 1.0


def test_dp_fedavg_charges_exactly_one_step_per_call():
    acc = dp.RDPAccountant()
    rng = np.random.default_rng(1)
    cfg = dp.DPConfig(clip_norm=1.0, noise_multiplier=2.0)
    updates = [np.ones(4), np.zeros(4)]
    dp.dp_fedavg(updates, cfg, rng=rng, accountant=acc)
    assert acc.steps == 1
    dp.dp_fedavg(updates, cfg, rng=rng, accountant=acc)
    assert acc.steps == 2


def test_dp_fedavg_noise_uses_replace_one_sensitivity():
    """Noise is calibrated to 2C, not C, for replace-one client adjacency."""
    cfg = dp.DPConfig(clip_norm=3.0, noise_multiplier=2.0)
    seed = 19
    actual = dp.dp_fedavg(
        [np.zeros(4)],
        cfg,
        rng=np.random.default_rng(seed),
    )
    expected = np.random.default_rng(seed).normal(0.0, 12.0, size=4)
    assert np.allclose(actual, expected)


def test_dp_fedavg_requires_at_least_one_update():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        dp.dp_fedavg([], dp.DPConfig(), rng=rng)


# ── Membership-inference distance signal (what DP defends against) ───────────
def test_mia_distance_signal_exists_without_dp():
    """On a small federation with no DP, the global average is CLOSER to its
    members than to fresh non-members — the membership signal the DP suite drives
    toward 0.5. We assert the directional signal, not a strict AUC threshold."""
    dsa = DSAConfig()
    qc = QLearnConfig(episodes=6)

    def client(seed: int) -> np.ndarray:
        return train_local_q(DSAEnv(cfg=dsa, seed=seed), qc, seed=seed)

    members = [client(1000 + i) for i in range(6)]
    nonmembers = [client(5000 + i) for i in range(6)]

    global_q = np.mean(np.stack(members), axis=0)
    g = global_q.ravel()
    mean_member_dist = float(np.mean([np.linalg.norm(g - m.ravel()) for m in members]))
    mean_nonmember_dist = float(np.mean([np.linalg.norm(g - m.ravel()) for m in nonmembers]))

    # a contributor is, on average, closer to the aggregate than a stranger
    assert mean_member_dist < mean_nonmember_dist
