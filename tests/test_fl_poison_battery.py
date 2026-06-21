"""Honest grading of the FL model-poisoning attack battery vs robust aggregators.

These tests assert the *literature-correct* behaviour of the shipped defences,
including where they FAIL:

* FedAvg is unbounded under scaling / sign-flip (breakdown point 0%).
* Every robust aggregator BOUNDS the naive Gaussian attack.
* Min-Max / Min-Sum (NDSS'21) produce materially larger Krum bias than the naive
  attack near the breakdown point — and force Krum to select a malicious update —
  demonstrating the defence's limitation honestly.
* Breakdown-point preconditions (n > 2f+2 for Krum, n > 2*beta for trimmed-mean)
  are respected by the attack/aggregator harness.

Fast: small dims, few seeds, no torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from horizon_ric.federated import poison_attacks as pa
from horizon_ric.federated import robust


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _benign(seed: int = 0, n_honest: int = 12, dim: int = 40, mu_scale: float = 2.0):
    """Honest population with a non-trivial (non-zero) mean direction."""
    rng = np.random.default_rng(seed)
    mu = mu_scale * np.ones(dim)
    return [mu + rng.normal(0.0, 1.0, size=dim) for _ in range(n_honest)]


def _l2(agg: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(agg - ref))


# ---------------------------------------------------------------------------
# Imports are wired correctly (ALIE/Fang are re-exported, not reimplemented)
# ---------------------------------------------------------------------------
def test_battery_reexports_alie_and_fang_without_reimplementation():
    # The battery must expose the imported attacks AND its own new ones.
    for name in ("alie", "fang_krum", "fang_median", "sign_flip", "scaling",
                 "gaussian", "min_max", "min_sum"):
        assert name in pa.ATTACK_BATTERY
    # alie_attack / fang_* are the very objects from spectrum.attacks (identity).
    from horizon_ric.spectrum import attacks as spectrum_attacks
    assert pa.alie_attack is spectrum_attacks.alie_attack
    assert pa.fang_attack_krum is spectrum_attacks.fang_attack_krum
    assert pa.fang_attack_median is spectrum_attacks.fang_attack_median


# ---------------------------------------------------------------------------
# 1. FedAvg is UNBOUNDED under scaling / sign-flip (breakdown point 0%)
# ---------------------------------------------------------------------------
def test_fedavg_unbounded_under_scaling():
    benign = _benign()
    honest_mean = np.mean(benign, axis=0)
    n_byz = 3
    # Default boost ~ n/f; a single boosted cluster already drags FedAvg far.
    mal = pa.scaling_attack(benign, n_byz)
    agg = robust.fedavg(list(benign) + list(mal))
    # Boost the malicious mass further -> error grows without bound.
    mal_big = pa.scaling_attack(benign, n_byz, boost=1000.0)
    agg_big = robust.fedavg(list(benign) + list(mal_big))
    d, d_big = _l2(agg, honest_mean), _l2(agg_big, honest_mean)
    assert d > 5.0                       # already badly poisoned
    assert d_big > 100.0 * d             # unbounded: scales with the boost factor


def test_fedavg_unbounded_under_sign_flip():
    benign = _benign()
    honest_mean = np.mean(benign, axis=0)
    n_byz = 3
    d_small = _l2(robust.fedavg(list(benign) + list(pa.sign_flip_attack(benign, n_byz, scale=1.0))),
                  honest_mean)
    d_big = _l2(robust.fedavg(list(benign) + list(pa.sign_flip_attack(benign, n_byz, scale=200.0))),
                honest_mean)
    assert d_small > 0.5
    assert d_big > 50.0 * d_small        # grows linearly with the flip scale -> unbounded


# ---------------------------------------------------------------------------
# 2. Robust aggregators BOUND the naive Gaussian attack
# ---------------------------------------------------------------------------
def test_robust_aggregators_bound_gaussian_attack():
    n_honest, n_byz, dim = 12, 4, 40
    for seed in range(4):
        benign = _benign(seed=seed, n_honest=n_honest, dim=dim)
        honest_mean = np.mean(benign, axis=0)
        mal = pa.gaussian_attack(benign, n_byz, sigma_mult=40.0, seed=seed)
        updates = list(benign) + list(mal)

        # FedAvg is wrecked by the noise...
        assert _l2(robust.fedavg(updates), honest_mean) > 5.0

        # ...but the robust aggregators stay near the honest mean (bounded).
        med = _l2(robust.coordinate_median(updates), honest_mean)
        trim = _l2(robust.trimmed_mean(updates, beta=n_byz), honest_mean)
        assert med < 3.0
        assert trim < 3.0

        # Krum must NOT select a malicious (noise) update; n > 2f+2 holds here.
        assert n_honest + n_byz > 2 * n_byz + 2
        kr = robust.krum(updates, f=n_byz)
        assert kr.selected_index < n_honest


# ---------------------------------------------------------------------------
# 3. Min-Max / Min-Sum produce MATERIALLY larger Krum bias than the naive
#    attack near the breakdown point (HONEST limitation of Krum)
# ---------------------------------------------------------------------------
def test_min_max_min_sum_defeat_krum_more_than_gaussian():
    # Near Krum's breakdown: f as large as allowed (n > 2f+2 => f < n_honest-2).
    n_honest, n_byz, dim = 10, 7, 60
    assert n_honest + n_byz > 2 * n_byz + 2  # precondition respected (n=17 > 16)

    gauss_bias, minmax_bias, minsum_bias = [], [], []
    minmax_byz, minsum_byz = [], []
    for seed in range(6):
        benign = _benign(seed=seed, n_honest=n_honest, dim=dim, mu_scale=1.0)
        honest_mean = np.mean(benign, axis=0)

        g = pa.gaussian_attack(benign, n_byz, seed=seed)
        mm = pa.min_max_attack(benign, n_byz)
        ms = pa.min_sum_attack(benign, n_byz)

        kr_g = robust.krum(list(benign) + list(g), f=n_byz)
        kr_mm = robust.krum(list(benign) + list(mm), f=n_byz)
        kr_ms = robust.krum(list(benign) + list(ms), f=n_byz)

        gauss_bias.append(_l2(kr_g.aggregate, honest_mean))
        minmax_bias.append(_l2(kr_mm.aggregate, honest_mean))
        minsum_bias.append(_l2(kr_ms.aggregate, honest_mean))
        minmax_byz.append(kr_mm.selected_index >= n_honest)
        minsum_byz.append(kr_ms.selected_index >= n_honest)

    g_mean = float(np.mean(gauss_bias))
    mm_mean = float(np.mean(minmax_bias))
    ms_mean = float(np.mean(minsum_bias))

    # Optimized attacks do materially MORE damage than the naive Gaussian on Krum.
    assert mm_mean > 1.2 * g_mean
    assert ms_mean > 1.2 * g_mean

    # And they routinely force Krum to SELECT a malicious update (it does not for
    # the naive Gaussian) — the honest demonstration that Krum is defeated.
    assert np.mean(minmax_byz) >= 0.5
    assert np.mean(minsum_byz) >= 0.5


# ---------------------------------------------------------------------------
# 4. Breakdown-point preconditions are respected
# ---------------------------------------------------------------------------
def test_krum_breakdown_precondition_enforced():
    # n = 3 honest + 1 byz = 4, NOT > 2*1+2=4  -> Krum must refuse.
    benign = _benign(n_honest=3, dim=8)
    mal = pa.min_sum_attack(benign, 1)
    with pytest.raises(ValueError):
        robust.krum(list(benign) + list(mal), f=1)


def test_trimmed_mean_breakdown_precondition_enforced():
    # n must exceed 2*beta. n = 4, beta = 2 -> not > 4 -> must refuse.
    benign = _benign(n_honest=2, dim=8)
    mal = pa.gaussian_attack(benign, 2, seed=0)
    with pytest.raises(ValueError):
        robust.trimmed_mean(list(benign) + list(mal), beta=2)


def test_breakdown_points_documented_for_every_aggregator():
    for agg in ("fedavg", "krum", "median", "trimmed_mean"):
        assert agg in pa.BREAKDOWN_POINTS
        assert isinstance(pa.BREAKDOWN_POINTS[agg], str)


# ---------------------------------------------------------------------------
# Attack constructions are well-formed (shape / determinism)
# ---------------------------------------------------------------------------
def test_attacks_return_n_malicious_correct_shape():
    benign = _benign(n_honest=8, dim=16)
    n_byz = 3
    for name, (fn, _cite) in pa.ATTACK_BATTERY.items():
        mal = fn(benign, n_byz)
        mal = list(mal)
        assert len(mal) == n_byz, name
        assert all(np.asarray(m).shape == (16,) for m in mal), name


def test_min_max_stays_within_benign_ball():
    # By construction the Min-Max point's max distance to benign must not exceed
    # the max benign-to-benign distance (that is the whole point of the attack).
    benign = _benign(seed=2, n_honest=12, dim=30, mu_scale=1.0)
    mat = np.asarray(benign)
    diff = mat[:, None, :] - mat[None, :, :]
    max_benign = float(np.max(np.sqrt(np.sum(diff**2, axis=2))))
    mal = pa.min_max_attack(benign, 4)[0]
    max_to_benign = float(np.max(np.linalg.norm(mat - mal, axis=1)))
    # Feasibility (with a tiny binary-search tolerance).
    assert max_to_benign <= max_benign + 1e-3
