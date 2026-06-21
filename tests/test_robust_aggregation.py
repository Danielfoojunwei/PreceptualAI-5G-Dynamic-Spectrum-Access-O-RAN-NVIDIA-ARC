"""Tests for Byzantine-robust + secure federated aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from horizon_ric.federated import (
    aggregate,
    coordinate_median,
    fedavg,
    flatten_state,
    krum,
    secure_mean,
    trimmed_mean,
    unflatten_state,
)


def _scenario(seed: int = 0, n_honest: int = 7, n_byz: int = 3, dim: int = 40):
    rng = np.random.default_rng(seed)
    honest = [rng.normal(0.0, 1.0, size=dim) for _ in range(n_honest)]
    byz = [np.full(dim, 100.0) for _ in range(n_byz)]
    return honest, byz


def test_fedavg_is_poisoned_by_byzantine_clients():
    honest, byz = _scenario()
    agg = fedavg(honest + byz)
    # 3 attackers at +100 drag the mean far from the honest ~0 mean.
    assert np.abs(agg).mean() > 20.0


def test_krum_rejects_byzantine_clients():
    honest, byz = _scenario()
    result = krum(honest + byz, f=3)
    # Krum must select one of the honest clients (indices 0..6), not an attacker.
    assert result.selected_index < len(honest)
    assert np.abs(result.aggregate).mean() < 3.0


def test_median_and_trimmed_mean_are_robust():
    honest, byz = _scenario(n_honest=9, n_byz=2)
    updates = honest + byz
    assert np.abs(coordinate_median(updates)).mean() < 3.0
    assert np.abs(trimmed_mean(updates, beta=2)).mean() < 3.0


def test_krum_requires_enough_clients():
    honest, byz = _scenario(n_honest=2, n_byz=1)
    with pytest.raises(ValueError):
        krum(honest + byz, f=1)  # n=3 not > 2*1+2=4


def test_aggregate_dispatch():
    honest, byz = _scenario()
    updates = honest + byz
    assert np.abs(aggregate(updates, "krum", f=3)).mean() < 3.0
    assert np.abs(aggregate(updates, "median")).mean() < 3.0
    assert np.abs(aggregate(updates, "fedavg")).mean() > 20.0


def test_flatten_unflatten_roundtrip():
    state = {"w": np.arange(6.0).reshape(2, 3), "b": np.array([1.0, 2.0])}
    flat, manifest = flatten_state(state)
    back = unflatten_state(flat, manifest)
    assert np.allclose(back["w"], state["w"])
    assert np.allclose(back["b"], state["b"])


def test_secure_mean_matches_plain_mean_privately():
    vecs = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]
    got = secure_mean(vecs, n_shares=5, threshold=3)
    expected = np.mean(vecs, axis=0)
    assert np.allclose(got, expected, atol=1e-4)
