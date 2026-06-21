"""Tests for the real neural receiver, the PGD attack, and the Shield fallback.

These prove the attack is genuine (finite-difference gradient check), that the
neural receiver actually learns, that PGD degrades it more than the certified
classical demapper, and that the Shield falls back under independent measurement.
"""

from __future__ import annotations

import numpy as np

from horizon_ric.phy import (
    NeuralReceiver,
    classical_ml_demap,
    make_dataset,
    pgd_attack,
)
from horizon_ric.shield.invariants import NeuralRxEnvelopeInvariant


def _ser(pred, y):
    return float(np.mean(pred != y))


def test_input_gradient_matches_finite_difference():
    # The PGD attack is only real if the input gradient is real. Verify the
    # hand-derived backprop against a finite-difference estimate.
    rng = np.random.default_rng(0)
    X, y = make_dataset(2000, 16, 20.0, rng)
    net = NeuralReceiver(16, hidden=32, seed=1).train(X, y, epochs=10, lr=0.3, seed=2)

    Xs, ys = X[:6].copy(), y[:6].copy()
    g = net.input_gradient(Xs, ys)

    eps = 1e-5
    num = np.zeros_like(Xs)
    for i in range(Xs.shape[0]):
        for j in range(Xs.shape[1]):
            Xp = Xs.copy()
            Xp[i, j] += eps
            Xm = Xs.copy()
            Xm[i, j] -= eps
            lp = -np.sum(np.log(net.probs(Xp)[np.arange(len(Xs)), ys] + 1e-12))
            lm = -np.sum(np.log(net.probs(Xm)[np.arange(len(Xs)), ys] + 1e-12))
            num[i, j] = (lp - lm) / (2 * eps)
    rel = np.abs(g - num).max() / (np.abs(num).max() + 1e-9)
    assert rel < 1e-4, f"input gradient does not match finite difference (rel={rel:.2e})"


def test_neural_receiver_learns():
    rng = np.random.default_rng(0)
    Xtr, ytr = make_dataset(20000, 16, 22.0, rng)
    Xte, yte = make_dataset(5000, 16, 22.0, rng)
    net = NeuralReceiver(16, hidden=128, seed=1).train(Xtr, ytr, epochs=40, lr=0.3, seed=2)
    # Near-optimal: clean error is tiny and close to the classical ML demapper.
    assert _ser(net.predict(Xte), yte) < 0.01
    assert _ser(classical_ml_demap(Xte, 16), yte) < 0.01


def test_pgd_degrades_neural_more_than_classical():
    rng = np.random.default_rng(0)
    Xtr, ytr = make_dataset(20000, 16, 22.0, rng)
    Xte, yte = make_dataset(8000, 16, 22.0, rng)
    net = NeuralReceiver(16, hidden=128, seed=1).train(Xtr, ytr, epochs=40, lr=0.3, seed=2)

    Xadv = pgd_attack(net, Xte, yte, epsilon=0.12, alpha=0.02, steps=30)
    neural = _ser(net.predict(Xadv), yte)
    classical = _ser(classical_ml_demap(Xadv, 16), yte)
    # The bounded perturbation barely moves the max-margin classical demapper but
    # flips the neural receiver much more often.
    assert neural > 3.0 * classical
    assert neural > 0.005


def test_shield_falls_back_when_measured_tbler_exceeds_baseline():
    # Independent CRC/HARQ measurement says the neural receiver is doing far worse
    # than the classical baseline → the envelope invariant must fall back.
    inv = NeuralRxEnvelopeInvariant(tolerance_dB=1.0)
    action = {"block": "neural_rx", "baseline_tbler": 0.001}
    ctx = {"measured_tbler": 0.05}  # measured neural TBLER >> baseline
    assert inv.evaluate(action, ctx).satisfied is False
    safe, corr = inv.project(action, ctx)
    assert safe["block"] == "classical_lmmse"
    assert corr
