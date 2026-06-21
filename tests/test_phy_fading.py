"""Tests for the realistic fading-channel PHY: LMMSE equaliser, the fading neural
receiver, the Y-only PGD attack, and the honest realism finding."""

from __future__ import annotations

import numpy as np

from horizon_ric.phy import (
    NeuralReceiver,
    classical_equalize_demap,
    fading_dataset,
    lmmse_equalize,
    pgd_attack,
)

MASK = np.array([1, 1, 0, 0])


def _ser(p, y):
    return float(np.mean(p != y))


def test_lmmse_equalizer_recovers_known_symbols():
    # Noise-free: LMMSE (≈ZF here) must invert the channel and recover X exactly.
    rng = np.random.default_rng(0)
    feats, labels, meta = fading_dataset(2000, 4, snr_dB=60.0, rng=rng)  # ~no noise
    pred = classical_equalize_demap(feats, 4, meta["N0"])
    assert _ser(pred, labels) < 1e-3

    # Direct equaliser identity: X̂ ≈ Y/H when N0→0.
    Y = feats[:, 0] + 1j * feats[:, 1]
    H = feats[:, 2] + 1j * feats[:, 3]
    xhat = lmmse_equalize(Y, H, meta["N0"])
    assert np.isfinite(xhat).all()


def test_classical_lmmse_low_error_at_high_snr():
    rng = np.random.default_rng(1)
    feats, labels, meta = fading_dataset(8000, 4, snr_dB=30.0, rng=rng)
    assert _ser(classical_equalize_demap(feats, 4, meta["N0"]), labels) < 0.03


def test_fading_neural_receiver_learns():
    rng = np.random.default_rng(0)
    Xtr, ytr, _ = fading_dataset(20000, 4, 28.0, rng)
    Xte, yte, _ = fading_dataset(5000, 4, 28.0, rng)
    net = NeuralReceiver(4, hidden=128, seed=1, n_features=4).train(
        Xtr, ytr, epochs=40, lr=0.3, seed=2
    )
    assert _ser(net.predict(Xte), yte) < 0.05


def test_fading_input_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    X, y, _ = fading_dataset(2000, 16, 28.0, rng)
    net = NeuralReceiver(16, hidden=32, seed=1, n_features=4).train(X, y, epochs=10, seed=2)
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
    assert rel < 1e-4


def test_pgd_mask_perturbs_received_signal_not_csi():
    # The attacker controls Y (cols 0,1) but never the channel estimate H (cols 2,3).
    rng = np.random.default_rng(0)
    X, y, _ = fading_dataset(3000, 16, 28.0, rng)
    net = NeuralReceiver(16, hidden=64, seed=1, n_features=4).train(X, y, epochs=20, seed=2)
    Xadv = pgd_attack(net, X, y, epsilon=0.08, alpha=0.02, steps=20, perturb_mask=MASK)
    assert np.allclose(Xadv[:, 2:], X[:, 2:])      # H columns untouched
    assert not np.allclose(Xadv[:, :2], X[:, :2])  # Y columns moved


def test_fading_neural_no_better_than_classical_under_attack():
    # Honest realism finding: under fading the neural receiver is at least as
    # vulnerable as the LMMSE baseline (the 22x AWGN gap does NOT survive fading).
    rng = np.random.default_rng(0)
    Xtr, ytr, _ = fading_dataset(30000, 16, 30.0, rng)
    Xte, yte, meta = fading_dataset(8000, 16, 30.0, rng)
    net = NeuralReceiver(16, hidden=128, seed=1, n_features=4).train(
        Xtr, ytr, epochs=50, lr=0.3, seed=2
    )
    Xa = pgd_attack(net, Xte, yte, epsilon=0.08, alpha=0.013, steps=40, perturb_mask=MASK)
    neural = _ser(net.predict(Xa), yte)
    classical = _ser(classical_equalize_demap(Xa, 16, meta["N0"]), yte)
    assert neural >= 0.9 * classical   # neural is not more robust than the baseline
    assert classical > 0.01            # the attack does bite both receivers
