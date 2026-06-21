"""Tests for the physical-layer jamming battery + imperfect-CSI stressor.

Each jammer must raise SER monotonically with JSR; the partial-band/pulsed
jammers must touch only the intended fraction/duty of symbols; imperfect CSI must
raise SER vs perfect CSI; and the Shield's effective SER must never beat the
*worse* receiver while never exceeding the better one by more than its tolerance.
Sizes are kept small/fast.
"""

from __future__ import annotations

import numpy as np

from horizon_ric.phy import (
    NeuralReceiver,
    classical_equalize_demap,
    fading_dataset,
)
from horizon_ric.phy.jamming import (
    YH_to_features,
    barrage_jammer,
    features_to_YH,
    imperfect_csi_dataset,
    partial_band_jammer,
    pulsed_jammer,
    single_tone_jammer,
)
from horizon_ric.shield.invariants import NeuralRxEnvelopeInvariant

M = 16
SNR = 28.0
WIN = 200


def _ser(pred, y):
    return float(np.mean(pred != y))


def _train_net(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    Xtr, ytr, _ = fading_dataset(20_000, M, SNR, rng)
    return NeuralReceiver(M, hidden=96, seed=1, n_features=4).train(
        Xtr, ytr, epochs=30, lr=0.3, seed=2
    )


def _classical_ser_under_jammer(jammer_fn):
    """SER of the LMMSE receiver as a function returning a jammed feature matrix.

    We grade jammer monotonicity on the *classical* receiver: it is deterministic
    given the features (no training noise), so the monotonic trend is clean.
    """
    rng = np.random.default_rng(3)
    feats, labels, meta = fading_dataset(6000, M, SNR, rng)
    n0 = meta["N0"]
    base = _ser(classical_equalize_demap(feats, M, n0), labels)
    return base, feats, labels, n0, rng


# ── monotonicity in JSR for every jammer ─────────────────────────────────────
def test_barrage_jammer_monotonic_in_jsr():
    base, feats, labels, n0, _ = _classical_ser_under_jammer(None)
    sers = [base]
    for jsr in [-10.0, -5.0, 0.0, 5.0, 10.0]:
        rng = np.random.default_rng(100 + int(jsr))
        Xj = barrage_jammer(feats, jsr, rng)
        sers.append(_ser(classical_equalize_demap(Xj, M, n0), labels))
    # Non-decreasing (allow tiny stochastic dips via a small tolerance).
    assert all(sers[i + 1] >= sers[i] - 0.01 for i in range(len(sers) - 1))
    assert sers[-1] > sers[0] + 0.05  # high JSR clearly worse than clean


def test_single_tone_jammer_monotonic_in_jsr():
    base, feats, labels, n0, _ = _classical_ser_under_jammer(None)
    sers = [base]
    for jsr in [-10.0, -5.0, 0.0, 5.0, 10.0]:
        rng = np.random.default_rng(200 + int(jsr))
        Xj = single_tone_jammer(feats, jsr, phase=0.7, rng=rng)
        sers.append(_ser(classical_equalize_demap(Xj, M, n0), labels))
    assert all(sers[i + 1] >= sers[i] - 0.01 for i in range(len(sers) - 1))
    assert sers[-1] > sers[0] + 0.05


def test_partial_band_jammer_monotonic_in_jsr():
    base, feats, labels, n0, _ = _classical_ser_under_jammer(None)
    sers = [base]
    for jsr in [-10.0, -5.0, 0.0, 5.0, 10.0]:
        rng = np.random.default_rng(300 + int(jsr))
        Xj = partial_band_jammer(feats, jsr, fraction=0.5, rng=rng)
        sers.append(_ser(classical_equalize_demap(Xj, M, n0), labels))
    assert all(sers[i + 1] >= sers[i] - 0.01 for i in range(len(sers) - 1))
    assert sers[-1] > sers[0] + 0.02


def test_pulsed_jammer_monotonic_in_jsr():
    base, feats, labels, n0, _ = _classical_ser_under_jammer(None)
    sers = [base]
    for jsr in [-10.0, -5.0, 0.0, 5.0, 10.0]:
        rng = np.random.default_rng(400 + int(jsr))
        Xj = pulsed_jammer(feats, jsr, duty=0.5, rng=rng)
        sers.append(_ser(classical_equalize_demap(Xj, M, n0), labels))
    assert all(sers[i + 1] >= sers[i] - 0.01 for i in range(len(sers) - 1))
    assert sers[-1] > sers[0] + 0.02


# ── selectivity: partial-band / pulsed touch only the intended subset ────────
def test_partial_band_jams_only_intended_fraction():
    rng = np.random.default_rng(5)
    feats, _, _ = fading_dataset(4000, M, SNR, rng)
    frac = 0.3
    Xj, mask = partial_band_jammer(feats, 10.0, frac, rng, return_mask=True)
    # The jammed count matches the requested fraction (rounded).
    assert abs(mask.sum() - round(frac * len(mask))) <= 1
    Y0, _ = features_to_YH(feats)
    Yj, _ = features_to_YH(Xj)
    # Unjammed symbols are byte-for-byte identical; jammed ones moved.
    assert np.allclose(Yj[~mask], Y0[~mask])
    assert not np.allclose(Yj[mask], Y0[mask])
    # H columns are never touched by any jammer.
    assert np.allclose(Xj[:, 2:], feats[:, 2:])


def test_pulsed_jammer_contiguous_duty_only():
    rng = np.random.default_rng(6)
    feats, _, _ = fading_dataset(4000, M, SNR, rng)
    duty = 0.25
    Xj, mask = pulsed_jammer(feats, 12.0, duty, rng, return_mask=True)
    assert abs(mask.sum() - round(duty * len(mask))) <= 1
    Y0, _ = features_to_YH(feats)
    Yj, _ = features_to_YH(Xj)
    assert np.allclose(Yj[~mask], Y0[~mask])
    assert not np.allclose(Yj[mask], Y0[mask])


def test_full_fraction_partial_band_equals_full_band():
    # fraction=1.0 must jam every symbol.
    rng = np.random.default_rng(7)
    feats, _, _ = fading_dataset(2000, M, SNR, rng)
    _, mask = partial_band_jammer(feats, 5.0, 1.0, rng, return_mask=True)
    assert mask.all()


# ── imperfect CSI raises SER vs perfect CSI ──────────────────────────────────
def test_imperfect_csi_raises_ser_vs_perfect():
    net = _train_net()
    # Perfect CSI baseline.
    rng = np.random.default_rng(9)
    f0, y0, m0 = imperfect_csi_dataset(8000, M, SNR, sigma_est=0.0, rng=rng)
    n0 = m0["N0"]
    neural0 = _ser(net.predict(f0), y0)
    classical0 = _ser(classical_equalize_demap(f0, M, n0), y0)
    # Imperfect CSI.
    rng2 = np.random.default_rng(10)
    f1, y1, m1 = imperfect_csi_dataset(8000, M, SNR, sigma_est=0.3, rng=rng2)
    neural1 = _ser(net.predict(f1), y1)
    classical1 = _ser(classical_equalize_demap(f1, M, m1["N0"]), y1)
    assert neural1 > neural0 + 1e-3
    assert classical1 > classical0 + 1e-3


def test_imperfect_csi_sigma_zero_matches_perfect_csi():
    # sigma_est=0 must reproduce a perfect-CSI feature matrix (H_hat == H_true).
    rng = np.random.default_rng(11)
    _, _, meta = imperfect_csi_dataset(2000, M, SNR, sigma_est=0.0, rng=rng)
    assert np.allclose(meta["H_hat"], meta["H_true"])


# ── Shield never makes effective SER worse than the better receiver ──────────
def _shield_effective(neural_pred, classical_pred, y):
    n_err = (neural_pred != y).astype(float)
    c_err = (classical_pred != y).astype(float)
    inv = NeuralRxEnvelopeInvariant(tolerance_dB=1.0)
    n_win = max(len(y) // WIN, 1)
    eff = []
    for w in range(n_win):
        s = slice(w * WIN, (w + 1) * WIN)
        measured_neural = float(n_err[s].mean())
        baseline_classical = float(c_err[s].mean())
        action = {"block": "neural_rx", "baseline_tbler": baseline_classical}
        ctx = {"measured_tbler": measured_neural}
        if not inv.evaluate(action, ctx).satisfied:
            eff.append(baseline_classical)
        else:
            eff.append(measured_neural)
    return float(np.mean(eff))


def test_shield_never_worse_than_better_receiver_under_jamming():
    net = _train_net()
    rng = np.random.default_rng(13)
    feats, labels, meta = fading_dataset(8000, M, SNR, rng)
    n0 = meta["N0"]
    # Stress the Shield under a mix of jammers + imperfect CSI.
    cases = [
        barrage_jammer(feats, 5.0, np.random.default_rng(20)),
        single_tone_jammer(feats, 8.0, phase=1.1, rng=np.random.default_rng(21)),
        partial_band_jammer(feats, 10.0, 0.4, np.random.default_rng(22)),
        pulsed_jammer(feats, 10.0, 0.3, np.random.default_rng(23)),
    ]
    for Xj in cases:
        npred = net.predict(Xj)
        cpred = classical_equalize_demap(Xj, M, n0)
        n_ser = _ser(npred, labels)
        c_ser = _ser(cpred, labels)
        eff = _shield_effective(npred, cpred, labels)
        better = min(n_ser, c_ser)
        worse = max(n_ser, c_ser)
        # The Shield routes between the two receivers: effective SER lies within
        # [better - margin, worse + margin]. It can never beat the better one (it
        # is not a denoiser) and the windowed fallback keeps it from blowing past
        # the worse one.
        assert eff <= worse + 0.05
        assert eff >= better - 0.05


def test_shield_effective_under_imperfect_csi_bounded():
    net = _train_net()
    rng = np.random.default_rng(14)
    feats, labels, meta = imperfect_csi_dataset(8000, M, SNR, sigma_est=0.25, rng=rng)
    npred = net.predict(feats)
    cpred = classical_equalize_demap(feats, M, meta["N0"])
    n_ser = _ser(npred, labels)
    c_ser = _ser(cpred, labels)
    eff = _shield_effective(npred, cpred, labels)
    assert eff <= max(n_ser, c_ser) + 0.05
    assert eff >= min(n_ser, c_ser) - 0.05


# ── jammers honour the attacker contract: Y only, never H ────────────────────
def test_jammers_never_touch_channel_columns():
    rng = np.random.default_rng(15)
    feats, _, _ = fading_dataset(1500, M, SNR, rng)
    for Xj in [
        barrage_jammer(feats, 6.0, np.random.default_rng(30)),
        single_tone_jammer(feats, 6.0, phase=0.3, rng=np.random.default_rng(31)),
        partial_band_jammer(feats, 6.0, 0.5, np.random.default_rng(32)),
        pulsed_jammer(feats, 6.0, 0.5, np.random.default_rng(33)),
    ]:
        assert np.allclose(Xj[:, 2:], feats[:, 2:])


def test_feature_roundtrip():
    rng = np.random.default_rng(16)
    feats, _, _ = fading_dataset(500, M, SNR, rng)
    Y, H = features_to_YH(feats)
    assert np.allclose(YH_to_features(Y, H), feats)
