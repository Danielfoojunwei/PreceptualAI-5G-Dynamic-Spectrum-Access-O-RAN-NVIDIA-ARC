"""Tests for the adversarial-evasion attack battery (`horizon_ric.phy.evasion`).

These prove each attack is REAL: it raises the neural receiver's symbol-error
rate above the clean baseline (the attack works); the black-box TRANSFER attack
is weaker than the corresponding white-box attack (no free lunch from a
surrogate); the gradient-sign / epsilon-ball geometry is respected; and the
``perturb_mask`` provably leaves the CSI feature columns untouched.

Training sizes are kept modest so the whole module runs in well under a minute.
"""

from __future__ import annotations

import numpy as np

from horizon_ric.phy import (
    NeuralReceiver,
    classical_ml_demap,
    fading_dataset,
    make_dataset,
)
from horizon_ric.phy.evasion import (
    bim,
    boundary_attack,
    fgsm,
    mim,
    transfer_attack,
)

MASK = np.array([1, 1, 0, 0])


def _ser(pred, y):
    return float(np.mean(pred != y))


def _awgn_net(snr=22.0, ntr=15000, nte=4000, seed=0):
    rng = np.random.default_rng(seed)
    Xtr, ytr = make_dataset(ntr, 16, snr, rng)
    Xte, yte = make_dataset(nte, 16, snr, rng)
    net = NeuralReceiver(16, hidden=128, seed=1).train(Xtr, ytr, epochs=35, lr=0.3, seed=2)
    return net, Xtr, ytr, Xte, yte


def _fading_net(snr=30.0, ntr=18000, nte=5000, seed=0):
    rng = np.random.default_rng(seed)
    Xtr, ytr, _ = fading_dataset(ntr, 16, snr, rng)
    Xte, yte, meta = fading_dataset(nte, 16, snr, rng)
    net = NeuralReceiver(16, hidden=128, seed=1, n_features=4).train(
        Xtr, ytr, epochs=35, lr=0.3, seed=2
    )
    return net, Xtr, ytr, Xte, yte, meta


# ── each white-box attack raises SER above clean (the attack works) ─────────
def test_fgsm_raises_ser_above_clean():
    net, _, _, Xte, yte = _awgn_net()
    clean = _ser(net.predict(Xte), yte)
    adv = _ser(net.predict(fgsm(net, Xte, yte, epsilon=0.12)), yte)
    assert adv > clean
    assert adv > 0.005  # meaningfully above the (near-zero) clean error


def test_bim_raises_ser_above_clean_and_above_fgsm():
    net, _, _, Xte, yte = _awgn_net()
    clean = _ser(net.predict(Xte), yte)
    fgsm_ser = _ser(net.predict(fgsm(net, Xte, yte, epsilon=0.12)), yte)
    bim_ser = _ser(net.predict(bim(net, Xte, yte, epsilon=0.12, steps=15)), yte)
    assert bim_ser > clean
    # Iterative BIM is at least as strong a white-box attack as single-step FGSM.
    assert bim_ser >= fgsm_ser - 1e-9


def test_mim_raises_ser_above_clean():
    net, _, _, Xte, yte = _awgn_net()
    clean = _ser(net.predict(Xte), yte)
    adv = _ser(net.predict(mim(net, Xte, yte, epsilon=0.12, steps=15)), yte)
    assert adv > clean
    assert adv > 0.005


def test_boundary_attack_raises_ser_above_clean_hard_label_only():
    # Decision-based black-box: uses only model.predict (no scores/gradients).
    net, _, _, Xte, yte = _awgn_net()
    Xs, ys = Xte[:2000], yte[:2000]
    clean = _ser(net.predict(Xs), ys)
    adv = _ser(net.predict(boundary_attack(net, Xs, ys, epsilon=0.12, steps=30, seed=0)), ys)
    assert adv > clean


# ── transfer attack is weaker than white-box ────────────────────────────────
def test_transfer_attack_no_stronger_than_whitebox():
    net, Xtr, ytr, Xte, yte = _awgn_net()
    # White-box reference (best of the gradient attacks on the TRUE target).
    wb = max(
        _ser(net.predict(fgsm(net, Xte, yte, epsilon=0.12)), yte),
        _ser(net.predict(bim(net, Xte, yte, epsilon=0.12, steps=20)), yte),
        _ser(net.predict(mim(net, Xte, yte, epsilon=0.12, steps=20)), yte),
    )
    Xa, surrogate = transfer_attack(
        net, Xtr, ytr, Xte, yte, M=16, epsilon=0.12, steps=20, attack="mim"
    )
    transfer_ser = _ser(net.predict(Xa), yte)
    # The black-box transfer still works (above clean)...
    assert transfer_ser > _ser(net.predict(Xte), yte)
    # ...but is no stronger than a white-box attack on the true target.
    assert transfer_ser <= wb + 1e-9
    # The surrogate is a genuinely different model (different width than target).
    assert surrogate.hidden != net.hidden


def test_transfer_surrogate_does_not_read_target_gradients():
    # Sanity: the surrogate's weights differ from the target's — the perturbation
    # is crafted on a model the attacker trained, not on the deployed target.
    net, Xtr, ytr, Xte, yte = _awgn_net(nte=2000)
    _, surrogate = transfer_attack(
        net, Xtr, ytr, Xte, yte, M=16, epsilon=0.12, steps=10, attack="bim"
    )
    assert surrogate.W1.shape != net.W1.shape  # different hidden width
    assert not np.allclose(surrogate.W2.ravel()[:10], net.W2.ravel()[:10])


# ── gradient / epsilon-ball sanity checks ───────────────────────────────────
def test_fgsm_is_exact_eps_corner_along_gradient_sign():
    # FGSM must place the perturbation on the L-inf eps-corner in the direction
    # of the gradient sign: X_adv - X == eps * sign(input_gradient).
    net, _, _, Xte, yte = _awgn_net(nte=500)
    eps = 0.1
    Xa = fgsm(net, Xte, yte, epsilon=eps)
    g = net.input_gradient(Xte, yte)
    expected = eps * np.sign(g)
    assert np.allclose(Xa - Xte, expected, atol=1e-12)


def test_iterative_attacks_stay_in_epsilon_ball():
    # BIM and MIM must never leave the L-inf eps-ball around the clean input.
    net, _, _, Xte, yte = _awgn_net(nte=1000)
    eps = 0.12
    for adv in (
        bim(net, Xte, yte, epsilon=eps, steps=25),
        mim(net, Xte, yte, epsilon=eps, steps=25),
    ):
        assert np.all(np.abs(adv - Xte) <= eps + 1e-9)


# ── perturb_mask leaves the CSI columns untouched (every attack) ────────────
def test_perturb_mask_leaves_csi_columns_untouched():
    net, Xtr, ytr, Xte, yte, _ = _fading_net()
    attacks = {
        "fgsm": fgsm(net, Xte, yte, epsilon=0.08, perturb_mask=MASK),
        "bim": bim(net, Xte, yte, epsilon=0.08, steps=20, perturb_mask=MASK),
        "mim": mim(net, Xte, yte, epsilon=0.08, steps=20, perturb_mask=MASK),
        "transfer": transfer_attack(
            net, Xtr, ytr, Xte, yte, M=16, epsilon=0.08, steps=15,
            perturb_mask=MASK, attack="mim",
        )[0],
        "boundary": boundary_attack(
            net, Xte[:1500], yte[:1500], epsilon=0.08, steps=20, seed=0,
            perturb_mask=MASK,
        ),
    }
    for name, Xa in attacks.items():
        ref = Xte if name != "boundary" else Xte[:1500]
        # CSI columns (H = cols 2,3) are provably untouched.
        assert np.allclose(Xa[:, 2:], ref[:, 2:]), f"{name} moved the CSI columns"
        # The Y columns (0,1) are actually perturbed (the attack did something).
        assert not np.allclose(Xa[:, :2], ref[:, :2]), f"{name} did not move Y"


def test_masked_attack_raises_neural_ser_under_fading():
    # Even with the CSI-frozen mask, the Y-only attack bites the neural receiver
    # under fading (above clean).
    net, _, _, Xte, yte, _ = _fading_net()
    clean = _ser(net.predict(Xte), yte)
    adv = _ser(net.predict(mim(net, Xte, yte, epsilon=0.08, steps=25, perturb_mask=MASK)), yte)
    assert adv > clean
    assert adv > 0.02
