"""A battery of REAL adversarial-evasion attacks on the neural receiver.

These are genuine implementations (no mocks/stubs), all torch-free / numpy, all
driven by the receiver's hand-derived input gradient
(:meth:`horizon_ric.phy.neural_rx.NeuralReceiver.input_gradient`) — the same
quantity verified by a finite-difference gradient check in the tests. They
extend the single white-box ``pgd_attack`` (Madry et al., 2018) already in
``horizon_ric.phy.pgd`` to the standard O-RAN WG11 / OWASP-ML "input
manipulation / adversarial evasion" threat surface against an ML-based receiver:

White-box (attacker has the target's gradients):
  * ``fgsm``  — Fast Gradient Sign Method, single step.
    Goodfellow, Shlens & Szegedy, "Explaining and Harnessing Adversarial
    Examples", ICLR 2015. https://arxiv.org/abs/1412.6572
  * ``bim``   — Basic Iterative Method / I-FGSM, iterated FGSM with L-inf
    projection. Kurakin, Goodfellow & Bengio, "Adversarial Examples in the
    Physical World", ICLR Workshop 2017. https://arxiv.org/abs/1607.02533
  * ``mim``   — Momentum Iterative Method, BIM with a gradient-momentum buffer
    (L1-normalised gradient accumulation). Dong et al., "Boosting Adversarial
    Attacks with Momentum", CVPR 2018. https://arxiv.org/abs/1710.06081

Black-box (attacker has NO target gradients):
  * ``transfer_attack`` — train a *surrogate* neural receiver (different seed /
    width), craft a white-box PGD perturbation on the surrogate, and transfer
    the adversarial features to the target. Transferability of adversarial
    examples: Papernot, McDaniel & Goodfellow, "Transferability in Machine
    Learning", 2016 (https://arxiv.org/abs/1605.07277); the transfer threat
    model is also Liu et al., ICLR 2017 (https://arxiv.org/abs/1611.02770).
  * ``boundary_attack`` — a cheap decision-based black-box attack: start from a
    large adversarial perturbation that flips the label, then random-walk toward
    the clean input while staying mis-classified, using ONLY the target's hard
    ``predict`` decision (no scores, no gradients). Brendel, Rauber & Bethge,
    "Decision-Based Adversarial Attacks", ICLR 2018.
    https://arxiv.org/abs/1712.04248

Every attack:
  * takes ``(model, X, y, ...)`` and returns adversarial features the same shape
    as ``X``, projected into an L-inf ``epsilon``-ball around ``X``;
  * supports a ``perturb_mask`` (e.g. ``[1, 1, 0, 0]`` for the fading model where
    the attacker controls the received signal Y but not the channel estimate H),
    so masked feature columns (the CSI) are provably left untouched.

The point of the battery — like the existing PGD benchmark — is to face the
Decision Safety Shield with attacks it was *not* hand-coded against, and to
report HONESTLY where each one succeeds (AWGN white-box) and where the gap
collapses (fading; black-box transfer being weaker than white-box).
"""

from __future__ import annotations

import numpy as np

from horizon_ric.phy.neural_rx import NeuralReceiver
from horizon_ric.phy.pgd import pgd_attack


def _mask_2d(perturb_mask: np.ndarray | None, n_features: int) -> np.ndarray | None:
    if perturb_mask is None:
        return None
    return np.asarray(perturb_mask, dtype=np.float64)[None, :]


def fgsm(
    model: NeuralReceiver,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float = 0.12,
    perturb_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fast Gradient Sign Method (Goodfellow et al., ICLR 2015).

    A single white-box step of size ``epsilon`` along the sign of the loss
    gradient: ``X_adv = X + epsilon * sign(∇_X L(model(X), y))``. The result is
    automatically inside the L-inf ``epsilon``-ball, so no extra projection is
    needed for the unmasked case. Masked columns are not perturbed.
    """
    X0 = np.asarray(X, dtype=np.float64)
    mask = _mask_2d(perturb_mask, X0.shape[1])
    g = model.input_gradient(X0, y)
    if mask is not None:
        g = g * mask
    return X0 + epsilon * np.sign(g)


def bim(
    model: NeuralReceiver,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float = 0.12,
    alpha: float | None = None,
    steps: int = 10,
    perturb_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Basic Iterative Method / I-FGSM (Kurakin et al., ICLR Workshop 2017).

    Iterated FGSM: take ``steps`` steps of size ``alpha`` along the sign of the
    gradient, clipping back into the L-inf ``epsilon``-ball after each step. This
    is exactly the un-randomised PGD already provided, so we delegate to the
    audited ``pgd_attack`` (Madry et al.) to keep a single projection path.
    BIM starts at the clean input (no random init), which ``pgd_attack`` does.

    Default per-step size follows Kurakin's rule of thumb ``alpha = epsilon /
    steps`` (rounded up via ``min`` so a few extra steps cannot leave the ball).
    """
    if alpha is None:
        alpha = epsilon / max(steps, 1)
    return pgd_attack(
        model, X, y, epsilon=epsilon, alpha=alpha, steps=steps, perturb_mask=perturb_mask
    )


def mim(
    model: NeuralReceiver,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float = 0.12,
    alpha: float | None = None,
    steps: int = 10,
    decay: float = 1.0,
    perturb_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Momentum Iterative Method (Dong et al., CVPR 2018).

    BIM with a velocity buffer that accumulates the L1-normalised gradient:

        g_{t+1} = decay * g_t + ∇_X L / ||∇_X L||_1
        X_{t+1} = clip_eps( X_t + alpha * sign(g_{t+1}) )

    The momentum stabilises the update direction and is what makes MIM transfer
    better than vanilla BIM (the paper's headline result). The L1 normalisation
    is per-sample so samples with tiny gradients are not drowned out. Masked
    columns are zeroed in the gradient before accumulation and so never move.
    """
    if alpha is None:
        alpha = epsilon / max(steps, 1)
    X0 = np.asarray(X, dtype=np.float64)
    X_adv = X0.copy()
    mask = _mask_2d(perturb_mask, X0.shape[1])
    g_accum = np.zeros_like(X0)
    for _ in range(steps):
        g = model.input_gradient(X_adv, y)
        if mask is not None:
            g = g * mask
        l1 = np.abs(g).sum(axis=1, keepdims=True)
        g_norm = g / np.where(l1 > 0, l1, 1.0)
        g_accum = decay * g_accum + g_norm
        X_adv = X_adv + alpha * np.sign(g_accum)
        X_adv = np.clip(X_adv, X0 - epsilon, X0 + epsilon)
    return X_adv


def transfer_attack(
    target: NeuralReceiver,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    M: int,
    epsilon: float = 0.12,
    alpha: float | None = None,
    steps: int = 30,
    surrogate_hidden: int | None = None,
    surrogate_seed: int = 17,
    train_epochs: int = 40,
    train_lr: float = 0.3,
    train_seed: int = 23,
    perturb_mask: np.ndarray | None = None,
    attack: str = "mim",
) -> tuple[np.ndarray, NeuralReceiver]:
    """Black-box TRANSFER attack — the attacker has NO target gradients.

    Threat model (O-RAN WG11 black-box evasion): the attacker can observe the
    same kind of training data the operator used, but cannot read the deployed
    target receiver's weights or gradients. They therefore train their OWN
    *surrogate* neural receiver — deliberately a DIFFERENT architecture (different
    seed and, by default, a different hidden width) so this is a real transfer
    and not a thinly-disguised white-box attack — craft an adversarial
    perturbation on the surrogate's gradients, and transfer the perturbed
    features to the target.

    Transferability of adversarial examples (Papernot et al. 2016; Liu et al.
    ICLR 2017): perturbations crafted on one model often fool another trained on
    similar data, but LESS reliably than a white-box attack on the true target —
    which is exactly the honest finding the benchmark/tests assert (transfer SER
    <= white-box SER).

    Args:
        target: the deployed receiver (used ONLY via ``.predict`` downstream; no
            gradients are read here — its weights are never touched).
        X_train, y_train: data the attacker uses to train the surrogate.
        X, y: the clean evaluation features/labels to perturb.
        M: constellation order (surrogate output dimension).
        attack: which white-box crafter to run on the surrogate
            (``"mim"`` — default, best transfer — ``"bim"`` or ``"fgsm"``).

    Returns:
        ``(X_adv, surrogate)`` — the adversarial features (to be fed to the
        target) and the trained surrogate (returned for auditing / reuse).
    """
    if surrogate_hidden is None:
        # A different width from the typical target (128/160) so the surrogate is
        # a genuinely different model — strengthens the "no target access" claim.
        surrogate_hidden = 96
    n_features = np.asarray(X).shape[1]
    surrogate = NeuralReceiver(
        M, hidden=surrogate_hidden, seed=surrogate_seed, n_features=n_features
    ).train(X_train, y_train, epochs=train_epochs, lr=train_lr, seed=train_seed)

    a = attack.lower()
    if a == "fgsm":
        X_adv = fgsm(surrogate, X, y, epsilon=epsilon, perturb_mask=perturb_mask)
    elif a == "bim":
        X_adv = bim(
            surrogate, X, y, epsilon=epsilon, alpha=alpha, steps=steps,
            perturb_mask=perturb_mask,
        )
    elif a == "mim":
        X_adv = mim(
            surrogate, X, y, epsilon=epsilon, alpha=alpha, steps=steps,
            perturb_mask=perturb_mask,
        )
    else:
        raise ValueError(f"unknown surrogate attack {attack!r}; use fgsm/bim/mim")
    # ``target`` is intentionally unused for crafting — that is the whole point of
    # a black-box transfer attack. We reference it only to validate compatibility.
    if target.M != M:
        raise ValueError("target/surrogate constellation order mismatch")
    return X_adv, surrogate


def boundary_attack(
    model: NeuralReceiver,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float = 0.12,
    steps: int = 40,
    sigma: float = 0.03,
    seed: int = 0,
    perturb_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Cheap decision-based black-box (boundary-style) attack.

    Uses ONLY the target's hard ``predict`` decision — no scores, no gradients
    (Brendel et al., ICLR 2018). For each sample:

      1. Find an initial adversarial point by pushing the feature toward the L-inf
         ball corner along random signs until the target mis-classifies (or use
         the full ``epsilon`` corner). Samples that never flip keep the clean
         feature (the attack simply fails on them — reported honestly).
      2. Random-walk: propose small Gaussian steps that move *toward* the clean
         input; accept a step only if the target still mis-classifies. This
         shrinks the perturbation while staying adversarial.

    The result stays within the L-inf ``epsilon``-ball and respects
    ``perturb_mask`` (masked columns never move). It is deliberately simple and
    cheap; it is weaker than the white-box attacks and is included to show a
    score-free black-box baseline.
    """
    rng = np.random.default_rng(seed)
    X0 = np.asarray(X, dtype=np.float64)
    n, f = X0.shape
    mask_row = (
        np.ones(f) if perturb_mask is None else np.asarray(perturb_mask, dtype=np.float64)
    )
    y = np.asarray(y)

    # --- 1. Initialise on the eps-corner with random signs; keep only flips. ---
    X_adv = X0.copy()
    is_adv = np.zeros(n, dtype=bool)
    for _ in range(8):  # a few random corners to find an initial adversarial point
        not_adv = ~is_adv
        if not not_adv.any():
            break
        signs = rng.choice([-1.0, 1.0], size=(n, f))
        cand = X0 + epsilon * signs * mask_row[None, :]
        pred = model.predict(cand)
        flipped = (pred != y) & not_adv
        X_adv[flipped] = cand[flipped]
        is_adv |= flipped

    # --- 2. Random-walk toward the clean input while staying adversarial. ----
    for _ in range(steps):
        # Step a fraction of the way back to X0 (shrinks the perturbation)...
        toward = X0 - X_adv
        proposal = X_adv + 0.5 * toward * mask_row[None, :]
        # ...plus a small orthogonal-ish Gaussian exploration step.
        proposal = proposal + sigma * rng.standard_normal((n, f)) * mask_row[None, :]
        proposal = np.clip(proposal, X0 - epsilon, X0 + epsilon)
        pred = model.predict(proposal)
        still_adv = (pred != y) & is_adv
        X_adv[still_adv] = proposal[still_adv]
    # Samples that never flipped stay at clean X0 (mask already guaranteed CSI
    # untouched because every update was multiplied by mask_row).
    return X_adv


__all__ = [
    "fgsm",
    "bim",
    "mim",
    "transfer_attack",
    "boundary_attack",
]
