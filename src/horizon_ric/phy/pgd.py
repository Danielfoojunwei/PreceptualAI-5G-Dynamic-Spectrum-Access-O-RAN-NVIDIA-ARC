"""Projected Gradient Descent (PGD) adversarial attack on the neural receiver.

A real white-box L-inf attack (Madry et al., 2018): iteratively step the received
symbol features along the sign of the model's input gradient to maximise the
demap loss, projecting back into an epsilon-ball each step. This models the
O-RAN WG11 / OWASP-ML "input manipulation / adversarial evasion" threat against
an ML-based receiver — an attacker who perturbs the received signal by a small,
bounded amount to force mis-demapping.

The perturbation is crafted on the *neural* receiver's gradients. The classical
ML demapper (nearest constellation point) has straight, max-margin boundaries, so
the same small perturbation flips it far less often — which is precisely why the
Shield's fallback to the classical receiver restores the link.
"""

from __future__ import annotations

import numpy as np

from horizon_ric.phy.neural_rx import NeuralReceiver


def pgd_attack(
    model: NeuralReceiver,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epsilon: float = 0.30,
    alpha: float = 0.05,
    steps: int = 20,
) -> np.ndarray:
    """Return adversarially-perturbed features within an L-inf epsilon-ball of X.

    Args:
        model: the target neural receiver.
        X: clean features (N, 2).
        y: true symbol labels (N,) — the attack maximises loss against these.
        epsilon: L-inf perturbation budget (in normalised symbol units).
        alpha: per-step size.
        steps: number of PGD iterations.
    """
    X0 = np.asarray(X, dtype=np.float64)
    X_adv = X0.copy()
    for _ in range(steps):
        g = model.input_gradient(X_adv, y)
        X_adv = X_adv + alpha * np.sign(g)
        # Project back into the L-inf epsilon-ball around X0.
        X_adv = np.clip(X_adv, X0 - epsilon, X0 + epsilon)
    return X_adv


__all__ = ["pgd_attack"]
