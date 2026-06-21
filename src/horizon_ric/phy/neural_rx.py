"""A real (torch-free) neural receiver: a 2-layer MLP symbol demapper.

Forward pass, cross-entropy loss, and backprop are all implemented by hand in
numpy — there is no autodiff framework and no mock. The point of this module is
not raw accuracy (the classical ML demapper is optimal for AWGN); it is to expose
a *differentiable* receiver whose input gradient drives a genuine PGD adversarial
attack (`horizon_ric.phy.pgd`), so the Decision Safety Shield can be shown facing
an attack it was not hand-coded against.

The network maps a received symbol's [Re, Im] features to a softmax over the M
constellation indices. ``input_gradient`` returns dL/dX for the cross-entropy
loss — the exact quantity PGD needs and the one verified by a finite-difference
gradient check in the tests.
"""

from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class NeuralReceiver:
    """2-layer MLP (2 → hidden → M) with ReLU and softmax output."""

    def __init__(self, M: int, hidden: int = 64, seed: int = 0) -> None:
        self.M = M
        self.hidden = hidden
        rng = np.random.default_rng(seed)
        # He initialisation.
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / 2.0), size=(2, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden, M))
        self.b2 = np.zeros(M)

    # ── forward ──────────────────────────────────────────────────────────
    def _forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        h = np.maximum(z1, 0.0)
        z2 = h @ self.W2 + self.b2
        p = _softmax(z2)
        return z1, h, z2, p

    def probs(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[3]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[3].argmax(axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = self.probs(X)
        return float(-np.mean(np.log(p[np.arange(len(X)), y] + 1e-12)))

    # ── training (mini-batch SGD) ────────────────────────────────────────
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 30,
        lr: float = 0.2,
        batch: int = 256,
        seed: int = 0,
    ) -> "NeuralReceiver":
        rng = np.random.default_rng(seed)
        n = len(X)
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                b = idx[s : s + batch]
                Xb, yb = X[b], y[b]
                z1, h, z2, p = self._forward(Xb)
                # dL/dz2 for mean cross-entropy.
                dz2 = p.copy()
                dz2[np.arange(len(b)), yb] -= 1.0
                dz2 /= len(b)
                dW2 = h.T @ dz2
                db2 = dz2.sum(axis=0)
                dh = dz2 @ self.W2.T
                dz1 = dh * (z1 > 0)
                dW1 = Xb.T @ dz1
                db1 = dz1.sum(axis=0)
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
        return self

    # ── input gradient (for PGD) ─────────────────────────────────────────
    def input_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """dL/dX of the (summed) cross-entropy loss — one gradient row per sample.

        Because samples are independent, this row i equals d(CE_i)/dX_i, which is
        exactly the gradient of the total loss w.r.t. X. Verified numerically in
        ``tests/test_neural_rx_pgd.py``.
        """
        z1, h, z2, p = self._forward(X)
        dz2 = p.copy()
        dz2[np.arange(len(X)), y] -= 1.0  # softmax-CE gradient (un-normalised)
        dh = dz2 @ self.W2.T
        dz1 = dh * (z1 > 0)
        dX = dz1 @ self.W1.T
        return dX


__all__ = ["NeuralReceiver"]
