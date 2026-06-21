"""QAM constellations + the classical (certified) maximum-likelihood demapper.

Torch-free. The classical demapper is the optimal AWGN receiver (nearest
constellation point); it has straight, max-margin Voronoi boundaries, so a small
bounded perturbation below the margin cannot flip it. That margin property is
exactly why it is the safe *fallback* when the neural receiver is shown — by
independent measurement — to be degraded under attack.
"""

from __future__ import annotations

import numpy as np


def qam_constellation(M: int) -> np.ndarray:
    """Unit-average-power QAM constellation points (complex), indexed 0..M-1."""
    if M == 4:
        pts = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex128) / np.sqrt(2)
    elif M == 16:
        levels = np.array([-3, -1, 1, 3], dtype=np.float64)
        pts = np.array([complex(i, q) for i in levels for q in levels], dtype=np.complex128)
        pts = pts / np.sqrt(10.0)
    else:
        raise ValueError(f"unsupported M={M}; use 4 (QPSK) or 16 (16-QAM)")
    return pts


def modulate(indices: np.ndarray, M: int) -> np.ndarray:
    """Map symbol indices to constellation points."""
    return qam_constellation(M)[np.asarray(indices)]


def symbols_to_features(rx: np.ndarray) -> np.ndarray:
    """Received complex symbols → real (N, 2) feature matrix [Re, Im]."""
    rx = np.asarray(rx)
    return np.stack([rx.real, rx.imag], axis=1).astype(np.float64)


def classical_ml_demap(features: np.ndarray, M: int) -> np.ndarray:
    """Optimal AWGN demapper: nearest constellation point. Input is (N, 2) [Re, Im]."""
    pts = qam_constellation(M)
    rx = features[:, 0] + 1j * features[:, 1]
    d = np.abs(rx[:, None] - pts[None, :])  # (N, M)
    return np.argmin(d, axis=1)


def awgn(symbols: np.ndarray, snr_dB: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN at the given Es/N0 (dB). Es is normalised to 1."""
    n0 = 1.0 / (10.0 ** (snr_dB / 10.0))  # Es=1
    sigma = np.sqrt(n0 / 2.0)  # per real dimension
    noise = sigma * (rng.standard_normal(symbols.shape) + 1j * rng.standard_normal(symbols.shape))
    return symbols + noise


def make_dataset(
    n: int, M: int, snr_dB: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (features (N,2), labels (N,)) of received QAM symbols under AWGN."""
    labels = rng.integers(0, M, size=n)
    tx = modulate(labels, M)
    rx = awgn(tx, snr_dB, rng)
    return symbols_to_features(rx), labels


__all__ = [
    "qam_constellation",
    "modulate",
    "symbols_to_features",
    "classical_ml_demap",
    "awgn",
    "make_dataset",
]
