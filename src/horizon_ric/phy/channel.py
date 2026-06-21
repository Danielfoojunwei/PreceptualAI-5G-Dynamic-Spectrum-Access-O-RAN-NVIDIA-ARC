"""Real frequency-selective (multipath) fading channel + LMMSE/ZF equalisers.

A step up in realism from the single-symbol AWGN model: an OFDM subcarrier sees
``Y_k = H_k * X_k + N_k`` where ``H_k`` is the frequency response of a random
multipath (Rayleigh tapped-delay-line) channel. The classical receiver does
per-subcarrier **LMMSE equalisation** then nearest-constellation demapping — the
optimal-ish linear receiver and the Shield's certified fallback. The neural
receiver instead learns to demap from ``[Re Y, Im Y, Re H, Im H]`` (perfect-CSI
assumption, standard for a neural OFDM receiver).

Everything is numpy. The attacker perturbs the *received* signal ``Y`` only (it
does not control the channel ``H``), which is enforced by a perturbation mask in
the PGD benchmark.
"""

from __future__ import annotations

import numpy as np

from horizon_ric.phy.constellation import modulate, qam_constellation


def rayleigh_taps(n_sym: int, n_taps: int, rng: np.random.Generator) -> np.ndarray:
    """Block-fading Rayleigh TDL taps, exponential power-delay profile, unit power.

    Returns (n_sym, n_taps) complex taps — one independent channel per symbol.
    """
    pdp = np.exp(-np.arange(n_taps) / max(n_taps / 3.0, 1.0))
    pdp = pdp / pdp.sum()
    std = np.sqrt(pdp / 2.0)  # per real/imag dim
    taps = (rng.standard_normal((n_sym, n_taps)) + 1j * rng.standard_normal((n_sym, n_taps)))
    return taps * std[None, :]


def freq_response(taps: np.ndarray, n_fft: int) -> np.ndarray:
    """Per-symbol frequency response sampled on one subcarrier (the carrier the
    symbol is sent on). Returns (n_sym,) complex H — the FFT of the zero-padded
    taps evaluated at a fixed subcarrier index per symbol (rotated by symbol)."""
    n_sym, n_taps = taps.shape
    # Full FFT, then pick a subcarrier that rotates across symbols so the dataset
    # spans the whole frequency response rather than one bin.
    H_full = np.fft.fft(taps, n=n_fft, axis=1)  # (n_sym, n_fft)
    k = np.arange(n_sym) % n_fft
    return H_full[np.arange(n_sym), k]


def fading_dataset(
    n: int, M: int, snr_dB: float, rng: np.random.Generator, n_taps: int = 4, n_fft: int = 64
):
    """Generate a fading OFDM demap dataset.

    Returns:
        features (n, 4) = [Re Y, Im Y, Re H, Im H]
        labels   (n,)   symbol indices
        meta dict with Y, H, N0 for the classical equaliser.
    """
    labels = rng.integers(0, M, size=n)
    X = modulate(labels, M)
    taps = rayleigh_taps(n, n_taps, rng)
    H = freq_response(taps, n_fft)
    n0 = 1.0 / (10.0 ** (snr_dB / 10.0))  # Es = 1
    noise = np.sqrt(n0 / 2.0) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    Y = H * X + noise
    feats = np.stack([Y.real, Y.imag, H.real, H.imag], axis=1).astype(np.float64)
    return feats, labels, {"N0": n0}


def lmmse_equalize(Y: np.ndarray, H: np.ndarray, n0: float) -> np.ndarray:
    """Per-subcarrier LMMSE equaliser: X̂ = conj(H)·Y / (|H|² + N0)."""
    return np.conj(H) * Y / (np.abs(H) ** 2 + n0)


def zf_equalize(Y: np.ndarray, H: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Zero-forcing equaliser: X̂ = Y / H."""
    return Y / (H + eps)


def classical_equalize_demap(features: np.ndarray, M: int, n0: float) -> np.ndarray:
    """LMMSE-equalise the received features, then nearest-constellation demap."""
    Y = features[:, 0] + 1j * features[:, 1]
    H = features[:, 2] + 1j * features[:, 3]
    Xhat = lmmse_equalize(Y, H, n0)
    pts = qam_constellation(M)
    d = np.abs(Xhat[:, None] - pts[None, :])
    return np.argmin(d, axis=1)


__all__ = [
    "rayleigh_taps",
    "freq_response",
    "fading_dataset",
    "lmmse_equalize",
    "zf_equalize",
    "classical_equalize_demap",
]
