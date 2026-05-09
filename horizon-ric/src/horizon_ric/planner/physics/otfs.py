"""OTFS (Orthogonal Time-Frequency Space) modulation primitives.

OTFS multiplexes information symbols on the *delay-Doppler* grid rather
than the time-frequency grid used by OFDM. The transmitter applies an
inverse symplectic finite Fourier transform (ISFFT) to lift the
delay-Doppler symbols into the time-frequency plane, then a Heisenberg
transform yields the time-domain waveform. The receiver inverts the
chain via Wigner + SFFT. Doubly-selective channels appear *quasi-static*
in delay-Doppler, which is why OTFS outperforms CP-OFDM at LEO Doppler.

Reference equations (Hadani et al., "Orthogonal Time Frequency Space
Modulation," IEEE WCNC-2017):

    (5)  X[n,m] = (1/√(NM)) Σ_{k=0}^{N−1} Σ_{l=0}^{M−1}
                 x[k,l] · exp( j2π(nk/N − ml/M) )       — ISFFT
    (6)  s(t) = Σ_{n=0}^{N−1} Σ_{m=0}^{M−1}
                X[n,m] · g_tx(t − nT) · exp( j2π m Δf (t − nT) )
    (7)  Y[n,m] = ⟨ y(t), g_rx(t − nT) e^{j2π m Δf (t − nT)} ⟩  — Wigner
    (8)  y[k,l] = (1/√(NM)) Σ_{n,m} Y[n,m] · exp( −j2π(nk/N − ml/M) )

    SFFT is the inverse map of ISFFT and *unitary*; eqns (5) and (8)
    therefore satisfy isfft∘sfft = sfft∘isfft = I (numerical
    round-off only).

This module provides only the discrete signal-processing kernels —
ISFFT/SFFT/channel application — needed by the planner for capacity
estimation and Doppler-robustness checks. The Heisenberg/Wigner step
collapses to identity when g_tx = g_rx = rect (the "ideal pulse"
assumption) which is what we use here.

Reference:
    R. Hadani et al., "Orthogonal Time Frequency Space Modulation,"
    IEEE WCNC, 2017.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OTFSGrid:
    """Discrete delay-Doppler grid parameters.

    The grid has `n_doppler` Doppler bins and `n_delay` delay bins; one
    OTFS frame therefore carries `n_doppler · n_delay` data symbols.

    Attributes
    ----------
    n_doppler:
        Number of Doppler bins (called *N* in eq. 5).
    n_delay:
        Number of delay bins (called *M* in eq. 5).
    sample_rate_hz:
        Sample rate of the underlying time-domain waveform; used to
        translate delay (in seconds) into integer taps.
    """

    n_doppler: int
    n_delay: int
    sample_rate_hz: float

    def __post_init__(self) -> None:
        if self.n_doppler <= 0 or self.n_delay <= 0:
            raise ValueError("n_doppler and n_delay must be positive")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

    @property
    def doppler_resolution_hz(self) -> float:
        """1 / (N · T)   with T = M / fs (eq. 6)."""
        frame_duration_s = self.n_delay / self.sample_rate_hz
        return 1.0 / (self.n_doppler * frame_duration_s)

    @property
    def delay_resolution_s(self) -> float:
        """1 / (M · Δf)   with Δf = fs / M ⇒ 1/fs."""
        return 1.0 / self.sample_rate_hz


@dataclass(frozen=True)
class OTFSChannelImpulse:
    """Sparse delay-Doppler channel description.

    `taps` is a list of (doppler_hz, delay_s, complex_gain) triples.
    Real LEO channels at Ka have ≤ 4 dominant taps; sparsity is the
    very property that motivates OTFS.
    """

    taps: list[tuple[float, float, complex]]

    def __post_init__(self) -> None:
        if not self.taps:
            raise ValueError("OTFSChannelImpulse requires at least one tap")
        for nu, tau, _ in self.taps:
            if tau < 0:
                raise ValueError(f"delay must be ≥ 0, got {tau}")
            if not np.isfinite(nu) or not np.isfinite(tau):
                raise ValueError("doppler/delay must be finite")


def otfs_isfft(x_doppler_delay: np.ndarray) -> np.ndarray:
    """Inverse Symplectic FFT precoder (eq. 5).

    Lifts a delay-Doppler symbol matrix `x[k, l]` of shape (N, M) into
    the time-frequency plane `X[n, m]` of the same shape. The transform
    is unitary:  sfft(isfft(x)) = x  to numerical precision.

    Implemented via two orthonormal 1-D FFTs:
        - inverse FFT along the Doppler (k → n) axis
        - forward FFT along the delay   (l → m) axis
    with `norm="ortho"` so both are unitary.
    """
    x = np.asarray(x_doppler_delay)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {x.shape}")
    # Doppler axis: inverse DFT (k → n). Delay axis: forward DFT (l → m).
    step1 = np.fft.ifft(x, axis=0, norm="ortho")
    step2 = np.fft.fft(step1, axis=1, norm="ortho")
    return step2


def otfs_sfft(y_time_freq: np.ndarray) -> np.ndarray:
    """Symplectic FFT decoder (eq. 8) — inverse of `otfs_isfft`.

    Forward FFT along Doppler axis, inverse FFT along delay axis, both
    with orthonormal scaling. Composing it with `otfs_isfft` gives the
    identity to within machine epsilon.
    """
    y = np.asarray(y_time_freq)
    if y.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {y.shape}")
    step1 = np.fft.fft(y, axis=0, norm="ortho")
    step2 = np.fft.ifft(step1, axis=1, norm="ortho")
    return step2


def otfs_channel_apply(
    x: np.ndarray,
    h: OTFSChannelImpulse,
    grid: OTFSGrid,
) -> np.ndarray:
    """Apply a doubly-selective channel in the delay-Doppler domain.

    Under the ideal-pulse assumption, the input-output relation in
    delay-Doppler is the *twisted convolution* (Hadani §III-C):

        y[k, l] = Σ_p  α_p · exp( j2π (k_p · l) / (N · M) )
                       · x[ (k − k_p) mod N, (l − l_p) mod M ]

    where (k_p, l_p) are the integer Doppler/delay indices of the p-th
    tap and α_p its complex gain. The exp() phase term is the
    discrete-grid signature of fractional-delay Doppler coupling.

    The transmitter must call `otfs_isfft` *before* this function and
    `otfs_sfft` after; for the planner we operate directly in the
    delay-Doppler grid which keeps the kernel compact and easy to
    reason about for capacity bounds.
    """
    x = np.asarray(x, dtype=complex)
    if x.ndim != 2 or x.shape != (grid.n_doppler, grid.n_delay):
        raise ValueError(
            f"x shape {x.shape} mismatch grid ({grid.n_doppler}, {grid.n_delay})"
        )

    n, m = grid.n_doppler, grid.n_delay
    delay_res = grid.delay_resolution_s
    doppler_res = grid.doppler_resolution_hz

    y = np.zeros_like(x)
    k_idx = np.arange(n).reshape(n, 1)
    l_idx = np.arange(m).reshape(1, m)

    for nu_hz, tau_s, alpha in h.taps:
        k_p = int(round(nu_hz / doppler_res)) % n
        l_p = int(round(tau_s / delay_res)) % m
        shifted = np.roll(np.roll(x, shift=k_p, axis=0), shift=l_p, axis=1)
        # Cross-phase from fractional-delay/Doppler coupling.
        phase = np.exp(2j * np.pi * (k_p * l_idx) / (n * m))
        y = y + alpha * phase * shifted
        # k_idx is referenced through the broadcast above (silence linters).
        _ = k_idx

    return y


__all__ = [
    "OTFSChannelImpulse",
    "OTFSGrid",
    "otfs_channel_apply",
    "otfs_isfft",
    "otfs_sfft",
]
