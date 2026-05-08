"""Frequency-Domain Spectrum Shaping (FDSS) for low-PAPR uplink.

FDSS is the DFT-s-OFDM enhancement standardised in 3GPP Rel-17 for
coverage-limited uplinks: the M-point DFT output is *extended* to a
larger Q-point spectrum (Q ≥ M, typically Q = 1.5 M or 2 M) and a
shaping filter is applied before the N-point IDFT. The shaping filter
trades a small SINR penalty for 1.5–3 dB PAPR reduction, which directly
translates to higher transmit power (the UE PA is back-off limited).

Pipeline (TX):

    bits → BPSK/QPSK mapping → M-point DFT (precoder)
         → spectrum extension M → Q (zero-pad or repeat)
         → frequency-domain filter F[k]
         → N-point IFFT
         → CP prepend

Pipeline (RX) is the matched inverse: CP strip → N-FFT → matched
filter F*[k] → spectrum de-extension → M-IFFT → demap.

Reference:
    3GPP TS 38.211 §6.3.1 (DFT-s-OFDM precoding) + Rel-17 spectrum
    shaping for FDSS. Rohde & Schwarz, "Frequency Domain Spectrum
    Shaping for 5G NR Uplink Coverage Enhancement," 2022 white paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FDSSConfig:
    """Configuration for one FDSS block.

    Attributes
    ----------
    m_subcarriers:
        Number of data subcarriers (DFT precoder size, *M*).
    q_extension:
        Extended spectrum length in subcarriers (*Q* ≥ *M*).
    filter_coeffs:
        Q-length complex filter `F[k]` applied after extension. The
        canonical raised-cosine choice is provided by
        `raised_cosine_filter()`.
    n_fft:
        Outer IFFT size (must be ≥ q_extension).
    cp_length:
        Cyclic-prefix length in samples.
    """

    m_subcarriers: int
    q_extension: int
    filter_coeffs: np.ndarray
    n_fft: int = 256
    cp_length: int = 16

    def __post_init__(self) -> None:
        if self.m_subcarriers <= 0:
            raise ValueError("m_subcarriers must be positive")
        if self.q_extension < self.m_subcarriers:
            raise ValueError("q_extension must be ≥ m_subcarriers")
        if self.n_fft < self.q_extension:
            raise ValueError("n_fft must be ≥ q_extension")
        if self.cp_length < 0:
            raise ValueError("cp_length must be ≥ 0")
        if self.filter_coeffs.shape != (self.q_extension,):
            raise ValueError(
                f"filter_coeffs must be length {self.q_extension}, "
                f"got {self.filter_coeffs.shape}"
            )


def raised_cosine_filter(q: int, rolloff: float = 0.25) -> np.ndarray:
    """Q-length raised-cosine spectral mask, real, magnitude in [0, 1].

    A flat passband occupies the centre (1 − rolloff) of the band; the
    edges roll off as cosine. This is the canonical FDSS filter choice
    in the R&S white paper and gives ~2 dB PAPR reduction at α = 0.25.
    """
    if q <= 0:
        raise ValueError("q must be positive")
    if not 0.0 <= rolloff <= 1.0:
        raise ValueError("rolloff must be in [0, 1]")
    k = np.arange(q)
    # Map subcarrier index to a normalized frequency in [-1, 1].
    f = 2.0 * (k - q / 2.0) / q
    abs_f = np.abs(f)
    flat = 1.0 - rolloff
    mask = np.zeros(q)
    mask[abs_f <= flat] = 1.0
    edge = (abs_f > flat) & (abs_f <= 1.0)
    mask[edge] = 0.5 * (
        1.0 + np.cos(np.pi * (abs_f[edge] - flat) / max(rolloff, 1e-12))
    )
    return mask


def _extend_spectrum(spec: np.ndarray, q: int) -> np.ndarray:
    """Cyclic spectrum extension M → Q (zero-pad symmetric)."""
    m = spec.shape[0]
    if q == m:
        return spec.copy()
    out = np.zeros(q, dtype=complex)
    half_low = m // 2
    half_high = m - half_low
    # Place low-frequency half at the start, high-frequency half at the end.
    out[:half_low] = spec[:half_low]
    out[q - half_high:] = spec[half_low:]
    return out


def _deextend_spectrum(spec: np.ndarray, m: int) -> np.ndarray:
    """Inverse of `_extend_spectrum`."""
    q = spec.shape[0]
    if q == m:
        return spec.copy()
    half_low = m // 2
    half_high = m - half_low
    out = np.zeros(m, dtype=complex)
    out[:half_low] = spec[:half_low]
    out[half_low:] = spec[q - half_high:]
    return out


def _bits_to_qpsk(bits: np.ndarray) -> np.ndarray:
    """Gray-coded QPSK mapping; bits.size must be even."""
    bits = np.asarray(bits).astype(int)
    if bits.size % 2 != 0:
        bits = np.concatenate([bits, [0]])
    pairs = bits.reshape(-1, 2)
    inphase = 1.0 - 2.0 * pairs[:, 0]
    quad = 1.0 - 2.0 * pairs[:, 1]
    return (inphase + 1j * quad) / np.sqrt(2.0)


def fdss_tx(bits: np.ndarray, cfg: FDSSConfig) -> np.ndarray:
    """FDSS transmitter — bits → time-domain samples with CP.

    Steps (TS 38.211 §6.3.1.1 + Rel-17 shaping):
        1. QPSK map bits to M complex symbols.
        2. M-point DFT (transform precoder, eq. 6.3.1.1-1).
        3. Symmetric spectrum extension M → Q.
        4. Apply FDSS filter `F[k]` element-wise.
        5. Place into N_FFT bins (DC + lower edges first).
        6. N-point IFFT, normalised by √N.
        7. Prepend cyclic prefix of length `cp_length`.
    """
    symbols = _bits_to_qpsk(bits)
    if symbols.size < cfg.m_subcarriers:
        # zero-pad data symbols to fill the DFT.
        pad = np.zeros(cfg.m_subcarriers - symbols.size, dtype=complex)
        symbols = np.concatenate([symbols, pad])
    else:
        symbols = symbols[: cfg.m_subcarriers]

    # 2) DFT precoder
    dft = np.fft.fft(symbols, norm="ortho")
    # 3) extend
    extended = _extend_spectrum(dft, cfg.q_extension)
    # 4) shape
    shaped = extended * cfg.filter_coeffs
    # 5) place into N_FFT bins
    spec = np.zeros(cfg.n_fft, dtype=complex)
    half_low = cfg.q_extension // 2
    half_high = cfg.q_extension - half_low
    spec[:half_low] = shaped[:half_low]
    spec[cfg.n_fft - half_high:] = shaped[half_low:]
    # 6) IFFT
    time_samples = np.fft.ifft(spec, norm="ortho")
    # 7) CP
    cp = time_samples[-cfg.cp_length:] if cfg.cp_length else np.zeros(0, dtype=complex)
    return np.concatenate([cp, time_samples])


def fdss_rx(samples: np.ndarray, cfg: FDSSConfig) -> np.ndarray:
    """FDSS receiver — inverse of `fdss_tx`, returning M soft symbols.

    Matched filter `F*[k]/(|F[k]|² + ε)` is applied after spectrum
    extraction (zero-forcing equaliser; no channel estimation here, the
    physics module exposes the raw chain only).
    """
    if samples.size < cfg.cp_length + cfg.n_fft:
        raise ValueError(
            f"samples too short: need {cfg.cp_length + cfg.n_fft}, got {samples.size}"
        )
    # Strip CP
    body = samples[cfg.cp_length: cfg.cp_length + cfg.n_fft]
    spec = np.fft.fft(body, norm="ortho")
    # Re-extract Q bins
    half_low = cfg.q_extension // 2
    half_high = cfg.q_extension - half_low
    extracted = np.zeros(cfg.q_extension, dtype=complex)
    extracted[:half_low] = spec[:half_low]
    extracted[half_low:] = spec[cfg.n_fft - half_high:]
    # Matched filter (ZF-style)
    eps = 1e-12
    matched = extracted * np.conj(cfg.filter_coeffs) / (
        np.abs(cfg.filter_coeffs) ** 2 + eps
    )
    # De-extend
    de_ext = _deextend_spectrum(matched, cfg.m_subcarriers)
    # Inverse precoder
    return np.fft.ifft(de_ext, norm="ortho")


def papr_db(signal: np.ndarray) -> float:
    """Peak-to-average power ratio of a complex baseband signal in dB.

    PAPR = 10·log10( max|x|² / mean|x|² ).  Returns 0 dB for an all-zero
    signal (avoids divide-by-zero in tests).
    """
    x = np.asarray(signal)
    if x.size == 0:
        return 0.0
    power = np.abs(x) ** 2
    mean_p = np.mean(power)
    if mean_p <= 0:
        return 0.0
    peak = np.max(power)
    return float(10.0 * np.log10(peak / mean_p))


__all__ = [
    "FDSSConfig",
    "fdss_rx",
    "fdss_tx",
    "papr_db",
    "raised_cosine_filter",
]
