"""Physical-layer jamming attacks + imperfect-CSI stress for the fading receivers.

A battery of REAL (torch-free numpy) PHY-layer adversaries that perturb the
*received* signal ``Y`` of an OFDM fading link, plus an imperfect channel-state-
information (CSI) stressor that is realism rather than an attack. These are the
classic electronic-warfare jammer archetypes (see e.g. R. Poisel, *Modern
Communications Jamming Principles and Techniques*, 2nd ed.; and the O-RAN WG11 /
3GPP threat taxonomy for RAN denial-of-service):

1. **Barrage / broadband AWGN jammer** — the jammer floods the whole band with
   wideband Gaussian noise, raising the effective noise floor. Parametrised by
   the **jammer-to-signal ratio (JSR)** in dB.
2. **Partial-band noise jammer** — concentrates its power into a *fraction* of
   the subcarriers/symbols, raising the per-bin JSR on the jammed fraction
   (classic against frequency-hopping / OFDM systems).
3. **Single-tone / narrowband (CW) jammer** — a continuous-wave complex tone is
   added to ``Y`` (a fixed bias / spur), the cheapest jammer to deploy.
4. **Pulsed (duty-cycled) jammer** — jams a duty-cycled subset of symbols at high
   instantaneous power, defeating average-power-based AGC/coding.
5. **Imperfect CSI** — the receiver's channel estimate is ``Ĥ = H + e`` with
   ``e ~ CN(0, σ²_est)``. The *true* channel ``H`` still shapes ``Y = H·X + N``,
   but BOTH receivers demap with the WRONG CSI ``Ĥ`` (neural features carry Ĥ;
   the LMMSE equaliser is fed Ĥ). This is estimation-error realism, not an
   attack — it is what every real receiver lives with.

The jammers act on the attacker's contract: they touch ``Y`` only (the received
signal), never the channel ``H`` itself — consistent with ``pgd.py`` and the
``[1,1,0,0]`` perturbation mask used elsewhere in this package.

Conventions
-----------
* A "feature matrix" is the ``(N, 4)`` array ``[Re Y, Im Y, Re H, Im H]`` produced
  by :func:`horizon_ric.phy.channel.fading_dataset` — the exact layout the neural
  receiver and the LMMSE demapper consume.
* JSR (jammer-to-signal ratio) is in dB. With unit-average-power symbols
  (``Es = 1``) and unit-average-power channel, the average received signal power
  is ``E[|H·X|²] = 1``, so a jammer power of ``10**(JSR/10)`` realises that JSR.
"""

from __future__ import annotations

import numpy as np

from horizon_ric.phy.channel import freq_response, rayleigh_taps
from horizon_ric.phy.constellation import modulate

# Standard fading feature layout: [Re Y, Im Y, Re H, Im H].
_Y_COLS = (0, 1)
_H_COLS = (2, 3)


# ── helpers to move between the complex domain and the feature layout ────────
def features_to_YH(features: np.ndarray):
    """Split a ``(N, 4)`` feature matrix into complex ``(Y, H)`` arrays."""
    f = np.asarray(features, dtype=np.float64)
    Y = f[:, 0] + 1j * f[:, 1]
    H = f[:, 2] + 1j * f[:, 3]
    return Y, H


def YH_to_features(Y: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Reassemble a ``(N, 4)`` feature matrix from complex ``(Y, H)``."""
    return np.stack([Y.real, Y.imag, H.real, H.imag], axis=1).astype(np.float64)


def _signal_power_from_features(features: np.ndarray) -> float:
    """Average received signal power E[|Y|²] estimated from the feature matrix.

    Used to scale jammer power to a requested JSR. With unit-power symbols and a
    unit-power channel this is ≈ 1 + N0, dominated by the signal at useful SNR.
    """
    Y, _ = features_to_YH(features)
    return float(np.mean(np.abs(Y) ** 2))


# ── 1. Barrage / broadband AWGN jammer ───────────────────────────────────────
def barrage_jammer(
    features: np.ndarray, jsr_dB: float, rng: np.random.Generator
) -> np.ndarray:
    """Broadband AWGN barrage jammer: raise the effective noise floor on ALL bins.

    Adds complex white Gaussian noise of power ``P_j = signal_power · 10**(JSR/10)``
    to every received symbol ``Y``. This is the classic full-band noise jammer; it
    degrades every subcarrier equally and is the worst case for a router-style
    defence because it hurts both receivers identically.

    Args:
        features: ``(N, 4)`` ``[Re Y, Im Y, Re H, Im H]``.
        jsr_dB: jammer-to-signal ratio in dB.
        rng: numpy Generator.
    Returns:
        new feature matrix with the jammed ``Y`` (``H`` columns untouched).
    """
    Y, H = features_to_YH(features)
    p_sig = _signal_power_from_features(features)
    p_j = p_sig * (10.0 ** (jsr_dB / 10.0))
    sigma = np.sqrt(p_j / 2.0)  # per real dimension
    jam = sigma * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
    return YH_to_features(Y + jam, H)


# ── 2. Partial-band noise jammer ─────────────────────────────────────────────
def partial_band_jammer(
    features: np.ndarray,
    jsr_dB: float,
    fraction: float,
    rng: np.random.Generator,
    return_mask: bool = False,
):
    """Partial-band noise jammer: jam only a ``fraction`` of the symbols/subcarriers.

    The jammer concentrates its noise on a random ``fraction`` ∈ (0, 1] of the
    received symbols. The ``jsr_dB`` here is the *per-jammed-bin* JSR (the local
    JSR experienced inside the jammed band), which is the standard way the
    partial-band jammer is parametrised: for a fixed total jammer power the
    narrower the band, the higher the in-band JSR.

    Args:
        features: ``(N, 4)`` feature matrix.
        jsr_dB: per-jammed-bin jammer-to-signal ratio (dB).
        fraction: fraction of bins to jam, in (0, 1].
        rng: numpy Generator.
        return_mask: also return the boolean jamming mask.
    Returns:
        jammed feature matrix, or ``(features, mask)`` if ``return_mask``.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    Y, H = features_to_YH(features)
    n = Y.shape[0]
    k = int(round(fraction * n))
    k = min(max(k, 0), n)
    mask = np.zeros(n, dtype=bool)
    if k > 0:
        idx = rng.choice(n, size=k, replace=False)
        mask[idx] = True
    p_sig = _signal_power_from_features(features)
    p_j = p_sig * (10.0 ** (jsr_dB / 10.0))
    sigma = np.sqrt(p_j / 2.0)
    jam = sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    Yj = Y.copy()
    Yj[mask] = Yj[mask] + jam[mask]
    out = YH_to_features(Yj, H)
    if return_mask:
        return out, mask
    return out


# ── 3. Single-tone / narrowband (CW) jammer ──────────────────────────────────
def single_tone_jammer(
    features: np.ndarray,
    jsr_dB: float,
    phase: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Single-tone / continuous-wave (CW) jammer: add a constant complex tone to Y.

    Models a narrowband CW jammer whose carrier lands in-band: a deterministic
    complex offset ``A·e^{jφ}`` is added to every received symbol, where the tone
    amplitude ``A`` realises the requested JSR (``A² = signal_power·10**(JSR/10)``).
    Unlike noise jammers this is a coherent bias, so the classical demapper's
    nearest-point decision shifts in a fixed direction.

    Args:
        features: ``(N, 4)`` feature matrix.
        jsr_dB: jammer-to-signal ratio in dB (tone power / signal power).
        phase: tone phase in radians; random if ``None`` (needs ``rng``).
        rng: used only when ``phase`` is None.
    Returns:
        jammed feature matrix.
    """
    Y, H = features_to_YH(features)
    p_sig = _signal_power_from_features(features)
    amp = np.sqrt(p_sig * (10.0 ** (jsr_dB / 10.0)))
    if phase is None:
        phase = float((rng or np.random.default_rng()).uniform(0.0, 2.0 * np.pi))
    tone = amp * np.exp(1j * phase)
    return YH_to_features(Y + tone, H)


# ── 4. Pulsed (duty-cycled) jammer ───────────────────────────────────────────
def pulsed_jammer(
    features: np.ndarray,
    jsr_dB: float,
    duty: float,
    rng: np.random.Generator,
    return_mask: bool = False,
):
    """Pulsed (duty-cycled) jammer: high-power noise on a duty-cycled symbol subset.

    The jammer is ON for a contiguous ``duty`` ∈ (0, 1] burst of symbols and OFF
    otherwise, delivering high *instantaneous* power on the ON symbols. ``jsr_dB``
    is the in-pulse (ON) JSR. A contiguous burst is used (rather than random bins)
    to model a real on/off radar-like pulse; the start offset is random.

    Args:
        features: ``(N, 4)`` feature matrix.
        jsr_dB: in-pulse jammer-to-signal ratio (dB).
        duty: duty cycle in (0, 1] — fraction of symbols jammed per burst.
        rng: numpy Generator.
        return_mask: also return the boolean ON mask.
    Returns:
        jammed feature matrix, or ``(features, mask)`` if ``return_mask``.
    """
    if not (0.0 < duty <= 1.0):
        raise ValueError(f"duty must be in (0, 1], got {duty}")
    Y, H = features_to_YH(features)
    n = Y.shape[0]
    k = int(round(duty * n))
    k = min(max(k, 0), n)
    mask = np.zeros(n, dtype=bool)
    if k > 0:
        start = int(rng.integers(0, n)) if n > 0 else 0
        idx = (start + np.arange(k)) % n  # contiguous (wrapping) ON burst
        mask[idx] = True
    p_sig = _signal_power_from_features(features)
    p_j = p_sig * (10.0 ** (jsr_dB / 10.0))
    sigma = np.sqrt(p_j / 2.0)
    jam = sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    Yj = Y.copy()
    Yj[mask] = Yj[mask] + jam[mask]
    out = YH_to_features(Yj, H)
    if return_mask:
        return out, mask
    return out


# ── 5. Imperfect CSI (realism stressor, not an attack) ───────────────────────
def imperfect_csi_dataset(
    n: int,
    M: int,
    snr_dB: float,
    sigma_est: float,
    rng: np.random.Generator,
    n_taps: int = 4,
    n_fft: int = 64,
):
    """Fading dataset whose features carry an *imperfect* channel estimate ``Ĥ``.

    The true channel ``H`` shapes the received signal ``Y = H·X + N`` exactly as in
    :func:`horizon_ric.phy.channel.fading_dataset`, but the CSI exposed to the
    receivers is ``Ĥ = H + e`` with ``e ~ CN(0, σ²_est)`` (per-complex-sample
    Gaussian channel-estimation error). Because the feature matrix's ``H`` columns
    hold ``Ĥ``:

    * the **neural receiver** demaps from ``[Re Y, Im Y, Re Ĥ, Im Ĥ]`` — it sees
      the wrong CSI;
    * the **LMMSE equaliser** (``classical_equalize_demap`` reads cols 2,3) also
      equalises with ``Ĥ``.

    Both receivers therefore demap against the same imperfect CSI, which is the
    honest realism condition. ``sigma_est = 0`` reproduces perfect CSI exactly.

    Args:
        n: number of symbols.
        M: constellation order (4 or 16).
        snr_dB: Es/N0 in dB (Es = 1).
        sigma_est: std of the per-dimension... see note. Total estimation-error
            variance is ``sigma_est**2`` per complex sample (split across Re/Im).
        rng: numpy Generator.
        n_taps, n_fft: channel model parameters (match ``fading_dataset``).
    Returns:
        ``(features, labels, meta)`` where ``features`` carry ``Ĥ``, and
        ``meta`` has ``N0`` plus the true ``H`` and the estimate ``H_hat`` (complex)
        for diagnostics.
    """
    labels = rng.integers(0, M, size=n)
    X = modulate(labels, M)
    taps = rayleigh_taps(n, n_taps, rng)
    H = freq_response(taps, n_fft)
    n0 = 1.0 / (10.0 ** (snr_dB / 10.0))  # Es = 1
    noise = np.sqrt(n0 / 2.0) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    Y = H * X + noise  # true channel shapes the received signal
    # Imperfect CSI: Ĥ = H + e, e ~ CN(0, sigma_est²).
    err = np.sqrt((sigma_est**2) / 2.0) * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )
    H_hat = H + err
    feats = YH_to_features(Y, H_hat)  # features carry the ESTIMATE, not the truth
    return feats, labels, {"N0": n0, "H_true": H, "H_hat": H_hat}


__all__ = [
    "features_to_YH",
    "YH_to_features",
    "barrage_jammer",
    "partial_band_jammer",
    "single_tone_jammer",
    "pulsed_jammer",
    "imperfect_csi_dataset",
]
