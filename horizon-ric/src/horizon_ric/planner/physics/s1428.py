"""ITU-R S.1428 reference earth-station antenna pattern.

Used by EPFD validation (ITU-R S.1503-3 §3.5) as the *victim* GSO ES
antenna gain mask: when an NGSO satellite is at off-axis angle φ from
the ES boresight (which points at its wanted GSO satellite), the ES
receives the NGSO at the gain G_rx(φ) given by this pattern.

S.1428-1 gives one of three patterns depending on D/λ:

    Case 1: D/λ ≥ 100   high-gain dish — Eqs (3a)-(3e)
    Case 2: 25 ≤ D/λ < 100 medium-gain dish — Eqs (4a)-(4d)
    Case 3: D/λ < 25    low-gain dish — single envelope

We implement all three, dispatched on D/λ. The functions are pure: no
state. Inputs are dish diameter (m), frequency (Hz), and off-axis angle
(deg). Output is gain (dBi).

Reference values cross-checked against:
    ITU-R BR S.1428-1 (02/2001) Annex 1.
"""

from __future__ import annotations

import math

_C_M_PER_S = 299_792_458.0


def _peak_gain_dBi(d_m: float, frequency_hz: float, eta: float = 0.65) -> float:
    """Peak boresight gain G_max for a circular aperture.

    G_max = η · (π D / λ)²    →    G_max_dB = 10 log10(η) + 20 log10(π D f / c)
    """
    if d_m <= 0 or frequency_hz <= 0:
        raise ValueError("dish diameter and frequency must be positive")
    return (
        10.0 * math.log10(eta)
        + 20.0 * math.log10(math.pi * d_m * frequency_hz / _C_M_PER_S)
    )


def s1428_gain_dBi(
    off_axis_deg: float,
    diameter_m: float,
    frequency_hz: float,
    aperture_efficiency: float = 0.65,
) -> float:
    """Earth-station receive gain at off-axis angle φ per ITU-R S.1428-1.

    Args:
        off_axis_deg: angle from the ES antenna boresight, in degrees.
            By convention φ ∈ [0, 180]; values are clamped on entry.
        diameter_m: ES dish diameter in metres.
        frequency_hz: receive frequency in Hz.
        aperture_efficiency: η in the G_max formula. Default 0.65 is the
            S.1428 reference for "well-designed" parabolic dishes.

    Returns:
        Receive gain in dBi at off-axis angle φ.
    """
    phi = abs(off_axis_deg)
    if phi > 180.0:
        phi = 360.0 - phi if phi <= 360.0 else 0.0

    wavelength_m = _C_M_PER_S / frequency_hz
    d_over_lambda = diameter_m / wavelength_m
    g_max = _peak_gain_dBi(diameter_m, frequency_hz, aperture_efficiency)

    if d_over_lambda >= 100.0:
        return _s1428_high_gain(phi, d_over_lambda, g_max)
    if d_over_lambda >= 25.0:
        return _s1428_medium_gain(phi, d_over_lambda, g_max)
    return _s1428_low_gain(phi, d_over_lambda, g_max)


def _s1428_high_gain(phi_deg: float, d_lam: float, g_max: float) -> float:
    """S.1428-1 Eqs (3a)–(3e) for D/λ ≥ 100.

    G_1 = 2 + 15 log10(D/λ)     (peak first sidelobe envelope, dBi)
    φ_m = 20 (λ/D) sqrt(G_max - G_1)   (transition angle, deg)
    φ_r = 15.85 (D/λ)^-0.6              (transition angle, deg)

    Branches:
      0      ≤ φ < φ_m :  G = G_max - 2.5e-3 (D/λ · φ)²
      φ_m    ≤ φ < φ_r :  G = G_1
      φ_r    ≤ φ < 10  :  G = 29 - 25 log10(φ)
      10     ≤ φ < 34.1:  G = 34 - 30 log10(φ)
      34.1   ≤ φ < 80  :  G = -12
      80     ≤ φ ≤ 180 :  G = -7
    """
    g1 = 2.0 + 15.0 * math.log10(d_lam)
    phi_m = (20.0 / d_lam) * math.sqrt(max(g_max - g1, 0.0))
    phi_r = 15.85 * d_lam ** -0.6

    if phi_deg < phi_m:
        return g_max - 2.5e-3 * (d_lam * phi_deg) ** 2
    if phi_deg < phi_r:
        return g1
    if phi_deg < 10.0:
        return 29.0 - 25.0 * math.log10(phi_deg)
    if phi_deg < 34.1:
        return 34.0 - 30.0 * math.log10(phi_deg)
    if phi_deg < 80.0:
        return -12.0
    if phi_deg < 120.0:
        return -7.0
    return -12.0


def _s1428_medium_gain(phi_deg: float, d_lam: float, g_max: float) -> float:
    """S.1428-1 Eqs (4a)–(4e) for 25 ≤ D/λ < 100.

    G_1 = -1 + 15 log10(D/λ)
    φ_m = 20 (λ/D) sqrt(G_max - G_1)
    φ_r = 100 · (λ/D)
    """
    g1 = -1.0 + 15.0 * math.log10(d_lam)
    phi_m = (20.0 / d_lam) * math.sqrt(max(g_max - g1, 0.0))
    phi_r = 100.0 / d_lam

    if phi_deg < phi_m:
        return g_max - 2.5e-3 * (d_lam * phi_deg) ** 2
    if phi_deg < phi_r:
        return g1
    if phi_deg < 10.0:
        return 29.0 - 25.0 * math.log10(phi_deg)
    if phi_deg < 34.1:
        return 34.0 - 30.0 * math.log10(phi_deg)
    if phi_deg < 80.0:
        return -12.0
    if phi_deg < 120.0:
        return -7.0
    return -12.0


def _s1428_low_gain(phi_deg: float, d_lam: float, g_max: float) -> float:
    """S.1428-1 envelope for D/λ < 25 (small dishes used in VSAT/UE)."""
    # First-sidelobe envelope. Coefficients for low-gain dishes use a
    # gentler roll-off per Annex 1 §3 figure.
    g1 = -2.0 + 15.0 * math.log10(d_lam)
    phi_m = (20.0 / d_lam) * math.sqrt(max(g_max - g1, 0.0))

    if phi_deg < phi_m:
        return g_max - 2.5e-3 * (d_lam * phi_deg) ** 2
    if phi_deg < 48.0:
        return max(g1, 32.0 - 25.0 * math.log10(phi_deg))
    return -10.0


__all__ = ["s1428_gain_dBi"]
