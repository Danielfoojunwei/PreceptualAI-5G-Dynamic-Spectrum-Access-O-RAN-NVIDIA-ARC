"""Vendored ITU-R reference tables.

These are the canonical tabulated values published by the ITU-R; they are
not curve-fit approximations. Functions in `propagation.py` use these
tables with log-log interpolation per the relevant Recommendation.

References:
    ITU-R P.838-3 (03/2005) Table 1 — k, α coefficients for specific
        rain attenuation, horizontal & vertical polarizations.
    ITU-R P.676-13 (08/2022) Annex 2 simplified γ_o, γ_w reference values
        at 1013.25 hPa, 15°C, ρ_w = 7.5 g/m³ (sampled from the simplified
        Eqs (1)–(2) in Annex 2; used for fast slant-path estimates).
"""

from __future__ import annotations

from bisect import bisect_left
from math import log10

# ─── ITU-R P.838-3 Table 1: rain k/α coefficients ────────────────────────
# Frequencies in GHz. Values copied from P.838-3 Table 1 (all 32 entries
# as published in the Recommendation, March 2005).
P838_FREQUENCIES_GHZ: tuple[float, ...] = (
    1.0, 2.0, 4.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0,
    25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0,
    100.0, 120.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0,
    900.0, 1000.0,
)

# k_H values (horizontal polarization).
P838_K_HORIZONTAL: tuple[float, ...] = (
    0.0000259, 0.0000847, 0.0001071, 0.0007056, 0.001915, 0.004115,
    0.01217, 0.02386, 0.04481, 0.09164, 0.1571, 0.2403, 0.3374, 0.4431,
    0.5521, 0.6600, 0.8606, 1.0315, 1.1704, 1.2807, 1.3671, 1.4866,
    1.5823, 1.6378, 1.6286, 1.5860, 1.5418, 1.5013, 1.4654, 1.4335,
    1.4050, 1.3795,
)

# α_H exponents (horizontal polarization).
P838_ALPHA_HORIZONTAL: tuple[float, ...] = (
    0.9691, 1.0664, 1.6009, 1.5900, 1.4810, 1.3905, 1.2571, 1.1825,
    1.1233, 1.0568, 0.9991, 0.9485, 0.9047, 0.8673, 0.8355, 0.8084,
    0.7656, 0.7345, 0.7115, 0.6944, 0.6815, 0.6640, 0.6494, 0.6382,
    0.6296, 0.6262, 0.6253, 0.6262, 0.6276, 0.6289, 0.6299, 0.6306,
)

# k_V values (vertical polarization).
P838_K_VERTICAL: tuple[float, ...] = (
    0.0000308, 0.0000998, 0.0002461, 0.0004878, 0.001425, 0.003450,
    0.01129, 0.02455, 0.05008, 0.09611, 0.1533, 0.2291, 0.3224, 0.4274,
    0.5375, 0.6472, 0.8515, 1.0253, 1.1668, 1.2795, 1.3680, 1.4911,
    1.5896, 1.6443, 1.6286, 1.5820, 1.5366, 1.4967, 1.4622, 1.4321,
    1.4056, 1.3822,
)

# α_V exponents (vertical polarization).
P838_ALPHA_VERTICAL: tuple[float, ...] = (
    0.8592, 0.9490, 1.2476, 1.5728, 1.4745, 1.3797, 1.2156, 1.1216,
    1.0440, 0.9847, 0.9491, 0.9129, 0.8761, 0.8421, 0.8123, 0.7871,
    0.7486, 0.7215, 0.7021, 0.6876, 0.6765, 0.6609, 0.6466, 0.6343,
    0.6262, 0.6256, 0.6272, 0.6293, 0.6315, 0.6334, 0.6348, 0.6358,
)


def _log_log_interp(
    x: float,
    xs: tuple[float, ...],
    ys: tuple[float, ...],
) -> float:
    """Log-log linear interpolation per ITU-R P.838 §2 footnote.

    Both x and y are positive. For x outside [xs[0], xs[-1]] we clamp to
    the nearest endpoint (no extrapolation).
    """
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect_left(xs, x)
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    lx0, lx1, lx = log10(x0), log10(x1), log10(x)
    ly0, ly1 = log10(y0), log10(y1)
    ly = ly0 + (ly1 - ly0) * (lx - lx0) / (lx1 - lx0)
    return 10.0 ** ly


def _linear_interp(
    x: float,
    xs: tuple[float, ...],
    ys: tuple[float, ...],
) -> float:
    """Linear interpolation in x and y. Used for α (the exponent itself)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect_left(xs, x)
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def p838_k_alpha(frequency_ghz: float, polarization: str) -> tuple[float, float]:
    """Return (k, α) per ITU-R P.838-3 §2 with proper interpolation.

    polarization ∈ {"horizontal", "vertical", "circular"}. For circular
    polarization with elevation θ and tilt τ=45°, P.838-3 Eqs (4)–(5)
    reduce to:
        k_c = (k_H + k_V) / 2
        α_c = (k_H α_H + k_V α_V) / (2 k_c)
    independent of θ (since cos(2·45°) = 0).
    """
    if frequency_ghz <= 0:
        raise ValueError(f"frequency_ghz must be > 0, got {frequency_ghz}")

    k_h = _log_log_interp(frequency_ghz, P838_FREQUENCIES_GHZ, P838_K_HORIZONTAL)
    a_h = _linear_interp(frequency_ghz, P838_FREQUENCIES_GHZ, P838_ALPHA_HORIZONTAL)
    k_v = _log_log_interp(frequency_ghz, P838_FREQUENCIES_GHZ, P838_K_VERTICAL)
    a_v = _linear_interp(frequency_ghz, P838_FREQUENCIES_GHZ, P838_ALPHA_VERTICAL)

    if polarization == "horizontal":
        return k_h, a_h
    if polarization == "vertical":
        return k_v, a_v
    if polarization == "circular":
        k_c = 0.5 * (k_h + k_v)
        a_c = (k_h * a_h + k_v * a_v) / (2.0 * k_c)
        return k_c, a_c
    raise ValueError(
        f"polarization must be 'horizontal', 'vertical', or 'circular'; got {polarization!r}"
    )


# ─── ITU-R P.676-13 Annex 2: γ_o, γ_w sampled reference values ───────────
# Sampled at 1013.25 hPa, 15°C, ρ_w = 7.5 g/m³. These were generated from
# the simplified equations (1)–(2) of P.676-13 Annex 2 at the listed
# frequencies; we interpolate (linear in dB) between samples. This is the
# reference "fast" model used by most NTN link-budget tools when the full
# line-by-line is not required. Documented accuracy of the Annex 2
# simplification is ±10% vs. Annex 1 line-by-line below 350 GHz.
P676_FREQUENCIES_GHZ: tuple[float, ...] = (
    1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0,
    22.235, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 53.0, 55.0, 57.0,
    60.0, 63.0, 66.0, 70.0, 80.0, 90.0, 100.0, 120.0, 150.0, 183.31,
    200.0, 250.0, 325.15, 350.0,
)

# γ_o(f) — dry-air specific attenuation, dB/km at sea level.
P676_GAMMA_O_DB_PER_KM: tuple[float, ...] = (
    0.0067, 0.0067, 0.0069, 0.0072, 0.0078, 0.0089, 0.0103, 0.0142, 0.0204, 0.0258,
    0.0314, 0.0421, 0.0717, 0.137, 0.300, 0.811, 2.20, 5.05, 8.50, 12.7,
    14.3, 11.5, 6.86, 1.91, 0.487, 0.349, 0.319, 0.314, 0.349, 0.395,
    0.419, 0.485, 0.582, 0.612,
)

# γ_w(f, ρ=7.5 g/m³) — water-vapour specific attenuation, dB/km.
P676_GAMMA_W_DB_PER_KM_RHO75: tuple[float, ...] = (
    0.000050, 0.00021, 0.00088, 0.00204, 0.00378, 0.00642, 0.00988, 0.01724, 0.02839, 0.03847,
    0.0651, 0.04413, 0.03026, 0.03128, 0.04004, 0.05405, 0.07393, 0.09232, 0.106, 0.122,
    0.142, 0.166, 0.196, 0.252, 0.435, 0.745, 1.151, 2.066, 3.497, 17.40,
    7.28, 7.35, 84.3, 24.2,
)


def p676_gamma_o_db_per_km(frequency_ghz: float) -> float:
    """Dry-air specific attenuation γ_o (dB/km) at sea level, 15°C, 1013.25 hPa.

    Returns 0 outside the tabulated range [1, 350] GHz; callers should
    treat this as 'unsupported' rather than 'no atmosphere'.
    """
    if frequency_ghz < P676_FREQUENCIES_GHZ[0] or frequency_ghz > P676_FREQUENCIES_GHZ[-1]:
        raise ValueError(
            f"frequency_ghz {frequency_ghz} outside tabulated range "
            f"[{P676_FREQUENCIES_GHZ[0]}, {P676_FREQUENCIES_GHZ[-1]}]"
        )
    return _linear_interp(frequency_ghz, P676_FREQUENCIES_GHZ, P676_GAMMA_O_DB_PER_KM)


def p676_gamma_w_db_per_km(
    frequency_ghz: float, rho_g_per_m3: float = 7.5
) -> float:
    """Water-vapour specific attenuation γ_w (dB/km) for the given ρ.

    Linear in ρ at the simplified Annex 2 level. Tabulated at ρ=7.5; we
    rescale linearly per Annex 2 (γ_w ∝ ρ to first order in the wings,
    excellent for the 22 / 183 / 325 GHz lines).
    """
    if frequency_ghz < P676_FREQUENCIES_GHZ[0] or frequency_ghz > P676_FREQUENCIES_GHZ[-1]:
        raise ValueError(
            f"frequency_ghz {frequency_ghz} outside tabulated range "
            f"[{P676_FREQUENCIES_GHZ[0]}, {P676_FREQUENCIES_GHZ[-1]}]"
        )
    if rho_g_per_m3 < 0:
        raise ValueError(f"rho_g_per_m3 must be ≥ 0, got {rho_g_per_m3}")
    base = _linear_interp(
        frequency_ghz, P676_FREQUENCIES_GHZ, P676_GAMMA_W_DB_PER_KM_RHO75
    )
    return base * (rho_g_per_m3 / 7.5)


# ─── ITU-R P.676-13 Annex 2 effective heights ────────────────────────────
# Eqs (25a)–(25b), simplified for the slant-path height integration.

def p676_effective_height_dry_km(pressure_hpa: float = 1013.25) -> float:
    """h_o ≈ 6 km · (p / 1013.25)^0.46. Annex 2 Eq (25a) simplified."""
    return 6.0 * (pressure_hpa / 1013.25) ** 0.46


def p676_effective_height_wet_km(
    frequency_ghz: float, rho_g_per_m3: float = 7.5
) -> float:
    """h_w ≈ 1.66 · (1 + 1.39 · σ_w(f)) km. Annex 2 Eq (25b) simplified.

    σ_w increases near the 22/183/325 GHz H₂O resonances.
    """
    f = frequency_ghz
    sigma = (
        1.013 / (1.0 + ((f - 22.235) / 9.0) ** 2)
        + 3.6   / (1.0 + ((f - 183.31) / 9.0) ** 2)
        + 1.6   / (1.0 + ((f - 325.15) / 9.0) ** 2)
    )
    # Light scaling with ρ: drier air → marginally higher effective height.
    rho_scale = 1.0 + 0.1 * (7.5 - rho_g_per_m3) / 7.5
    return 1.66 * (1.0 + 1.39 * sigma) * rho_scale
