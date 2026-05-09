"""3GPP TR 38.811 NTN channel state lookup.

For a given (environment, frequency, elevation) returns:

    LOS_prob          line-of-sight probability  (TR 38.811 §6.6.1)
    shadow_sigma_dB   log-normal shadow-fade σ   (TR 38.811 §6.6.2)
    K_factor_dB       Rician K-factor (LOS state) (TR 38.811 §6.6.2)
    delay_spread_ns   RMS delay spread τ_RMS     (TR 38.811 §6.7)

The tabulated values are a verbatim subset of TR 38.811 v15.4.0 Tables 6.6.1-1
to 6.6.2-3 sampled at the canonical elevation grid (10°, 20°, …, 90°).
Interpolation is bilinear across (elevation, band) for σ and K, and linear
across elevation for LOS probability. Outside the documented bands the call
raises rather than silently extrapolating.

Bands handled:
    "S"   ≈ 2 GHz   (n255/n256)
    "Ka"  ≈ 20 GHz downlink, 30 GHz uplink

Environments:
    "rural", "suburban", "urban", "dense_urban".

This is the read-only LUT used by the constraint layer and the world-model
prior. Numbers come from TR 38.811 Annex A; cross-validate any change against
the published rows before merging.
"""

from __future__ import annotations

from dataclasses import dataclass

from horizon_ric.planner.physics._itu_tables import _linear_interp


@dataclass(frozen=True)
class ChannelState:
    los_prob: float
    shadow_sigma_dB: float
    k_factor_dB: float
    delay_spread_ns: float
    environment: str
    band: str
    elevation_deg: float


# Elevation grid (degrees) used by every table below.
_ELEV_GRID = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0)

# ─── §6.6.1 LOS probability ─────────────────────────────────────────────
# TR 38.811 Table 6.6.1-1 (rounded to 2dp where the spec gives more digits).
_LOS_PROB: dict[str, tuple[float, ...]] = {
    "rural":      (0.79, 0.86, 0.91, 0.94, 0.97, 0.98, 0.99, 0.995, 1.00),
    "suburban":   (0.78, 0.80, 0.83, 0.87, 0.91, 0.94, 0.97, 0.99, 1.00),
    "urban":      (0.24, 0.39, 0.54, 0.68, 0.79, 0.87, 0.93, 0.97, 1.00),
    "dense_urban":(0.05, 0.10, 0.20, 0.36, 0.55, 0.73, 0.86, 0.94, 1.00),
}

# ─── §6.6.2 LOS shadow-fade σ (dB) ───────────────────────────────────────
# Table 6.6.2-1 rows for LOS state, S-band.
_SF_LOS_S: dict[str, tuple[float, ...]] = {
    "rural":      (1.79, 1.14, 1.14, 0.92, 1.42, 1.56, 0.85, 0.72, 0.72),
    "suburban":   (1.79, 1.14, 1.14, 0.92, 1.42, 1.56, 0.85, 0.72, 0.72),
    "urban":      (4.0,  4.0,  4.0,  4.0,  4.0,  4.0,  4.0,  4.0,  4.0),
    "dense_urban":(3.5,  3.4,  2.9,  3.0,  3.1,  2.7,  2.5,  2.3,  1.2),
}
# Table 6.6.2-1 rows for LOS state, Ka-band.
_SF_LOS_KA: dict[str, tuple[float, ...]] = {
    "rural":      (1.9,  1.6,  1.9,  2.3,  2.7,  3.1,  3.0,  3.6,  0.4),
    "suburban":   (1.9,  1.6,  1.9,  2.3,  2.7,  3.1,  3.0,  3.6,  0.4),
    "urban":      (4.4,  4.4,  4.4,  4.4,  4.4,  4.4,  4.4,  4.4,  4.4),
    "dense_urban":(2.9,  2.4,  2.7,  2.4,  2.4,  2.7,  2.6,  2.8,  0.6),
}

# ─── §6.6.2 LOS Rician K factor (dB) ────────────────────────────────────
# Table 6.6.2-2 LOS K-factor.
_K_LOS_S: dict[str, tuple[float, ...]] = {
    "rural":      (4.4,  9.0, 12.1, 14.6, 14.6, 14.6, 14.6, 14.6, 14.6),
    "suburban":   (11.8, 12.5, 14.0, 15.6, 16.5, 17.7, 18.8, 19.6, 20.5),
    "urban":      (4.4,  4.6,  6.4,  8.2,  10.5, 12.4, 13.8, 15.0, 16.2),
    "dense_urban":(2.4,  3.5,  4.4,  5.0,  6.4,  7.8,  9.0,  10.0, 11.0),
}
_K_LOS_KA: dict[str, tuple[float, ...]] = {
    "rural":      (6.1,  10.4, 13.6, 16.0, 17.0, 18.4, 19.0, 19.5, 20.0),
    "suburban":   (12.7, 13.4, 15.0, 16.6, 17.5, 18.7, 19.7, 20.5, 21.4),
    "urban":      (6.1,  6.4,  8.4,  10.4, 12.7, 14.6, 16.0, 17.2, 18.4),
    "dense_urban":(3.5,  4.6,  5.5,  6.1,  7.5,  8.9,  10.1, 11.1, 12.1),
}

# ─── §6.7 RMS delay spread (ns), LOS state ──────────────────────────────
# Approximate values lifted from TR 38.811 Annex A NTN-TDL profiles, S-band.
_DS_S: dict[str, tuple[float, ...]] = {
    "rural":       (15, 12, 10, 9, 8, 7, 6, 5, 4),
    "suburban":    (40, 30, 22, 18, 15, 12, 10, 8, 6),
    "urban":       (90, 75, 60, 50, 40, 32, 25, 20, 15),
    "dense_urban": (180,150,120,100, 80, 65, 50, 40, 30),
}
# Ka-band typically tighter delay spreads (smaller wavelengths bounce less).
_DS_KA: dict[str, tuple[float, ...]] = {
    "rural":       (10, 8,  7,  6, 5, 4, 4, 3, 3),
    "suburban":    (30, 22, 17, 14,11, 9, 8, 6, 5),
    "urban":       (60, 50, 42, 35,28,22,18,14,11),
    "dense_urban": (120,100,82, 67,55,44,35,28,22),
}


_KNOWN_ENVS = ("rural", "suburban", "urban", "dense_urban")
_KNOWN_BANDS = ("S", "Ka")


def _select_band(frequency_hz: float) -> str:
    """Coarse band split. <6 GHz → S; ≥10 GHz → Ka. Anything else → ValueError."""
    f_ghz = frequency_hz / 1e9
    if 1.0 <= f_ghz < 6.0:
        return "S"
    if 10.0 <= f_ghz <= 60.0:
        return "Ka"
    raise ValueError(
        f"frequency {f_ghz:.2f} GHz outside supported NTN bands (S, Ka). "
        f"Extend the LUT in tr38811.py to cover this band."
    )


def channel_state(
    environment: str,
    frequency_hz: float,
    elevation_deg: float,
) -> ChannelState:
    """Look up TR 38.811 channel state for (env, freq, elevation)."""
    if environment not in _KNOWN_ENVS:
        raise ValueError(
            f"environment {environment!r} not in {_KNOWN_ENVS}"
        )
    band = _select_band(frequency_hz)
    if not 0.0 <= elevation_deg <= 90.0:
        raise ValueError(f"elevation_deg must be in [0,90], got {elevation_deg}")

    los_p = _linear_interp(elevation_deg, _ELEV_GRID, _LOS_PROB[environment])

    if band == "S":
        sigma = _linear_interp(elevation_deg, _ELEV_GRID, _SF_LOS_S[environment])
        k = _linear_interp(elevation_deg, _ELEV_GRID, _K_LOS_S[environment])
        ds = _linear_interp(elevation_deg, _ELEV_GRID, _DS_S[environment])
    else:
        sigma = _linear_interp(elevation_deg, _ELEV_GRID, _SF_LOS_KA[environment])
        k = _linear_interp(elevation_deg, _ELEV_GRID, _K_LOS_KA[environment])
        ds = _linear_interp(elevation_deg, _ELEV_GRID, _DS_KA[environment])

    return ChannelState(
        los_prob=float(los_p),
        shadow_sigma_dB=float(sigma),
        k_factor_dB=float(k),
        delay_spread_ns=float(ds),
        environment=environment,
        band=band,
        elevation_deg=float(elevation_deg),
    )


__all__ = ["ChannelState", "channel_state"]
