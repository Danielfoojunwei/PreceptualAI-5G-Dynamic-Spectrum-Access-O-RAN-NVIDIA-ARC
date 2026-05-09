"""Propagation physics — ITU-R P.525 / P.676 / P.838 closed-form equations.

These functions are pure: no learnable parameters, no state. They compose
into total path loss for the Compositional World Model (CWM, paradigm H2).

Implementation notes — what is rigorous vs. simplified, and where the
seams are:

  * `free_space_path_loss_dB`   exact P.525-4 form, no approximation.
  * `gas_attenuation_dB`        uses the ITU-R P.676-13 Annex 2 simplified
                                model: γ_o + γ_w sampled from the published
                                Annex 2 reference equations (`_itu_tables`),
                                with the Eq (25a)/(25b) effective heights
                                and a curved-Earth slant-path term.
                                Annex 2 documented accuracy is ±10 % vs. the
                                Annex 1 line-by-line model below 350 GHz.
  * `rain_attenuation_dB`       k, α from the canonical P.838-3 Table 1
                                (vendored verbatim, log-log interpolated).
                                Path-length reduction follows P.618-13
                                §2.2.1.2 with a P.839-4 default rain height
                                that the caller can override.

Functions out of scope (not yet implemented; do NOT call as if present):
  * P.452 NLoS terrestrial interference
  * P.840 cloud + fog attenuation
  * P.618 scintillation, depolarization
The module docstring used to claim P.452; that claim has been removed.

References:
    ITU-R P.525-4   — Free-space path loss
    ITU-R P.676-13  — Atmospheric gas attenuation (Annex 2 simplified)
    ITU-R P.838-3   — Specific rain attenuation k/α
    ITU-R P.618-13  — Slant-path total attenuation
    ITU-R P.839-4   — Rain height
"""

from __future__ import annotations

import math

from horizon_ric.planner.physics._itu_tables import (
    p676_effective_height_dry_km,
    p676_effective_height_wet_km,
    p676_gamma_o_db_per_km,
    p676_gamma_w_db_per_km,
    p838_k_alpha,
)

# Speed of light in m/s, exact per CGPM 1983 definition.
_C_M_PER_S = 299_792_458.0


def free_space_path_loss_dB(distance_m: float, frequency_hz: float) -> float:
    """ITU-R P.525-4 free-space path loss in dB.

    FSPL_dB = 20 log10(4π · d · f / c)

    Args:
        distance_m: distance in metres (>0).
        frequency_hz: carrier frequency in Hz (>0).

    Returns:
        Free-space path loss in dB.

    Raises:
        ValueError: if distance or frequency is non-positive.
    """
    if distance_m <= 0:
        raise ValueError(f"distance_m must be > 0, got {distance_m}")
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be > 0, got {frequency_hz}")

    return 20.0 * math.log10(4.0 * math.pi * distance_m * frequency_hz / _C_M_PER_S)


_EARTH_RADIUS_KM = 6371.0  # mean spherical-Earth radius (P.834 reference)


def _slant_path_factor(elevation_deg: float, effective_height_km: float) -> float:
    """Curved-Earth slant-path factor per ITU-R P.676-13 Annex 2 Eq (28).

    For high elevations (≥10°) this collapses to csc(elevation). At low
    elevations the curvature correction matters: a 5° slant is ~4–5%
    longer than 1/sin(5°) suggests.
    """
    if elevation_deg >= 10.0:
        return 1.0 / math.sin(math.radians(elevation_deg))
    # Annex 2 Eq (28) curved-Earth approximation.
    h = effective_height_km
    Re = _EARTH_RADIUS_KM
    sin_el = math.sin(math.radians(elevation_deg))
    return math.sqrt((Re / h) ** 2 * sin_el ** 2 + 2.0 * Re / h + 1.0) - (Re / h) * sin_el


def gas_attenuation_dB(
    elevation_deg: float,
    frequency_ghz: float,
    surface_water_vapor_g_per_m3: float = 7.5,
    surface_pressure_hpa: float = 1013.25,
    *,
    es_lat_deg: float | None = None,
    es_lon_deg: float | None = None,
    use_itu_maps: bool = False,
) -> float:
    """Earth-to-space gas attenuation per ITU-R P.676-13 Annex 2.

    Implementation:
        γ_o(f), γ_w(f, ρ)  from the vendored P.676-13 Annex 2 reference table
        h_o, h_w           from Eqs (25a)/(25b) (pressure- and freq-dependent)
        slant factor       Eq (28) curved-Earth at low elevation, csc at high
        A_total            (γ_o · h_o + γ_w · h_w) · slant_factor

    Args:
        elevation_deg: elevation angle in degrees (0–90).
        frequency_ghz: frequency in GHz (1–350; raises outside).
        surface_water_vapor_g_per_m3: ρ_w at surface (default 7.5 g/m³).
        surface_pressure_hpa: total air pressure at surface (default 1013.25).

    Returns:
        One-way slant-path gas attenuation in dB.

    Raises:
        ValueError: if elevation is out of [0, 90] or frequency outside the
            tabulated [1, 350] GHz range.
    """
    if elevation_deg < 0 or elevation_deg > 90:
        raise ValueError(f"elevation_deg must be in [0,90], got {elevation_deg}")

    # Optional: replace the 7.5 g/m³ engineering default with the P.836
    # gridded annual map at the earth station's geographic location. Falls
    # back to the input value if the map isn't on disk.
    if use_itu_maps and es_lat_deg is not None and es_lon_deg is not None:
        from horizon_ric.data.itu_r import P836Map  # noqa: PLC0415

        map_value = P836Map().lookup(es_lat_deg, es_lon_deg)
        if map_value == map_value:  # not NaN
            surface_water_vapor_g_per_m3 = map_value

    gamma_o = p676_gamma_o_db_per_km(frequency_ghz)
    gamma_w = p676_gamma_w_db_per_km(frequency_ghz, surface_water_vapor_g_per_m3)
    h_o = p676_effective_height_dry_km(surface_pressure_hpa)
    h_w = p676_effective_height_wet_km(frequency_ghz, surface_water_vapor_g_per_m3)

    # Slant factor uses the heavier of the two effective heights — gas
    # attenuation is dominated by dry air at most frequencies, so h_o is the
    # right scale to use for the curvature correction.
    factor = _slant_path_factor(max(elevation_deg, 0.5), max(h_o, h_w))
    return factor * (gamma_o * h_o + gamma_w * h_w)


def rain_attenuation_dB(
    rain_rate_mm_per_hr: float,
    elevation_deg: float,
    frequency_ghz: float,
    polarization: str = "circular",
    rain_height_km: float = 3.0,
    earth_station_height_km: float = 0.0,
    *,
    es_lat_deg: float | None = None,
    es_lon_deg: float | None = None,
    use_itu_maps: bool = False,
) -> float:
    """Slant-path rain attenuation per ITU-R P.838-3 + P.618-13.

    Specific attenuation:
        γ_R = k · R^α                 (dB/km, P.838-3 Eq (1))
    where (k, α) are interpolated from the canonical P.838-3 Table 1
    (`_itu_tables.p838_k_alpha`). For circular polarization the table
    uses Eqs (4)–(5) with τ=45°, which collapses the elevation term.

    Slant geometry (P.618-13 §2.2.1):
        L_s = (h_R − h_s) / sin(el)               (≥10°)
              with curved-Earth correction below 10°.
        Path-length reduction factor (Eqs 32–35) with R₀.₀₁ = R input.

    Args:
        rain_rate_mm_per_hr: 0.01 % exceedance rain rate (mm/hr).
        elevation_deg: elevation angle (deg).
        frequency_ghz: frequency in GHz.
        polarization: "horizontal", "vertical", or "circular".
        rain_height_km: melting-layer height above sea level (P.839-4
            default ≈3 km, varies 1–5 km by latitude). Override with
            P.839-4 lat/lon lookup for high-fidelity work.
        earth_station_height_km: altitude of the earth station above MSL.

    Returns:
        One-way slant-path rain attenuation in dB.

    Raises:
        ValueError: if elevation, height geometry, or polarization is invalid.
    """
    # Optional: pull rain rate (R0.01) and rain height from ITU maps before
    # the early-return guard, so an es_lat/lon can supply rain even when
    # the caller passed 0.
    if use_itu_maps and es_lat_deg is not None and es_lon_deg is not None:
        from horizon_ric.data.itu_r import P837Map, P839Map  # noqa: PLC0415

        rh = P839Map().lookup(es_lat_deg, es_lon_deg)
        if rh == rh and rh > earth_station_height_km:
            rain_height_km = rh
        if rain_rate_mm_per_hr <= 0:
            rr = P837Map().lookup(es_lat_deg, es_lon_deg, percentage_time=0.01)
            if rr == rr and rr > 0:
                rain_rate_mm_per_hr = rr

    if rain_rate_mm_per_hr <= 0:
        return 0.0
    if elevation_deg <= 0 or elevation_deg > 90:
        raise ValueError(f"elevation_deg must be in (0, 90], got {elevation_deg}")
    if rain_height_km <= earth_station_height_km:
        raise ValueError(
            f"rain_height_km ({rain_height_km}) must exceed earth_station_height_km "
            f"({earth_station_height_km}); no rain layer above the station."
        )

    k, alpha = p838_k_alpha(frequency_ghz, polarization)
    gamma_R = k * (rain_rate_mm_per_hr ** alpha)

    # Slant length to top of rain cell. P.618-13 §2.2.1.1, Eq (1)/(2).
    delta_h = rain_height_km - earth_station_height_km
    el_rad = math.radians(elevation_deg)
    if elevation_deg >= 5.0:
        L_s = delta_h / math.sin(el_rad)
    else:
        # Curved-Earth correction at low elevation, P.618-13 Eq (2).
        Re = _EARTH_RADIUS_KM
        L_s = 2.0 * delta_h / (
            math.sqrt(math.sin(el_rad) ** 2 + 2.0 * delta_h / Re) + math.sin(el_rad)
        )

    # Horizontal projection.
    L_g = L_s * math.cos(el_rad)

    # Path-length reduction factor r₀.₀₁ — P.618-13 Eq (32).
    r = 1.0 / (1.0 + 0.78 * math.sqrt(L_g * gamma_R / frequency_ghz)
               - 0.38 * (1.0 - math.exp(-2.0 * L_g)))
    r = max(min(r, 2.5), 0.0)  # ITU-R cap

    return gamma_R * L_s * r


def total_path_loss_dB(
    distance_m: float,
    frequency_hz: float,
    elevation_deg: float = 90.0,
    rain_rate_mm_per_hr: float = 0.0,
    surface_water_vapor_g_per_m3: float = 7.5,
    polarization: str = "circular",
) -> dict[str, float]:
    """Compose full earth-to-space path loss from physics submodules.

    Useful for the CWM: returns a per-component breakdown so that operators
    can audit which physics drove the prediction.

    Args:
        distance_m: slant range in metres.
        frequency_hz: carrier frequency in Hz.
        elevation_deg: elevation angle (default 90° = zenith).
        rain_rate_mm_per_hr: rain rate (default 0, no rain).
        surface_water_vapor_g_per_m3: water vapor at surface (default 7.5).
        polarization: "horizontal" / "vertical" / "circular".

    Returns:
        dict with keys:
            "fspl_dB": free-space path loss
            "gas_dB": gas attenuation
            "rain_dB": rain attenuation
            "total_dB": sum of all losses
    """
    f_ghz = frequency_hz / 1e9
    fspl = free_space_path_loss_dB(distance_m, frequency_hz)
    gas = gas_attenuation_dB(elevation_deg, f_ghz, surface_water_vapor_g_per_m3)
    rain = rain_attenuation_dB(rain_rate_mm_per_hr, elevation_deg, f_ghz, polarization)
    return {
        "fspl_dB": fspl,
        "gas_dB": gas,
        "rain_dB": rain,
        "total_dB": fspl + gas + rain,
    }
