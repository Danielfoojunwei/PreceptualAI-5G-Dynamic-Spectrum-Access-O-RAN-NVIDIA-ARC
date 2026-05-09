"""Doppler shift + Doppler rate from orbital state.

For LEO at v ≈ 7.6 km/s, β = v/c ≈ 2.5e-5; relativistic correction is < 0.5
Hz at 28 GHz, so we use the first-order classical formula with documented
bound. For tighter pre-comp use cases switch to the full relativistic form.

References:
    3GPP TR 38.811 §6.3 (NTN channel; Doppler pre-compensation contract).
    Vallado §11 (range-rate from osculating elements).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta

from horizon_ric.planner.physics.geodesy import (
    ECEF,
    geodetic_to_ecef,
    look_angles_to_target,
)
from horizon_ric.planner.physics.orbital import (
    OrbitalState,
    eci_to_ecef_km,
)

_C_M_PER_S = 299_792_458.0
_OMEGA_EARTH_RAD_S = 7.2921151467e-5  # IERS-2010 mean Earth rotation rate


@dataclass(frozen=True)
class DopplerSample:
    """Single Doppler observation at one instant."""

    t_utc: object  # datetime
    elevation_deg: float
    range_m: float
    range_rate_m_s: float
    shift_hz: float
    rate_hz_s: float


def _es_velocity_eci_km_s(
    es_lat_deg: float, es_lon_deg: float, es_height_m: float = 0.0
) -> tuple[float, float, float]:
    """Earth-station velocity in ECI ≈ ω⊕ × r_es. ECI ≈ ECEF for the
    instantaneous rotation; the small extra term from precession/nutation is
    negligible for Doppler at LEO scales.
    """
    es_ecef = geodetic_to_ecef(es_lat_deg, es_lon_deg, es_height_m)
    # ω × r where ω = (0, 0, ω_E)
    vx = -_OMEGA_EARTH_RAD_S * es_ecef.y
    vy = _OMEGA_EARTH_RAD_S * es_ecef.x
    vz = 0.0
    return (vx / 1000.0, vy / 1000.0, vz / 1000.0)


def range_rate_m_s(
    sat_state: OrbitalState,
    es_lat_deg: float,
    es_lon_deg: float,
    es_height_m: float = 0.0,
) -> tuple[float, float]:
    """Return (range_m, range_rate_m_s) from satellite ECI state to ES.

    Range-rate is the line-of-sight projection of relative velocity:
        ṙ = (v_sat - v_es) · r̂_los
    Both states are converted to a common frame (ECEF at the sample time).
    """
    sat_r_ecef_km = eci_to_ecef_km(sat_state.r_eci_km, sat_state.epoch_utc)

    # Velocity in ECEF: v_ecef = R(-θ) v_eci - ω × r_ecef
    # We rotate the velocity vector by -GMST (same as position) to get the
    # *instantaneous* ECEF velocity ignoring frame transport, then subtract
    # the ω×r term so the result is ground-relative.
    from horizon_ric.planner.physics.orbital import gmst_rad

    theta = gmst_rad(sat_state.epoch_utc)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    vx_eci, vy_eci, vz_eci = sat_state.v_eci_km_s
    vx_e = cos_t * vx_eci + sin_t * vy_eci + _OMEGA_EARTH_RAD_S * sat_r_ecef_km[1]
    vy_e = -sin_t * vx_eci + cos_t * vy_eci - _OMEGA_EARTH_RAD_S * sat_r_ecef_km[0]
    vz_e = vz_eci

    es_ecef = geodetic_to_ecef(es_lat_deg, es_lon_deg, es_height_m)
    sat_ecef_m = ECEF(
        sat_r_ecef_km[0] * 1000.0,
        sat_r_ecef_km[1] * 1000.0,
        sat_r_ecef_km[2] * 1000.0,
    )
    los_x = sat_ecef_m.x - es_ecef.x
    los_y = sat_ecef_m.y - es_ecef.y
    los_z = sat_ecef_m.z - es_ecef.z
    rng = math.sqrt(los_x * los_x + los_y * los_y + los_z * los_z)
    if rng < 1e-9:
        return 0.0, 0.0
    rng_rate = (
        (vx_e * 1000.0) * los_x
        + (vy_e * 1000.0) * los_y
        + (vz_e * 1000.0) * los_z
    ) / rng
    return rng, rng_rate


def doppler_shift_hz(
    sat_state: OrbitalState,
    es_lat_deg: float,
    es_lon_deg: float,
    carrier_hz: float,
    es_height_m: float = 0.0,
) -> float:
    """First-order Doppler: Δf = -f_c · ṙ / c (sign convention: positive Δf
    means the satellite is approaching the ES)."""
    _, rng_rate = range_rate_m_s(sat_state, es_lat_deg, es_lon_deg, es_height_m)
    return -carrier_hz * rng_rate / _C_M_PER_S


def doppler_rate_hz_s(
    sat_state: OrbitalState,
    es_lat_deg: float,
    es_lon_deg: float,
    carrier_hz: float,
    es_height_m: float = 0.0,
    dt_s: float = 1.0,
) -> float:
    """Numerical Doppler rate: finite-difference of doppler_shift_hz over dt_s.

    Uses the orbital propagator to advance the state by dt_s and retake the
    measurement. dt_s = 1 s is a good default for LEO; smaller values amplify
    numerical noise from the SGP4 / Kepler+J2 step.
    """
    f0 = doppler_shift_hz(sat_state, es_lat_deg, es_lon_deg, carrier_hz, es_height_m)
    advanced_epoch = sat_state.epoch_utc + timedelta(seconds=dt_s)
    # Re-derive state at advanced epoch using Kepler+J2 from current osculating.
    # Approximate by extrapolating with current velocity (good for dt_s ≪ T_orb).
    rx, ry, rz = sat_state.r_eci_km
    vx, vy, vz = sat_state.v_eci_km_s
    advanced = OrbitalState(
        epoch_utc=advanced_epoch,
        r_eci_km=(rx + vx * dt_s, ry + vy * dt_s, rz + vz * dt_s),
        v_eci_km_s=sat_state.v_eci_km_s,
    )
    f1 = doppler_shift_hz(advanced, es_lat_deg, es_lon_deg, carrier_hz, es_height_m)
    return (f1 - f0) / dt_s


def pass_envelope(
    initial_state: OrbitalState,
    es_lat_deg: float,
    es_lon_deg: float,
    carrier_hz: float,
    duration_s: float = 600.0,
    step_s: float = 5.0,
    es_height_m: float = 0.0,
) -> list[DopplerSample]:
    """Sweep an entire LEO pass at `step_s` cadence.

    Returns list of DopplerSamples covering `duration_s` from the initial
    state. Useful for serving DU pre-compensation tables and UE-budget
    feasibility checks.
    """
    samples: list[DopplerSample] = []
    n = max(int(duration_s // step_s), 1)
    rx, ry, rz = initial_state.r_eci_km
    vx, vy, vz = initial_state.v_eci_km_s
    for i in range(n + 1):
        t = i * step_s
        new_state = OrbitalState(
            epoch_utc=initial_state.epoch_utc + timedelta(seconds=t),
            r_eci_km=(rx + vx * t, ry + vy * t, rz + vz * t),
            v_eci_km_s=initial_state.v_eci_km_s,
        )
        rng, rate = range_rate_m_s(
            new_state, es_lat_deg, es_lon_deg, es_height_m
        )
        # Look angle for elevation
        from horizon_ric.planner.physics.orbital import state_ecef_m

        sat_ecef = state_ecef_m(new_state)
        look = look_angles_to_target(
            es_lat_deg, es_lon_deg, sat_ecef, es_height_m
        )
        shift = -carrier_hz * rate / _C_M_PER_S
        # Rate by finite difference vs the previous sample (or 0 at t=0).
        if samples:
            rate_hz_s = (shift - samples[-1].shift_hz) / step_s
        else:
            rate_hz_s = 0.0
        samples.append(
            DopplerSample(
                t_utc=new_state.epoch_utc,
                elevation_deg=look.elevation_deg,
                range_m=rng,
                range_rate_m_s=rate,
                shift_hz=shift,
                rate_hz_s=rate_hz_s,
            )
        )
    return samples


__all__ = [
    "DopplerSample",
    "doppler_rate_hz_s",
    "doppler_shift_hz",
    "pass_envelope",
    "range_rate_m_s",
]
