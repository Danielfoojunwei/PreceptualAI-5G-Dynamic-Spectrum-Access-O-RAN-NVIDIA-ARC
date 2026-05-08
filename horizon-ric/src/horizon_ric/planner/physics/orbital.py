"""Orbital propagation — SGP4 + Walker constellations + Kepler+J2.

Three propagation paths (cost vs accuracy):

    sgp4_state(tle, t_utc)           Vallado-correct, all perturbations,
                                      ~30 µs per call (vendored python-sgp4).
    kepler_j2_step(state, dt_s)      Two-body + J2 secular precession,
                                      ~1 µs per call. Use BETWEEN sgp4 calls.
    walker_delta_constellation(...)  Closed-form expansion of a Walker spec
                                      into N initial Keplerian elements.

Returned states are ECI (Earth-Centred Inertial, J2000) by default;
`eci_to_ecef(state, t_utc)` converts using IAU 1982 GMST.

References:
    Vallado D.A. *Fundamentals of Astrodynamics & Applications*, 4 e.
    Hoots & Roehrich 1980, *Spacetrack Report #3* (SGP4).
    Walker J.G. 1984, *Satellite Constellations*, J. Br. Interplanet. Soc. 37.
    IAU 1982 GMST formula (USNO Circular 179).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from sgp4.api import WGS84, Satrec, jday

from horizon_ric.planner.physics.geodesy import ECEF, WGS84_A_M

# Earth gravitational parameter (km³/s²) — WGS-84 / EGM-96 value used by SGP4.
_MU_KM3_S2 = 398_600.4418
_J2 = 1.0826267e-3
_R_E_KM = WGS84_A_M / 1000.0
_OMEGA_EARTH_RAD_S = 7.2921151467e-5  # IERS 2010 mean Earth rotation rate


@dataclass(frozen=True)
class OrbitalState:
    """Position + velocity in ECI (km, km/s) at `epoch_utc`."""

    epoch_utc: datetime
    r_eci_km: tuple[float, float, float]
    v_eci_km_s: tuple[float, float, float]

    def speed_km_s(self) -> float:
        vx, vy, vz = self.v_eci_km_s
        return math.sqrt(vx * vx + vy * vy + vz * vz)

    def altitude_km(self) -> float:
        x, y, z = self.r_eci_km
        return math.sqrt(x * x + y * y + z * z) - _R_E_KM


@dataclass(frozen=True)
class KeplerianElements:
    """Classical orbital elements (radians for angles, km for a)."""

    a_km: float            # semi-major axis
    e: float               # eccentricity
    i_rad: float           # inclination
    raan_rad: float        # right ascension of ascending node (Ω)
    argp_rad: float        # argument of periapsis (ω)
    M_rad: float           # mean anomaly at epoch
    epoch_utc: datetime


# ─── SGP4 wrapper ────────────────────────────────────────────────────────


def sgp4_state(tle_line1: str, tle_line2: str, t_utc: datetime) -> OrbitalState:
    """Propagate a TLE to UTC `t_utc` using vendored python-sgp4 (Vallado).

    Returns ECI (TEME, equivalent to J2000 within sub-mm at LEO altitudes
    over the centuries this software will exist). Conversion to ECEF should
    use `eci_to_ecef(state, t_utc)`.
    """
    sat = Satrec.twoline2rv(tle_line1, tle_line2, WGS84)
    if t_utc.tzinfo is None:
        raise ValueError("t_utc must be timezone-aware")
    t_utc = t_utc.astimezone(timezone.utc)
    jd, fr = jday(
        t_utc.year, t_utc.month, t_utc.day,
        t_utc.hour, t_utc.minute,
        t_utc.second + t_utc.microsecond / 1e6,
    )
    err, r, v = sat.sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 propagation failed (code {err})")
    return OrbitalState(epoch_utc=t_utc, r_eci_km=tuple(r), v_eci_km_s=tuple(v))


# ─── Walker constellation ────────────────────────────────────────────────


def walker_delta_constellation(
    inclination_deg: float,
    total_sats: int,
    num_planes: int,
    phasing_F: int,
    altitude_km: float,
    eccentricity: float = 0.0,
    arg_perigee_deg: float = 0.0,
    epoch_utc: datetime | None = None,
) -> list[KeplerianElements]:
    """Expand a Walker-Δ (i:T/P/F) constellation into T element sets.

    Walker's parameters (1984):
        i    inclination
        T    total number of satellites
        P    number of orbital planes (must divide T)
        F    relative phasing parameter ∈ {0, …, P−1}.
             Δν_inter = (360° · F) / T

    Plane k has Ω_k = (360°·k)/P; satellite j in plane k has
    M_{kj} = (360°·j)/(T/P) + (360°·F·k)/T.
    """
    if total_sats <= 0 or num_planes <= 0:
        raise ValueError("total_sats and num_planes must be positive")
    if total_sats % num_planes != 0:
        raise ValueError(
            f"total_sats ({total_sats}) must be divisible by num_planes "
            f"({num_planes})"
        )
    if not 0 <= phasing_F < num_planes:
        raise ValueError(f"phasing_F must be in [0, {num_planes - 1}]")

    sats_per_plane = total_sats // num_planes
    a_km = _R_E_KM + altitude_km
    epoch = epoch_utc or datetime.now(timezone.utc)
    inc_rad = math.radians(inclination_deg)
    argp_rad = math.radians(arg_perigee_deg)

    elements: list[KeplerianElements] = []
    for k in range(num_planes):
        raan = (2.0 * math.pi * k) / num_planes
        for j in range(sats_per_plane):
            M = (
                (2.0 * math.pi * j) / sats_per_plane
                + (2.0 * math.pi * phasing_F * k) / total_sats
            ) % (2.0 * math.pi)
            elements.append(
                KeplerianElements(
                    a_km=a_km, e=eccentricity, i_rad=inc_rad,
                    raan_rad=raan, argp_rad=argp_rad, M_rad=M,
                    epoch_utc=epoch,
                )
            )
    return elements


# ─── Kepler + J2 propagation ─────────────────────────────────────────────


def mean_motion_rad_s(a_km: float) -> float:
    """n = √(μ/a³)."""
    return math.sqrt(_MU_KM3_S2 / (a_km ** 3))


def j2_secular_rates(elem: KeplerianElements) -> tuple[float, float, float]:
    """Secular rates dΩ/dt, dω/dt, dM_J2/dt (rad/s) per Vallado §9.6."""
    n = mean_motion_rad_s(elem.a_km)
    p = elem.a_km * (1.0 - elem.e * elem.e)
    factor = 1.5 * n * _J2 * (_R_E_KM / p) ** 2
    cos_i = math.cos(elem.i_rad)
    sin_i = math.sin(elem.i_rad)
    sin2_i = sin_i * sin_i
    d_raan = -factor * cos_i
    d_argp = 0.5 * factor * (5.0 * cos_i * cos_i - 1.0)
    d_M = 0.5 * factor * math.sqrt(1.0 - elem.e * elem.e) * (3.0 * cos_i * cos_i - 1.0)
    # The Mean-anomaly rate is n + the J2 secular contribution:
    return d_raan, d_argp, n + d_M


def keplerian_to_eci(elem: KeplerianElements) -> tuple[tuple[float, float, float],
                                                        tuple[float, float, float]]:
    """Convert classical elements at epoch to (r_ECI, v_ECI) in km, km/s."""
    a, e, i, raan, argp, M = (
        elem.a_km, elem.e, elem.i_rad, elem.raan_rad, elem.argp_rad, elem.M_rad,
    )
    # Solve Kepler's equation M = E - e sin E (Newton-Raphson)
    E = M if e < 0.8 else math.pi
    for _ in range(10):
        delta = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= delta
        if abs(delta) < 1e-12:
            break
    cos_E, sin_E = math.cos(E), math.sin(E)
    nu = math.atan2(math.sqrt(1.0 - e * e) * sin_E, cos_E - e)
    r_pf = a * (1.0 - e * cos_E)
    p = a * (1.0 - e * e)
    # Position in perifocal frame
    cos_nu, sin_nu = math.cos(nu), math.sin(nu)
    r_pqw = (r_pf * cos_nu, r_pf * sin_nu, 0.0)
    h_term = math.sqrt(_MU_KM3_S2 / p)
    v_pqw = (-h_term * sin_nu, h_term * (e + cos_nu), 0.0)
    # Rotate PQW → ECI: R_z(-Ω) R_x(-i) R_z(-ω)
    cos_O, sin_O = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(i), math.sin(i)
    cos_w, sin_w = math.cos(argp), math.sin(argp)
    R11 = cos_O * cos_w - sin_O * sin_w * cos_i
    R12 = -cos_O * sin_w - sin_O * cos_w * cos_i
    R21 = sin_O * cos_w + cos_O * sin_w * cos_i
    R22 = -sin_O * sin_w + cos_O * cos_w * cos_i
    R31 = sin_w * sin_i
    R32 = cos_w * sin_i

    def rotate(v: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, _z = v
        return (
            R11 * x + R12 * y,
            R21 * x + R22 * y,
            R31 * x + R32 * y,
        )

    return rotate(r_pqw), rotate(v_pqw)


def kepler_j2_step(elem: KeplerianElements, dt_s: float) -> KeplerianElements:
    """Advance a Keplerian element set by `dt_s` using two-body + J2 secular
    rates (Vallado §9.6). Periodic terms are dropped (~ km error over hours)."""
    d_raan, d_argp, d_M = j2_secular_rates(elem)
    new_raan = (elem.raan_rad + d_raan * dt_s) % (2.0 * math.pi)
    new_argp = (elem.argp_rad + d_argp * dt_s) % (2.0 * math.pi)
    new_M = (elem.M_rad + d_M * dt_s) % (2.0 * math.pi)
    new_epoch = elem.epoch_utc.fromtimestamp(
        elem.epoch_utc.timestamp() + dt_s, tz=timezone.utc
    )
    return KeplerianElements(
        a_km=elem.a_km, e=elem.e, i_rad=elem.i_rad,
        raan_rad=new_raan, argp_rad=new_argp, M_rad=new_M,
        epoch_utc=new_epoch,
    )


def propagate_batch(
    states: Iterable[KeplerianElements], dt_s: float
) -> list[KeplerianElements]:
    """J2-secular Kepler step for every element set in `states`."""
    return [kepler_j2_step(s, dt_s) for s in states]


def keplerian_state(elem: KeplerianElements) -> OrbitalState:
    r, v = keplerian_to_eci(elem)
    return OrbitalState(epoch_utc=elem.epoch_utc, r_eci_km=r, v_eci_km_s=v)


# ─── ECI → ECEF (IAU 1982 GMST) ──────────────────────────────────────────


def gmst_rad(t_utc: datetime) -> float:
    """Greenwich Mean Sidereal Time at `t_utc`, radians, IAU 1982 model.

    Sufficient (sub-arcsecond) for any orbit work above LEO altitude.
    """
    if t_utc.tzinfo is None:
        raise ValueError("t_utc must be timezone-aware")
    t_utc = t_utc.astimezone(timezone.utc)
    jd, fr = jday(
        t_utc.year, t_utc.month, t_utc.day,
        t_utc.hour, t_utc.minute,
        t_utc.second + t_utc.microsecond / 1e6,
    )
    jd_ut1 = jd + fr  # we treat UTC as UT1 (sub-second drift, fine here)
    T_U = (jd_ut1 - 2_451_545.0) / 36_525.0
    gmst_seconds = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T_U
        + 0.093104 * T_U * T_U
        - 6.2e-6 * (T_U ** 3)
    )
    gmst_seconds = gmst_seconds % 86400.0
    if gmst_seconds < 0:
        gmst_seconds += 86400.0
    return (gmst_seconds / 86400.0) * 2.0 * math.pi


def eci_to_ecef_km(
    r_eci_km: tuple[float, float, float],
    t_utc: datetime,
) -> tuple[float, float, float]:
    """Rotate ECI by -GMST around Z axis to get ECEF (km)."""
    theta = gmst_rad(t_utc)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x, y, z = r_eci_km
    return (cos_t * x + sin_t * y, -sin_t * x + cos_t * y, z)


def state_ecef_m(state: OrbitalState) -> ECEF:
    """Convenience: OrbitalState → ECEF (in metres, ready for `geodesy`)."""
    x, y, z = eci_to_ecef_km(state.r_eci_km, state.epoch_utc)
    return ECEF(x * 1000.0, y * 1000.0, z * 1000.0)


__all__ = [
    "KeplerianElements",
    "OrbitalState",
    "eci_to_ecef_km",
    "gmst_rad",
    "j2_secular_rates",
    "kepler_j2_step",
    "keplerian_state",
    "keplerian_to_eci",
    "mean_motion_rad_s",
    "propagate_batch",
    "sgp4_state",
    "state_ecef_m",
    "walker_delta_constellation",
]
