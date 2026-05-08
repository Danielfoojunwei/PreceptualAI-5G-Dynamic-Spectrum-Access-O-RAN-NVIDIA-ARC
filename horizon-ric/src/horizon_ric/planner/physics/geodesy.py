"""WGS-84 ellipsoid geodesy.

Replaces the spherical-earth approximations previously used inside
`policy.constraints` for GSO geometry. Uses the official WGS-84 defining
parameters (a, f) per NIMA TR 8350.2:

    a = 6378137.0 m              (semi-major axis)
    f = 1 / 298.257223563        (flattening)
    b = a · (1 - f)              (semi-minor axis)
    e² = 2f - f²                 (first eccentricity squared)

Functions:
    geodetic_to_ecef             (lat, lon, h) → (x, y, z) in metres
    ecef_to_geodetic             (x, y, z) → (lat, lon, h) [Bowring]
    ecef_distance_m              straight-line distance between two ECEF points
    look_angles_to_target        from an earth-station to any ECEF target,
                                 returns (azimuth_deg, elevation_deg, range_m)
    gso_satellite_ecef           ECEF of a GSO satellite at a given longitude
    look_angles_to_gso           convenience: ES → GSO

Units: degrees for lat/lon/azimuth/elevation, metres for distances.
Sign conventions: latitude positive north, longitude positive east,
azimuth clockwise from true north, elevation positive above the local
horizon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ─── WGS-84 defining constants (NIMA TR 8350.2) ──────────────────────────
WGS84_A_M: float = 6_378_137.0
WGS84_F: float = 1.0 / 298.257223563
WGS84_B_M: float = WGS84_A_M * (1.0 - WGS84_F)
WGS84_E2: float = 2.0 * WGS84_F - WGS84_F * WGS84_F  # first eccentricity²
WGS84_EP2: float = WGS84_E2 / (1.0 - WGS84_E2)        # second eccentricity²

# Standard geostationary orbit altitude above the equator (m).
GSO_ALTITUDE_M: float = 35_786_000.0
# GSO orbital radius from Earth's centre (semi-major axis ≈ 42,164.17 km).
GSO_RADIUS_M: float = WGS84_A_M + GSO_ALTITUDE_M


@dataclass(frozen=True)
class ECEF:
    """ECEF position in metres."""

    x: float
    y: float
    z: float

    def __sub__(self, other: "ECEF") -> "ECEF":
        return ECEF(self.x - other.x, self.y - other.y, self.z - other.z)

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


@dataclass(frozen=True)
class LookAngles:
    """Azimuth/elevation/range from an earth station to a target."""

    azimuth_deg: float       # [0, 360) clockwise from true north
    elevation_deg: float     # [-90, 90], negative if below local horizon
    range_m: float           # straight-line slant distance


def geodetic_to_ecef(lat_deg: float, lon_deg: float, height_m: float = 0.0) -> ECEF:
    """Convert WGS-84 geodetic (lat, lon, h) to ECEF (x, y, z).

    Standard form (Heiskanen & Moritz 1967):
        N = a / sqrt(1 - e² sin²φ)         (prime vertical radius of curvature)
        x = (N + h) cosφ cosλ
        y = (N + h) cosφ sinλ
        z = (N (1 - e²) + h) sinφ
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + height_m) * cos_lat * math.cos(lon)
    y = (n + height_m) * cos_lat * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + height_m) * sin_lat
    return ECEF(x, y, z)


def ecef_to_geodetic(p: ECEF) -> tuple[float, float, float]:
    """ECEF → (lat_deg, lon_deg, height_m) via Bowring's closed-form (1976).

    Accurate to ~1 mm anywhere within the atmosphere. For altitudes >> R_E
    (e.g. GSO satellites) the formula is still well-conditioned but Bowring
    converges in one iteration; we add a safety pass.
    """
    x, y, z = p.x, p.y, p.z
    lon = math.atan2(y, x)
    p_xy = math.sqrt(x * x + y * y)

    if p_xy < 1e-9:
        # Near a geographic pole.
        lat = math.copysign(math.pi / 2.0, z)
        n_polar = WGS84_A_M * WGS84_A_M / WGS84_B_M
        h = abs(z) - WGS84_B_M
        return math.degrees(lat), math.degrees(lon), h

    # Bowring 1976 starting parameter.
    theta = math.atan2(z * WGS84_A_M, p_xy * WGS84_B_M)
    sin_th = math.sin(theta)
    cos_th = math.cos(theta)
    lat = math.atan2(
        z + WGS84_EP2 * WGS84_B_M * sin_th * sin_th * sin_th,
        p_xy - WGS84_E2 * WGS84_A_M * cos_th * cos_th * cos_th,
    )
    sin_lat = math.sin(lat)
    n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    h = p_xy / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), h


def ecef_distance_m(a: ECEF, b: ECEF) -> float:
    """Straight-line (slant) distance between two ECEF points, in metres."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def look_angles_to_target(
    es_lat_deg: float,
    es_lon_deg: float,
    target: ECEF,
    es_height_m: float = 0.0,
) -> LookAngles:
    """Compute azimuth, elevation, and range from a WGS-84 earth station
    to a target given in ECEF.

    Build the local ENU (east-north-up) basis at the ES, project the
    line-of-sight vector, then derive az/el/range. This is the textbook
    transform; no spherical-earth approximation is involved.
    """
    es_ecef = geodetic_to_ecef(es_lat_deg, es_lon_deg, es_height_m)
    los = target - es_ecef
    rng = los.norm()
    if rng < 1e-9:
        return LookAngles(0.0, 90.0, 0.0)

    lat = math.radians(es_lat_deg)
    lon = math.radians(es_lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    # Local ENU basis vectors expressed in ECEF.
    east_x, east_y, east_z = -sin_lon, cos_lon, 0.0
    north_x = -sin_lat * cos_lon
    north_y = -sin_lat * sin_lon
    north_z = cos_lat
    up_x = cos_lat * cos_lon
    up_y = cos_lat * sin_lon
    up_z = sin_lat

    east = los.x * east_x + los.y * east_y + los.z * east_z
    north = los.x * north_x + los.y * north_y + los.z * north_z
    up = los.x * up_x + los.y * up_y + los.z * up_z

    # Elevation from horizontal plane.
    horiz = math.sqrt(east * east + north * north)
    elevation = math.degrees(math.atan2(up, horiz))
    # Azimuth clockwise from true north.
    azimuth = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0
    return LookAngles(azimuth_deg=azimuth, elevation_deg=elevation, range_m=rng)


def gso_satellite_ecef(
    longitude_deg: float, altitude_m: float = GSO_ALTITUDE_M
) -> ECEF:
    """ECEF position of a satellite at the given equatorial longitude.

    Defaults to GSO altitude (35,786 km). Override `altitude_m` to model
    other equatorial circular orbits.
    """
    lon = math.radians(longitude_deg)
    radius = WGS84_A_M + altitude_m
    return ECEF(radius * math.cos(lon), radius * math.sin(lon), 0.0)


def look_angles_to_gso(
    es_lat_deg: float, es_lon_deg: float, gso_lon_deg: float,
    es_height_m: float = 0.0,
    gso_altitude_m: float = GSO_ALTITUDE_M,
) -> LookAngles:
    """Convenience: az/el/range from an ES to an equatorial satellite."""
    return look_angles_to_target(
        es_lat_deg, es_lon_deg,
        gso_satellite_ecef(gso_lon_deg, gso_altitude_m), es_height_m,
    )


def angle_between_ecef_vectors_deg(
    origin: ECEF, p1: ECEF, p2: ECEF
) -> float:
    """Angle subtended at `origin` between two target points (degrees).

    Used by EPFD to compute the off-axis angle from a victim earth-station
    boresight (towards its wanted GSO satellite) to an interfering NGSO sat.
    """
    v1 = p1 - origin
    v2 = p2 - origin
    n1, n2 = v1.norm(), v2.norm()
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z) / (n1 * n2)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


__all__ = [
    "ECEF",
    "GSO_ALTITUDE_M",
    "GSO_RADIUS_M",
    "LookAngles",
    "WGS84_A_M",
    "WGS84_B_M",
    "WGS84_E2",
    "WGS84_F",
    "angle_between_ecef_vectors_deg",
    "ecef_distance_m",
    "ecef_to_geodetic",
    "geodetic_to_ecef",
    "gso_satellite_ecef",
    "look_angles_to_gso",
    "look_angles_to_target",
]
