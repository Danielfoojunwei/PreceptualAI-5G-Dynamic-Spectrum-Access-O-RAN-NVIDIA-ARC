"""Antenna beam patterns — 3GPP TR 38.901 + ITU-R F.1336 reference patterns.

Used by the Compositional World Model to compute the antenna gain
contribution along any direction. Critical input to the GSO PFD constraint
layer (we need to know how much energy points toward each GSO arc).

References:
    3GPP TR 38.901 v17.0 §7.3   — Antenna model for NR
    ITU-R F.1336-5 (2019)       — Reference patterns for omni / sectoral
    ITU-R S.580 (2004)          — FSS earth station antenna pattern
"""

from __future__ import annotations

import math


def beam_gain_dB(
    azimuth_deg: float,
    elevation_deg: float,
    boresight_az_deg: float = 0.0,
    boresight_el_deg: float = 90.0,
    half_power_az_deg: float = 65.0,
    half_power_el_deg: float = 65.0,
    max_gain_dBi: float = 14.0,
    sidelobe_floor_dBi: float = -25.0,
) -> float:
    """3GPP TR 38.901 §7.3.1 single-element antenna pattern.

    A_E(θ', φ') = -min[ -(A_E,V(θ') + A_E,H(φ')) , A_max ]

    where A_E,V and A_E,H are the vertical and horizontal cuts.

    Args:
        azimuth_deg: query azimuth in degrees [-180, 180].
        elevation_deg: query elevation in degrees [0, 180]; 90° = horizon.
        boresight_az_deg: beam pointing azimuth.
        boresight_el_deg: beam pointing elevation.
        half_power_az_deg: 3-dB azimuth beamwidth (default 65° per 3GPP).
        half_power_el_deg: 3-dB elevation beamwidth (default 65° per 3GPP).
        max_gain_dBi: peak gain at boresight.
        sidelobe_floor_dBi: floor gain (front-to-back; default -25 dBi).

    Returns:
        Gain in dBi at the queried (az, el) direction.
    """
    # Compute angular offsets from boresight, accounting for azimuth wrap-around
    delta_az = ((azimuth_deg - boresight_az_deg + 180.0) % 360.0) - 180.0
    delta_el = elevation_deg - boresight_el_deg

    # Vertical pattern (Eq 7.3-4 in 3GPP TR 38.901)
    a_v = -min(12.0 * (delta_el / half_power_el_deg) ** 2, 30.0)
    # Horizontal pattern (Eq 7.3-5)
    a_h = -min(12.0 * (delta_az / half_power_az_deg) ** 2, 30.0)
    # Composite pattern (Eq 7.3-6)
    a_total = -min(-(a_v + a_h), 30.0)

    gain_dBi = max_gain_dBi + a_total
    return max(gain_dBi, sidelobe_floor_dBi)


def omni_pattern_dB(elevation_deg: float, max_gain_dBi: float = 8.0) -> float:
    """ITU-R F.1336-5 reference omnidirectional radiation pattern.

    Used for satellite earth-station omni / sector reference comparisons.
    Simplified for control-loop use; full F.1336 has multiple cases.

    Args:
        elevation_deg: elevation angle [0, 90].
        max_gain_dBi: peak omnidirectional gain.

    Returns:
        Gain in dBi.
    """
    # Above horizon: full peak gain
    if elevation_deg >= 0:
        # Smoother roll-off above horizon (F.1336 Annex 2)
        return max_gain_dBi - 0.05 * abs(elevation_deg - 0.0)
    return max_gain_dBi - 12.0  # below horizon: heavy attenuation


def angle_between_vectors_deg(
    az1_deg: float,
    el1_deg: float,
    az2_deg: float,
    el2_deg: float,
) -> float:
    """Angle between two pointing directions (great-circle / spherical).

    Args:
        az1_deg, el1_deg: first direction.
        az2_deg, el2_deg: second direction.

    Returns:
        Angle in degrees [0, 180].
    """
    az1, el1 = math.radians(az1_deg), math.radians(el1_deg)
    az2, el2 = math.radians(az2_deg), math.radians(el2_deg)
    cos_angle = (
        math.sin(el1) * math.sin(el2)
        + math.cos(el1) * math.cos(el2) * math.cos(az1 - az2)
    )
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))
