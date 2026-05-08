"""Physics modules for the Compositional World Model (paradigm H2).

Each module wraps a closed-form ITU-R / 3GPP equation. They are pure
functions — no learnable parameters, no state.

Implemented today:
    propagation:  ITU-R P.525 (FSPL), P.676-13 Annex 2 (gas), P.838-3
                  (rain specific atten.), P.618-13 (slant geometry).
    beam_pattern: 3GPP TR 38.901 antenna patterns + ITU-R F.1336.
    geodesy:      WGS-84 ellipsoid + ECEF + look-angles (no spherical-Earth).
    s1428:        ITU-R S.1428-1 victim earth-station antenna gain mask.
    epfd:         ITU-R S.1503-3 EPFD-down aggregation across NGSO sats.

Reserved for later phases (NOT implemented):
    P.452 NLoS interference, P.840 cloud/fog, queueing, cluster sim.

References:
    ITU-R P.525-4   — Free-space path loss
    ITU-R P.676-13  — Gas attenuation (Annex 2 simplified)
    ITU-R P.838-3   — Rain specific attenuation k/α
    ITU-R P.618-13  — Slant-path geometry & reduction factors
    ITU-R S.1428-1  — Reference earth-station antenna pattern
    ITU-R S.1503-3  — EPFD validation framework
    3GPP TR 38.901  — NR channel models
    ITU-R F.1336-5  — Reference radiation patterns
"""

from horizon_ric.planner.physics.beam_pattern import (
    angle_between_vectors_deg,
    beam_gain_dB,
    omni_pattern_dB,
)
from horizon_ric.planner.physics.coexistence import (
    BandEnvelope,
    InlineEventStat,
    aggregate_aclr_leakage_dBm,
    compose_feasibility,
    ngso_inline_event_probability,
    nru_fair_share_airtime,
    ntn_to_terrestrial_required_guard_db,
    p_servicelink_rain_given_gateway,
)
from horizon_ric.planner.physics.epfd import (
    EPFDResult,
    EPFDTimeCDF,
    NGSOEmitter,
    epfd_down,
    epfd_time_cdf,
)
from horizon_ric.planner.physics.geodesy import (
    ECEF,
    GSO_ALTITUDE_M,
    GSO_RADIUS_M,
    LookAngles,
    angle_between_ecef_vectors_deg,
    ecef_distance_m,
    ecef_to_geodetic,
    geodetic_to_ecef,
    gso_satellite_ecef,
    look_angles_to_gso,
    look_angles_to_target,
)
from horizon_ric.planner.physics.propagation import (
    free_space_path_loss_dB,
    gas_attenuation_dB,
    rain_attenuation_dB,
    total_path_loss_dB,
)
from horizon_ric.planner.physics.s1428 import s1428_gain_dBi

__all__ = [
    "BandEnvelope",
    "ECEF",
    "EPFDResult",
    "EPFDTimeCDF",
    "GSO_ALTITUDE_M",
    "GSO_RADIUS_M",
    "InlineEventStat",
    "LookAngles",
    "NGSOEmitter",
    "aggregate_aclr_leakage_dBm",
    "angle_between_ecef_vectors_deg",
    "angle_between_vectors_deg",
    "beam_gain_dB",
    "compose_feasibility",
    "ecef_distance_m",
    "ecef_to_geodetic",
    "epfd_down",
    "epfd_time_cdf",
    "free_space_path_loss_dB",
    "gas_attenuation_dB",
    "geodetic_to_ecef",
    "gso_satellite_ecef",
    "look_angles_to_gso",
    "look_angles_to_target",
    "ngso_inline_event_probability",
    "ntn_to_terrestrial_required_guard_db",
    "nru_fair_share_airtime",
    "omni_pattern_dB",
    "p_servicelink_rain_given_gateway",
    "rain_attenuation_dB",
    "s1428_gain_dBi",
    "total_path_loss_dB",
]
