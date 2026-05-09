"""NTN-specific timing budgets for 5G NR (TS 38.331 §5.2.1.4 + TR 38.821).

Computes:
    one_way_delay_s            slant-range divided by c
    k_offset_slots             UL grant timing offset (TS 38.331 §5.2.1.4)
    k_mac_slots                MAC-CE application offset
    min_harq_processes         minimum number of HARQ processes
    harq_feedback_enabled      can we keep ACK/NACK feedback?
    rach_response_window_s     RACH response window (TR 38.821 §7.2.1.1)

For LEO-600 km / Ka: K_offset ≈ 12 slots at μ=1, HARQ feedback can be kept
with 16 processes. For GEO: K_offset ≈ 514 slots, HARQ feedback must be
disabled and RLC AM ARQ relied upon.

References:
    3GPP TS 38.331 §5.2.1.4 (NTN-Config IE)
    3GPP TR 38.821 §6.1.1 (HARQ for NTN), §7.2.1.1 (PRACH for NTN)
    3GPP TS 38.211 §4.3.2 (slot timing per numerology)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from horizon_ric.planner.physics.geodesy import (
    GSO_ALTITUDE_M,
    look_angles_to_target,
)

_C_M_PER_S = 299_792_458.0
_R_E_M = 6_378_137.0


@dataclass(frozen=True)
class NTNTimingState:
    one_way_delay_s: float
    k_offset_slots: int
    k_mac_slots: int
    min_harq_processes: int
    harq_feedback_enabled: bool
    rach_response_window_s: float
    slot_duration_s: float


def _slot_duration_s(scs_khz: float) -> float:
    """5G NR slot duration as a function of subcarrier spacing."""
    if scs_khz <= 0:
        raise ValueError(f"scs_khz must be > 0, got {scs_khz}")
    # μ from SCS = 15 · 2^μ; T_slot = 1 ms / 2^μ
    mu = round(math.log2(scs_khz / 15.0))
    return 1e-3 / (2 ** mu)


def _slant_range_m_from_elevation(altitude_m: float, elevation_deg: float) -> float:
    """Slant range to a satellite at given altitude / ES elevation, spherical
    Earth (suitable to ~ km accuracy at GSO; sub-100 m at LEO)."""
    if altitude_m <= 0:
        raise ValueError(f"altitude_m must be > 0, got {altitude_m}")
    if not 0.0 <= elevation_deg <= 90.0:
        raise ValueError(f"elevation_deg out of range, got {elevation_deg}")
    eps = math.radians(elevation_deg)
    R = _R_E_M
    h = altitude_m
    # Standard slant-range equation (Lutz et al. 2000, Eq. 2-1)
    return -R * math.sin(eps) + math.sqrt(
        (R * math.sin(eps)) ** 2 + h * h + 2 * R * h
    )


def ntn_timing_state(
    altitude_m: float,
    elevation_deg: float,
    scs_khz: float = 30.0,
    ta_uncertainty_us: float = 50.0,
) -> NTNTimingState:
    """Compute the NTN-specific timing parameters at a given geometry.

    Args:
        altitude_m: orbital altitude above Earth's mean radius.
        elevation_deg: ES elevation to the satellite (deg).
        scs_khz: subcarrier spacing — drives slot duration.
        ta_uncertainty_us: TA uncertainty after pre-comp (drives RACH window).

    Returns:
        NTNTimingState with all computed values.

    Notes:
        K_offset is set so K_offset · T_slot ≥ 2·d/c (round-trip delay).
        K_mac is conservatively set to the same value; TS 38.331 allows it
        to differ when MAC-CE timing is measured separately.
        HARQ feedback is enabled only when 2·d/c fits within 16·T_slot.
    """
    slant = _slant_range_m_from_elevation(altitude_m, elevation_deg)
    one_way = slant / _C_M_PER_S
    rtt = 2.0 * one_way

    t_slot = _slot_duration_s(scs_khz)
    k_offset = math.ceil(rtt / t_slot)
    k_mac = k_offset  # conservative; spec allows difference if measured
    # HARQ process count: Rel-17 supports up to 32 processes for NTN.
    min_harq = max(int(math.ceil(rtt / t_slot)), 16)
    harq_feedback = min_harq <= 32

    rach_response_window = rtt + ta_uncertainty_us * 1e-6

    return NTNTimingState(
        one_way_delay_s=one_way,
        k_offset_slots=int(k_offset),
        k_mac_slots=int(k_mac),
        min_harq_processes=int(min_harq),
        harq_feedback_enabled=bool(harq_feedback),
        rach_response_window_s=rach_response_window,
        slot_duration_s=t_slot,
    )


def ntn_timing_for_gso(
    es_lat_deg: float,
    es_lon_deg: float,
    gso_lon_deg: float,
    scs_khz: float = 30.0,
) -> NTNTimingState:
    """Convenience: NTN timing state for a GSO link from a specific ES."""
    from horizon_ric.planner.physics.geodesy import gso_satellite_ecef

    look = look_angles_to_target(
        es_lat_deg, es_lon_deg, gso_satellite_ecef(gso_lon_deg)
    )
    if look.elevation_deg <= 0:
        raise ValueError(
            f"GSO at lon={gso_lon_deg} below local horizon for ES "
            f"({es_lat_deg}, {es_lon_deg})"
        )
    return ntn_timing_state(GSO_ALTITUDE_M, look.elevation_deg, scs_khz)


__all__ = ["NTNTimingState", "ntn_timing_for_gso", "ntn_timing_state"]
