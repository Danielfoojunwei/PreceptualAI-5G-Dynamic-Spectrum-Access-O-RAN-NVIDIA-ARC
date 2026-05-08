"""Multi-band coexistence physics (NTN ↔ terrestrial ↔ WiFi ↔ ISAC).

Six closed-form predictors used by the constraint layer to keep policy
decisions feasible across all bands at once. Each function cites its
spec section + paragraph and rejects pathological inputs rather than
silently returning a misleading number.

References:
    ITU-R S.1325-3 §3.2 — NGSO event-based methodology.
    ITU-R RR Article 9.7A — NGSO-NGSO coordination trigger.
    3GPP TS 38.108 — NTN UE radio.
    3GPP TS 38.104 §6.6, §7.4–7.5 — BS ACLR + ACS limits.
    3GPP TS 38.101-1 §6.6.2 — UE ACLR.
    ETSI EN 303 687 §4.2.7 — LBT in 6 GHz UNII.
    3GPP TS 37.213 §4 — Cat-4 LBT for NR-U.
    Bianchi G., *Performance Analysis of the IEEE 802.11 DCF*,
        IEEE J-SAC 2000.
    ITU-R P.1815-2 §3 — joint exceedance, rain spatial correlation.
    ITU-R P.452-17 §4.2 — clear-air interference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ─── 1. NGSO ↔ NGSO in-line interference ────────────────────────────────


@dataclass(frozen=True)
class InlineEventStat:
    """Snapshot statistics of NGSO-vs-NGSO in-line risk."""

    p_inline: float
    """Probability ≥1 in-line event at any instant ∈ [0, 1]."""
    expected_seconds_per_orbit: float
    """Expected total in-line duration per 90-min orbital pass."""
    n_visible_a: int
    n_visible_b: int


def ngso_inline_event_probability(
    visible_a: int,
    visible_b: int,
    in_line_angle_deg: float = 1.0,
    min_elevation_deg: float = 25.0,
    pass_duration_s: float = 5400.0,
) -> InlineEventStat:
    """Probability that any sat in B falls within `in_line_angle_deg` of any
    sat in A, as seen from a fixed earth station, at any instant.

    Per ITU-R S.1325-3 §3.2, model A and B as Poisson distributions
    uniformly visible above the ES horizon over solid angle
        Ω_vis = 2π · (1 − cos(90° − ε_min)).
    Cone solid angle around one B sat:  ω(α) = 2π · (1 − cos α).

    p_per_pair  ≈ ω(α) / Ω_vis
    p_inline   ≈ 1 − exp(−N_A · N_B · p_per_pair)

    Returns an `InlineEventStat` with probability and expected per-orbit
    duration (probability × pass_duration). The result is the regime the
    rApp must coordinate the ITU-R RR Article 9.7A coordination request
    against.
    """
    if visible_a < 0 or visible_b < 0:
        raise ValueError("visible_a and visible_b must be non-negative")
    if not 0.0 < in_line_angle_deg < 180.0:
        raise ValueError(
            f"in_line_angle_deg must be in (0,180), got {in_line_angle_deg}"
        )
    if not 0.0 <= min_elevation_deg < 90.0:
        raise ValueError(
            f"min_elevation_deg must be in [0,90), got {min_elevation_deg}"
        )

    half_visible_cap_rad = math.radians(90.0 - min_elevation_deg)
    omega_visible_sr = 2.0 * math.pi * (1.0 - math.cos(half_visible_cap_rad))
    cone_sr = 2.0 * math.pi * (1.0 - math.cos(math.radians(in_line_angle_deg)))
    if omega_visible_sr <= 0:
        return InlineEventStat(0.0, 0.0, visible_a, visible_b)

    p_per_pair = cone_sr / omega_visible_sr
    p_inline = 1.0 - math.exp(-visible_a * visible_b * p_per_pair)
    return InlineEventStat(
        p_inline=p_inline,
        expected_seconds_per_orbit=p_inline * pass_duration_s,
        n_visible_a=visible_a,
        n_visible_b=visible_b,
    )


# ─── 2. NTN ↔ terrestrial guard-band ────────────────────────────────────


def ntn_to_terrestrial_required_guard_db(
    ue_tx_power_dBm: float,
    ue_back_lobe_gain_dBi: float,
    distance_to_gnb_m: float,
    frequency_hz: float,
    gnb_acs_dBm_per_mhz: float = -98.0,
    nlos_path_loss_offset_dB: float = 0.0,
) -> float:
    """How many dB of OOB suppression are needed at the UE to keep its
    upward-pointing NTN UL emission below the terrestrial gNB's ACS limit.

    First-order: free-space path loss + fixed UMa-NLOS offset. The UE
    transmits +ue_tx_power_dBm with effective antenna gain
    ue_back_lobe_gain_dBi towards the horizon; the gNB sees:

        I_dBm = P_UE + G_UE_back − PL(d, f)

    Required ACLR/OOB suppression = max(0, I_dBm − ACS_dBm/MHz).
    """
    if distance_to_gnb_m <= 0 or frequency_hz <= 0:
        raise ValueError("distance_to_gnb_m and frequency_hz must be positive")
    # FSPL in dB
    fspl = 20.0 * math.log10(
        4.0 * math.pi * distance_to_gnb_m * frequency_hz / 299_792_458.0
    )
    interference_dBm = (
        ue_tx_power_dBm + ue_back_lobe_gain_dBi - fspl - nlos_path_loss_offset_dB
    )
    return max(0.0, interference_dBm - gnb_acs_dBm_per_mhz)


# ─── 3. WiFi 6E/7 ↔ NR-U airtime under LBT ──────────────────────────────


def nru_fair_share_airtime(
    n_wifi_stations: int,
    wifi_tau: float = 0.04,
    cross_rat_deferral_pct: float = 9.0,
) -> float:
    """Bianchi-2000 saturation throughput estimate of how much 6 GHz airtime
    NR-U can claim under fair-share LBT.

    ρ_w = (N_w·τ_w·(1−τ_w)^(N_w−1)) /
          (1 − (1−τ_w)^N_w + N_w·τ_w·(1−τ_w)^(N_w−1))
    NR-U effective TXOP fraction = (1 − ρ_w) · (1 − P_def)

    Returns the NR-U airtime as a fraction in [0, 1].
    """
    if n_wifi_stations < 0:
        raise ValueError("n_wifi_stations must be non-negative")
    if not 0.0 <= cross_rat_deferral_pct <= 100.0:
        raise ValueError("cross_rat_deferral_pct must be in [0,100]")
    if n_wifi_stations == 0:
        return 1.0 - cross_rat_deferral_pct / 100.0

    n = n_wifi_stations
    t = wifi_tau
    numer = n * t * (1.0 - t) ** (n - 1)
    denom = 1.0 - (1.0 - t) ** n + numer
    rho_w = numer / denom if denom > 0 else 0.0
    return max(0.0, (1.0 - rho_w) * (1.0 - cross_rat_deferral_pct / 100.0))


# ─── 4. Aggregate ACLR leakage from co-scheduled UEs ────────────────────


def aggregate_aclr_leakage_dBm(
    ue_powers_dBm: list[float],
    ue_aclr_dB: float = 30.0,
    victim_path_loss_dB: float = 0.0,
) -> float:
    """Per-UE first-adjacent leakage = P_UE − ACLR_UE; aggregate is the
    log-sum of linear powers (TS 38.101-1 §6.6.2.1.1)."""
    if not ue_powers_dBm:
        return -300.0
    if ue_aclr_dB < 0:
        raise ValueError("ue_aclr_dB must be ≥ 0")
    leakages_linear = [
        10.0 ** ((p - ue_aclr_dB - victim_path_loss_dB) / 10.0)
        for p in ue_powers_dBm
    ]
    total_linear = sum(leakages_linear)
    return 10.0 * math.log10(total_linear) if total_linear > 0 else -300.0


# ─── 5. Conditional rain probability (gateway → service link) ───────────


def p_servicelink_rain_given_gateway(
    distance_km: float,
    rain_rate_gateway_mm_hr: float,
    rain_rate_threshold_mm_hr: float = 5.0,
    mu_log_rain: float = 0.0,
    sigma_log_rain: float = 1.0,
) -> float:
    """ITU-R P.1815 / P.452 conditional rain-rate exceedance.

    Bivariate log-normal model on rain rate. Correlation:
        ρ(d) = 0.59·exp(−d/31) + 0.41·exp(−d/800)   (km, ITU-R P.1815-2 §3)

    Returns P(R_service > r_2 | R_gateway > r_1) ∈ [0, 1]. The rApp uses
    this to pre-emptively shift load to a different gateway when heavy
    rain is detected at the current one.
    """
    if distance_km < 0:
        raise ValueError(f"distance_km must be ≥ 0, got {distance_km}")
    if rain_rate_gateway_mm_hr <= 0 or rain_rate_threshold_mm_hr <= 0:
        return 0.0

    # P.1815 correlation envelope.
    rho = 0.59 * math.exp(-distance_km / 31.0) + 0.41 * math.exp(-distance_km / 800.0)
    rho = max(min(rho, 1.0 - 1e-9), 0.0)

    z1 = (math.log(rain_rate_gateway_mm_hr) - mu_log_rain) / sigma_log_rain
    z2 = (math.log(rain_rate_threshold_mm_hr) - mu_log_rain) / sigma_log_rain
    arg = (rho * z1 - z2) / math.sqrt(max(1.0 - rho * rho, 1e-12))

    # Standard normal CDF via erf.
    return 0.5 * (1.0 + math.erf(arg / math.sqrt(2.0)))


# ─── 6. Feasibility envelope composer ───────────────────────────────────


@dataclass(frozen=True)
class BandEnvelope:
    """Operating envelope for one band/regulatory regime.

    `name` matches a band name your scheduler knows about. `f_low_hz`
    and `f_high_hz` define the spectral extent. `max_eirp_dBW` is the
    instantaneous EIRP cap. Optional callbacks let callers express
    time- or geometry-dependent caps (e.g. EPFD masks vary with the
    off-axis angle from the GSO arc).
    """

    name: str
    f_low_hz: float
    f_high_hz: float
    max_eirp_dBW: float
    spec_reference: str
    note: str = ""


def compose_feasibility(
    envelopes: list[BandEnvelope],
    proposed_freq_hz: float,
    proposed_eirp_dBW: float,
) -> BandEnvelope | None:
    """Find the first envelope (if any) that admits a (frequency, EIRP)
    proposal; returns None when no band fits.

    Linear-time scan; the rApp typically has < 20 envelopes (one per
    licensed band per region). For tighter validation use the dedicated
    `policy.constraints` projection layer.
    """
    if not envelopes:
        return None
    for env in envelopes:
        if env.f_low_hz <= proposed_freq_hz <= env.f_high_hz:
            if proposed_eirp_dBW <= env.max_eirp_dBW:
                return env
    return None


__all__ = [
    "BandEnvelope",
    "InlineEventStat",
    "aggregate_aclr_leakage_dBm",
    "compose_feasibility",
    "ngso_inline_event_probability",
    "ntn_to_terrestrial_required_guard_db",
    "nru_fair_share_airtime",
    "p_servicelink_rain_given_gateway",
]
