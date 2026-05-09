"""ITU-R S.1503-3 EPFD aggregation.

EPFD (Equivalent Power Flux Density, dBW/m²/ref-bandwidth) is the
regulatory metric for NGSO-vs-GSO coexistence. For EPFD-down at a victim
GSO earth-station, each NGSO satellite contributes:

    pfd_i = EIRP_i / (4 π d_i²)            (downlink PFD at the ES)
    g_rx_i = G_rx(φ_i)                      (ES gain at off-axis angle to NGSO)
    epfd_i = pfd_i · g_rx_i / g_rx_max     (normalised to peak ES gain)

The aggregate EPFD is the sum over all visible NGSO satellites:

    EPFD = Σ_i  pfd_i · g_rx_i / g_rx_max

This module computes a *snapshot* aggregate. The full S.1503-3 framework
also requires a time-statistical analysis (CDF over orbital propagation),
which is too heavy for control-loop work; the snapshot is what the rApp
constraint layer evaluates per decision.

Inputs:
    es_lat/lon/height                    victim earth-station
    wanted_gso_lon                       direction the ES is pointing
    es_diameter_m, es_frequency_hz       ES dish for S.1428 mask
    ngso_satellites                      list of (ECEF position, EIRP_dBW)
    reference_bandwidth_hz               normalisation bandwidth (1 MHz default)

Output:
    EPFD in dBW/m² / ref_bandwidth (one snapshot value)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from horizon_ric.planner.physics.geodesy import (
    ECEF,
    LookAngles,
    angle_between_ecef_vectors_deg,
    ecef_distance_m,
    geodetic_to_ecef,
    gso_satellite_ecef,
    look_angles_to_target,
)
from horizon_ric.planner.physics.s1428 import s1428_gain_dBi


@dataclass(frozen=True)
class NGSOEmitter:
    """One NGSO satellite contributing to the EPFD aggregate."""

    position_ecef: ECEF
    eirp_dBW: float           # EIRP toward the victim ES (dBW)
    name: str = "ngso"


@dataclass(frozen=True)
class EPFDResult:
    """Per-snapshot EPFD aggregate plus per-emitter breakdown."""

    epfd_dBW_per_m2: float                # aggregate (in 0 dB-Hz reference)
    per_emitter_dBW_per_m2: list[tuple[str, float]]
    n_visible: int                        # NGSOs above ES horizon


def _peak_es_gain_dBi(diameter_m: float, frequency_hz: float, eta: float = 0.65) -> float:
    """Replicate the S.1428 peak-gain formula for normalisation."""
    c = 299_792_458.0
    return (
        10.0 * math.log10(eta)
        + 20.0 * math.log10(math.pi * diameter_m * frequency_hz / c)
    )


def epfd_down(
    es_lat_deg: float,
    es_lon_deg: float,
    wanted_gso_lon_deg: float,
    es_diameter_m: float,
    es_frequency_hz: float,
    ngso_satellites: list[NGSOEmitter],
    es_height_m: float = 0.0,
    aperture_efficiency: float = 0.65,
    reference_bandwidth_hz: float = 1e6,
    transmit_bandwidth_hz: float = 1e6,
) -> EPFDResult:
    """Compute EPFD-down aggregate at the victim ES from all visible NGSOs.

    The classical formula (S.1503-3 Annex 1 Eq (1)):

        epfd = 10 log10( Σ_i 10^(P_i / 10) · G_rx(φ_i) / G_rx,max )

    where P_i is the per-NGSO downlink PFD at the ES (dBW/m²) computed
    from EIRP_i, slant range d_i, and the spreading factor 4πd². We add
    a bandwidth normalisation (dB) so the result is in
    dBW/m² / `reference_bandwidth_hz`.

    Args:
        es_lat_deg, es_lon_deg, es_height_m: WGS-84 victim ES location.
        wanted_gso_lon_deg: longitude of the GSO satellite the ES is
            pointing at (defines its boresight).
        es_diameter_m: ES dish diameter (m), drives the S.1428 mask.
        es_frequency_hz: receive frequency (Hz), drives the S.1428 mask.
        ngso_satellites: list of NGSO emitters with ECEF position + EIRP.
        aperture_efficiency: η for the ES dish (default 0.65 per S.1428).
        reference_bandwidth_hz: normalisation bandwidth (default 1 MHz).
        transmit_bandwidth_hz: NGSO emission bandwidth (default 1 MHz);
            EPFD is scaled by ref/tx bandwidth ratio per S.1503 §3.2.

    Returns:
        `EPFDResult` with aggregate EPFD (dBW/m² / ref_bw) and per-emitter
        breakdown for diagnostics.
    """
    # Reject zero/empty inputs early.
    if es_diameter_m <= 0:
        raise ValueError(f"es_diameter_m must be > 0, got {es_diameter_m}")
    if es_frequency_hz <= 0:
        raise ValueError(f"es_frequency_hz must be > 0, got {es_frequency_hz}")

    es_ecef = geodetic_to_ecef(es_lat_deg, es_lon_deg, es_height_m)
    boresight_target = gso_satellite_ecef(wanted_gso_lon_deg)

    # Confirm wanted GSO is above the ES horizon. If not, the ES isn't
    # actually pointing at it and EPFD aggregation is undefined.
    wanted_look = look_angles_to_target(
        es_lat_deg, es_lon_deg, boresight_target, es_height_m
    )
    if wanted_look.elevation_deg < 0.0:
        # Treat as no aggregation possible; return -inf-equivalent.
        return EPFDResult(
            epfd_dBW_per_m2=-300.0,
            per_emitter_dBW_per_m2=[],
            n_visible=0,
        )

    g_rx_max = _peak_es_gain_dBi(es_diameter_m, es_frequency_hz, aperture_efficiency)

    # Bandwidth normalisation per S.1503 §3.2.
    bw_norm_dB = 10.0 * math.log10(reference_bandwidth_hz / transmit_bandwidth_hz)

    contributions: list[tuple[str, float]] = []
    linear_sum = 0.0
    n_visible = 0

    for sat in ngso_satellites:
        # Visibility from ES.
        sat_look: LookAngles = look_angles_to_target(
            es_lat_deg, es_lon_deg, sat.position_ecef, es_height_m
        )
        if sat_look.elevation_deg < 0.0:
            continue
        n_visible += 1

        # Off-axis angle at the ES boresight: angle between
        # (ES → wanted GSO) and (ES → NGSO sat).
        phi_deg = angle_between_ecef_vectors_deg(
            es_ecef, boresight_target, sat.position_ecef
        )

        # ES receive gain at that off-axis angle.
        g_rx_phi = s1428_gain_dBi(
            phi_deg, es_diameter_m, es_frequency_hz, aperture_efficiency
        )

        # PFD at the ES from this NGSO.
        d_m = ecef_distance_m(es_ecef, sat.position_ecef)
        spreading_dB = 10.0 * math.log10(4.0 * math.pi * d_m * d_m)
        pfd_dBW_per_m2 = sat.eirp_dBW - spreading_dB

        # Per-emitter EPFD contribution (gain ratio in dB) + bandwidth.
        epfd_i_dB = pfd_dBW_per_m2 + (g_rx_phi - g_rx_max) + bw_norm_dB
        contributions.append((sat.name, epfd_i_dB))
        linear_sum += 10.0 ** (epfd_i_dB / 10.0)

    if linear_sum <= 0.0:
        return EPFDResult(
            epfd_dBW_per_m2=-300.0,
            per_emitter_dBW_per_m2=contributions,
            n_visible=n_visible,
        )
    aggregate_dB = 10.0 * math.log10(linear_sum)
    return EPFDResult(
        epfd_dBW_per_m2=aggregate_dB,
        per_emitter_dBW_per_m2=contributions,
        n_visible=n_visible,
    )


@dataclass(frozen=True)
class EPFDTimeCDF:
    """Time-statistical EPFD distribution over a finite window.

    `samples_dB` is the sorted list of EPFD snapshots taken at `step_s`
    cadence. The CDF is empirical: P(EPFD ≤ x) = (rank(x) + 1) / (N+1)
    where rank uses ITU-R S.1503 percentile convention.
    """

    samples_dB: tuple[float, ...]
    duration_s: float
    step_s: float
    n_visible_max: int
    n_visible_mean: float

    def percentile(self, p_pct: float) -> float:
        """Return the EPFD level exceeded for `p_pct` % of the window.

        ITU-R Article 22 EPFD masks are typically expressed as
        EPFD-not-to-be-exceeded for a given percentage of time
        (e.g. 0.001 % → near-worst-case spike).
        """
        if not 0.0 < p_pct < 100.0:
            raise ValueError(f"p_pct must be in (0, 100), got {p_pct}")
        if not self.samples_dB:
            return -300.0
        # Sort ascending, take the (1 − p) percentile from the top.
        sorted_db = sorted(self.samples_dB)
        # We want EPFD level that is exceeded p_pct of the time —
        # = (100 − p_pct) percentile of the empirical distribution.
        idx = int(round((1.0 - p_pct / 100.0) * (len(sorted_db) - 1)))
        idx = max(0, min(idx, len(sorted_db) - 1))
        return sorted_db[idx]

    def max_dB(self) -> float:
        return max(self.samples_dB) if self.samples_dB else -300.0

    def mean_dB(self) -> float:
        if not self.samples_dB:
            return -300.0
        # Aggregate in linear power, then back to dB (ITU convention).
        linear = [10.0 ** (s / 10.0) for s in self.samples_dB]
        return 10.0 * math.log10(sum(linear) / len(linear))


def epfd_time_cdf(
    es_lat_deg: float,
    es_lon_deg: float,
    wanted_gso_lon_deg: float,
    es_diameter_m: float,
    es_frequency_hz: float,
    constellation_at: Callable[[datetime], list[NGSOEmitter]],
    t_start_utc: datetime,
    duration_s: float = 86_400.0,
    step_s: float = 60.0,
    es_height_m: float = 0.0,
    aperture_efficiency: float = 0.65,
    reference_bandwidth_hz: float = 1e6,
    transmit_bandwidth_hz: float = 1e6,
) -> EPFDTimeCDF:
    """Time-statistical EPFD per ITU-R S.1503-3 §4.

    `constellation_at(t)` is a caller-supplied callable that returns the
    NGSOEmitters visible at time `t`. Wire this to:

        from horizon_ric.planner.physics.orbital import (
            walker_delta_constellation, kepler_j2_step, keplerian_state,
        )

    and pass a closure that propagates the elements forward and converts
    to ECEF. We deliberately do NOT bake one specific propagator in;
    callers can mix SGP4 (TLE-based) with Kepler+J2 (Walker-based) as the
    accuracy/cost trade-off requires.

    Args:
        constellation_at: callable mapping UTC datetime → list of
            NGSOEmitter(position_ecef, eirp_dBW, name). Must accept the
            datetime instance unchanged from the iteration loop.
        duration_s: total window. Default = 1 day; ITU-R Article 22
            statistical analysis uses 0.001% percentiles over a year, so
            a real compliance run wants `duration_s = 86400 * 365`.
        step_s: sampling cadence. 60 s is a reasonable default for LEO
            (sub-minute pass dynamics smooth out at this cadence); reduce
            for tighter spikes, increase to amortise propagator cost.

    Returns:
        EPFDTimeCDF with all snapshots, queryable via `.percentile()` etc.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    if step_s <= 0:
        raise ValueError(f"step_s must be > 0, got {step_s}")

    n_steps = int(duration_s / step_s) + 1
    samples: list[float] = []
    n_visible_total = 0
    n_visible_max = 0

    for i in range(n_steps):
        t = t_start_utc + timedelta(seconds=i * step_s)
        emitters = constellation_at(t)
        snapshot = epfd_down(
            es_lat_deg=es_lat_deg,
            es_lon_deg=es_lon_deg,
            wanted_gso_lon_deg=wanted_gso_lon_deg,
            es_diameter_m=es_diameter_m,
            es_frequency_hz=es_frequency_hz,
            ngso_satellites=emitters,
            es_height_m=es_height_m,
            aperture_efficiency=aperture_efficiency,
            reference_bandwidth_hz=reference_bandwidth_hz,
            transmit_bandwidth_hz=transmit_bandwidth_hz,
        )
        samples.append(snapshot.epfd_dBW_per_m2)
        n_visible_total += snapshot.n_visible
        n_visible_max = max(n_visible_max, snapshot.n_visible)

    return EPFDTimeCDF(
        samples_dB=tuple(samples),
        duration_s=duration_s,
        step_s=step_s,
        n_visible_max=n_visible_max,
        n_visible_mean=n_visible_total / max(n_steps, 1),
    )


__all__ = [
    "EPFDResult",
    "EPFDTimeCDF",
    "NGSOEmitter",
    "epfd_down",
    "epfd_time_cdf",
]
