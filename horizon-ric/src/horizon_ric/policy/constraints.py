"""Hard constraint layer — GSO PFD + ITU spectral mask + edge GPU ceiling.

Implements the differentiable projection layer for PreceptualAI. Three classes
of constraint:

1. GSO arc PFD ≤ -140 dBW (regulatory, ITU-R Article 22 + S.1503).
2. ITU spectral mask compliance (regulatory).
3. Edge GPU memory + bandwidth ceilings (operational, but absolute).

References:
    ITU-R Article 22       — EPFD masks for NGSO-vs-GSO coexistence
    ITU-R S.1503-3 (2018)  — Functional description of EPFD validation
    FCC 47 CFR §25.146     — NGSO EPFD compliance (US)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from horizon_ric.contracts.constraint_layer import (
    ConstraintLayer,
    ConstraintViolation,
)
from horizon_ric.planner.physics.beam_pattern import (
    beam_gain_dB,
)
from horizon_ric.planner.physics.epfd import NGSOEmitter, epfd_down
from horizon_ric.planner.physics.geodesy import (
    GSO_ALTITUDE_M,
    WGS84_A_M,
    look_angles_to_gso,
)
from horizon_ric.planner.physics.propagation import free_space_path_loss_dB


@dataclass
class GSOArcEntry:
    """One GSO satellite's geostationary slot for the constraint check."""

    satellite_id: str
    longitude_deg: float  # GSO longitude (positive east)
    operator: str = "unknown"


@dataclass
class PreceptualAIConstraintConfig:
    """Configuration for the PreceptualAI constraint layer.

    The PFD floor is a single representative value; for real Article 22
    compliance use the band-specific EPFD masks loaded into
    `epfd_floor_dBW_per_m2_per_ref_bw`. When that mask is provided AND
    `victim_es` is configured, the EPFD aggregator runs in addition to
    the per-arc PFD check.
    """

    # Single-arc PFD floor (representative, used as a fast pre-check).
    pfd_floor_dBW_per_m2: float = -140.0

    # WGS-84 ellipsoid (legacy constants kept for backward compatibility;
    # geometry now uses `planner.physics.geodesy` directly).
    earth_radius_km: float = WGS84_A_M / 1000.0
    gso_altitude_km: float = GSO_ALTITUDE_M / 1000.0

    # Edge-compute capacity. Defaults are intentionally low (Jetson Orin
    # Nano per project memory); override per deployment.
    edge_gpu_memory_gb_max: float = 8.0
    edge_gpu_bandwidth_gbps_max: float = 102.0

    # GSO arcs to consider for the per-arc PFD pre-check.
    gso_arcs: list[GSOArcEntry] = field(default_factory=list)

    # EPFD-down compliance (ITU-R S.1503-3, Article 22 EPFD masks).
    # When `epfd_floor_dBW_per_m2_per_ref_bw` is None, EPFD aggregation
    # is skipped (only per-arc PFD enforced).
    epfd_floor_dBW_per_m2_per_ref_bw: float | None = None
    epfd_reference_bandwidth_hz: float = 1e6
    epfd_transmit_bandwidth_hz: float = 1e6
    victim_es_diameter_m: float = 1.2
    victim_es_aperture_efficiency: float = 0.65


class PreceptualAIConstraintLayer(ConstraintLayer):
    """Hard-constraint layer for the NTN+AI-RAN vertical.

    Enforces three classes:
        1. GSO arc PFD floor (regulatory).
        2. ITU spectral mask (regulatory; simplified to band edges).
        3. Edge GPU capacity ceiling (operational).

    Verifier composition
    --------------------
    Pass ``verifier_chain=["gso_pfd", "itu_spectral_mask", ...]`` (and
    optionally ``verifier_configs={...}``) at construction time to delegate
    feasibility to the named registry entries. The default chain is
    ``None``, in which case the layer uses the in-class implementation
    below (preserving existing behaviour for legacy callers).
    """

    def __init__(
        self,
        config: PreceptualAIConstraintConfig | None = None,
        *,
        verifier_chain: list[str] | None = None,
        verifier_configs: dict[str, Any] | None = None,
    ):
        self.cfg = config or PreceptualAIConstraintConfig()
        self._verifier_chain = list(verifier_chain) if verifier_chain else None
        self._verifier_configs = dict(verifier_configs or {})

    # ─── ConstraintLayer contract ────────────────────────────────────

    def hard_constraint_ids(self) -> list[str]:
        ids = ["gso_pfd_floor", "itu_spectral_mask", "edge_gpu_capacity"]
        if self.cfg.epfd_floor_dBW_per_m2_per_ref_bw is not None:
            ids.append("itu_epfd_down")
        return ids

    def check_feasibility(
        self, action: dict[str, Any], context: dict[str, Any]
    ) -> list[ConstraintViolation]:
        """Return list of all violations for the proposed action."""
        if self._verifier_chain is not None:
            return self._check_via_chain(action, context)
        violations: list[ConstraintViolation] = []

        # 1. GSO PFD check
        max_pfd = self._max_pfd_at_any_gso(action, context)
        margin = self.cfg.pfd_floor_dBW_per_m2 - max_pfd
        if max_pfd > self.cfg.pfd_floor_dBW_per_m2:
            violations.append(
                ConstraintViolation(
                    constraint_id="gso_pfd_floor",
                    severity="hard",
                    margin_dB=margin,  # negative when violated
                    message=(
                        f"PFD at nearest GSO arc {max_pfd:.2f} dBW/m² exceeds "
                        f"floor {self.cfg.pfd_floor_dBW_per_m2:.2f} dBW/m² "
                        f"by {-margin:.2f} dB."
                    ),
                )
            )

        # 2. ITU spectral mask (simplified: band-edge check)
        freq_hz = action.get("frequency_hz", 0.0)
        band = context.get("permitted_band_hz", (0.0, 1e12))
        if freq_hz < band[0] or freq_hz > band[1]:
            violations.append(
                ConstraintViolation(
                    constraint_id="itu_spectral_mask",
                    severity="hard",
                    margin_dB=None,
                    message=(
                        f"frequency {freq_hz / 1e9:.3f} GHz outside permitted "
                        f"band {band[0] / 1e9:.3f}–{band[1] / 1e9:.3f} GHz."
                    ),
                )
            )

        # 3. Edge GPU capacity
        gpu_demand_gb = action.get("ai_workload_gpu_gb", 0.0)
        gpu_demand_gbps = action.get("ai_workload_gpu_bw_gbps", 0.0)
        if gpu_demand_gb > self.cfg.edge_gpu_memory_gb_max:
            violations.append(
                ConstraintViolation(
                    constraint_id="edge_gpu_capacity",
                    severity="hard",
                    margin_dB=None,
                    message=(
                        f"requested GPU memory {gpu_demand_gb:.1f} GB exceeds "
                        f"edge ceiling {self.cfg.edge_gpu_memory_gb_max:.1f} GB."
                    ),
                )
            )
        if gpu_demand_gbps > self.cfg.edge_gpu_bandwidth_gbps_max:
            violations.append(
                ConstraintViolation(
                    constraint_id="edge_gpu_capacity",
                    severity="hard",
                    margin_dB=None,
                    message=(
                        f"requested GPU bandwidth {gpu_demand_gbps:.1f} Gbps exceeds "
                        f"edge ceiling {self.cfg.edge_gpu_bandwidth_gbps_max:.1f} Gbps."
                    ),
                )
            )

        # 4. EPFD-down (ITU-R S.1503-3 / Article 22) — only runs when both a
        # mask floor and an NGSO emitter list are provided in context.
        epfd_floor = self.cfg.epfd_floor_dBW_per_m2_per_ref_bw
        ngso_emitters: list[NGSOEmitter] | None = context.get("ngso_emitters")
        wanted_gso_lon = context.get("wanted_gso_longitude_deg")
        if (
            epfd_floor is not None
            and ngso_emitters
            and wanted_gso_lon is not None
        ):
            es_lat = float(context.get("earth_station_latitude_deg", 0.0))
            es_lon = float(context.get("earth_station_longitude_deg", 0.0))
            es_freq = float(action.get("frequency_hz", 12e9))
            result = epfd_down(
                es_lat_deg=es_lat,
                es_lon_deg=es_lon,
                wanted_gso_lon_deg=float(wanted_gso_lon),
                es_diameter_m=self.cfg.victim_es_diameter_m,
                es_frequency_hz=es_freq,
                ngso_satellites=list(ngso_emitters),
                aperture_efficiency=self.cfg.victim_es_aperture_efficiency,
                reference_bandwidth_hz=self.cfg.epfd_reference_bandwidth_hz,
                transmit_bandwidth_hz=self.cfg.epfd_transmit_bandwidth_hz,
            )
            margin = epfd_floor - result.epfd_dBW_per_m2
            if result.epfd_dBW_per_m2 > epfd_floor:
                violations.append(
                    ConstraintViolation(
                        constraint_id="itu_epfd_down",
                        severity="hard",
                        margin_dB=margin,
                        message=(
                            f"aggregate EPFD {result.epfd_dBW_per_m2:.2f} dBW/m² across "
                            f"{result.n_visible} visible NGSO sat(s) exceeds Article 22 "
                            f"floor {epfd_floor:.2f} dBW/m² by {-margin:.2f} dB."
                        ),
                    )
                )

        return violations

    def project(
        self, action: dict[str, Any], context: dict[str, Any]
    ) -> tuple[dict[str, Any], list[ConstraintViolation]]:
        """Project an infeasible action onto the feasible set."""
        if self._verifier_chain is not None:
            return self._project_via_chain(action, context)
        feasible = dict(action)
        corrections: list[ConstraintViolation] = []

        # 1. GSO PFD: reduce EIRP until PFD floor is satisfied. We assert
        # monotone decrease per iteration; if it ever fails to decrease we
        # disable the transmitter rather than loop forever.
        prev_pfd = float("inf")
        max_iter = 30
        converged = False
        for _ in range(max_iter):
            current_pfd = self._max_pfd_at_any_gso(feasible, context)
            if current_pfd <= self.cfg.pfd_floor_dBW_per_m2:
                converged = True
                break
            if current_pfd >= prev_pfd - 1e-6:
                # Not making progress (e.g. action has no controllable EIRP term).
                break
            prev_pfd = current_pfd
            drop_dB = current_pfd - self.cfg.pfd_floor_dBW_per_m2 + 0.5  # 0.5 dB margin
            new_power = feasible.get("tx_power_dBm", 0.0) - drop_dB
            corrections.append(
                ConstraintViolation(
                    constraint_id="gso_pfd_floor",
                    severity="hard",
                    margin_dB=drop_dB,
                    message=(
                        f"projected: TX power reduced by {drop_dB:.2f} dB to "
                        f"satisfy GSO arc PFD floor."
                    ),
                )
            )
            feasible["tx_power_dBm"] = new_power
        if not converged:
            # Hard refusal — disable transmitter via an explicit flag so
            # downstream math doesn't see -inf and crash.
            feasible["transmitter_enabled"] = False
            feasible["tx_power_dBm"] = -1000.0  # finite floor in dBm; unambiguous "off"
            corrections.append(
                ConstraintViolation(
                    constraint_id="gso_pfd_floor",
                    severity="hard",
                    margin_dB=None,
                    message=(
                        "projection failed: action cannot satisfy GSO PFD floor. "
                        "Transmitter disabled (transmitter_enabled=False)."
                    ),
                )
            )

        # 2. Spectral mask: clip frequency to permitted band.
        band = context.get("permitted_band_hz", (0.0, 1e12))
        freq_hz = feasible.get("frequency_hz", 0.0)
        if freq_hz < band[0]:
            feasible["frequency_hz"] = band[0]
            corrections.append(
                ConstraintViolation(
                    constraint_id="itu_spectral_mask",
                    severity="hard",
                    margin_dB=None,
                    message=f"frequency clipped to band low {band[0] / 1e9:.3f} GHz",
                )
            )
        elif freq_hz > band[1]:
            feasible["frequency_hz"] = band[1]
            corrections.append(
                ConstraintViolation(
                    constraint_id="itu_spectral_mask",
                    severity="hard",
                    margin_dB=None,
                    message=f"frequency clipped to band high {band[1] / 1e9:.3f} GHz",
                )
            )

        # 3. Edge GPU: deny requests over ceiling (cannot magically conjure compute).
        gpu_demand = feasible.get("ai_workload_gpu_gb", 0.0)
        if gpu_demand > self.cfg.edge_gpu_memory_gb_max:
            feasible["ai_workload_gpu_gb"] = 0.0
            feasible["workload_placement"] = "regional_edge_or_cloud"
            corrections.append(
                ConstraintViolation(
                    constraint_id="edge_gpu_capacity",
                    severity="hard",
                    margin_dB=None,
                    message=(
                        "AI workload exceeds local edge GPU ceiling; "
                        "deferred to regional_edge_or_cloud."
                    ),
                )
            )

        return feasible, corrections

    def lagrangian_violation(
        self, action: torch.Tensor, context: dict[str, Any]
    ) -> torch.Tensor:
        """Differentiable Lagrangian signal for CSAC training.

        Action convention (B, action_dim):
            action[:, 0] = tx_power_dBm
            action[:, 1] = antenna_max_gain_dBi
            action[:, 2] = beam_az_deg
            action[:, 3] = beam_el_deg

        We compute a torch-native PFD surrogate that varies with all four
        dimensions:

            EIRP_dBW   = tx_power_dBm − 30 + antenna_max_gain_dBi
            G_off-axis = max_gain − 12 ((Δaz/HPBW)² + (Δel/HPBW)²)  (clipped)
            spreading  = 10 log10(4π d_GSO²)
            PFD        = EIRP + G_off_axis − spreading

        where ``Δaz``, ``Δel`` are the angles between the beam pointing and the
        worst-case GSO arc longitude (configured via context["gso_arcs_deg"] or
        ``self.cfg.gso_arcs``). Returns ReLU(PFD − floor).

        Args:
            action: (B, ≥4) tensor.
            context: optional ``earth_station_latitude_deg``,
                     ``earth_station_longitude_deg``, ``gso_arcs_deg``.
        Returns:
            (B,) non-negative tensor of PFD violation magnitude in dB.
        """
        if action.ndim != 2:
            raise ValueError(f"action must be (B, action_dim), got {tuple(action.shape)}")
        if action.shape[1] < 4:
            raise ValueError(
                f"action must have ≥4 dims (tx_dBm, gain_dBi, beam_az, beam_el); "
                f"got {action.shape[1]}"
            )
        B = action.shape[0]

        tx_power_dBm = action[:, 0]
        gain_dBi = action[:, 1]
        beam_az = action[:, 2]
        beam_el = action[:, 3]

        # Pull GSO arc longitudes from context or config.
        arc_lons_deg = context.get("gso_arcs_deg")
        if arc_lons_deg is None and self.cfg.gso_arcs:
            arc_lons_deg = [a.longitude_deg for a in self.cfg.gso_arcs]
        if not arc_lons_deg:
            # No GSO arcs to consider → no PFD constraint active.
            return torch.zeros(B, dtype=action.dtype, device=action.device)

        es_lat = float(context.get("earth_station_latitude_deg", 0.0))
        es_lon = float(context.get("earth_station_longitude_deg", 0.0))

        # Compute az/el to each arc (host-side; small constant cost).
        arc_az = torch.tensor(
            [self._gso_azimuth_from_es(es_lat, es_lon, lon) for lon in arc_lons_deg],
            dtype=action.dtype,
            device=action.device,
        )  # (A,)
        arc_el = torch.tensor(
            [self._gso_elevation_from_es(es_lat, es_lon, lon) for lon in arc_lons_deg],
            dtype=action.dtype,
            device=action.device,
        )  # (A,)
        slant_km = torch.tensor(
            [self._slant_range_km(es_lat, es_lon, lon) for lon in arc_lons_deg],
            dtype=action.dtype,
            device=action.device,
        )  # (A,)

        # Off-axis angle between (beam_az, beam_el) and (arc_az, arc_el),
        # both broadcast to (B, A).
        d_az = ((beam_az[:, None] - arc_az[None, :] + 180.0) % 360.0) - 180.0
        d_el = beam_el[:, None] - arc_el[None, :]

        # 3GPP 38.901 single-element approximation: A = 12 ((Δ/HPBW)²),
        # capped at 30 dB attenuation.
        hpbw_deg = 65.0  # default 38.901 single-element HPBW
        att_dB = 12.0 * ((d_az / hpbw_deg) ** 2 + (d_el / hpbw_deg) ** 2)
        att_dB = att_dB.clamp(max=30.0)
        gain_at_arc_dBi = gain_dBi[:, None] - att_dB  # (B, A)

        # Mask arcs below horizon (no contribution).
        visible = (arc_el >= 0.0).to(action.dtype)  # (A,)

        # Spreading loss 10 log10(4π d²) with d in metres.
        slant_m = slant_km * 1000.0
        spreading_dB = 10.0 * torch.log10(
            4.0 * torch.pi * slant_m * slant_m
        )  # (A,)

        eirp_dBW = (tx_power_dBm - 30.0)[:, None] + gain_at_arc_dBi  # (B, A)
        pfd_dBW_per_m2 = eirp_dBW - spreading_dB[None, :]  # (B, A)

        # Worst-case across visible arcs.
        very_low = pfd_dBW_per_m2.new_full(pfd_dBW_per_m2.shape, -300.0)
        masked = torch.where(visible[None, :].bool(), pfd_dBW_per_m2, very_low)
        worst_pfd, _ = masked.max(dim=1)  # (B,)

        floor = action.new_full((B,), self.cfg.pfd_floor_dBW_per_m2)
        return torch.relu(worst_pfd - floor)

    # ─── Internal helpers ────────────────────────────────────────────

    def _max_pfd_at_any_gso(
        self, action: dict[str, Any], context: dict[str, Any]
    ) -> float:
        """Compute max PFD across all known GSO arcs for the proposed action.

        Returns:
            PFD in dBW/m² at the worst-case GSO arc.
        """
        eirp_dBm = action.get("tx_power_dBm", 0.0) + action.get("antenna_gain_dBi", 0.0)
        eirp_dBW = eirp_dBm - 30.0  # dBm → dBW
        if eirp_dBW <= -200:
            return -300.0  # transmitter effectively off

        # Earth station position (ECEF longitudes are simplest)
        es_lat = context.get("earth_station_latitude_deg", 0.0)
        es_lon = context.get("earth_station_longitude_deg", 0.0)
        beam_az = action.get("beam_azimuth_deg", 0.0)
        beam_el = action.get("beam_elevation_deg", 90.0)
        freq_hz = action.get("frequency_hz", 4e9)

        worst_pfd = -300.0
        for arc in self.cfg.gso_arcs:
            # GSO subpoint at equator at given longitude. Compute az/el from ES.
            gso_az_from_es = self._gso_azimuth_from_es(es_lat, es_lon, arc.longitude_deg)
            gso_el_from_es = self._gso_elevation_from_es(es_lat, es_lon, arc.longitude_deg)
            if gso_el_from_es < 0:
                # GSO satellite below ES horizon → not visible, no PFD contribution.
                continue

            # Antenna gain in the direction of this GSO. The 38.901 pattern
            # already encodes the off-axis fall-off via the angle between the
            # beam boresight and the (gso_az, gso_el) direction.
            gain_at_gso_dBi = beam_gain_dB(
                azimuth_deg=gso_az_from_es,
                elevation_deg=gso_el_from_es,
                boresight_az_deg=beam_az,
                boresight_el_deg=beam_el,
                max_gain_dBi=action.get("antenna_gain_dBi", 14.0),
            )

            # Slant range from earth station to GSO satellite.
            slant_km = self._slant_range_km(es_lat, es_lon, arc.longitude_deg)
            if slant_km <= 0:
                continue
            slant_m = slant_km * 1000.0

            # PFD at the GSO satellite footprint:
            #   PFD = EIRP / (4 π d²)  → in dB: EIRP_dBW − 10 log10(4 π d²)
            spreading_dB = 10.0 * math.log10(4.0 * math.pi * slant_m * slant_m)
            pfd = eirp_dBW + gain_at_gso_dBi - spreading_dB

            # The free-space path loss is captured implicitly in the spreading
            # term for PFD (which is power per unit area, not received power).
            # We only compute fspl_dB here as a sanity check that the slant
            # geometry is consistent; it is not added to PFD.
            _ = free_space_path_loss_dB(slant_m, freq_hz)

            worst_pfd = max(worst_pfd, pfd)

        return worst_pfd

    def _check_via_chain(self, action, context):
        from horizon_ric.policy.verifier_registry import get_verifier
        out: list[ConstraintViolation] = []
        for name in self._verifier_chain:  # type: ignore[union-attr]
            cfg = self._verifier_configs.get(name, self.cfg)
            v = get_verifier(name, cfg)
            out.extend(v.check(action, context))
        return out

    def _project_via_chain(self, action, context):
        from horizon_ric.policy.verifier_registry import get_verifier
        feasible = dict(action)
        corrections: list[ConstraintViolation] = []
        for name in self._verifier_chain:  # type: ignore[union-attr]
            cfg = self._verifier_configs.get(name, self.cfg)
            v = get_verifier(name, cfg)
            feasible, corr = v.project(feasible, context)
            corrections.extend(corr)
        return feasible, corrections

    def _gso_azimuth_from_es(
        self, es_lat_deg: float, es_lon_deg: float, gso_lon_deg: float
    ) -> float:
        """Azimuth (deg, [0,360) clockwise from true north) — WGS-84 geometry."""
        return look_angles_to_gso(
            es_lat_deg, es_lon_deg, gso_lon_deg,
            gso_altitude_m=self.cfg.gso_altitude_km * 1000.0,
        ).azimuth_deg

    def _gso_elevation_from_es(
        self, es_lat_deg: float, es_lon_deg: float, gso_lon_deg: float
    ) -> float:
        """Elevation (deg, [-90, 90]) — WGS-84 geometry, returns negative
        when the GSO sub-point is below the ES local horizon."""
        return look_angles_to_gso(
            es_lat_deg, es_lon_deg, gso_lon_deg,
            gso_altitude_m=self.cfg.gso_altitude_km * 1000.0,
        ).elevation_deg

    def _slant_range_km(
        self, es_lat_deg: float, es_lon_deg: float, gso_lon_deg: float
    ) -> float:
        """Slant range km — WGS-84 ECEF straight-line distance."""
        return (
            look_angles_to_gso(
                es_lat_deg, es_lon_deg, gso_lon_deg,
                gso_altitude_m=self.cfg.gso_altitude_km * 1000.0,
            ).range_m
            / 1000.0
        )
