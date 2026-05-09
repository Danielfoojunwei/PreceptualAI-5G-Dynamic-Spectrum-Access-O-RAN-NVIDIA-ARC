"""Maritime vertical playbook — thin example over the generic NTN + AI-RAN
framework.

Maritime is one example vertical, not the framework's centre of gravity.
See `ntn_air_ran.py` for the generic, vertical-neutral generator and
`rural.py` / `disaster.py` / `enterprise.py` for sibling examples.

Backward compatibility: `MaritimeScenarioConfig`, `MaritimeWindow`, and
`MaritimeSyntheticGenerator` keep their public surface so older callers and
tests do not need to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from horizon_ric.scenarios.ntn_air_ran import (
    NTNAIRANConfig,
    NTNAIRANSyntheticGenerator,
)


@dataclass
class MaritimeScenarioConfig:
    """Maritime-flavoured config. Defaults are Port of Rotterdam (Maasvlakte
    II). The Frankfurt landlocked-default trap is rejected at construction.
    """

    port_lat: float = 51.9244
    port_lon: float = 4.4777
    num_terrestrial_cells: int = 8
    num_ships: int = 12
    num_ai_cameras_per_ship: int = 4

    edge_gpu_count: int = 6
    edge_gpu_memory_gb: float = 8.0

    ntn_beam_count: int = 3
    ntn_capacity_mbps_per_beam: float = 200.0
    ntn_path_loss_dB_nominal: float = 175.0

    window_seconds: int = 300
    step_seconds: int = 5

    ship_arrival_lambda_per_hour: float = 4.0
    weather_storm_probability: float = 0.05
    sla_breach_threshold: float = 0.85
    sla_breach_slope: float = 8.0
    rng_seed: int = 0

    synthetic_only: bool = True

    def __post_init__(self) -> None:
        if abs(self.port_lat - 50.110) < 1e-3 and abs(self.port_lon - 8.682) < 1e-3:
            raise ValueError(
                "port at (50.110, 8.682) is Frankfurt — landlocked. Choose a "
                "real port (defaults are now Rotterdam). This refusal is an "
                "explicit guard against the prior placeholder."
            )

    def to_generic(self) -> NTNAIRANConfig:
        """Project the maritime-flavoured fields onto the generic config."""
        return NTNAIRANConfig(
            site_lat=self.port_lat,
            site_lon=self.port_lon,
            n_ground_cells=self.num_terrestrial_cells,
            n_user_terminals=self.num_ships,
            n_ai_workloads_per_terminal=self.num_ai_cameras_per_ship,
            edge_gpu_count=self.edge_gpu_count,
            edge_gpu_memory_gb=self.edge_gpu_memory_gb,
            ntn_beam_count=self.ntn_beam_count,
            ntn_capacity_mbps_per_beam=self.ntn_capacity_mbps_per_beam,
            ntn_path_loss_dB_nominal=self.ntn_path_loss_dB_nominal,
            window_seconds=self.window_seconds,
            step_seconds=self.step_seconds,
            arrival_lambda_per_hour=self.ship_arrival_lambda_per_hour,
            weather_storm_probability=self.weather_storm_probability,
            sla_breach_threshold=self.sla_breach_threshold,
            sla_breach_slope=self.sla_breach_slope,
            rng_seed=self.rng_seed,
            synthetic_only=self.synthetic_only,
        )


@dataclass
class MaritimeWindow:
    """Maritime-flavoured window. Field names preserved for back-compat;
    this is the maritime alias for NTNAIRANWindow.
    """

    timestamps_s: np.ndarray
    cell_prb_utilization: np.ndarray
    ship_uplink_demand_mbps: np.ndarray
    edge_gpu_load: np.ndarray
    ntn_beam_capacity_used: np.ndarray
    weather_rain_mm_per_hr: np.ndarray
    sla_breaches: np.ndarray
    metadata: dict = field(default_factory=dict)


class MaritimeSyntheticGenerator(NTNAIRANSyntheticGenerator):
    """Thin maritime subclass of the generic generator. Identical model,
    maritime-flavoured field names and metadata for back-compat.
    """

    def __init__(self, config: MaritimeScenarioConfig | None = None):
        cfg = config or MaritimeScenarioConfig()
        self._maritime_cfg = cfg
        super().__init__(cfg.to_generic())

    def _build_metadata(self, arrivals, active_workloads, is_storm):
        return {
            "vertical": "maritime",
            "config_seed": self.cfg.rng_seed,
            "is_storm": bool(is_storm),
            "ship_arrivals": int(arrivals.sum()),
            "active_cameras": int(active_workloads),
        }

    def _wrap_window(
        self,
        timestamps_s,
        cell_prb_utilization,
        terminal_uplink_demand_mbps,
        edge_gpu_load,
        ntn_beam_capacity_used,
        weather_rain_mm_per_hr,
        sla_breaches,
        metadata,
    ):
        return MaritimeWindow(
            timestamps_s=timestamps_s,
            cell_prb_utilization=cell_prb_utilization,
            ship_uplink_demand_mbps=terminal_uplink_demand_mbps,
            edge_gpu_load=edge_gpu_load,
            ntn_beam_capacity_used=ntn_beam_capacity_used,
            weather_rain_mm_per_hr=weather_rain_mm_per_hr,
            sla_breaches=sla_breaches,
            metadata=metadata,
        )
