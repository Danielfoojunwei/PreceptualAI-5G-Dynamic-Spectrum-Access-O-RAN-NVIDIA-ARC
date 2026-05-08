"""Generic NTN + AI-RAN synthetic scenario framework.

Field-agnostic baseline used by every vertical playbook. The naming and
metadata in this module deliberately avoid any single-vertical jargon —
subclasses re-frame the same statistical process for their domain.

The generator emits one 5-minute window of telemetry that exercises:
    - terrestrial cell PRB utilisation under user-terminal demand
    - edge GPU load when AI inference workloads schedule
    - NTN beam capacity under weather-induced attenuation
    - SLA breach ground-truth labels for risk-head training

SYNTHETIC ONLY: never feed the output to a production policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class NTNAIRANConfig:
    """Configuration for the generic NTN + AI-RAN synthetic generator.

    Site coordinates are REQUIRED — there is no default site, by design,
    so the caller must consciously place the deployment.
    """

    # Site coordinates — REQUIRED
    site_lat: float
    site_lon: float

    # Topology
    n_ground_cells: int = 8
    n_user_terminals: int = 12
    n_ai_workloads_per_terminal: int = 4

    # Compute (Jetson Orin Nano edge target — see UHCI memory)
    edge_gpu_count: int = 6
    edge_gpu_memory_gb: float = 8.0

    # NTN — Ka-band LEO link-budget representative
    ntn_beam_count: int = 3
    ntn_capacity_mbps_per_beam: float = 200.0
    ntn_path_loss_dB_nominal: float = 175.0

    # Time series
    window_seconds: int = 300
    step_seconds: int = 5

    # Behaviour
    arrival_lambda_per_hour: float = 4.0
    weather_storm_probability: float = 0.05
    sla_breach_threshold: float = 0.85
    sla_breach_slope: float = 8.0
    rng_seed: int = 0

    # Hard guard against accidental production wiring
    synthetic_only: bool = True


@dataclass
class NTNAIRANWindow:
    """One window of generic NTN + AI-RAN telemetry."""

    timestamps_s: np.ndarray  # (T,)
    cell_prb_utilization: np.ndarray  # (T, n_ground_cells)
    terminal_uplink_demand_mbps: np.ndarray  # (T, n_user_terminals)
    edge_gpu_load: np.ndarray  # (T, edge_gpu_count)
    ntn_beam_capacity_used: np.ndarray  # (T, ntn_beam_count)
    weather_rain_mm_per_hr: np.ndarray  # (T,)
    sla_breaches: np.ndarray  # (T,) binary
    metadata: dict = field(default_factory=dict)


class NTNAIRANSyntheticGenerator:
    """Phase-1 synthetic telemetry generator, vertical-neutral.

    Subclasses customise the default config (site coords, mix, weather) but
    do not change the statistical model — that lives here.
    """

    def __init__(self, config: NTNAIRANConfig | None = None):
        if config is None:
            raise TypeError(
                "NTNAIRANSyntheticGenerator requires an explicit config "
                "(no implicit site coordinates)."
            )
        self.cfg = config
        self._rng = np.random.default_rng(self.cfg.rng_seed)

    # --- public API -----------------------------------------------------

    def sample(self):
        """Generate one synthetic window."""
        T = self.cfg.window_seconds // self.cfg.step_seconds
        ts = np.arange(T) * self.cfg.step_seconds

        arrivals = self._rng.poisson(
            lam=self.cfg.arrival_lambda_per_hour * (self.cfg.window_seconds / 3600),
            size=self.cfg.n_user_terminals,
        )
        terminal_demand = self._terminal_uplink_demand(T, arrivals)

        prb_base = 0.3 + 0.05 * self._rng.standard_normal(
            (T, self.cfg.n_ground_cells)
        )
        terminal_total = terminal_demand.sum(axis=1, keepdims=True)
        cell_prb = np.clip(prb_base + 0.4 * (terminal_total / 1000.0), 0.0, 1.0)

        active_workloads = arrivals.sum() * self.cfg.n_ai_workloads_per_terminal
        gpu_base = 0.4 + 0.05 * self._rng.standard_normal(
            (T, self.cfg.edge_gpu_count)
        )
        gpu_load = np.clip(
            gpu_base + 0.05 * (active_workloads / max(self.cfg.edge_gpu_count, 1)),
            0.0,
            1.0,
        )

        is_storm = self._rng.random() < self.cfg.weather_storm_probability
        rain = np.zeros(T)
        if is_storm:
            peak_t = T // 2
            sigma = T / 4
            rain = 30.0 * np.exp(-((np.arange(T) - peak_t) / sigma) ** 2)

        ntn_used = np.zeros((T, max(self.cfg.ntn_beam_count, 1)))
        if self.cfg.ntn_beam_count > 0:
            for b in range(self.cfg.ntn_beam_count):
                base = 0.5 + 0.1 * self._rng.standard_normal(T)
                ntn_used[:, b] = np.clip(base + 0.01 * rain, 0.0, 1.0)
        else:
            ntn_used = np.zeros((T, 0))

        worst_components = [cell_prb.max(axis=1), gpu_load.max(axis=1)]
        if ntn_used.shape[1] > 0:
            worst_components.append(ntn_used.max(axis=1))
        worst_resource = np.maximum.reduce(worst_components)
        sla_breach_prob = self._sigmoid(
            self.cfg.sla_breach_slope
            * (worst_resource - self.cfg.sla_breach_threshold)
        )
        sla_breaches = (self._rng.random(T) < sla_breach_prob).astype(np.int8)

        metadata = self._build_metadata(arrivals, active_workloads, is_storm)

        return self._wrap_window(
            timestamps_s=ts,
            cell_prb_utilization=cell_prb.astype(np.float32),
            terminal_uplink_demand_mbps=terminal_demand.astype(np.float32),
            edge_gpu_load=gpu_load.astype(np.float32),
            ntn_beam_capacity_used=ntn_used.astype(np.float32),
            weather_rain_mm_per_hr=rain.astype(np.float32),
            sla_breaches=sla_breaches,
            metadata=metadata,
        )

    def sample_batch(self, n_windows: int):
        """Generate n_windows independent samples."""
        return [self.sample() for _ in range(n_windows)]

    # --- subclass hooks -------------------------------------------------

    def _build_metadata(self, arrivals, active_workloads, is_storm):
        return {
            "config_seed": self.cfg.rng_seed,
            "is_storm": bool(is_storm),
            "terminal_arrivals": int(arrivals.sum()),
            "active_workloads": int(active_workloads),
        }

    def _wrap_window(self, **kwargs):
        return NTNAIRANWindow(**kwargs)

    # --- internals ------------------------------------------------------

    def _terminal_uplink_demand(self, T: int, arrivals: np.ndarray) -> np.ndarray:
        """Per-terminal uplink demand in Mbps over the window."""
        demand = np.zeros((T, self.cfg.n_user_terminals))
        for s in range(self.cfg.n_user_terminals):
            if arrivals[s] == 0:
                continue
            arrive_t = self._rng.integers(0, T)
            duration = self._rng.integers(T // 4, T)
            end_t = min(arrive_t + duration, T)
            base_demand = 50.0 + 30.0 * self._rng.standard_normal()
            for t in range(arrive_t, end_t):
                ramp = (t - arrive_t) / max(end_t - arrive_t, 1)
                demand[t, s] = max(0.0, base_demand * math.sin(math.pi * ramp))
        return demand

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
