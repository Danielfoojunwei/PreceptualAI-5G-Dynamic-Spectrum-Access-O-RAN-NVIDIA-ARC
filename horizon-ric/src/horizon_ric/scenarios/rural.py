"""Rural coverage playbook — sparse cells, more terminals, low storm rate."""

from __future__ import annotations

from dataclasses import replace

from horizon_ric.scenarios.ntn_air_ran import (
    NTNAIRANConfig,
    NTNAIRANSyntheticGenerator,
)


def _default_config(rng_seed: int = 0) -> NTNAIRANConfig:
    return NTNAIRANConfig(
        site_lat=39.0,
        site_lon=-98.0,  # central Kansas, USA
        n_ground_cells=4,
        n_user_terminals=30,
        n_ai_workloads_per_terminal=2,
        edge_gpu_count=2,
        ntn_beam_count=4,
        weather_storm_probability=0.02,
        arrival_lambda_per_hour=2.0,
        rng_seed=rng_seed,
    )


class RuralCoverageGenerator(NTNAIRANSyntheticGenerator):
    """Wide-area rural deployment with NTN backhaul and few edge GPUs."""

    def __init__(self, config: NTNAIRANConfig | None = None):
        cfg = config or _default_config()
        super().__init__(cfg)

    def _build_metadata(self, arrivals, active_workloads, is_storm):
        meta = super()._build_metadata(arrivals, active_workloads, is_storm)
        meta["vertical"] = "rural"
        return meta


def default_rural_config(rng_seed: int = 0) -> NTNAIRANConfig:
    """Public helper: rural defaults at the Kansas reference site."""
    return replace(_default_config(rng_seed))
