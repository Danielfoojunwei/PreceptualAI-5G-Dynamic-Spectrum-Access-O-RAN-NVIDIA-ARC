"""Enterprise AI-RAN playbook — indoor mid-band, high edge compute, no NTN."""

from __future__ import annotations

from horizon_ric.scenarios.ntn_air_ran import (
    NTNAIRANConfig,
    NTNAIRANSyntheticGenerator,
)


def _default_config(rng_seed: int = 0) -> NTNAIRANConfig:
    return NTNAIRANConfig(
        site_lat=37.7749,
        site_lon=-122.4194,  # San Francisco enterprise reference
        n_ground_cells=12,
        n_user_terminals=200,
        n_ai_workloads_per_terminal=1,
        edge_gpu_count=16,
        ntn_beam_count=0,
        weather_storm_probability=0.0,
        arrival_lambda_per_hour=12.0,
        rng_seed=rng_seed,
    )


class EnterpriseAIRANGenerator(NTNAIRANSyntheticGenerator):
    """Indoor mid-band (3.5 GHz) enterprise deployment with rich edge GPU."""

    BAND_GHZ: float = 3.5

    def __init__(self, config: NTNAIRANConfig | None = None):
        cfg = config or _default_config()
        super().__init__(cfg)

    def _build_metadata(self, arrivals, active_workloads, is_storm):
        meta = super()._build_metadata(arrivals, active_workloads, is_storm)
        meta["vertical"] = "enterprise"
        meta["band_ghz"] = self.BAND_GHZ
        meta["indoor"] = True
        return meta
