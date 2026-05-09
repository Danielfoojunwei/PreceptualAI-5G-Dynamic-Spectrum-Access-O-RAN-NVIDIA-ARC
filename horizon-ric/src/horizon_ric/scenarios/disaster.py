"""Disaster-recovery playbook — burst arrivals, harsh weather, NTN fallback."""

from __future__ import annotations

from horizon_ric.scenarios.ntn_air_ran import (
    NTNAIRANConfig,
    NTNAIRANSyntheticGenerator,
)


def _default_config(rng_seed: int = 0) -> NTNAIRANConfig:
    return NTNAIRANConfig(
        site_lat=29.95,
        site_lon=-90.07,  # New Orleans (hurricane-prone reference site)
        n_ground_cells=3,
        n_user_terminals=20,
        n_ai_workloads_per_terminal=3,
        edge_gpu_count=4,
        ntn_beam_count=4,
        weather_storm_probability=0.4,
        arrival_lambda_per_hour=30.0,
        rng_seed=rng_seed,
    )


class DisasterRecoveryGenerator(NTNAIRANSyntheticGenerator):
    """Surge-load deployment for emergency response.

    NTN-only fallback flag indicates the terrestrial network may be down;
    consumers should prefer NTN beams when this is set.
    """

    def __init__(
        self,
        config: NTNAIRANConfig | None = None,
        *,
        ntn_only_fallback: bool = True,
    ):
        cfg = config or _default_config()
        self.ntn_only_fallback = bool(ntn_only_fallback)
        super().__init__(cfg)

    def _build_metadata(self, arrivals, active_workloads, is_storm):
        meta = super()._build_metadata(arrivals, active_workloads, is_storm)
        meta["vertical"] = "disaster"
        meta["ntn_only_fallback"] = self.ntn_only_fallback
        return meta
