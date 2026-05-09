"""NTN link-state composer (E2E_DEBUG gap 8).

Compose the (channel_state, ntn_timing_state, doppler_envelope) physics
modules into a single feature vector the encoder's `entity_tokenizer`
can consume as a satellite asset's attributes. Without this, the world
model sees a satellite as just an ECEF position; with it, the model can
reason about which links are HARQ-feasible and what Doppler residual
the UE's PHY can still cope with.

Output is a stable 12-dim feature vector — the satellite-asset
`attr_dim=8` default in `EntityTokenizerConfig` is therefore widened to
match. Composition contract:

    [0]  one_way_delay_s         — physics.ntn_timing
    [1]  k_offset_slots          — physics.ntn_timing
    [2]  min_harq_processes      — physics.ntn_timing
    [3]  harq_feedback_enabled   — physics.ntn_timing  (0/1)
    [4]  doppler_shift_kHz       — physics.doppler
    [5]  doppler_rate_Hz_per_s   — physics.doppler  (signed)
    [6]  channel_los_prob        — physics.tr38811
    [7]  channel_shadow_sigma    — physics.tr38811
    [8]  channel_K_factor_dB     — physics.tr38811
    [9]  channel_delay_spread_ns — physics.tr38811
    [10] elevation_deg           — physics.geodesy
    [11] slant_range_km          — physics.geodesy

All values are normalised before being returned to keep the encoder's
input distribution well-conditioned.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from horizon_ric.planner.physics.doppler import (
    doppler_rate_hz_s,
    doppler_shift_hz,
)
from horizon_ric.planner.physics.geodesy import (
    look_angles_to_target,
)
from horizon_ric.planner.physics.ntn_timing import ntn_timing_state
from horizon_ric.planner.physics.orbital import OrbitalState, state_ecef_m
from horizon_ric.planner.physics.tr38811 import channel_state

LINK_STATE_DIM: int = 12


@dataclass(frozen=True)
class LinkStateInputs:
    sat_state: OrbitalState
    es_lat_deg: float
    es_lon_deg: float
    es_height_m: float = 0.0
    carrier_hz: float = 28e9
    environment: str = "rural"


def _normalise(values: list[float]) -> torch.Tensor:
    """Per-feature normalisation tuned for the 12-dim layout above.

    Uses scales chosen to land each feature roughly in [-3, 3] for the
    NTN regimes the rApp targets (LEO 400–1500 km, GSO 35,786 km, S/Ka
    bands). These constants are documented in the module docstring and
    can be tightened from data once we have a real distribution.
    """
    if len(values) != LINK_STATE_DIM:
        raise ValueError(
            f"expected {LINK_STATE_DIM} values, got {len(values)}"
        )
    scales = [
        0.130,    # one_way_delay_s — GSO ~0.13s; LEO ~5e-3
        514.0,    # k_offset_slots — GSO worst case
        32.0,     # min_harq_processes
        1.0,      # harq_feedback (boolean)
        700.0,    # doppler_shift_kHz — LEO Ka peak
        6.0,      # doppler_rate_kHz_s
        1.0,      # los_prob
        5.0,      # shadow_sigma_dB
        25.0,     # K_factor_dB
        180.0,    # delay_spread_ns
        90.0,     # elevation_deg
        40_000.0, # slant_range_km — covers LEO + GSO
    ]
    return torch.tensor(
        [v / s for v, s in zip(values, scales, strict=True)],
        dtype=torch.float32,
    )


def compose_link_state(inputs: LinkStateInputs) -> torch.Tensor:
    """Build the 12-dim link-state vector for one satellite ↔ ES link.

    Returns a (12,) torch.float32 tensor ready to drop into the encoder
    as the satellite asset's `attributes`.
    """
    # Geometry
    sat_ecef = state_ecef_m(inputs.sat_state)
    look = look_angles_to_target(
        inputs.es_lat_deg, inputs.es_lon_deg, sat_ecef, inputs.es_height_m,
    )
    elevation = look.elevation_deg
    range_km = look.range_m / 1000.0

    # If the satellite is below the horizon, return a zero vector with a
    # sentinel elevation; downstream layers that ALSO read the spatial
    # prior's los_mask will mask this asset out anyway.
    if elevation < 0:
        return torch.zeros(LINK_STATE_DIM, dtype=torch.float32)

    # Doppler
    doppler_hz = doppler_shift_hz(
        inputs.sat_state, inputs.es_lat_deg, inputs.es_lon_deg,
        inputs.carrier_hz, inputs.es_height_m,
    )
    doppler_rate = doppler_rate_hz_s(
        inputs.sat_state, inputs.es_lat_deg, inputs.es_lon_deg,
        inputs.carrier_hz, inputs.es_height_m,
    )

    # NTN timing (slot-coarse). Use μ=1 default.
    altitude_m = inputs.sat_state.altitude_km() * 1000.0
    timing = ntn_timing_state(altitude_m=altitude_m, elevation_deg=elevation)

    # 38.811 channel
    cs = channel_state(inputs.environment, inputs.carrier_hz, elevation)

    return _normalise([
        timing.one_way_delay_s,
        float(timing.k_offset_slots),
        float(timing.min_harq_processes),
        1.0 if timing.harq_feedback_enabled else 0.0,
        doppler_hz / 1000.0,
        doppler_rate / 1000.0,
        cs.los_prob,
        cs.shadow_sigma_dB,
        cs.k_factor_dB,
        cs.delay_spread_ns,
        elevation,
        range_km,
    ])


__all__ = ["LINK_STATE_DIM", "LinkStateInputs", "compose_link_state"]
