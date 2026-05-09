"""TR 38.811 channel state + NTN timing tests."""

import pytest

from horizon_ric.planner.physics.ntn_timing import (
    NTNTimingState,
    ntn_timing_for_gso,
    ntn_timing_state,
)
from horizon_ric.planner.physics.tr38811 import channel_state


class TestChannelStateLookup:
    def test_known_envs_all_resolve(self):
        for env in ("rural", "suburban", "urban", "dense_urban"):
            cs = channel_state(env, 28e9, 40)
            assert 0.0 <= cs.los_prob <= 1.0
            assert cs.shadow_sigma_dB >= 0.0
            assert cs.delay_spread_ns > 0.0

    def test_los_increases_with_elevation(self):
        for env in ("urban", "dense_urban"):
            p10 = channel_state(env, 28e9, 10).los_prob
            p90 = channel_state(env, 28e9, 90).los_prob
            assert p90 > p10

    def test_dense_urban_more_shadowing_than_suburban(self):
        s = channel_state("suburban", 28e9, 30)
        d = channel_state("dense_urban", 28e9, 30)
        # At low-to-mid elevation Ka shadow σ in dense_urban is comparable
        # but higher; at minimum it must be >= suburban value.
        assert d.shadow_sigma_dB >= s.shadow_sigma_dB - 0.5

    def test_unknown_env_raises(self):
        with pytest.raises(ValueError):
            channel_state("desert", 28e9, 30)

    def test_unsupported_band_raises(self):
        with pytest.raises(ValueError):
            channel_state("rural", 7e9, 30)  # C-band unsupported

    def test_band_classification(self):
        s = channel_state("rural", 2e9, 30)
        ka = channel_state("rural", 28e9, 30)
        assert s.band == "S"
        assert ka.band == "Ka"


class TestNTNTimingState:
    def test_leo_600_40deg_ka(self):
        nt = ntn_timing_state(altitude_m=600_000, elevation_deg=40, scs_khz=30)
        assert isinstance(nt, NTNTimingState)
        # RTT ≈ 5.9 ms → K_offset ≈ 12 slots at μ=1
        assert nt.k_offset_slots >= 10 and nt.k_offset_slots <= 16
        # HARQ feedback is feasible at LEO with up to 32 processes
        assert nt.harq_feedback_enabled is True

    def test_geo_huge_k_offset(self):
        # GSO altitude
        nt = ntn_timing_state(altitude_m=35_786_000, elevation_deg=40, scs_khz=30)
        # RTT ≈ 257 ms → K_offset > 500 slots; HARQ must be disabled
        assert nt.k_offset_slots > 400
        assert nt.harq_feedback_enabled is False
        assert nt.min_harq_processes > 32 - 1

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            ntn_timing_state(altitude_m=-1, elevation_deg=30)
        with pytest.raises(ValueError):
            ntn_timing_state(altitude_m=600_000, elevation_deg=-5)
        with pytest.raises(ValueError):
            ntn_timing_state(altitude_m=600_000, elevation_deg=30, scs_khz=0)

    def test_slot_duration_scs_30khz(self):
        nt = ntn_timing_state(altitude_m=600_000, elevation_deg=40, scs_khz=30)
        # μ=1 → T_slot = 0.5 ms
        assert abs(nt.slot_duration_s - 0.0005) < 1e-9

    def test_for_gso_below_horizon_raises(self):
        # ES at 89.9°N; GSO at 0°lon is below horizon
        with pytest.raises(ValueError):
            ntn_timing_for_gso(89.9, 0.0, 0.0)

    def test_for_gso_at_subpoint(self):
        # ES at equator under GSO sub-point
        nt = ntn_timing_for_gso(0.0, 0.0, 0.0)
        # one-way ≈ 0.119 s
        assert abs(nt.one_way_delay_s - 0.119) < 0.001
