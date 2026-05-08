"""Maritime synthetic generator tests."""

import numpy as np

from horizon_ric.scenarios.maritime import (
    MaritimeScenarioConfig,
    MaritimeSyntheticGenerator,
    MaritimeWindow,
)


class TestMaritimeGenerator:
    def test_window_shapes(self):
        cfg = MaritimeScenarioConfig(rng_seed=1)
        gen = MaritimeSyntheticGenerator(cfg)
        w = gen.sample()
        assert isinstance(w, MaritimeWindow)
        T = cfg.window_seconds // cfg.step_seconds
        assert w.timestamps_s.shape == (T,)
        assert w.cell_prb_utilization.shape == (T, cfg.num_terrestrial_cells)
        assert w.ship_uplink_demand_mbps.shape == (T, cfg.num_ships)
        assert w.edge_gpu_load.shape == (T, cfg.edge_gpu_count)
        assert w.ntn_beam_capacity_used.shape == (T, cfg.ntn_beam_count)
        assert w.weather_rain_mm_per_hr.shape == (T,)
        assert w.sla_breaches.shape == (T,)

    def test_values_in_expected_ranges(self):
        gen = MaritimeSyntheticGenerator(MaritimeScenarioConfig(rng_seed=2))
        w = gen.sample()
        assert np.all(w.cell_prb_utilization >= 0.0)
        assert np.all(w.cell_prb_utilization <= 1.0)
        assert np.all(w.edge_gpu_load >= 0.0)
        assert np.all(w.edge_gpu_load <= 1.0)
        assert np.all(w.ntn_beam_capacity_used >= 0.0)
        assert np.all(w.ntn_beam_capacity_used <= 1.0)
        assert np.all(w.weather_rain_mm_per_hr >= 0.0)
        assert set(np.unique(w.sla_breaches).tolist()) <= {0, 1}

    def test_seed_determinism(self):
        a = MaritimeSyntheticGenerator(MaritimeScenarioConfig(rng_seed=42)).sample()
        b = MaritimeSyntheticGenerator(MaritimeScenarioConfig(rng_seed=42)).sample()
        np.testing.assert_array_equal(a.cell_prb_utilization, b.cell_prb_utilization)
        np.testing.assert_array_equal(a.sla_breaches, b.sla_breaches)

    def test_different_seeds_diverge(self):
        a = MaritimeSyntheticGenerator(MaritimeScenarioConfig(rng_seed=1)).sample()
        b = MaritimeSyntheticGenerator(MaritimeScenarioConfig(rng_seed=2)).sample()
        # At least one of these grids should differ between seeds
        assert not np.array_equal(a.cell_prb_utilization, b.cell_prb_utilization)

    def test_storm_increases_ntn_load(self):
        # Force a storm via probability=1.0
        gen = MaritimeSyntheticGenerator(
            MaritimeScenarioConfig(rng_seed=7, weather_storm_probability=1.0)
        )
        w = gen.sample()
        assert w.metadata["is_storm"] is True
        assert w.weather_rain_mm_per_hr.max() > 5.0  # storm peak

    def test_no_storm_zero_rain(self):
        gen = MaritimeSyntheticGenerator(
            MaritimeScenarioConfig(rng_seed=11, weather_storm_probability=0.0)
        )
        w = gen.sample()
        assert w.metadata["is_storm"] is False
        assert np.all(w.weather_rain_mm_per_hr == 0.0)

    def test_sample_batch(self):
        gen = MaritimeSyntheticGenerator(MaritimeScenarioConfig(rng_seed=0))
        batch = gen.sample_batch(3)
        assert len(batch) == 3
        assert all(isinstance(w, MaritimeWindow) for w in batch)
