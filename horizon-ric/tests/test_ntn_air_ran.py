"""Tests for the generic NTN + AI-RAN framework and the rural / disaster /
enterprise vertical playbooks.
"""

from __future__ import annotations

import numpy as np
import pytest

from horizon_ric.scenarios import (
    DisasterRecoveryGenerator,
    EnterpriseAIRANGenerator,
    NTNAIRANConfig,
    NTNAIRANSyntheticGenerator,
    NTNAIRANWindow,
    RuralCoverageGenerator,
)
from horizon_ric.scenarios.disaster import _default_config as _disaster_defaults
from horizon_ric.scenarios.enterprise import _default_config as _enterprise_defaults
from horizon_ric.scenarios.rural import _default_config as _rural_defaults

# ---------------------------------------------------------------------------
# Generic framework
# ---------------------------------------------------------------------------


class TestGenericFramework:
    def _cfg(self, **overrides) -> NTNAIRANConfig:
        base = dict(site_lat=10.0, site_lon=20.0, rng_seed=0)
        base.update(overrides)
        return NTNAIRANConfig(**base)

    def test_window_shapes(self):
        cfg = self._cfg(rng_seed=1)
        gen = NTNAIRANSyntheticGenerator(cfg)
        w = gen.sample()
        assert isinstance(w, NTNAIRANWindow)
        T = cfg.window_seconds // cfg.step_seconds
        assert w.timestamps_s.shape == (T,)
        assert w.cell_prb_utilization.shape == (T, cfg.n_ground_cells)
        assert w.terminal_uplink_demand_mbps.shape == (T, cfg.n_user_terminals)
        assert w.edge_gpu_load.shape == (T, cfg.edge_gpu_count)
        assert w.ntn_beam_capacity_used.shape == (T, cfg.ntn_beam_count)
        assert w.weather_rain_mm_per_hr.shape == (T,)
        assert w.sla_breaches.shape == (T,)

    def test_values_in_expected_ranges(self):
        gen = NTNAIRANSyntheticGenerator(self._cfg(rng_seed=2))
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
        a = NTNAIRANSyntheticGenerator(self._cfg(rng_seed=42)).sample()
        b = NTNAIRANSyntheticGenerator(self._cfg(rng_seed=42)).sample()
        np.testing.assert_array_equal(a.cell_prb_utilization, b.cell_prb_utilization)
        np.testing.assert_array_equal(a.sla_breaches, b.sla_breaches)

    def test_different_seeds_diverge(self):
        a = NTNAIRANSyntheticGenerator(self._cfg(rng_seed=1)).sample()
        b = NTNAIRANSyntheticGenerator(self._cfg(rng_seed=2)).sample()
        assert not np.array_equal(
            a.cell_prb_utilization, b.cell_prb_utilization
        )

    def test_storm_increases_rain(self):
        gen = NTNAIRANSyntheticGenerator(
            self._cfg(rng_seed=7, weather_storm_probability=1.0)
        )
        w = gen.sample()
        assert w.metadata["is_storm"] is True
        assert w.weather_rain_mm_per_hr.max() > 5.0

    def test_no_storm_zero_rain(self):
        gen = NTNAIRANSyntheticGenerator(
            self._cfg(rng_seed=11, weather_storm_probability=0.0)
        )
        w = gen.sample()
        assert w.metadata["is_storm"] is False
        assert np.all(w.weather_rain_mm_per_hr == 0.0)

    def test_sample_batch(self):
        gen = NTNAIRANSyntheticGenerator(self._cfg(rng_seed=0))
        batch = gen.sample_batch(3)
        assert len(batch) == 3
        assert all(isinstance(w, NTNAIRANWindow) for w in batch)

    def test_site_coords_required(self):
        with pytest.raises(TypeError):
            NTNAIRANConfig()  # type: ignore[call-arg]

    def test_no_implicit_config(self):
        with pytest.raises(TypeError):
            NTNAIRANSyntheticGenerator()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Rural
# ---------------------------------------------------------------------------


class TestRuralCoverage:
    def test_default_site_is_rural_kansas(self):
        cfg = _rural_defaults()
        assert cfg.site_lat == pytest.approx(39.0)
        assert cfg.site_lon == pytest.approx(-98.0)

    def test_rural_storm_probability_is_realistic(self):
        cfg = _rural_defaults()
        assert cfg.weather_storm_probability < 0.05  # realistic, not 50%

    def test_rural_terminal_count_is_high(self):
        cfg = _rural_defaults()
        assert cfg.n_user_terminals >= 30

    def test_rural_metadata_marks_vertical(self):
        gen = RuralCoverageGenerator()
        w = gen.sample()
        assert w.metadata["vertical"] == "rural"

    def test_rural_window_shape(self):
        gen = RuralCoverageGenerator()
        w = gen.sample()
        T = w.timestamps_s.shape[0]
        assert w.cell_prb_utilization.shape == (T, gen.cfg.n_ground_cells)
        assert w.terminal_uplink_demand_mbps.shape == (T, gen.cfg.n_user_terminals)

    def test_rural_low_gpu_count(self):
        cfg = _rural_defaults()
        assert cfg.edge_gpu_count <= 2


# ---------------------------------------------------------------------------
# Disaster
# ---------------------------------------------------------------------------


class TestDisasterRecovery:
    def test_disaster_high_arrival_rate(self):
        cfg = _disaster_defaults()
        assert cfg.arrival_lambda_per_hour >= 30.0

    def test_disaster_high_storm_probability(self):
        cfg = _disaster_defaults()
        assert cfg.weather_storm_probability >= 0.4

    def test_disaster_metadata_includes_fallback(self):
        gen = DisasterRecoveryGenerator()
        w = gen.sample()
        assert w.metadata["vertical"] == "disaster"
        assert "ntn_only_fallback" in w.metadata
        assert w.metadata["ntn_only_fallback"] is True

    def test_disaster_fallback_can_be_disabled(self):
        gen = DisasterRecoveryGenerator(ntn_only_fallback=False)
        w = gen.sample()
        assert w.metadata["ntn_only_fallback"] is False

    def test_disaster_window_sane(self):
        gen = DisasterRecoveryGenerator()
        w = gen.sample()
        assert w.cell_prb_utilization.max() <= 1.0
        assert w.cell_prb_utilization.min() >= 0.0

    def test_disaster_batch(self):
        gen = DisasterRecoveryGenerator()
        batch = gen.sample_batch(2)
        assert len(batch) == 2


# ---------------------------------------------------------------------------
# Enterprise
# ---------------------------------------------------------------------------


class TestEnterpriseAIRAN:
    def test_enterprise_no_ntn_beams(self):
        cfg = _enterprise_defaults()
        assert cfg.ntn_beam_count == 0

    def test_enterprise_high_gpu(self):
        cfg = _enterprise_defaults()
        assert cfg.edge_gpu_count == 16

    def test_enterprise_indoor_metadata(self):
        gen = EnterpriseAIRANGenerator()
        w = gen.sample()
        assert w.metadata["vertical"] == "enterprise"
        assert w.metadata["indoor"] is True
        assert w.metadata["band_ghz"] == pytest.approx(3.5)

    def test_enterprise_high_terminal_count(self):
        cfg = _enterprise_defaults()
        assert cfg.n_user_terminals >= 200

    def test_enterprise_zero_rain(self):
        gen = EnterpriseAIRANGenerator()
        w = gen.sample()
        # storm probability is 0 -> no rain
        assert np.all(w.weather_rain_mm_per_hr == 0.0)

    def test_enterprise_window_has_no_ntn_columns(self):
        gen = EnterpriseAIRANGenerator()
        w = gen.sample()
        assert w.ntn_beam_capacity_used.shape[1] == 0
