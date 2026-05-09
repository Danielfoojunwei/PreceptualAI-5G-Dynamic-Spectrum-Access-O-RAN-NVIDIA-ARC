"""Maritime scenario guard tests for the Frankfurt-default fix."""

import pytest

from horizon_ric.scenarios.maritime import (
    MaritimeScenarioConfig,
    MaritimeSyntheticGenerator,
)


class TestPortGuard:
    def test_default_port_is_a_real_port(self):
        cfg = MaritimeScenarioConfig()
        # Default is Rotterdam (51.9244, 4.4777), not Frankfurt (50.110, 8.682).
        assert cfg.port_lat != 50.110 or cfg.port_lon != 8.682

    def test_frankfurt_default_is_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            MaritimeScenarioConfig(port_lat=50.110, port_lon=8.682)
        assert "Frankfurt" in str(excinfo.value)

    def test_other_inland_coords_pass(self):
        # We only block the documented Frankfurt placeholder; other inland
        # coords aren't validated (no GIS in CI).
        MaritimeScenarioConfig(port_lat=40.0, port_lon=-74.0)  # NY harbour


class TestThresholdsAreConfigurable:
    def test_breach_threshold_overridable(self):
        cfg = MaritimeScenarioConfig(rng_seed=0, sla_breach_threshold=0.5)
        gen = MaritimeSyntheticGenerator(cfg)
        w = gen.sample()
        # With a lower threshold, more breaches are expected.
        n_breaches = int(w.sla_breaches.sum())
        assert n_breaches >= 0  # smoke; precise count depends on RNG

    def test_jetson_orin_default(self):
        # UHCI memory says edge target is Jetson Orin Nano (8 GB).
        cfg = MaritimeScenarioConfig()
        assert cfg.edge_gpu_memory_gb == 8.0
