"""ITU-R propagation physics — sanity tests.

These are bounded-tolerance tests against well-known reference values from
ITU-R Recommendations. They are NOT a substitute for the full conformance
suite (which validates against the ITU-Rpy reference Python implementation
under test option [itur]); they are a fast smoke-test that runs in CI.

Pre-registered tolerances per `STANDARDS.md` §5.3:
    closed-form modules match ITU-Rpy reference within ≤0.1 dB,
    closed-form modules match ITU published table within ≤0.5 dB.
"""

import math

import pytest

from horizon_ric.planner.physics import (
    free_space_path_loss_dB,
    gas_attenuation_dB,
    rain_attenuation_dB,
    total_path_loss_dB,
)


class TestFreeSpacePathLoss:
    def test_one_meter_one_ghz_reference(self):
        # FSPL = 20 log10(4π·1·1e9/c) = 32.448 dB at 1 m, 1 GHz
        # (This is the well-known FSPL constant.)
        loss = free_space_path_loss_dB(distance_m=1.0, frequency_hz=1e9)
        assert abs(loss - 32.448) < 0.01

    def test_geo_uplink_ka_band(self):
        # GSO slant ~38500 km at 30 GHz → ~213.7 dB
        loss = free_space_path_loss_dB(distance_m=38_500_000, frequency_hz=30e9)
        assert 213 < loss < 215

    def test_doubling_distance_adds_6dB(self):
        loss_1 = free_space_path_loss_dB(1000.0, 2e9)
        loss_2 = free_space_path_loss_dB(2000.0, 2e9)
        assert abs((loss_2 - loss_1) - 6.0) < 0.05

    def test_doubling_frequency_adds_6dB(self):
        loss_1 = free_space_path_loss_dB(1000.0, 1e9)
        loss_2 = free_space_path_loss_dB(1000.0, 2e9)
        assert abs((loss_2 - loss_1) - 6.0) < 0.05

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            free_space_path_loss_dB(0.0, 1e9)
        with pytest.raises(ValueError):
            free_space_path_loss_dB(1.0, 0.0)


class TestGasAttenuation:
    def test_low_frequency_negligible(self):
        # Below 10 GHz, gas attenuation is < 0.5 dB at zenith
        assert gas_attenuation_dB(elevation_deg=90, frequency_ghz=2.0) < 0.5

    def test_60_ghz_o2_peak(self):
        # 60 GHz is the O2 absorption peak — should be highly attenuating
        atten = gas_attenuation_dB(elevation_deg=90, frequency_ghz=60.0)
        assert atten > 30.0  # tens of dB at zenith

    def test_low_elevation_amplifies(self):
        # Slant path at 5° vs 90° → ~10x longer path
        zenith = gas_attenuation_dB(elevation_deg=90, frequency_ghz=20.0)
        low_el = gas_attenuation_dB(elevation_deg=5, frequency_ghz=20.0)
        assert low_el > 5 * zenith

    def test_invalid_elevation_raises(self):
        with pytest.raises(ValueError):
            gas_attenuation_dB(elevation_deg=-1, frequency_ghz=10.0)


class TestRainAttenuation:
    def test_no_rain_returns_zero(self):
        assert rain_attenuation_dB(0.0, 30, 20.0) == 0.0

    def test_higher_rain_more_attenuation(self):
        a = rain_attenuation_dB(5.0, 30, 20.0)
        b = rain_attenuation_dB(50.0, 30, 20.0)
        assert b > a

    def test_higher_freq_more_attenuation(self):
        # At 5 mm/hr, Ka-band (30 GHz) should attenuate more than C-band (4 GHz)
        c_band = rain_attenuation_dB(5.0, 30, 4.0)
        ka_band = rain_attenuation_dB(5.0, 30, 30.0)
        assert ka_band > 5 * c_band

    def test_polarization_circular_is_average(self):
        h = rain_attenuation_dB(20.0, 30, 20.0, polarization="horizontal")
        v = rain_attenuation_dB(20.0, 30, 20.0, polarization="vertical")
        c = rain_attenuation_dB(20.0, 30, 20.0, polarization="circular")
        assert min(h, v) <= c <= max(h, v) * 1.1


class TestTotalPathLoss:
    def test_decomposition_sums(self):
        result = total_path_loss_dB(
            distance_m=38_500_000,
            frequency_hz=30e9,
            elevation_deg=30,
            rain_rate_mm_per_hr=10.0,
        )
        assert math.isclose(
            result["total_dB"],
            result["fspl_dB"] + result["gas_dB"] + result["rain_dB"],
            abs_tol=0.001,
        )

    def test_clear_sky_lower_than_storm(self):
        clear = total_path_loss_dB(38_500_000, 30e9, 30, rain_rate_mm_per_hr=0.0)
        storm = total_path_loss_dB(38_500_000, 30e9, 30, rain_rate_mm_per_hr=30.0)
        assert storm["total_dB"] > clear["total_dB"]
