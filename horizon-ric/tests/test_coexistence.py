"""Multi-band coexistence physics tests."""

import pytest

from horizon_ric.planner.physics.coexistence import (
    BandEnvelope,
    InlineEventStat,
    aggregate_aclr_leakage_dBm,
    compose_feasibility,
    ngso_inline_event_probability,
    ntn_to_terrestrial_required_guard_db,
    nru_fair_share_airtime,
    p_servicelink_rain_given_gateway,
)


class TestNGSOInLine:
    def test_basic_probability(self):
        # Design brief: 12 Starlink, 3 OneWeb, α=1°, ε_min=25° → ~0.94%
        s = ngso_inline_event_probability(
            visible_a=12, visible_b=3, in_line_angle_deg=1.0, min_elevation_deg=25.0
        )
        assert isinstance(s, InlineEventStat)
        assert 0.005 < s.p_inline < 0.02
        assert s.expected_seconds_per_orbit > 20

    def test_zero_visible_zero_prob(self):
        s = ngso_inline_event_probability(0, 5, 1.0)
        assert s.p_inline == 0.0
        s = ngso_inline_event_probability(5, 0, 1.0)
        assert s.p_inline == 0.0

    def test_larger_cone_more_prob(self):
        s_narrow = ngso_inline_event_probability(10, 10, in_line_angle_deg=0.5)
        s_wide = ngso_inline_event_probability(10, 10, in_line_angle_deg=5.0)
        assert s_wide.p_inline > s_narrow.p_inline

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            ngso_inline_event_probability(-1, 5, 1.0)
        with pytest.raises(ValueError):
            ngso_inline_event_probability(5, 5, 0.0)
        with pytest.raises(ValueError):
            ngso_inline_event_probability(5, 5, 1.0, min_elevation_deg=95.0)


class TestNRUAirtime:
    def test_no_wifi_full_airtime_minus_deferral(self):
        a = nru_fair_share_airtime(0, cross_rat_deferral_pct=0.0)
        assert a == 1.0

    def test_airtime_in_unit_interval(self):
        # Bianchi with fixed τ has a non-monotonic ρ_w(N) — peak around the
        # mid-range. We only assert that the result is a sensible fraction.
        for n in (1, 2, 5, 10, 50):
            a = nru_fair_share_airtime(n)
            assert 0.0 <= a <= 1.0

    def test_brief_value(self):
        # Brief: 8 WiFi stations → ~ 50% NR-U airtime
        a = nru_fair_share_airtime(8)
        assert 0.4 < a < 0.6

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            nru_fair_share_airtime(-1)
        with pytest.raises(ValueError):
            nru_fair_share_airtime(5, cross_rat_deferral_pct=200.0)


class TestACLRAggregate:
    def test_two_ues_3db_higher(self):
        # Two equal-power leakers should aggregate +3 dB.
        single = aggregate_aclr_leakage_dBm([20.0])
        pair = aggregate_aclr_leakage_dBm([20.0, 20.0])
        assert abs((pair - single) - 3.0) < 0.05

    def test_empty_returns_floor(self):
        assert aggregate_aclr_leakage_dBm([]) == -300.0

    def test_path_loss_subtracts(self):
        with_pl = aggregate_aclr_leakage_dBm([20.0], victim_path_loss_dB=70.0)
        no_pl = aggregate_aclr_leakage_dBm([20.0], victim_path_loss_dB=0.0)
        assert no_pl - with_pl == pytest.approx(70.0, abs=0.05)

    def test_invalid_aclr(self):
        with pytest.raises(ValueError):
            aggregate_aclr_leakage_dBm([20.0], ue_aclr_dB=-5.0)


class TestNTNToTerrestrialGuardBand:
    def test_close_distance_more_suppression(self):
        close = ntn_to_terrestrial_required_guard_db(
            ue_tx_power_dBm=23, ue_back_lobe_gain_dBi=-10,
            distance_to_gnb_m=100, frequency_hz=2e9,
        )
        far = ntn_to_terrestrial_required_guard_db(
            ue_tx_power_dBm=23, ue_back_lobe_gain_dBi=-10,
            distance_to_gnb_m=10_000, frequency_hz=2e9,
        )
        assert close > far

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            ntn_to_terrestrial_required_guard_db(0.0, 0.0, 0.0, 1e9)
        with pytest.raises(ValueError):
            ntn_to_terrestrial_required_guard_db(0.0, 0.0, 100.0, 0.0)


class TestRainConditional:
    def test_zero_distance_full_correlation(self):
        # At d=0, ρ ≈ 1 → conditional probability ≈ 1 when threshold ≤ gateway
        p = p_servicelink_rain_given_gateway(0.0, rain_rate_gateway_mm_hr=10,
                                              rain_rate_threshold_mm_hr=5)
        assert p > 0.9

    def test_distance_decreases_correlation(self):
        near = p_servicelink_rain_given_gateway(10, 10, 5)
        far = p_servicelink_rain_given_gateway(500, 10, 5)
        assert near > far

    def test_zero_gateway_returns_zero(self):
        p = p_servicelink_rain_given_gateway(20, 0, 5)
        assert p == 0.0

    def test_negative_distance_rejected(self):
        with pytest.raises(ValueError):
            p_servicelink_rain_given_gateway(-1, 10, 5)


class TestComposeFeasibility:
    def test_finds_admitting_band(self):
        envs = [
            BandEnvelope("n78", 3.3e9, 3.8e9, 50.0, "TS 38.104"),
            BandEnvelope("n258", 24.25e9, 27.5e9, 65.0, "TS 38.104"),
        ]
        match = compose_feasibility(envs, 3.5e9, 45.0)
        assert match is not None
        assert match.name == "n78"

    def test_eirp_too_high_no_match(self):
        envs = [
            BandEnvelope("n78", 3.3e9, 3.8e9, 50.0, "TS 38.104"),
        ]
        match = compose_feasibility(envs, 3.5e9, 60.0)  # over the cap
        assert match is None

    def test_freq_outside_no_match(self):
        envs = [
            BandEnvelope("n78", 3.3e9, 3.8e9, 50.0, "TS 38.104"),
        ]
        match = compose_feasibility(envs, 5.0e9, 30.0)
        assert match is None

    def test_empty_envelopes(self):
        assert compose_feasibility([], 3.5e9, 30.0) is None
