"""Doppler shift / rate / pass-envelope tests."""

import math
from datetime import datetime, timezone

from horizon_ric.planner.physics.doppler import (
    DopplerSample,
    doppler_shift_hz,
    pass_envelope,
    range_rate_m_s,
)
from horizon_ric.planner.physics.orbital import (
    KeplerianElements,
    keplerian_state,
)

_T0 = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)


def _starlink_passing_overhead() -> KeplerianElements:
    """A Starlink-altitude satellite passing roughly over (0°, 0°) ES."""
    return KeplerianElements(
        a_km=6378.137 + 550.0,
        e=0.0,
        i_rad=math.radians(53.0),
        raan_rad=0.0,
        argp_rad=0.0,
        M_rad=0.0,
        epoch_utc=_T0,
    )


class TestDopplerShift:
    def test_finite_at_28ghz(self):
        state = keplerian_state(_starlink_passing_overhead())
        f = doppler_shift_hz(state, es_lat_deg=0.0, es_lon_deg=0.0,
                             carrier_hz=28e9)
        # Magnitude should be < 1 MHz at LEO/Ka — sanity bound.
        assert -1e6 < f < 1e6

    def test_scales_with_carrier(self):
        state = keplerian_state(_starlink_passing_overhead())
        f1 = doppler_shift_hz(state, 0.0, 0.0, 1e9)
        f2 = doppler_shift_hz(state, 0.0, 0.0, 2e9)
        # Doppler is linear in carrier frequency.
        assert abs(f2 / f1 - 2.0) < 1e-6 or abs(f1) < 1e-3


class TestRangeRate:
    def test_reasonable_range_at_overhead(self):
        state = keplerian_state(_starlink_passing_overhead())
        rng, rate = range_rate_m_s(state, 0.0, 0.0)
        # Range to a 550 km LEO sat from below: ≥ 550 km
        assert rng >= 550_000 - 1000
        # Range rate magnitude bounded by orbital speed (7.6 km/s)
        assert abs(rate) < 7700.0


class TestPassEnvelope:
    def test_envelope_structure(self):
        state = keplerian_state(_starlink_passing_overhead())
        samples = pass_envelope(
            state,
            es_lat_deg=0.0, es_lon_deg=0.0,
            carrier_hz=28e9,
            duration_s=60.0,
            step_s=10.0,
        )
        assert len(samples) >= 6
        for s in samples:
            assert isinstance(s, DopplerSample)
            assert s.range_m > 0
        # Doppler shift should change over the pass.
        shifts = {s.shift_hz for s in samples}
        assert len(shifts) > 2
