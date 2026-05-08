"""EPFD time-CDF tests using a synthetic constellation propagator."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from horizon_ric.planner.physics.epfd import (
    EPFDTimeCDF,
    NGSOEmitter,
    epfd_time_cdf,
)
from horizon_ric.planner.physics.geodesy import ECEF, GSO_RADIUS_M


_T0 = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)


def _moving_emitter_factory(eirp_dBW: float = -10.0):
    """Closure: at each `t`, return a single NGSO emitter on a circle
    of radius 800 km above the equator, rotating uniformly. Visibility
    from (0, 0) ES is intermittent, giving a non-trivial CDF."""

    R = 6_378_137.0 + 800_000.0
    omega_rad_s = 2 * math.pi / 6000.0  # period 100 min

    def constellation_at(t: datetime) -> list[NGSOEmitter]:
        dt_s = (t - _T0).total_seconds()
        theta = omega_rad_s * dt_s
        return [NGSOEmitter(
            position_ecef=ECEF(R * math.cos(theta), R * math.sin(theta), 0.0),
            eirp_dBW=eirp_dBW,
            name="orbiter",
        )]

    return constellation_at


class TestEPFDTimeCDF:
    def test_runs_and_returns_correct_shape(self):
        cdf = epfd_time_cdf(
            es_lat_deg=0.0, es_lon_deg=0.0,
            wanted_gso_lon_deg=10.0,
            es_diameter_m=1.2, es_frequency_hz=12e9,
            constellation_at=_moving_emitter_factory(),
            t_start_utc=_T0,
            duration_s=600.0, step_s=60.0,
        )
        assert isinstance(cdf, EPFDTimeCDF)
        assert len(cdf.samples_dB) == 11   # 600/60 + 1
        assert cdf.duration_s == 600.0
        assert cdf.step_s == 60.0

    def test_percentile_monotonic(self):
        cdf = epfd_time_cdf(
            es_lat_deg=0.0, es_lon_deg=0.0,
            wanted_gso_lon_deg=10.0,
            es_diameter_m=1.2, es_frequency_hz=12e9,
            constellation_at=_moving_emitter_factory(),
            t_start_utc=_T0,
            duration_s=600.0, step_s=30.0,
        )
        # Higher percentile threshold → lower EPFD value (less time exceeded).
        worst = cdf.percentile(0.001)
        common = cdf.percentile(50.0)
        # 0.001% is the near-worst spike; 50% is the median.
        assert worst >= common

    def test_max_and_mean_in_sample_range(self):
        cdf = epfd_time_cdf(
            es_lat_deg=0.0, es_lon_deg=0.0,
            wanted_gso_lon_deg=10.0,
            es_diameter_m=1.2, es_frequency_hz=12e9,
            constellation_at=_moving_emitter_factory(),
            t_start_utc=_T0,
            duration_s=300.0, step_s=30.0,
        )
        assert cdf.max_dB() >= cdf.mean_dB() - 1e-6
        assert cdf.max_dB() == max(cdf.samples_dB)

    def test_invalid_durations_rejected(self):
        with pytest.raises(ValueError):
            epfd_time_cdf(
                es_lat_deg=0, es_lon_deg=0, wanted_gso_lon_deg=0,
                es_diameter_m=1.2, es_frequency_hz=12e9,
                constellation_at=lambda _: [],
                t_start_utc=_T0, duration_s=-1, step_s=60,
            )
        with pytest.raises(ValueError):
            epfd_time_cdf(
                es_lat_deg=0, es_lon_deg=0, wanted_gso_lon_deg=0,
                es_diameter_m=1.2, es_frequency_hz=12e9,
                constellation_at=lambda _: [],
                t_start_utc=_T0, duration_s=60, step_s=0,
            )

    def test_percentile_invalid(self):
        cdf = epfd_time_cdf(
            es_lat_deg=0.0, es_lon_deg=0.0,
            wanted_gso_lon_deg=10.0,
            es_diameter_m=1.2, es_frequency_hz=12e9,
            constellation_at=_moving_emitter_factory(),
            t_start_utc=_T0,
            duration_s=120.0, step_s=30.0,
        )
        with pytest.raises(ValueError):
            cdf.percentile(0.0)
        with pytest.raises(ValueError):
            cdf.percentile(100.0)

    def test_empty_constellation_floor(self):
        cdf = epfd_time_cdf(
            es_lat_deg=0.0, es_lon_deg=0.0,
            wanted_gso_lon_deg=10.0,
            es_diameter_m=1.2, es_frequency_hz=12e9,
            constellation_at=lambda _: [],   # never any emitters
            t_start_utc=_T0,
            duration_s=120.0, step_s=30.0,
        )
        # All samples should be at the -300 dB floor.
        assert cdf.max_dB() == -300.0
        assert cdf.n_visible_max == 0
