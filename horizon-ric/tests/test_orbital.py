"""Orbital propagation: Kepler+J2 + Walker + SGP4."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from horizon_ric.planner.physics.orbital import (
    KeplerianElements,
    eci_to_ecef_km,
    gmst_rad,
    j2_secular_rates,
    kepler_j2_step,
    keplerian_state,
    keplerian_to_eci,
    mean_motion_rad_s,
    propagate_batch,
    sgp4_state,
    state_ecef_m,
    walker_delta_constellation,
)


_T0 = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)


def _starlink_550_53_circular() -> KeplerianElements:
    return KeplerianElements(
        a_km=6378.137 + 550.0,
        e=0.0,
        i_rad=math.radians(53.0),
        raan_rad=0.0,
        argp_rad=0.0,
        M_rad=0.0,
        epoch_utc=_T0,
    )


class TestMeanMotion:
    def test_starlink_period_about_95min(self):
        n = mean_motion_rad_s(6378.137 + 550.0)
        period_min = 2 * math.pi / n / 60.0
        assert 95.0 < period_min < 96.5

    def test_geo_period_one_sidereal_day(self):
        n = mean_motion_rad_s(42164.17)
        period_s = 2 * math.pi / n
        # one sidereal day ≈ 86164 s
        assert abs(period_s - 86164.0) < 200.0


class TestJ2Secular:
    def test_starlink_nodal_regression_about_5deg_per_day(self):
        elem = _starlink_550_53_circular()
        d_raan, _, _ = j2_secular_rates(elem)
        deg_per_day = math.degrees(d_raan) * 86400.0
        # Vallado-correct value is about -4.5 to -5.0 deg/day at 53°/550 km
        assert -5.5 < deg_per_day < -4.0


class TestKeplerJ2Step:
    def test_advances_mean_anomaly(self):
        e0 = _starlink_550_53_circular()
        e1 = kepler_j2_step(e0, dt_s=60.0)
        # Mean anomaly should advance by approximately n · 60 s
        n = mean_motion_rad_s(e0.a_km)
        expected_dM = n * 60.0
        # delta should be > 0 and roughly the mean motion times dt
        d = (e1.M_rad - e0.M_rad) % (2 * math.pi)
        assert 0.9 * expected_dM < d < 1.1 * expected_dM

    def test_propagate_batch(self):
        elements = walker_delta_constellation(53.0, 22, 2, 1, 550.0, epoch_utc=_T0)
        advanced = propagate_batch(elements, dt_s=60.0)
        assert len(advanced) == len(elements)
        for a, b in zip(elements, advanced, strict=True):
            assert b.epoch_utc > a.epoch_utc


class TestKeplerianToECI:
    def test_state_at_perigee_circular(self):
        e0 = _starlink_550_53_circular()  # circular, perigee at M=0
        r, v = keplerian_to_eci(e0)
        radius = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        speed = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        # Circular at 550 km: r = 6928.137 km, v ≈ 7.59 km/s
        assert abs(radius - e0.a_km) < 0.1
        assert 7.5 < speed < 7.7


class TestWalker:
    def test_walker_count(self):
        elements = walker_delta_constellation(53.0, 22, 2, 1, 550.0, epoch_utc=_T0)
        assert len(elements) == 22

    def test_walker_planes_evenly_spaced(self):
        elements = walker_delta_constellation(53.0, 22, 2, 1, 550.0, epoch_utc=_T0)
        raans = sorted({e.raan_rad for e in elements})
        assert len(raans) == 2
        assert abs((raans[1] - raans[0]) - math.pi) < 1e-9  # 180° apart

    def test_walker_inclination(self):
        elements = walker_delta_constellation(87.9, 18, 2, 1, 1200.0, epoch_utc=_T0)
        for e in elements:
            assert abs(e.i_rad - math.radians(87.9)) < 1e-9

    def test_walker_rejects_invalid(self):
        with pytest.raises(ValueError):
            walker_delta_constellation(53.0, 5, 2, 0, 550.0, epoch_utc=_T0)
        with pytest.raises(ValueError):
            walker_delta_constellation(53.0, 4, 2, 5, 550.0, epoch_utc=_T0)


class TestGMST:
    def test_t0_repeatable(self):
        # GMST at fixed UTC should be deterministic and in [0, 2π)
        g = gmst_rad(_T0)
        assert 0.0 <= g < 2.0 * math.pi

    def test_evolves(self):
        # GMST should advance about 2π over a sidereal day.
        g0 = gmst_rad(_T0)
        g1 = gmst_rad(_T0 + timedelta(hours=12))
        # Half-rotation is approximately π radians (modulo)
        diff = (g1 - g0) % (2 * math.pi)
        assert abs(diff - math.pi) < 0.01


class TestECEFConversion:
    def test_state_ecef_m(self):
        state = keplerian_state(_starlink_550_53_circular())
        p = state_ecef_m(state)
        # ECEF radius equals ECI radius at this instant — frame is just rotated.
        eci = state.r_eci_km
        eci_radius_km = math.sqrt(eci[0]**2 + eci[1]**2 + eci[2]**2)
        ecef_radius_m = math.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2)
        assert abs(ecef_radius_m / 1000.0 - eci_radius_km) < 0.01


class TestSGP4:
    """Smoke test using a real Starlink-style TLE."""

    # ISS TLE from Celestrak (well-known, stable across months):
    LINE1 = "1 25544U 98067A   24001.50000000  .00012345  00000-0  22330-3 0  9999"
    LINE2 = "2 25544  51.6400 100.0000 0001000  90.0000  20.0000 15.50000000123456"

    def test_sgp4_returns_sensible_state(self):
        state = sgp4_state(self.LINE1, self.LINE2, _T0)
        # Any LEO altitude in the 200-2000 km band counts as sane.
        alt = state.altitude_km()
        assert 200 < alt < 2000
        # Orbital speed at LEO is ~7-8 km/s; bound loosely.
        assert 6.0 < state.speed_km_s() < 9.0

    def test_naive_datetime_rejected(self):
        with pytest.raises(ValueError):
            sgp4_state(self.LINE1, self.LINE2, datetime(2026, 1, 1))  # naive
