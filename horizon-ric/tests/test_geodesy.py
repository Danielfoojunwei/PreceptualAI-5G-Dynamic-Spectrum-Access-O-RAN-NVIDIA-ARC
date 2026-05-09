"""WGS-84 geodesy + look-angles tests."""

import math

from horizon_ric.planner.physics.geodesy import (
    ECEF,
    GSO_RADIUS_M,
    WGS84_A_M,
    angle_between_ecef_vectors_deg,
    ecef_distance_m,
    ecef_to_geodetic,
    geodetic_to_ecef,
    gso_satellite_ecef,
    look_angles_to_gso,
    look_angles_to_target,
)


class TestGeodeticECEFRoundTrip:
    def test_equator_zero_lon(self):
        p = geodetic_to_ecef(0.0, 0.0, 0.0)
        assert abs(p.x - WGS84_A_M) < 1e-3
        assert abs(p.y) < 1e-6
        assert abs(p.z) < 1e-6

    def test_round_trip(self):
        for lat, lon, h in [
            (0.0, 0.0, 0.0),
            (45.0, 90.0, 100.0),
            (-30.0, -60.0, 1000.0),
            (60.0, 180.0, 10_000.0),
        ]:
            p = geodetic_to_ecef(lat, lon, h)
            lat2, lon2, h2 = ecef_to_geodetic(p)
            assert abs(lat - lat2) < 1e-6
            assert abs(((lon - lon2) + 180) % 360 - 180) < 1e-6
            assert abs(h - h2) < 0.01


class TestLookAngles:
    def test_overhead_target_zenith(self):
        # ES at (0, 0); target straight up.
        p_target = ECEF(WGS84_A_M + 1_000_000, 0.0, 0.0)
        look = look_angles_to_target(0.0, 0.0, p_target)
        assert abs(look.elevation_deg - 90.0) < 1e-3
        assert abs(look.range_m - 1_000_000) < 1.0

    def test_below_horizon_negative_elevation(self):
        # ES at (89.9°N, 0°); GSO at (0°, 0°lon) is below horizon.
        look = look_angles_to_gso(89.9, 0.0, 0.0)
        assert look.elevation_deg < 0.0

    def test_gso_at_subpoint(self):
        # ES at (0, 0); GSO at (0°, 0°lon) directly overhead (zenith).
        look = look_angles_to_gso(0.0, 0.0, 0.0)
        assert abs(look.elevation_deg - 90.0) < 0.5
        assert look.range_m > 35e6  # at GSO altitude


class TestECEFDistance:
    def test_zero_distance(self):
        a = ECEF(1.0, 2.0, 3.0)
        assert ecef_distance_m(a, a) == 0.0

    def test_known_distance(self):
        a = ECEF(0, 0, 0)
        b = ECEF(3, 4, 0)
        assert abs(ecef_distance_m(a, b) - 5.0) < 1e-9


class TestAngleBetweenECEFVectors:
    def test_zero_for_same_point(self):
        origin = ECEF(0, 0, 0)
        p = ECEF(1, 0, 0)
        assert angle_between_ecef_vectors_deg(origin, p, p) < 1e-3

    def test_90_degrees(self):
        origin = ECEF(0, 0, 0)
        p1 = ECEF(1, 0, 0)
        p2 = ECEF(0, 1, 0)
        assert abs(angle_between_ecef_vectors_deg(origin, p1, p2) - 90.0) < 1e-3

    def test_180_degrees(self):
        origin = ECEF(0, 0, 0)
        p1 = ECEF(1, 0, 0)
        p2 = ECEF(-1, 0, 0)
        assert abs(angle_between_ecef_vectors_deg(origin, p1, p2) - 180.0) < 1e-3


class TestGSOSatellite:
    def test_radius(self):
        for lon in [0.0, 90.0, -45.0, 180.0]:
            p = gso_satellite_ecef(lon)
            assert abs(p.norm() - GSO_RADIUS_M) < 1.0

    def test_z_zero_equatorial(self):
        for lon in [0.0, 30.0, -90.0]:
            p = gso_satellite_ecef(lon)
            assert abs(p.z) < 1e-6

    def test_altitude_override(self):
        p550 = gso_satellite_ecef(0.0, altitude_m=550_000)
        assert abs(p550.norm() - (WGS84_A_M + 550_000)) < 1.0
