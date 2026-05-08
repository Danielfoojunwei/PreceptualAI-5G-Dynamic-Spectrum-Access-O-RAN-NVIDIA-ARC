"""Additional constraint-layer tests for the post-audit fixes."""

import math

import torch

from horizon_ric.policy.constraints import (
    GSOArcEntry,
    PreceptualAIConstraintConfig,
    PreceptualAIConstraintLayer,
)


def _ctx():
    return {
        "earth_station_latitude_deg": 30.0,
        "earth_station_longitude_deg": 0.0,
        "permitted_band_hz": (3.4e9, 4.2e9),
    }


class TestGSOAzimuthFormula:
    """GSO at sub-equator longitude: ES at 30°N looking at GSO south of it
    should produce an azimuth between 90° and 270° (south half-plane)."""

    def test_azimuth_due_south_for_es_directly_north(self):
        cl = PreceptualAIConstraintLayer()
        # ES at 45°N, 0°E; GSO at 0°E. Bearing must be 180° (due south).
        az = cl._gso_azimuth_from_es(45.0, 0.0, 0.0)
        assert abs(az - 180.0) < 0.5

    def test_azimuth_in_south_half_plane(self):
        cl = PreceptualAIConstraintLayer()
        # ES at 30°N, 0°E; GSO at 30°E. Should be SE quadrant (90°–180°).
        az = cl._gso_azimuth_from_es(30.0, 0.0, 30.0)
        assert 90.0 < az < 180.0

    def test_azimuth_normalised_to_0_360(self):
        cl = PreceptualAIConstraintLayer()
        for es_lat in [-45.0, 0.0, 30.0, 60.0]:
            for gso_lon in [-90.0, -10.0, 10.0, 100.0]:
                az = cl._gso_azimuth_from_es(es_lat, 0.0, gso_lon)
                assert 0.0 <= az < 360.0


class TestGSOElevationGeometry:
    def test_es_at_pole_sees_satellite_below_horizon(self):
        cl = PreceptualAIConstraintLayer()
        # GSO is in the equatorial plane; from the geographic pole it's
        # below the horizon (negative elevation).
        el = cl._gso_elevation_from_es(89.9, 0.0, 0.0)
        assert el < 0.0

    def test_es_at_subpoint_sees_satellite_at_zenith(self):
        cl = PreceptualAIConstraintLayer()
        el = cl._gso_elevation_from_es(0.0, 0.0, 0.0)
        assert abs(el - 90.0) < 0.5

    def test_uses_config_for_ratio(self):
        # Override gso_altitude_km — elevation must change.
        c1 = PreceptualAIConstraintConfig(gso_altitude_km=35_786.0)
        c2 = PreceptualAIConstraintConfig(gso_altitude_km=20_000.0)
        cl1 = PreceptualAIConstraintLayer(c1)
        cl2 = PreceptualAIConstraintLayer(c2)
        el1 = cl1._gso_elevation_from_es(30.0, 0.0, 30.0)
        el2 = cl2._gso_elevation_from_es(30.0, 0.0, 30.0)
        assert el1 != el2  # different altitudes give different geometry


class TestProjectionDoesNotEmitInf:
    def test_projection_uses_finite_off_sentinel(self):
        cl = PreceptualAIConstraintLayer(
            PreceptualAIConstraintConfig(
                gso_arcs=[GSOArcEntry(satellite_id="s", longitude_deg=0.0)],
                pfd_floor_dBW_per_m2=-1000.0,  # impossible — force fail-out
            )
        )
        action = {
            "frequency_hz": 3.7e9,
            "tx_power_dBm": 30.0,
            "antenna_gain_dBi": 14.0,
            "beam_azimuth_deg": 180.0,
            "beam_elevation_deg": 60.0,
        }
        feasible, viols = cl.project(action, _ctx())
        # Must not contain -inf in any numeric field
        assert math.isfinite(feasible["tx_power_dBm"])
        # Must explicitly disable transmitter on hard refusal
        assert feasible.get("transmitter_enabled") is False


class TestLagrangianGradientFlow:
    def test_grad_flows_through_tx_power(self):
        cl = PreceptualAIConstraintLayer(
            PreceptualAIConstraintConfig(
                gso_arcs=[GSOArcEntry(satellite_id="s", longitude_deg=0.0)]
            )
        )
        action = torch.tensor(
            [[40.0, 14.0, 180.0, 60.0]], requires_grad=True
        )  # tx_dBm, gain, beam_az, beam_el
        v = cl.lagrangian_violation(action, _ctx())
        v.sum().backward()
        # Gradient w.r.t. tx_power_dBm should be exactly 1 when violating
        assert action.grad is not None
        assert action.grad[0, 0].item() > 0

    def test_grad_flows_through_beam_direction(self):
        cl = PreceptualAIConstraintLayer(
            PreceptualAIConstraintConfig(
                gso_arcs=[GSOArcEntry(satellite_id="s", longitude_deg=0.0)]
            )
        )
        # Pointing very close to GSO direction → high off-axis gain → violation
        action = torch.tensor(
            [[40.0, 14.0, 180.0, 60.0]], requires_grad=True
        )
        v = cl.lagrangian_violation(action, _ctx())
        v.sum().backward()
        # At least one of the beam direction grads should be non-zero
        assert action.grad is not None
        assert action.grad[0, 2:].abs().sum().item() > 0

    def test_no_arcs_returns_zero(self):
        cl = PreceptualAIConstraintLayer()  # no arcs configured
        action = torch.tensor([[50.0, 14.0, 0.0, 30.0]])
        v = cl.lagrangian_violation(action, _ctx())
        assert torch.all(v == 0.0)

    def test_rejects_too_few_dims(self):
        cl = PreceptualAIConstraintLayer()
        bad = torch.zeros(2, 2)  # only 2 dims, need 4
        try:
            cl.lagrangian_violation(bad, _ctx())
        except ValueError:
            return
        assert False, "expected ValueError for action with <4 dims"
