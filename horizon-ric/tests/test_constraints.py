"""Tests for the GSO PFD / spectral mask / edge GPU constraint layer."""

import torch

from horizon_ric.policy.constraints import (
    GSOArcEntry,
    PreceptualAIConstraintConfig,
    PreceptualAIConstraintLayer,
)


def _ctx(**overrides):
    base = {
        "earth_station_latitude_deg": 30.0,
        "earth_station_longitude_deg": 0.0,
        "permitted_band_hz": (3.4e9, 4.2e9),  # representative C-band downlink
    }
    base.update(overrides)
    return base


def _arcs():
    return [
        GSOArcEntry(satellite_id="sat-A", longitude_deg=0.0, operator="opA"),
        GSOArcEntry(satellite_id="sat-B", longitude_deg=20.0, operator="opB"),
    ]


class TestConstraintIDs:
    def test_advertises_three_hard_constraints(self):
        cl = PreceptualAIConstraintLayer()
        ids = cl.hard_constraint_ids()
        assert {"gso_pfd_floor", "itu_spectral_mask", "edge_gpu_capacity"} <= set(ids)


class TestSpectralMask:
    def test_below_band_violates(self):
        cl = PreceptualAIConstraintLayer(
            PreceptualAIConstraintConfig(gso_arcs=_arcs())
        )
        action = {"frequency_hz": 1.0e9, "tx_power_dBm": -200.0}
        viols = cl.check_feasibility(action, _ctx())
        ids = [v.constraint_id for v in viols]
        assert "itu_spectral_mask" in ids

    def test_in_band_passes_mask(self):
        cl = PreceptualAIConstraintLayer(
            PreceptualAIConstraintConfig(gso_arcs=_arcs())
        )
        action = {"frequency_hz": 3.7e9, "tx_power_dBm": -200.0}
        viols = cl.check_feasibility(action, _ctx())
        ids = [v.constraint_id for v in viols]
        assert "itu_spectral_mask" not in ids

    def test_projection_clips_frequency(self):
        cl = PreceptualAIConstraintLayer(
            PreceptualAIConstraintConfig(gso_arcs=_arcs())
        )
        action = {"frequency_hz": 5.0e9, "tx_power_dBm": -200.0}
        feasible, _ = cl.project(action, _ctx())
        assert feasible["frequency_hz"] == 4.2e9


class TestEdgeGPU:
    def test_excessive_memory_demand_violates(self):
        cl = PreceptualAIConstraintLayer(
            PreceptualAIConstraintConfig(
                gso_arcs=_arcs(),
                edge_gpu_memory_gb_max=80.0,
            )
        )
        action = {
            "frequency_hz": 3.7e9,
            "tx_power_dBm": -200.0,
            "ai_workload_gpu_gb": 200.0,
        }
        viols = cl.check_feasibility(action, _ctx())
        assert any(v.constraint_id == "edge_gpu_capacity" for v in viols)

    def test_projection_defers_excess_workload(self):
        cl = PreceptualAIConstraintLayer(
            PreceptualAIConstraintConfig(
                gso_arcs=_arcs(),
                edge_gpu_memory_gb_max=80.0,
            )
        )
        action = {
            "frequency_hz": 3.7e9,
            "tx_power_dBm": -200.0,
            "ai_workload_gpu_gb": 200.0,
        }
        feasible, _ = cl.project(action, _ctx())
        assert feasible["ai_workload_gpu_gb"] == 0.0
        assert feasible["workload_placement"] == "regional_edge_or_cloud"


class TestGSOPFDFloor:
    def test_zero_power_does_not_violate(self):
        cl = PreceptualAIConstraintLayer(
            PreceptualAIConstraintConfig(gso_arcs=_arcs())
        )
        action = {
            "frequency_hz": 3.7e9,
            "tx_power_dBm": -300.0,  # transmitter effectively off
            "antenna_gain_dBi": 14.0,
            "beam_azimuth_deg": 90.0,  # point away from GSO arc
            "beam_elevation_deg": 30.0,
        }
        viols = cl.check_feasibility(action, _ctx())
        assert not any(v.constraint_id == "gso_pfd_floor" for v in viols)


class TestLagrangianViolation:
    def test_returns_per_batch_nonneg(self):
        cl = PreceptualAIConstraintLayer(
            PreceptualAIConstraintConfig(gso_arcs=_arcs())
        )
        action = torch.tensor(
            [
                [10.0, 0.5, 30.0, 0.0],
                [50.0, 0.5, 30.0, 0.0],  # high TX power → larger violation
            ],
            dtype=torch.float32,
        )
        v = cl.lagrangian_violation(action, _ctx())
        assert v.shape == (2,)
        assert torch.all(v >= 0.0)
        # Higher TX power should produce a non-smaller violation
        assert v[1].item() >= v[0].item()

    def test_rejects_wrong_rank(self):
        cl = PreceptualAIConstraintLayer()
        bad = torch.zeros(4)
        try:
            cl.lagrangian_violation(bad, _ctx())
        except ValueError:
            return
        assert False, "expected ValueError on 1-D action"
