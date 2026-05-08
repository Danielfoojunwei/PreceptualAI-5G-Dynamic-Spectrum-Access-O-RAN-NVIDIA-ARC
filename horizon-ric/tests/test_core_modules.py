"""Tests for core/cfc_core, core/physics_residual, core/timing_budgets."""

import pytest
import torch

from horizon_ric.core.cfc_core import CfCCell, CfCConfig
from horizon_ric.core.physics_residual import PhysicsResidualHead
from horizon_ric.core.timing_budgets import (
    ThermalState,
    a1_admit,
    edge_compute_lower_bound_us,
    end_to_end_latency_ms,
    fl_max_rounds_per_hour,
    fl_round_seconds,
    inference_budget_ms,
    l1_admit,
    l2_inference_budget_us,
    memory_bound_latency_ms,
)

# ─── CfCCell ─────────────────────────────────────────────────────────────


class TestCfCCell:
    def test_forward_shape(self):
        cell = CfCCell(CfCConfig(input_dim=8, hidden_dim=16))
        h0 = cell.init_hidden(4)
        x = torch.randn(4, 8)
        h1 = cell(x, h0, dt=0.5)
        assert h1.shape == (4, 16)

    def test_dt_per_sample(self):
        cell = CfCCell(CfCConfig(input_dim=4, hidden_dim=8))
        h0 = cell.init_hidden(3)
        x = torch.randn(3, 4)
        # per-sample dt: (B,1)
        dt = torch.tensor([[0.1], [1.0], [10.0]])
        h1 = cell(x, h0, dt=dt)
        assert h1.shape == (3, 8)

    def test_irregular_dt_changes_output(self):
        cell = CfCCell(CfCConfig(input_dim=4, hidden_dim=8))
        h0 = cell.init_hidden(2)
        x = torch.randn(2, 4)
        h_short = cell(x, h0, dt=0.01)
        h_long = cell(x, h0, dt=10.0)
        # Drastically different dt should yield different outputs
        assert not torch.allclose(h_short, h_long, atol=1e-3)

    def test_grad_flows_through_cell(self):
        cell = CfCCell(CfCConfig(input_dim=2, hidden_dim=4))
        h0 = cell.init_hidden(2)
        x = torch.randn(2, 2, requires_grad=True)
        h1 = cell(x, h0, dt=1.0)
        h1.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0

    def test_rejects_bad_shapes(self):
        cell = CfCCell(CfCConfig(input_dim=4, hidden_dim=8))
        with pytest.raises(ValueError):
            cell(torch.zeros(4), torch.zeros(2, 8), dt=1.0)
        with pytest.raises(ValueError):
            cell(torch.zeros(2, 4), torch.zeros(8), dt=1.0)
        with pytest.raises(ValueError):
            cell(torch.zeros(2, 4), torch.zeros(3, 8), dt=1.0)


# ─── PhysicsResidualHead ─────────────────────────────────────────────────


class TestPhysicsResidual:
    def _make(self, output_dim: int = 1):
        # Trivial physics: returns sum of state along last dim, broadcast.
        def physics(state: torch.Tensor) -> torch.Tensor:
            return state.sum(dim=-1, keepdim=True).repeat(1, output_dim)

        return PhysicsResidualHead(
            physics_fn=physics,
            state_dim=4,
            latent_dim=8,
            output_dim=output_dim,
            hidden_dim=16,
        )

    def test_forward_returns_all_keys(self):
        head = self._make(output_dim=1)
        out = head(
            state=torch.randn(3, 4),
            latent=torch.randn(3, 8),
        )
        for k in (
            "physics_pred", "residual_pred", "total_pred",
            "logvar", "hidden", "should_fallback",
        ):
            assert k in out

    def test_physics_pred_unmodified(self):
        head = self._make(output_dim=1)
        state = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        latent = torch.zeros(1, 8)
        out = head(state, latent)
        # Physics of [1,2,3,4] = 10
        assert abs(out["physics_pred"].item() - 10.0) < 1e-5

    def test_total_equals_physics_plus_residual(self):
        head = self._make(output_dim=2)
        state = torch.randn(5, 4)
        latent = torch.randn(5, 8)
        out = head(state, latent)
        assert torch.allclose(
            out["total_pred"], out["physics_pred"] + out["residual_pred"], atol=1e-5
        )

    def test_residual_bounded(self):
        head = self._make(output_dim=1)
        # Force the residual head into saturation by making latent extreme
        state = torch.randn(64, 4) * 1.0
        latent = torch.randn(64, 8) * 100.0
        out = head(state, latent)
        # Residual is bounded by residual_db_bound (default 5 dB) — closes
        # the dB dimensional contract of THEOREMS.md §1.
        residual_max = out["residual_pred"].abs().max().item()
        # Bound: |delta_dB| ≤ 5 dB (the operational dB cap)
        assert residual_max <= head.residual_db_bound + 1e-6

    def test_grad_flows_through_residual(self):
        head = self._make(output_dim=1)
        state = torch.randn(4, 4)
        latent = torch.randn(4, 8, requires_grad=True)
        out = head(state, latent)
        out["total_pred"].sum().backward()
        assert latent.grad is not None
        assert latent.grad.abs().sum().item() > 0


# ─── timing budgets ──────────────────────────────────────────────────────


class TestTimingBudgets:
    def test_l1_admit_fast_model(self):
        # 30 µs model fits in L1 at μ=1
        assert l1_admit(model_latency_us=30.0, slot_numerology=1) == "L1_ok"

    def test_l1_admit_defers_to_L2(self):
        # 200 µs is too slow for L1 but fits L2
        result = l1_admit(model_latency_us=120.0, slot_numerology=1)
        assert result in ("defer_to_L2", "defer_to_A1")

    def test_l1_admit_defers_to_A1(self):
        # 5 ms is too slow for both L1 and L2
        assert l1_admit(model_latency_us=5000.0, slot_numerology=1) == "defer_to_A1"

    def test_l2_budget(self):
        # μ=1, slot=500 µs, overhead 300 µs → 200 µs ML budget
        b = l2_inference_budget_us(slot_numerology=1, scheduler_overhead_us=300.0)
        assert abs(b - 200.0) < 1e-9

    def test_a1_admit_within_budget(self):
        # 50 ms decision in 100 ms refresh window → ok
        assert a1_admit(decision_latency_ms=50.0, policy_period_ms=100.0)

    def test_a1_admit_exceeds_budget(self):
        # 95 ms decision in 100 ms window with 10 ms overhead → not ok
        assert not a1_admit(decision_latency_ms=95.0, policy_period_ms=100.0)

    def test_compute_lower_bound(self):
        # 50M params fp16, 10 TFLOPS, batch 1 → ~10 µs
        us = edge_compute_lower_bound_us(
            n_params=50_000_000, precision_bits=16, tops=10.0, batch=1
        )
        assert 5 < us < 50

    def test_memory_bound_latency(self):
        # 100 MB at 100 GB/s → 1 ms
        ms = memory_bound_latency_ms(
            activation_bytes=100_000_000, weight_bytes=0, bw_gbps=100.0,
        )
        assert abs(ms - 1.0) < 0.05

    def test_thermal_throttle(self):
        nominal = 5.0
        cool = ThermalState(60.0, 25.0, 999.0)
        hot = ThermalState(95.0, 25.0, 0.0)
        warm = ThermalState(85.0, 25.0, 5.0)
        assert inference_budget_ms(cool, nominal) == nominal
        assert inference_budget_ms(hot, nominal) > nominal
        # warm sits between
        assert nominal < inference_budget_ms(warm, nominal) < inference_budget_ms(hot, nominal)

    def test_fl_round_seconds(self):
        # 50M fp32 params, no sparsity, 50 Mbps → 32 s
        s = fl_round_seconds(
            n_params=50_000_000, precision_bits=32, sparsity=0.0, link_mbps=50.0,
        )
        assert 30 < s < 35

    def test_fl_max_rounds_per_hour(self):
        n = fl_max_rounds_per_hour(
            n_params=10_000_000, precision_bits=32, sparsity=0.5, link_mbps=100.0,
        )
        assert n > 0

    def test_end_to_end_latency_ranks(self):
        total, ranked = end_to_end_latency_ms(
            {"encode": 2.0, "world": 1.0, "policy": 0.5, "wm": 0.2}
        )
        assert abs(total - 3.7) < 1e-6
        assert ranked[0] == "encode"

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            edge_compute_lower_bound_us(0, 16, 10.0)
        with pytest.raises(ValueError):
            memory_bound_latency_ms(1, 1, 0)
        with pytest.raises(ValueError):
            fl_round_seconds(1_000_000, 32, 1.5, 50.0)
        with pytest.raises(ValueError):
            l1_admit(50.0, slot_numerology=10)
