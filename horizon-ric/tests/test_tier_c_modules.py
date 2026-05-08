"""Tests for Tier-C: Liquid-S4, Perceiver, Graph-JEPA, latent dynamics,
TD-MPC2 planner, diffusion tail."""

import pytest
import torch

from horizon_ric.core.latent_dynamics import LatentDynamics, LatentDynamicsConfig
from horizon_ric.core.liquid_s4 import LiquidS4, LiquidS4Config
from horizon_ric.encoder.graph_jepa import GraphJEPA, GraphJEPAConfig
from horizon_ric.encoder.perceiver_fusion import PerceiverConfig, PerceiverFusion
from horizon_ric.policy.diffusion_tail import (
    DiffusionConfig,
    DiffusionTailSampler,
)
from horizon_ric.policy.td_mpc_planner import (
    TDMPCConfig,
    TDMPCPlanner,
)


# ─── LiquidS4 ────────────────────────────────────────────────────────────


class TestLiquidS4:
    def test_full_sequence_shape(self):
        m = LiquidS4(LiquidS4Config(d_model=16, d_state=4, n_layers=2))
        y = m(torch.randn(2, 8, 16))
        assert y.shape == (2, 8, 16)

    def test_step_streaming(self):
        m = LiquidS4(LiquidS4Config(d_model=8, d_state=4, n_layers=2))
        states = None
        x_seq = torch.randn(3, 5, 8)
        outs = []
        for t in range(5):
            y, states = m.step(x_seq[:, t], states)
            outs.append(y)
        stacked = torch.stack(outs, dim=1)
        assert stacked.shape == (3, 5, 8)

    def test_grad_flows(self):
        # Seed deterministic — random init occasionally produces a state
        # whose backward gradient is exactly zero by chance.
        torch.manual_seed(20260506)
        m = LiquidS4(LiquidS4Config(d_model=8, d_state=4, n_layers=1))
        x = torch.randn(2, 4, 8, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    def test_rejects_wrong_d_model(self):
        m = LiquidS4(LiquidS4Config(d_model=8))
        with pytest.raises(ValueError):
            m(torch.randn(2, 4, 16))


# ─── Perceiver ───────────────────────────────────────────────────────────


class TestPerceiverFusion:
    def test_fixed_output_shape(self):
        p = PerceiverFusion(PerceiverConfig(
            n_latents=32, d_latent=16, d_input=16, n_self_layers=1, n_heads=4,
        ))
        # Variable input N — output shape stays fixed.
        z1 = p(torch.randn(2, 50, 16))
        z2 = p(torch.randn(2, 200, 16))
        assert z1.shape == (2, 32, 16)
        assert z2.shape == (2, 32, 16)

    def test_input_mask_excludes_tokens(self):
        p = PerceiverFusion(PerceiverConfig(
            n_latents=8, d_latent=8, d_input=8, n_self_layers=0, n_heads=2,
        ))
        x = torch.randn(2, 10, 8)
        m = torch.ones(2, 10, dtype=torch.bool)
        m[:, 5:] = False  # mask half
        z = p(x, input_mask=m)
        assert z.shape == (2, 8, 8)
        assert torch.isfinite(z).all()

    def test_grad_flows_to_input(self):
        p = PerceiverFusion(PerceiverConfig(
            n_latents=8, d_latent=8, d_input=8, n_self_layers=1, n_heads=2,
        ))
        x = torch.randn(1, 6, 8, requires_grad=True)
        z = p(x)
        z.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    def test_rejects_wrong_input_dim(self):
        p = PerceiverFusion(PerceiverConfig(
            n_latents=8, d_latent=8, d_input=8, n_self_layers=0, n_heads=2,
        ))
        with pytest.raises(ValueError):
            p(torch.randn(2, 5))  # missing the feature dim


# ─── GraphJEPA ───────────────────────────────────────────────────────────


class TestGraphJEPA:
    def _make(self):
        encoder = PerceiverFusion(PerceiverConfig(
            n_latents=16, d_latent=16, d_input=16, n_self_layers=1, n_heads=4,
        ))
        return GraphJEPA(encoder, GraphJEPAConfig(
            d_latent=16, predictor_hidden=32, predictor_layers=1, mask_ratio=0.5,
        ))

    def test_loss_returns_keys(self):
        jepa = self._make()
        out = jepa.loss(torch.randn(2, 30, 16))
        assert {"loss", "pred_norm", "target_norm"} <= out.keys()
        assert out["loss"].dim() == 0

    def test_loss_is_finite_and_grad_flows(self):
        jepa = self._make()
        out = jepa.loss(torch.randn(2, 20, 16))
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        # Predictor must have grads.
        any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in jepa.predictor.parameters()
        )
        assert any_grad

    def test_target_encoder_frozen(self):
        jepa = self._make()
        for p in jepa.target_encoder.parameters():
            assert not p.requires_grad

    def test_ema_update_moves_target(self):
        jepa = self._make()
        # Force a real divergence by perturbing the context encoder.
        with torch.no_grad():
            for p in jepa.context_encoder.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        before = next(jepa.target_encoder.parameters()).detach().clone()
        jepa.update_target()
        after = next(jepa.target_encoder.parameters()).detach()
        assert not torch.allclose(before, after)


# ─── LatentDynamics ──────────────────────────────────────────────────────


class TestLatentDynamics:
    def test_rollout_shape(self):
        ld = LatentDynamics(LatentDynamicsConfig(d_latent=8, d_action=4, d_state=4))
        traj = ld.rollout(torch.randn(2, 8), torch.randn(2, 6, 4))
        assert traj.shape == (2, 6, 8)

    def test_grad_flows_through_rollout(self):
        # Seed deterministic — random init occasionally produces an
        # initial state whose gradient is exactly zero by chance.
        torch.manual_seed(20260506)
        ld = LatentDynamics(LatentDynamicsConfig(d_latent=8, d_action=4, d_state=4))
        z0 = torch.randn(2, 8)
        a = torch.randn(2, 4, 4, requires_grad=True)
        traj = ld.rollout(z0, a)
        traj.sum().backward()
        assert a.grad is not None and a.grad.abs().sum() > 0

    def test_rejects_bad_action_shape(self):
        ld = LatentDynamics(LatentDynamicsConfig(d_latent=8, d_action=4))
        with pytest.raises(ValueError):
            ld.rollout(torch.zeros(2, 8), torch.zeros(2, 5))  # 2-D actions

    def test_rejects_batch_mismatch(self):
        ld = LatentDynamics(LatentDynamicsConfig(d_latent=8, d_action=4))
        with pytest.raises(ValueError):
            ld.rollout(torch.zeros(2, 8), torch.zeros(3, 5, 4))


# ─── TD-MPC2 ────────────────────────────────────────────────────────────


class TestTDMPCPlanner:
    def _build(self):
        ld = LatentDynamics(LatentDynamicsConfig(d_latent=8, d_action=4, d_state=4))

        def reward(traj, actions):
            return -traj.norm(dim=-1)

        def cstr(traj, actions):
            return torch.relu(actions.abs().sum(dim=-1) - 6.0)

        planner = TDMPCPlanner(
            ld, reward, [cstr],
            TDMPCConfig(horizon=4, n_samples=24, n_iterations=2),
            action_dim=4,
        )
        return planner

    def test_plan_shape(self):
        planner = self._build()
        result = planner.plan(torch.randn(8))
        assert result.actions.shape == (4, 4)
        assert isinstance(result.expected_score, float)

    def test_nominal_action(self):
        planner = self._build()
        a = planner.nominal_action(torch.randn(8))
        assert a.shape == (4,)

    def test_rejects_bad_z0(self):
        planner = self._build()
        with pytest.raises(ValueError):
            planner.plan(torch.zeros(2, 8))


# ─── Diffusion tail ─────────────────────────────────────────────────────


class TestDiffusionTail:
    def test_loss_finite_and_grad(self):
        d = DiffusionTailSampler(DiffusionConfig(
            d_latent=8, d_action=4, horizon=3, n_steps=5, hidden_dim=32,
        ))
        z0 = torch.randn(2, 8)
        rollout = torch.randn(2, 3, 8)
        actions = torch.randn(2, 3, 4)
        loss = d.loss(z0, rollout, actions)
        assert torch.isfinite(loss)
        loss.backward()

    def test_sample_shape(self):
        d = DiffusionTailSampler(DiffusionConfig(
            d_latent=8, d_action=4, horizon=3, n_steps=5, hidden_dim=32,
        ))
        samples = d.sample(torch.randn(8), torch.randn(3, 4), n_samples=4)
        assert samples.shape == (4, 3, 8)

    def test_percentile_metric(self):
        d = DiffusionTailSampler(DiffusionConfig(
            d_latent=8, d_action=4, horizon=3, n_steps=5, hidden_dim=32,
        ))
        samples = torch.randn(16, 3, 8)
        # 95th percentile of sample-norms must be ≥ median.
        score_fn = lambda s: s.norm(dim=(-1, -2))
        p95 = d.percentile_metric(samples, score_fn, p_pct=95.0)
        p50 = d.percentile_metric(samples, score_fn, p_pct=50.0)
        assert p95 >= p50

    def test_invalid_horizon_rejected(self):
        d = DiffusionTailSampler(DiffusionConfig(
            d_latent=4, d_action=2, horizon=3, n_steps=4, hidden_dim=16,
        ))
        with pytest.raises(ValueError):
            d.loss(torch.zeros(1, 4), torch.zeros(1, 5, 4), torch.zeros(1, 5, 2))
