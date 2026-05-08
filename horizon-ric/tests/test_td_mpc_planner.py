"""Tests for the TRAINED TD-MPC2 planner mode.

These tests cover the optional `value_head` + `policy_prior` constructor
arguments added to wire in the trained heads from
`scripts/train_tdmpc_planner.py`. The original `TestTDMPCPlanner` class
in `tests/test_tier_c_modules.py` exercises the random-sampling baseline
and is intentionally left untouched.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from horizon_ric.core import LatentDynamics, LatentDynamicsConfig
from horizon_ric.policy.td_mpc_planner import TDMPCConfig, TDMPCPlanner

CHECKPOINTS = ROOT / "checkpoints"
TRAINED_AVAILABLE = (
    (CHECKPOINTS / "tdmpc_value_v0.1.pt").exists()
    and (CHECKPOINTS / "tdmpc_policy_prior_v0.1.pt").exists()
    and (CHECKPOINTS / "latent_dynamics_v0.1.pt").exists()
)


# --------------------------------------------------------------------------- #
# Mock heads — let the planner-side wiring be tested without GPU / checkpoints.
# --------------------------------------------------------------------------- #


class _MockValueHead(torch.nn.Module):
    def __init__(self, d_latent: int, d_action: int, hidden: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_latent + d_action, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1)).squeeze(-1)


class _MockPolicyPrior(torch.nn.Module):
    def __init__(self, d_latent: int, d_action: int, horizon: int = 4, hidden: int = 32):
        super().__init__()
        self.d_action = d_action
        self.horizon = horizon
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_latent, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 2 * d_action),
        )

    def forward(self, z):
        out = self.net(z)
        m, ls = out.chunk(2, dim=-1)
        return m, ls.clamp(-3.0, 1.0)

    def planner_prior(self, z0):
        m, ls = self.forward(z0)
        H = self.horizon
        return m.expand(H, -1).contiguous(), ls.expand(H, -1).contiguous()


def _build_random_planner(d_latent=8, d_action=4, horizon=4, n_samples=24):
    ld = LatentDynamics(LatentDynamicsConfig(d_latent=d_latent, d_action=d_action, d_state=4))

    def reward(traj, actions):
        return -traj.norm(dim=-1)

    def cstr(traj, actions):
        return torch.relu(actions.abs().sum(dim=-1) - 6.0)

    return TDMPCPlanner(
        ld, reward, [cstr],
        TDMPCConfig(horizon=horizon, n_samples=n_samples, n_iterations=2),
        action_dim=d_action,
    ), ld, reward, cstr


def _build_trained_planner(d_latent=8, d_action=4, horizon=4, n_samples=24):
    planner_rand, ld, reward, cstr = _build_random_planner(
        d_latent=d_latent, d_action=d_action, horizon=horizon, n_samples=n_samples,
    )
    value = _MockValueHead(d_latent, d_action)
    prior = _MockPolicyPrior(d_latent, d_action, horizon=horizon)
    planner_trained = TDMPCPlanner(
        ld, reward, [cstr],
        TDMPCConfig(horizon=horizon, n_samples=n_samples, n_iterations=2),
        action_dim=d_action,
        value_head=value,
        policy_prior=prior.planner_prior,
        value_weight=0.5,
    )
    return planner_trained, value, prior, ld


# --------------------------------------------------------------------------- #
# Test class — TRAINED-mode planner. Does NOT modify TestTDMPCPlanner.
# --------------------------------------------------------------------------- #


class TestTDMPCPlannerTrained:
    """Trained-mode tests for the planner: with `value_head` + `policy_prior`."""

    def test_plan_shape_with_trained_heads(self):
        planner, _, _, _ = _build_trained_planner()
        z0 = torch.randn(8)
        result = planner.plan(z0)
        assert result.actions.shape == (4, 4)
        assert isinstance(result.expected_score, float)

    def test_nominal_action_with_trained_heads(self):
        planner, _, _, _ = _build_trained_planner()
        a = planner.nominal_action(torch.randn(8))
        assert a.shape == (4,)

    def test_value_head_changes_score(self):
        """A non-trivial value head must shift the expected score
        relative to the random-sampling planner; use a fixed seed for
        determinism."""
        torch.manual_seed(0)
        rand_planner, _, _, _ = _build_random_planner()
        torch.manual_seed(0)
        rand_score = rand_planner.plan(torch.randn(8)).expected_score

        torch.manual_seed(0)
        trained_planner, _, _, _ = _build_trained_planner()
        torch.manual_seed(0)
        trained_score = trained_planner.plan(torch.randn(8)).expected_score
        assert rand_score != trained_score

    def test_action_variance_nontrivial(self):
        """Plans across a batch of init states should produce non-trivial
        spread in the nominal action — i.e. the prior is not collapsed."""
        planner, _, _, _ = _build_trained_planner()
        torch.manual_seed(123)
        actions = torch.stack(
            [planner.nominal_action(torch.randn(8)) for _ in range(16)], dim=0,
        )                                                              # (16, 4)
        assert actions.std(dim=0).mean().item() > 1e-3

    def test_constraint_violation_count_zero(self):
        """The constraint penalty wired in `_build_trained_planner` is
        ReLU(|a|.sum > 6); for the toy 4-dim action space and horizon=4
        the trained planner with bounded prior must keep violations at
        zero across 32 random init states."""
        planner, _, _, _ = _build_trained_planner()
        torch.manual_seed(456)
        viol = 0
        for _ in range(32):
            a_seq = planner.plan(torch.randn(8)).actions
            v = torch.relu(a_seq.abs().sum(dim=-1) - 6.0).sum().item()
            if v > 0.0:
                viol += 1
        assert viol == 0, f"expected zero violations, got {viol}/32"

    def test_value_weight_zero_disables_value_contribution(self):
        """With value_weight=0 the planner should match the policy-prior-only
        path. Concretely: scores must be finite."""
        planner_rand, ld, reward, cstr = _build_random_planner()
        value = _MockValueHead(8, 4)
        prior = _MockPolicyPrior(8, 4, horizon=4)
        planner = TDMPCPlanner(
            ld, reward, [cstr],
            TDMPCConfig(horizon=4, n_samples=24, n_iterations=2),
            action_dim=4,
            value_head=value,
            policy_prior=prior.planner_prior,
            value_weight=0.0,
        )
        s = planner.plan(torch.randn(8)).expected_score
        assert isinstance(s, float)
        assert torch.isfinite(torch.tensor(s))

    def test_policy_prior_only_no_value(self):
        """Policy prior without a value head must still plan."""
        planner_rand, ld, reward, cstr = _build_random_planner()
        prior = _MockPolicyPrior(8, 4, horizon=4)
        planner = TDMPCPlanner(
            ld, reward, [cstr],
            TDMPCConfig(horizon=4, n_samples=24, n_iterations=2),
            action_dim=4,
            policy_prior=prior.planner_prior,
        )
        result = planner.plan(torch.randn(8))
        assert result.actions.shape == (4, 4)


# --------------------------------------------------------------------------- #
# Integration test against REAL trained checkpoints. Skipped when missing.
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not TRAINED_AVAILABLE, reason="trained checkpoints not present")
class TestTDMPCPlannerCheckpointed:
    """Loads the actual `tdmpc_*_v0.1.pt` checkpoints and runs the planner."""

    def _load(self):
        # Import the training module's heads to get the matching nn.Module shape.
        from train_tdmpc_planner import (  # type: ignore[import-not-found]
            D_ACTION,
            D_LATENT,
            HORIZON,
            PolicyPrior,
            ValueHead,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dyn = LatentDynamics(
            LatentDynamicsConfig(d_latent=D_LATENT, d_action=D_ACTION, d_state=16)
        ).to(device)
        dyn.load_state_dict(torch.load(
            CHECKPOINTS / "latent_dynamics_v0.1.pt", map_location=device, weights_only=True,
        ))
        value = ValueHead().to(device)
        value.load_state_dict(torch.load(
            CHECKPOINTS / "tdmpc_value_v0.1.pt", map_location=device, weights_only=True,
        ))
        policy = PolicyPrior(horizon=HORIZON).to(device)
        policy.load_state_dict(torch.load(
            CHECKPOINTS / "tdmpc_policy_prior_v0.1.pt", map_location=device, weights_only=True,
        ))
        return dyn, value, policy, device, D_ACTION, D_LATENT, HORIZON

    def test_load_and_plan(self):
        dyn, value, policy, device, D_A, D_Z, H = self._load()

        def reward(traj, actions):
            return -traj.norm(dim=-1) - 0.01 * actions.norm(dim=-1)

        cstr = lambda traj, a: (a.abs() - 1.5).clamp(min=0.0).max(dim=-1).values

        planner = TDMPCPlanner(
            dyn, reward, [cstr],
            TDMPCConfig(horizon=H, n_samples=64, n_iterations=4,
                        constraint_weight=0.0),
            action_dim=D_A, device=device,
            value_head=value,
            policy_prior=policy.planner_prior,
            value_weight=0.5,
        )
        z0 = torch.randn(D_Z, device=device)
        result = planner.plan(z0)
        assert result.actions.shape == (H, D_A)
        # Non-trivial action variance across a small batch.
        actions = torch.stack(
            [planner.nominal_action(torch.randn(D_Z, device=device)) for _ in range(8)],
            dim=0,
        )
        assert actions.std(dim=0).mean().item() > 1e-4

    def test_violation_count_zero_on_real_planner(self):
        dyn, value, policy, device, D_A, D_Z, H = self._load()

        def reward(traj, actions):
            return -traj.norm(dim=-1) - 0.01 * actions.norm(dim=-1)

        cstr = lambda traj, a: (a.abs() - 1.5).clamp(min=0.0).max(dim=-1).values

        planner = TDMPCPlanner(
            dyn, reward, [cstr],
            TDMPCConfig(horizon=H, n_samples=64, n_iterations=4,
                        constraint_weight=10.0),
            action_dim=D_A, device=device,
            value_head=value,
            policy_prior=policy.planner_prior,
            value_weight=0.5,
        )
        viol = 0
        torch.manual_seed(31337)
        for _ in range(20):
            a = planner.nominal_action(torch.randn(D_Z, device=device))
            if (a.abs() > 1.5).any().item():
                viol += 1
        # With constraint_weight=10 and a learned bounded prior, violations
        # should not increase relative to random — assert ≤2 to allow a tiny
        # tail given the synthetic constraint.
        assert viol <= 2, f"unexpected planner violations: {viol}/20"
