"""Counterfactual reproducibility tests (Devil-A/C Findings 9, 26).

A regulator replaying ``(state, action_space, planner_config, random_seed)``
must obtain byte-identical rejected alternatives. The pinning is:

    * ``TDMPCConfig.random_seed`` (or ``plan(z0, random_seed=...)``) →
      pure planner output.
    * ``RejectedAlternative.random_seed`` field on every emitted record.
    * Same seed → identical alternatives. Different seed → different
      alternatives (with probability ≥ 1 − negl(d)).
"""

from __future__ import annotations

from typing import Sequence

import pytest
import torch

from horizon_ric.evidence.schema import (
    PredictedOutcome,
    RejectedAlternative,
    RejectionReasonMachine,
)
from horizon_ric.policy.counterfactual import (
    AlternativeCandidate,
    build_rejected_alternatives,
    candidates_from_planner_elite,
)
from horizon_ric.policy.td_mpc_planner import (
    PlanResult,
    TDMPCConfig,
    TDMPCPlanner,
)


# ---------------------------------------------------------------------------
# Shared fixture: a tiny linear "dynamics" so we can exercise the planner
# deterministically with no learned weights.
# ---------------------------------------------------------------------------
class _LinearDynamics:
    """Trivial latent dynamics: z' = 0.9·z + a (broadcast)."""

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # z0: (B, Z)  actions: (B, H, A)  → traj: (B, H, Z)
        B, H, A = actions.shape
        Z = z0.shape[-1]
        traj = torch.zeros(B, H, Z, device=z0.device)
        z = z0
        for h in range(H):
            # broadcast action onto z by repeating / truncating
            a = actions[:, h, : min(A, Z)]
            if a.shape[-1] < Z:
                pad = torch.zeros(B, Z - a.shape[-1], device=z0.device)
                a = torch.cat([a, pad], dim=-1)
            z = 0.9 * z + a
            traj[:, h] = z
        return traj


def _reward(traj: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    # Penalise large states + large actions.
    return -(traj.norm(dim=-1) ** 2) - 0.01 * (actions.norm(dim=-1) ** 2)


def _make_planner(seed: int | None) -> TDMPCPlanner:
    cfg = TDMPCConfig(
        horizon=4, n_samples=32, n_iterations=2, init_std=1.0,
        random_seed=seed,
    )
    return TDMPCPlanner(
        dynamics=_LinearDynamics(),
        reward_fn=_reward,
        config=cfg,
        action_dim=2,
        device="cpu",
    )


def _alternatives_from_plan(
    planner: TDMPCPlanner, z0: torch.Tensor, seed: int,
) -> tuple[PlanResult, list[RejectedAlternative]]:
    plan = planner.plan(z0, random_seed=seed)
    cands = candidates_from_planner_elite(
        elite_actions=plan.elite_actions[:, 0, :],
        elite_rewards=plan.elite_rewards,
        elite_violations=plan.elite_violations,
        action_keys=["a0", "a1"],
    )
    alts = build_rejected_alternatives(
        cands,
        chosen_score=plan.expected_score,
        random_seed=seed,
    )
    return plan, alts


# ---------------------------------------------------------------------------
# 1. Same seed → identical alternatives (deep equality).
# ---------------------------------------------------------------------------
def test_same_seed_identical_alternatives() -> None:
    planner = _make_planner(seed=None)
    z0 = torch.tensor([0.5, -0.3, 0.1, 0.7])

    plan_a, alts_a = _alternatives_from_plan(planner, z0, seed=42)
    plan_b, alts_b = _alternatives_from_plan(planner, z0, seed=42)

    # Same plan actions byte-for-byte.
    torch.testing.assert_close(plan_a.actions, plan_b.actions)
    assert plan_a.random_seed == plan_b.random_seed == 42

    # Same alternatives at every position.
    assert len(alts_a) == len(alts_b)
    for x, y in zip(alts_a, alts_b):
        assert x.action == y.action
        assert x.rejection_reason_machine == y.rejection_reason_machine
        assert x.random_seed == y.random_seed == 42


# ---------------------------------------------------------------------------
# 2. Different seed → different alternatives (with overwhelming probability).
# ---------------------------------------------------------------------------
def test_different_seed_different_alternatives() -> None:
    planner = _make_planner(seed=None)
    z0 = torch.tensor([0.5, -0.3, 0.1, 0.7])

    plan_a, alts_a = _alternatives_from_plan(planner, z0, seed=42)
    plan_b, alts_b = _alternatives_from_plan(planner, z0, seed=99)

    # Plan actions differ.
    assert not torch.allclose(plan_a.actions, plan_b.actions, atol=1e-6)
    # Seeds are recorded distinctly.
    assert alts_a[0].random_seed != alts_b[0].random_seed


# ---------------------------------------------------------------------------
# 3. Seed is pinned in every emitted RejectedAlternative.
# ---------------------------------------------------------------------------
def test_seed_pinned_in_every_record() -> None:
    cands: list[AlternativeCandidate] = [
        AlternativeCandidate(
            action={"x": 1.0}, reward_sum=0.5,
            constraint_violation_sum=0.0, sla_risk_30s=0.05,
        ),
        AlternativeCandidate(
            action={"x": 2.0}, reward_sum=0.3,
            constraint_violation_sum=0.0, sla_risk_30s=0.10,
        ),
        AlternativeCandidate(
            action={"x": 3.0}, reward_sum=0.1,
            constraint_violation_sum=0.0, sla_risk_30s=0.25,
        ),
    ]
    alts = build_rejected_alternatives(cands, chosen_score=1.0, random_seed=7)
    assert len(alts) == 3
    for a in alts:
        assert a.random_seed == 7


# ---------------------------------------------------------------------------
# 4. The schema enforces random_seed (regulator-facing contract).
# ---------------------------------------------------------------------------
def test_schema_requires_random_seed() -> None:
    with pytest.raises(Exception):  # pydantic ValidationError
        RejectedAlternative(
            rank=1,
            action={"x": 1.0},
            predicted_outcome=PredictedOutcome(
                sla_risk_30s=0.0, sla_risk_1min=0.0, sla_risk_5min=0.0,
            ),
            rejection_reason_machine=RejectionReasonMachine(
                primary_cause="energy_cost",
                primary_metric="score",
                predicted_value=0.0,
                threshold=0.0,
                horizon="30s",
            ),
            rejection_reason_human="x",
            # NO random_seed → must fail validation
        )
