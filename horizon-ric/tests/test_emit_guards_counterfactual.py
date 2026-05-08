"""Pre-emit guards + counterfactual builder + ConstraintCorrection round-trip."""

import math

import pytest
import torch

from horizon_ric.evidence.schema import (
    ConstraintCorrection,
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.policy.constraints import (
    ConstraintViolation,
    GSOArcEntry,
    PreceptualAIConstraintConfig,
    PreceptualAIConstraintLayer,
)
from horizon_ric.policy.counterfactual import (
    AlternativeCandidate,
    build_rejected_alternatives,
    candidates_from_planner_elite,
)
from horizon_ric.policy.emit_guards import (
    GuardFailure,
    guard_constraint_context_complete,
    guard_corrections_recorded,
    guard_decision_within_a1_budget,
    guard_predicted_outcome_pretrained,
    run_guard_chain,
)


def _layer_with_epfd():
    return PreceptualAIConstraintLayer(
        PreceptualAIConstraintConfig(
            gso_arcs=[GSOArcEntry(satellite_id="g1", longitude_deg=0.0)],
            epfd_floor_dBW_per_m2_per_ref_bw=-145.0,
        )
    )


# ─── individual guards ────────────────────────────────────────────────────


class TestGuards:
    def test_pretrained_pass(self):
        assert guard_predicted_outcome_pretrained(True) is None

    def test_pretrained_fail(self):
        f = guard_predicted_outcome_pretrained(False)
        assert isinstance(f, GuardFailure)
        assert f.guard_id == "sla_head_not_pretrained"

    def test_context_complete_pass(self):
        layer = _layer_with_epfd()
        ctx = {"ngso_emitters": [object()], "wanted_gso_longitude_deg": 0.0}
        assert guard_constraint_context_complete(ctx, layer) is None

    def test_context_missing_emitters(self):
        layer = _layer_with_epfd()
        ctx = {"wanted_gso_longitude_deg": 0.0}
        f = guard_constraint_context_complete(ctx, layer)
        assert f is not None and "ngso_emitters" in f.message

    def test_context_pass_when_no_epfd(self):
        layer = PreceptualAIConstraintLayer()  # no EPFD configured
        f = guard_constraint_context_complete({}, layer)
        assert f is None

    def test_a1_budget_pass(self):
        assert guard_decision_within_a1_budget(50.0, 100.0) is None

    def test_a1_budget_fail(self):
        f = guard_decision_within_a1_budget(150.0, 100.0)
        assert f is not None and f.guard_id == "decision_over_budget"

    def test_corrections_recorded_pass_no_corrections(self):
        assert guard_corrections_recorded([], []) is None

    def test_corrections_recorded_pass_logged(self):
        c = [ConstraintViolation("epfd", "hard", 0.1, "msg")]
        audit = [{"constraint_id": "epfd"}]
        assert guard_corrections_recorded(c, audit) is None

    def test_corrections_recorded_fail(self):
        c = [ConstraintViolation("epfd", "hard", 0.1, "msg")]
        f = guard_corrections_recorded(c, None)
        assert f is not None
        assert f.guard_id == "corrections_not_audited"


# ─── chain ────────────────────────────────────────────────────────────────


class TestGuardChain:
    def test_chain_collects_all_failures(self):
        layer = _layer_with_epfd()
        failures = run_guard_chain(
            head_is_pretrained=False,
            constraint_context={},
            constraint_layer=layer,
            elapsed_ms=999.0,
            corrections=[ConstraintViolation("epfd", "hard", 0.1, "msg")],
            audit_corrections_field=None,
            policy_period_ms=100.0,
        )
        ids = {f.guard_id for f in failures}
        assert {
            "sla_head_not_pretrained",
            "incomplete_constraint_context",
            "decision_over_budget",
            "corrections_not_audited",
        } <= ids

    def test_chain_passes_when_clean(self):
        layer = _layer_with_epfd()
        failures = run_guard_chain(
            head_is_pretrained=True,
            constraint_context={
                "ngso_emitters": [object()],
                "wanted_gso_longitude_deg": 0.0,
            },
            constraint_layer=layer,
            elapsed_ms=10.0,
            corrections=None,
            audit_corrections_field=None,
        )
        assert failures == []


# ─── counterfactual ──────────────────────────────────────────────────────


class TestCounterfactual:
    def test_basic_build(self):
        cands = [
            AlternativeCandidate(
                action={"tx": 30.0},
                reward_sum=-5.0, constraint_violation_sum=2.0,
                primary_metric="epfd", primary_value=2.0,
                threshold=0.0, horizon="60s",
            ),
            AlternativeCandidate(
                action={"tx": 10.0},
                reward_sum=-1.0, constraint_violation_sum=0.0,
                sla_risk_30s=0.30,
                primary_metric="sla", primary_value=0.30,
                threshold=0.20, horizon="30s",
            ),
        ]
        out = build_rejected_alternatives(cands, chosen_score=0.0)
        assert len(out) == 2
        assert out[0].rank == 1 and out[1].rank == 2
        # Reward-sorted: rank 1 has highest reward_sum (-1 > -5).
        assert out[0].action["tx"] == 10.0
        # Causes are properly classified.
        causes = {alt.rejection_reason_machine.primary_cause for alt in out}
        assert "constraint_violation_hard" in causes
        assert "sla_breach_predicted" in causes

    def test_empty_candidates(self):
        assert build_rejected_alternatives([], chosen_score=0.0) == []

    def test_planner_elite_round_trip(self):
        elite_actions = torch.tensor([[1.0, 2.0], [-1.0, 0.0]])
        elite_rewards = torch.tensor([-2.0, -1.0])
        elite_violations = torch.tensor([0.0, 0.0])
        cands = candidates_from_planner_elite(
            elite_actions, elite_rewards, elite_violations,
            action_keys=["tx_dBm", "freq_norm"],
        )
        assert len(cands) == 2
        assert cands[0].action == {"tx_dBm": 1.0, "freq_norm": 2.0}

    def test_planner_elite_rejects_mismatch(self):
        with pytest.raises(ValueError):
            candidates_from_planner_elite(
                torch.zeros(3, 4), torch.zeros(3), torch.zeros(3),
                action_keys=["a", "b"],   # only 2 names for a 4-dim action
            )


# ─── ConstraintCorrection round-trip on DecisionRecord ──────────────────


class TestConstraintCorrectionAudit:
    def test_serialise_with_corrections(self):
        rec = DecisionRecord.new(
            decision_id="d-1",
            rapp_instance_id="rapp",
            state_hash="h",
            chosen_action={"tx_power_dBm": 25.0},
            predicted_outcome_chosen=PredictedOutcome(
                sla_risk_30s=0.05, sla_risk_1min=0.05, sla_risk_5min=0.05,
            ),
            rejected_alternatives=[],
            constraint_corrections=[
                ConstraintCorrection(
                    constraint_id="gso_pfd_floor",
                    severity="hard",
                    margin_dB=1.5,
                    message="reduced TX power by 1.5 dB",
                ),
            ],
            model_versions=ModelVersions(
                encoder="e", risk_heads="r", dyna="d",
                policy="p", constraint_layer="c", rapp="ra",
            ),
        )
        blob = rec.model_dump_json()
        rec2 = DecisionRecord.model_validate_json(blob)
        assert len(rec2.constraint_corrections) == 1
        assert rec2.constraint_corrections[0].constraint_id == "gso_pfd_floor"
        assert math.isclose(rec2.constraint_corrections[0].margin_dB, 1.5)
