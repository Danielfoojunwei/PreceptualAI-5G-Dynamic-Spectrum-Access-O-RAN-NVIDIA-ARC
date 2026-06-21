"""Evidence schema + counterfactual explanation tests."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from horizon_ric.evidence.explanation import generate_human_explanation
from horizon_ric.evidence.schema import (
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
    RejectedAlternative,
    RejectionReasonMachine,
)


def _versions() -> ModelVersions:
    return ModelVersions(
        encoder="enc-0.1.0",
        risk_heads="rh-0.1.0",
        dyna="dy-0.1.0",
        policy="pol-0.1.0",
        constraint_layer="cl-0.1.0",
        rapp="horizon-ric-0.1.0",
    )


def _outcome(risk: float = 0.05) -> PredictedOutcome:
    return PredictedOutcome(
        sla_risk_30s=risk,
        sla_risk_1min=risk,
        sla_risk_5min=risk,
        gateway_loads={"G1": 0.5},
        ntn_capacity_used=0.4,
        energy_kwh=1.2,
    )


class TestPredictedOutcome:
    def test_risk_must_be_in_unit_interval(self):
        with pytest.raises(ValidationError):
            PredictedOutcome(
                sla_risk_30s=1.5,
                sla_risk_1min=0.1,
                sla_risk_5min=0.1,
            )

    def test_negative_energy_rejected(self):
        with pytest.raises(ValidationError):
            PredictedOutcome(
                sla_risk_30s=0.1,
                sla_risk_1min=0.1,
                sla_risk_5min=0.1,
                energy_kwh=-1.0,
            )


class TestRejectionReasonMachine:
    def test_unknown_cause_rejected(self):
        with pytest.raises(ValidationError):
            RejectionReasonMachine(
                primary_cause="cosmic_ray",  # not in Literal set
                primary_metric="g1",
                predicted_value=0.9,
                threshold=0.8,
                horizon="30s",
            )

    def test_horizon_must_be_canonical(self):
        with pytest.raises(ValidationError):
            RejectionReasonMachine(
                primary_cause="gateway_overload",
                primary_metric="g1",
                predicted_value=0.9,
                threshold=0.8,
                horizon="42s",
            )


class TestDecisionRecord:
    def test_construct_and_round_trip(self):
        rej = RejectedAlternative(
            rank=1,
            action={"tx_power_dBm": 30.0, "frequency_hz": 3.7e9},
            predicted_outcome=_outcome(0.91),
            rejection_reason_machine=RejectionReasonMachine(
                primary_cause="sla_breach_predicted",
                primary_metric="sla_risk_30s",
                predicted_value=0.91,
                threshold=0.20,
                horizon="30s",
            ),
            rejection_reason_human="Predicted SLA breach probability 91% on sla_risk_30s.",
            random_seed=42,
        )
        rec = DecisionRecord.new(
            decision_id="d-1",
            rapp_instance_id="rapp-1",
            state_hash="abc123",
            chosen_action={"tx_power_dBm": 25.0, "frequency_hz": 3.7e9},
            predicted_outcome_chosen=_outcome(0.05),
            rejected_alternatives=[rej],
            model_versions=_versions(),
        )
        assert rec.decision_id == "d-1"
        assert isinstance(rec.timestamp, datetime)
        assert rec.timestamp.tzinfo is timezone.utc
        # Round-trip JSON
        blob = rec.model_dump_json()
        rec2 = DecisionRecord.model_validate_json(blob)
        assert rec2.decision_id == "d-1"
        assert rec2.rejected_alternatives[0].rank == 1

    def test_actual_outcomes_optional(self):
        rec = DecisionRecord.new(
            decision_id="d-2",
            rapp_instance_id="rapp-1",
            state_hash="abc",
            chosen_action={},
            predicted_outcome_chosen=_outcome(0.05),
            rejected_alternatives=[],
            model_versions=_versions(),
        )
        assert rec.actual_outcome_30s is None
        assert rec.actual_outcome_1min is None
        assert rec.actual_outcome_5min is None


class TestExplanation:
    def test_known_cause_uses_template(self):
        reason = RejectionReasonMachine(
            primary_cause="gateway_overload",
            primary_metric="gateway_load_G3",
            predicted_value=0.92,
            threshold=0.85,
            horizon="60s",
        )
        s = generate_human_explanation(reason)
        assert "gateway_load_G3" in s
        assert "92%" in s
        assert "85%" in s
        assert "60s" in s

    def test_all_known_causes_have_templates(self):
        causes = [
            "gateway_overload",
            "sla_breach_predicted",
            "compute_overload",
            "energy_cost",
            "ntn_capacity_exhausted",
            "spectrum_unavailable",
            "constraint_violation_hard",
            "constraint_violation_soft",
            "policy_oscillation",
        ]
        for cause in causes:
            reason = RejectionReasonMachine(
                primary_cause=cause,
                primary_metric="m",
                predicted_value=0.5,
                threshold=0.4,
                horizon="30s",
            )
            s = generate_human_explanation(reason)
            assert isinstance(s, str)
            assert len(s) > 20
