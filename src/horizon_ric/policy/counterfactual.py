"""Counterfactual builder.

Convert a set of candidate (rejected) actions — the alternatives an AI-RAN
decision head considered but did not pick — into the DecisionRecord's
``rejected_alternatives`` field, with structured rejection reasons keyed off the
per-candidate score and constraint violation. This is what makes a decision
*replayable*: a regulator handed (state, action_space, random_seed) reconstructs
the same chosen action and the same rejected list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from horizon_ric.evidence.explanation import generate_human_explanation
from horizon_ric.evidence.schema import (
    PredictedOutcome,
    RejectedAlternative,
    RejectionReasonMachine,
)

_HORIZON = Literal["30s", "60s", "300s"]


@dataclass
class AlternativeCandidate:
    """One non-chosen candidate with its scoring inputs."""

    action: dict
    """Serialisable action dict that would have been emitted."""
    reward_sum: float
    constraint_violation_sum: float
    sla_risk_30s: float = 0.0
    sla_risk_1min: float = 0.0
    sla_risk_5min: float = 0.0
    primary_metric: str = "score"
    primary_value: float | None = None
    threshold: float = 0.0
    horizon: _HORIZON = "30s"


def build_rejected_alternatives(
    candidates: Iterable[AlternativeCandidate],
    chosen_score: float,
    soft_constraint_threshold: float = 1e-6,
    random_seed: int = 0,
) -> list[RejectedAlternative]:
    """Materialise an audit-ready list of ``RejectedAlternative``s.

    Cause classification:
        * violation > threshold   → "constraint_violation_hard"
        * sla_risk_30s   > 0.20    → "sla_breach_predicted"
        * reward_sum < chosen      → "energy_cost" (catch-all)
        * else                      → "policy_oscillation"

    ``random_seed`` is the RNG seed pinned during the decision that produced
    these candidates; it is stamped onto every emitted ``RejectedAlternative``
    so replay with the same seed yields an identical list.
    """
    out: list[RejectedAlternative] = []
    sorted_cands = sorted(candidates, key=lambda c: c.reward_sum, reverse=True)

    for rank, cand in enumerate(sorted_cands, start=1):
        if cand.constraint_violation_sum > soft_constraint_threshold:
            cause = "constraint_violation_hard"
        elif cand.sla_risk_30s > 0.20:
            cause = "sla_breach_predicted"
        elif cand.reward_sum < chosen_score:
            cause = "energy_cost"
        else:
            cause = "policy_oscillation"

        primary_value = (
            cand.primary_value
            if cand.primary_value is not None
            else cand.constraint_violation_sum
            if cause == "constraint_violation_hard"
            else cand.sla_risk_30s
        )
        reason = RejectionReasonMachine(
            primary_cause=cause,
            primary_metric=cand.primary_metric,
            predicted_value=float(primary_value),
            threshold=float(cand.threshold),
            horizon=cand.horizon,
        )
        out.append(
            RejectedAlternative(
                rank=rank,
                action=cand.action,
                predicted_outcome=PredictedOutcome(
                    sla_risk_30s=float(cand.sla_risk_30s),
                    sla_risk_1min=float(cand.sla_risk_1min),
                    sla_risk_5min=float(cand.sla_risk_5min),
                ),
                rejection_reason_machine=reason,
                rejection_reason_human=generate_human_explanation(reason),
                random_seed=int(random_seed),
            )
        )
    return out


__all__ = [
    "AlternativeCandidate",
    "build_rejected_alternatives",
]
