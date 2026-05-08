"""Pre-emit guards (E2E_DEBUG gaps 2, 4, 5, 7).

The planner happily produces an action; the constraint layer happily
projects it; the SLA head happily emits a probability. If any of those
were untrained, ill-configured, or missing context, the rApp would still
emit a policy and the regulator would never know. These guards are the
production-grade gates that sit between the planner output and the A1
emission so an unsafe / un-evidenced policy NEVER ships.

Each guard returns either `None` (pass) or a `GuardFailure` describing
exactly what failed. The orchestrator in `rapp.lifecycle` runs the full
chain before calling `A1Adapter.emit_policy`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from horizon_ric.policy.constraints import (
    ConstraintViolation,
    PreceptualAIConstraintLayer,
)


@dataclass(frozen=True)
class GuardFailure:
    """Fatal pre-emit refusal. The rApp logs and aborts the emit."""

    guard_id: str
    message: str
    detail: dict[str, Any] | None = None


@dataclass
class EmitTrace:
    """Wall-clock + correction record from running the guard chain."""

    decision_started_ns: int
    decision_completed_ns: int | None = None
    corrections: list[ConstraintViolation] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.corrections is None:
            self.corrections = []

    @property
    def elapsed_ms(self) -> float:
        end = self.decision_completed_ns or time.monotonic_ns()
        return (end - self.decision_started_ns) / 1e6


def guard_predicted_outcome_pretrained(
    head_is_pretrained: bool,
) -> GuardFailure | None:
    """Gap 2 — refuse emit when the SLA head hasn't been pretrained.

    `head_is_pretrained` should reflect either a checkpoint loader's
    success flag or an explicit operator override (e.g. for canary
    rollouts that only exercise the action surface)."""
    if head_is_pretrained:
        return None
    return GuardFailure(
        guard_id="sla_head_not_pretrained",
        message=(
            "SLA risk head is not pretrained — emitted predictions would "
            "be near-uniform 0.5 noise. Refuse emit until a checkpoint is "
            "loaded or the operator explicitly opts into canary mode."
        ),
    )


def guard_constraint_context_complete(
    context: dict[str, Any],
    constraint_layer: PreceptualAIConstraintLayer,
) -> GuardFailure | None:
    """Gap 4 — when the constraint layer can run EPFD, the caller MUST
    supply NGSO emitters + wanted GSO longitude. Omitting them silently
    skips the EPFD floor — that's how a regulatory violation slips out.
    """
    ids = constraint_layer.hard_constraint_ids()
    if "itu_epfd_down" not in ids:
        return None  # EPFD not configured at all — fine.

    missing = []
    for key in ("ngso_emitters", "wanted_gso_longitude_deg"):
        if context.get(key) is None:
            missing.append(key)
    if not missing:
        return None
    return GuardFailure(
        guard_id="incomplete_constraint_context",
        message=(
            "EPFD floor is configured but the context lacks "
            + ", ".join(missing)
            + ". Refuse emit — silent EPFD bypass is unsafe."
        ),
        detail={"missing_keys": missing},
    )


def guard_decision_within_a1_budget(
    elapsed_ms: float,
    policy_period_ms: float = 100.0,
) -> GuardFailure | None:
    """Gap 5 — refuse emit if the decision exceeded its A1 budget.

    A late policy is by definition stale by the time it reaches the
    Near-RT RIC; emitting it just thrashes the data plane.
    """
    if elapsed_ms <= policy_period_ms:
        return None
    return GuardFailure(
        guard_id="decision_over_budget",
        message=(
            f"decision took {elapsed_ms:.1f} ms; A1 refresh window is "
            f"{policy_period_ms:.0f} ms. Aborting emit — policy is stale."
        ),
        detail={"elapsed_ms": elapsed_ms, "budget_ms": policy_period_ms},
    )


def guard_corrections_recorded(
    corrections: list[ConstraintViolation] | None,
    audit_corrections_field: list[dict[str, Any]] | None,
) -> GuardFailure | None:
    """Gap 7 — every constraint correction must land in the audit record.

    Returns failure when there were corrections but the audit field is
    missing or empty.
    """
    if not corrections:
        return None
    if not audit_corrections_field:
        return GuardFailure(
            guard_id="corrections_not_audited",
            message=(
                f"{len(corrections)} constraint correction(s) were applied "
                "but none persisted to the DecisionRecord. The audit trail "
                "would hide the projection — regulator cannot reproduce."
            ),
            detail={
                "n_corrections": len(corrections),
                "constraint_ids": sorted({c.constraint_id for c in corrections}),
            },
        )
    return None


def run_guard_chain(
    *,
    head_is_pretrained: bool,
    constraint_context: dict[str, Any],
    constraint_layer: PreceptualAIConstraintLayer,
    elapsed_ms: float,
    corrections: list[ConstraintViolation] | None,
    audit_corrections_field: list[dict[str, Any]] | None,
    policy_period_ms: float = 100.0,
) -> list[GuardFailure]:
    """Apply every guard. Returns the (possibly empty) list of failures."""
    failures: list[GuardFailure] = []
    for fail in (
        guard_predicted_outcome_pretrained(head_is_pretrained),
        guard_constraint_context_complete(constraint_context, constraint_layer),
        guard_decision_within_a1_budget(elapsed_ms, policy_period_ms),
        guard_corrections_recorded(corrections, audit_corrections_field),
    ):
        if fail is not None:
            failures.append(fail)
    return failures


__all__ = [
    "EmitTrace",
    "GuardFailure",
    "guard_constraint_context_complete",
    "guard_corrections_recorded",
    "guard_decision_within_a1_budget",
    "guard_predicted_outcome_pretrained",
    "run_guard_chain",
]
