"""Pre-emit guards — the last gate between a decision and the A1 wire.

The Shield produces a safe action and a :class:`SafetyCertificate`; these guards
make the final go/no-go call before the policy is emitted. Each guard returns
either ``None`` (pass) or a :class:`GuardFailure`. The orchestrator runs the
whole chain and refuses to emit if any guard fails — so an unsafe or
un-evidenced policy NEVER ships.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from horizon_ric.shield.certificate import ConstraintViolation, SafetyCertificate


@dataclass(frozen=True)
class GuardFailure:
    """Fatal pre-emit refusal. The rApp logs and aborts the emit."""

    guard_id: str
    message: str
    detail: Optional[dict[str, Any]] = None


@dataclass
class EmitTrace:
    """Wall-clock + correction record from running the decision pipeline."""

    decision_started_ns: int
    decision_completed_ns: Optional[int] = None
    corrections: list[ConstraintViolation] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.corrections is None:
            self.corrections = []

    @property
    def elapsed_ms(self) -> float:
        end = self.decision_completed_ns or time.monotonic_ns()
        return (end - self.decision_started_ns) / 1e6


def guard_shield_certificate_safe(
    certificate: SafetyCertificate | None,
) -> GuardFailure | None:
    """Refuse emit unless the Shield certified the action as safe.

    This is the core security gate: a missing certificate, a blocked emit, or
    any residual hard-invariant violation means the decision does not ship.
    """
    if certificate is None:
        return GuardFailure(
            guard_id="missing_safety_certificate",
            message="no SafetyCertificate attached — the action never passed the Shield.",
        )
    if certificate.emit_blocked or not certificate.safe:
        return GuardFailure(
            guard_id="shield_blocked",
            message=(
                "Shield could not make the action safe; emit refused. "
                f"violated={certificate.violated_ids}"
            ),
            detail={"violated_ids": certificate.violated_ids},
        )
    return None


def guard_predicted_outcome_pretrained(
    head_is_pretrained: bool,
) -> GuardFailure | None:
    """Refuse emit when the predictive head has not been pretrained.

    An untrained head emits near-uniform noise; better to refuse than to ship a
    confident-looking but meaningless prediction into the audit chain.
    """
    if head_is_pretrained:
        return None
    return GuardFailure(
        guard_id="head_not_pretrained",
        message=(
            "predictive head is not pretrained — refuse emit until a checkpoint "
            "is loaded or the operator explicitly opts into canary mode."
        ),
    )


def guard_decision_within_a1_budget(
    elapsed_ms: float,
    policy_period_ms: float = 100.0,
) -> GuardFailure | None:
    """Refuse emit if the decision exceeded its A1 refresh budget.

    A late policy is stale by the time it reaches the Near-RT RIC; emitting it
    just thrashes the data plane.
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
    """Every Shield correction must land in the audit record.

    Returns failure when corrections were applied but the audit field is empty —
    a hidden projection means the regulator cannot reproduce the decision.
    """
    if not corrections:
        return None
    if not audit_corrections_field:
        return GuardFailure(
            guard_id="corrections_not_audited",
            message=(
                f"{len(corrections)} Shield correction(s) were applied but none "
                "persisted to the DecisionRecord. The audit trail would hide the "
                "projection — regulator cannot reproduce."
            ),
            detail={
                "n_corrections": len(corrections),
                "constraint_ids": sorted({c.constraint_id for c in corrections}),
            },
        )
    return None


def run_guard_chain(
    *,
    certificate: SafetyCertificate | None,
    head_is_pretrained: bool = True,
    elapsed_ms: float = 0.0,
    audit_corrections_field: list[dict[str, Any]] | None = None,
    policy_period_ms: float = 100.0,
) -> list[GuardFailure]:
    """Apply every guard. Returns the (possibly empty) list of failures."""
    corrections = certificate.corrections if certificate is not None else None
    failures: list[GuardFailure] = []
    for fail in (
        guard_shield_certificate_safe(certificate),
        guard_predicted_outcome_pretrained(head_is_pretrained),
        guard_decision_within_a1_budget(elapsed_ms, policy_period_ms),
        guard_corrections_recorded(corrections, audit_corrections_field),
    ):
        if fail is not None:
            failures.append(fail)
    return failures


__all__ = [
    "EmitTrace",
    "GuardFailure",
    "guard_shield_certificate_safe",
    "guard_predicted_outcome_pretrained",
    "guard_decision_within_a1_budget",
    "guard_corrections_recorded",
    "run_guard_chain",
]
