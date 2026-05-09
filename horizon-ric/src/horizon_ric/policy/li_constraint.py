"""Lawful Intercept (LI) jurisdiction constraint hook.

Compliance context
------------------
The PreceptualAI rApp does not itself perform interception. However, an rApp
that controls beam steering, gateway placement, slice assignment, or
hand-off can disturb the operator's Lawful Intercept Mediation Function
(LIMF). For any EU/UK/US operator pilot, the operator's LI architect will
refuse sign-off until the rApp exposes a constraint surface that:

  1. Refuses any action that touches a UE flagged by the LIMF as protected
     (i.e. an active intercept target whose flow path must not be
     reconfigured by an external rApp).
  2. Refuses any action that re-homes a protected slice into a jurisdiction
     not on the operator's `allowed_jurisdictions` list (e.g. a DE-issued
     warrant requires the bearer remains anchored in DE/EU).

This module adds the surface. The actual LIMF UE/slice catalog is injected
at construction time — the rApp does not see warrant content, only the
opaque tag set.

References
----------
    ETSI TS 103 221-1 / -2  — LI handover interface
    3GPP TS 33.126 / 33.127 — LI requirements / architecture
    O-RAN.WG11 §LI guidance — rApp/xApp non-interference with LIMF

Design notes
------------
* This is a HARD-only constraint. Soft Lagrangian penalties are explicitly
  forbidden — a regulator-mandated intercept must never be a tunable
  trade-off in a learned policy. ``lagrangian_violation`` therefore returns
  a zero tensor unconditionally; the projection layer is the only
  enforcement path.
* ``project`` is intentionally conservative: it strips protected UE IDs from
  the action's ``affected_ue_ids`` set rather than attempting to rewrite the
  jurisdiction. Reasoning: silently mutating the jurisdiction of an action
  could mask a buggy planner that is trying to move a warrant target across
  borders. We'd rather refuse loudly (drop the UE) and let the planner
  retry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from horizon_ric.contracts.constraint_layer import (
    ConstraintLayer,
    ConstraintViolation,
)


@dataclass
class LIJurisdictionRule:
    """One LI jurisdiction rule, sourced from the operator's LIMF.

    Attributes:
        rule_id: Stable opaque identifier (e.g. warrant tag, NOT warrant content).
        protected_ue_ids: UE IDs the LIMF has flagged as active intercept targets.
            Any action touching one of these UEs must be refused.
        protected_slice_ids: Slice IDs whose anchor jurisdiction is constrained.
        allowed_jurisdictions: Set of ISO-3166 alpha-2 codes (or supranational
            codes like ``"EU"``) where the protected slices may be anchored.
        description: Human-readable note for audit logs (NOT shown to the
            policy network).
    """

    rule_id: str
    protected_ue_ids: set[str] = field(default_factory=set)
    protected_slice_ids: set[str] = field(default_factory=set)
    allowed_jurisdictions: set[str] = field(default_factory=set)
    description: str = ""


class LIConstraint(ConstraintLayer):
    """LI-jurisdiction hard-constraint layer.

    Action contract (dict keys read from ``action``):
        affected_ue_ids: Iterable[str]   — UEs that this action would impact.
        affected_slice_ids: Iterable[str] — Slices that this action would impact.
        target_jurisdiction: str         — ISO-3166 alpha-2 (or "EU"/"UK") of
                                            the cell/gateway the action selects.

    Any of these fields can be absent — a missing field is treated as the
    empty set / ``None`` and cannot trigger a violation. (The constraint
    cannot "make up" UE IDs; it can only refuse what the planner declares.)

    Fail-closed default
    -------------------
    By default ``fail_closed=True``: when ``rules`` is empty, every action
    is rejected with the synthetic constraint id ``li_fail_closed`` (severity
    "hard"). This closes the Devil-A Finding #1 zero-config bypass: a
    deployment without an LIMF rule catalogue is not authorised to emit. To
    permit no-LI test/lab deployments, the operator must explicitly construct
    ``LIConstraint(rules=[], fail_closed=False)`` AND record that election in
    the audit chain (``deployment_audit_record``); the rApp lifecycle layer
    enforces this audit-record requirement.
    """

    _CID_UE = "li_protected_ue"
    _CID_JX = "li_jurisdiction"
    _CID_FAIL_CLOSED = "li_fail_closed"

    def __init__(
        self,
        rules: list[LIJurisdictionRule],
        *,
        fail_closed: bool = True,
        deployment_audit_record: str | None = None,
    ):
        self.rules: list[LIJurisdictionRule] = list(rules)
        self.fail_closed: bool = bool(fail_closed)
        self.deployment_audit_record: str | None = deployment_audit_record
        # If the operator opts out of fail-closed, the choice MUST be recorded
        # in the audit chain. This is enforced at construction time so any
        # downstream caller that loads an ``LIConstraint`` from a manifest is
        # guaranteed to have an evidence trail.
        if not self.fail_closed and not self.rules:
            if not deployment_audit_record:
                raise ValueError(
                    "LIConstraint(fail_closed=False) with empty rules requires "
                    "a non-empty deployment_audit_record (audit-chain entry id) "
                    "documenting the operator's explicit no-LI election. See "
                    "docs/compliance/li_applicability.md §6.1."
                )

    # ─── ConstraintLayer contract ────────────────────────────────────

    def hard_constraint_ids(self) -> list[str]:
        return [self._CID_UE, self._CID_JX, self._CID_FAIL_CLOSED]

    def check_feasibility(
        self, action: dict[str, Any], context: dict[str, Any]
    ) -> list[ConstraintViolation]:
        violations: list[ConstraintViolation] = []

        # Fail-closed gate: empty rule catalogue + fail_closed=True ⇒
        # reject every action. This is the production-safe default for any
        # deployment that has not wired the LIMF rule catalogue at boot.
        if self.fail_closed and not self.rules:
            violations.append(
                ConstraintViolation(
                    constraint_id=self._CID_FAIL_CLOSED,
                    severity="hard",
                    margin_dB=None,
                    message=(
                        "LIConstraint is in fail-closed mode with an empty rule "
                        "catalogue: every action is rejected until the operator "
                        "wires the LIMF/ADMF rule set. To permit no-LI lab "
                        "deployments, construct LIConstraint(rules=[], "
                        "fail_closed=False, deployment_audit_record=<id>)."
                    ),
                )
            )
            return violations

        affected_ues = self._as_str_set(action.get("affected_ue_ids"))
        affected_slices = self._as_str_set(action.get("affected_slice_ids"))
        target_jx = action.get("target_jurisdiction")

        for rule in self.rules:
            # 1. Protected UE intersection — always refused.
            if affected_ues and rule.protected_ue_ids:
                hit = affected_ues & rule.protected_ue_ids
                if hit:
                    violations.append(
                        ConstraintViolation(
                            constraint_id=self._CID_UE,
                            severity="hard",
                            margin_dB=None,
                            message=(
                                f"action affects LI-protected UE(s) {sorted(hit)} "
                                f"under rule {rule.rule_id!r}; refused."
                            ),
                        )
                    )

            # 2. Protected slice + wrong jurisdiction.
            if affected_slices and rule.protected_slice_ids:
                hit = affected_slices & rule.protected_slice_ids
                if hit and target_jx not in rule.allowed_jurisdictions:
                    violations.append(
                        ConstraintViolation(
                            constraint_id=self._CID_JX,
                            severity="hard",
                            margin_dB=None,
                            message=(
                                f"action would anchor LI-protected slice(s) "
                                f"{sorted(hit)} in jurisdiction {target_jx!r}, "
                                f"not in allowed set "
                                f"{sorted(rule.allowed_jurisdictions)} "
                                f"(rule {rule.rule_id!r}); refused."
                            ),
                        )
                    )

        return violations

    def project(
        self, action: dict[str, Any], context: dict[str, Any]
    ) -> tuple[dict[str, Any], list[ConstraintViolation]]:
        """Conservative projection: strip protected UEs from ``affected_ue_ids``.

        We deliberately do NOT attempt to rewrite ``target_jurisdiction``,
        because silently moving a warrant target across borders is exactly
        the failure mode the constraint exists to prevent. If a slice
        violation is detected, we leave it in place and report it — the
        caller (emit-guard chain) will refuse to emit. This is the
        loud-failure path by design.
        """
        feasible = dict(action)
        corrections: list[ConstraintViolation] = []

        # Fail-closed gate (mirrors check_feasibility).
        if self.fail_closed and not self.rules:
            corrections.append(
                ConstraintViolation(
                    constraint_id=self._CID_FAIL_CLOSED,
                    severity="hard",
                    margin_dB=None,
                    message=(
                        "LIConstraint fail-closed: empty rule catalogue rejects "
                        "all actions; project() returns the action unmodified "
                        "alongside the fail-closed violation so the emit-guard "
                        "chain refuses to emit."
                    ),
                )
            )
            return feasible, corrections

        affected_ues = self._as_str_set(feasible.get("affected_ue_ids"))
        if affected_ues:
            protected_union: set[str] = set()
            for rule in self.rules:
                protected_union |= rule.protected_ue_ids
            removed = affected_ues & protected_union
            if removed:
                feasible["affected_ue_ids"] = sorted(affected_ues - removed)
                corrections.append(
                    ConstraintViolation(
                        constraint_id=self._CID_UE,
                        severity="hard",
                        margin_dB=None,
                        message=(
                            f"projected: removed LI-protected UE(s) "
                            f"{sorted(removed)} from affected_ue_ids."
                        ),
                    )
                )

        # Surface (but do not silently fix) slice/jurisdiction violations.
        slice_violations = [
            v for v in self.check_feasibility(feasible, context)
            if v.constraint_id == self._CID_JX
        ]
        corrections.extend(slice_violations)

        return feasible, corrections

    def lagrangian_violation(
        self, action: torch.Tensor, context: dict[str, Any]
    ) -> torch.Tensor:
        """LI is hard-only — return zero unconditionally.

        Soft-penalising LI in the reward would let the policy "trade off"
        an intercept warrant against throughput, which is exactly the
        anti-pattern WG11 / ETSI TS 103 221 forbid. The training signal
        for LI must come exclusively from the projection layer (rejected
        actions never enter the replay buffer).
        """
        if action.ndim < 1:
            return torch.zeros((), dtype=action.dtype, device=action.device)
        return torch.zeros(action.shape[0], dtype=action.dtype, device=action.device)

    # ─── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _as_str_set(value: Any) -> set[str]:
        """Coerce an iterable / None into a ``set[str]`` defensively."""
        if value is None:
            return set()
        if isinstance(value, set):
            return {str(x) for x in value}
        if isinstance(value, (list, tuple, frozenset)):
            return {str(x) for x in value}
        # Single string is treated as a one-element set (not character-iterated).
        if isinstance(value, str):
            return {value}
        try:
            return {str(x) for x in value}
        except TypeError:
            return set()


__all__ = [
    "LIConstraint",
    "LIJurisdictionRule",
]
