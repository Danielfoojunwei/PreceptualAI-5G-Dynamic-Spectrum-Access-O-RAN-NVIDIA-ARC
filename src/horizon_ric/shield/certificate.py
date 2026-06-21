"""Data model for the Decision Safety Shield.

A *decision* (the data a neural-PHY block or RIC policy head emits) enters the
Shield; a *safe action* plus a *SafetyCertificate* come out. The certificate is
the AI-RAN-native trust object: it records, for one decision, which invariants
were checked, the margin to each bound, what was projected/fell back, and the
model provenance — so the decision is auditable and reproducible *by
construction*, not retroactively.

This module is intentionally dependency-light (stdlib only) so it can be
imported on the real-time path and serialised into the evidence chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# Loop tiers mirror the O-RAN control-loop hierarchy. Auditability is matched
# to the timescale: sub-ms dApp loops sample/aggregate; Non-RT rApp loops carry
# the full counterfactual.
LOOP_TIERS = ("rt", "near_rt", "non_rt")


@dataclass(frozen=True)
class ConstraintViolation:
    """A single invariant breach (or a correction applied to fix one).

    ``margin_dB`` is the signed distance to the bound in dB when the invariant
    is a power/ratio quantity (negative ⇒ violated); ``None`` for invariants
    whose natural unit is not dB (frequency, set membership, jurisdiction).
    """

    constraint_id: str
    severity: str  # "hard" | "soft"
    margin_dB: Optional[float]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "severity": self.severity,
            "margin_dB": self.margin_dB,
            "message": self.message,
        }


@dataclass(frozen=True)
class InvariantCheck:
    """Per-invariant evaluation result, with the margin to the bound.

    ``margin`` is a generic signed distance to the bound (positive ⇒ inside the
    safe set); ``unit`` documents its meaning ("dB", "Hz", "ratio", "bool").
    """

    invariant_id: str
    satisfied: bool
    margin: Optional[float] = None
    unit: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "satisfied": self.satisfied,
            "margin": self.margin,
            "unit": self.unit,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SafetyCertificate:
    """The signed-by-construction record of one shielded decision."""

    decision_id: str
    issued_at: str  # ISO-8601 UTC
    loop_tier: str
    block: str  # the AI-PHY / RIC block that produced the proposed action
    action_proposed: dict[str, Any]
    action_safe: dict[str, Any]
    invariants: list[InvariantCheck]
    corrections: list[ConstraintViolation]
    violated_ids: list[str]
    projected: bool
    fallback_used: bool
    fallback_to: Optional[str]
    safe: bool  # final action satisfies every HARD invariant
    emit_blocked: bool  # Shield could not make the action safe → refuse emit
    rng_seed: Optional[int] = None
    model_provenance: Optional[dict[str, Any]] = None

    @property
    def min_margin_dB(self) -> Optional[float]:
        """Tightest dB margin across invariants (the worst-case headroom)."""
        dbs = [c.margin for c in self.invariants if c.unit == "dB" and c.margin is not None]
        return min(dbs) if dbs else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "issued_at": self.issued_at,
            "loop_tier": self.loop_tier,
            "block": self.block,
            "action_proposed": self.action_proposed,
            "action_safe": self.action_safe,
            "invariants": [c.to_dict() for c in self.invariants],
            "corrections": [c.to_dict() for c in self.corrections],
            "violated_ids": list(self.violated_ids),
            "projected": self.projected,
            "fallback_used": self.fallback_used,
            "fallback_to": self.fallback_to,
            "safe": self.safe,
            "emit_blocked": self.emit_blocked,
            "min_margin_dB": self.min_margin_dB,
            "rng_seed": self.rng_seed,
            "model_provenance": self.model_provenance,
        }


@dataclass(frozen=True)
class ShieldDisposition:
    """Return value of ``Shield.dispose`` — the safe action and its certificate."""

    safe_action: dict[str, Any]
    certificate: SafetyCertificate


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "LOOP_TIERS",
    "ConstraintViolation",
    "InvariantCheck",
    "SafetyCertificate",
    "ShieldDisposition",
    "utc_now_iso",
]
