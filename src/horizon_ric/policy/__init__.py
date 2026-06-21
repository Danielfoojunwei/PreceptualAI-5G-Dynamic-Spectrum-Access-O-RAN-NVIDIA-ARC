"""Policy layer — the decision artefacts the Shield validates and audits.

    li_constraint   Lawful-Intercept jurisdiction hard constraint (TS 33.127).
    counterfactual  Builds the rejected-alternatives audit field (replayable).
    emit_guards     Pre-emit go/no-go gates (incl. the Shield-certificate gate).
    guard_registry  Pluggable guard chain.

Wire: AI decision → Shield.dispose → counterfactual + emit_guards → A1 emit.
"""

from horizon_ric.policy.counterfactual import (
    AlternativeCandidate,
    build_rejected_alternatives,
)
from horizon_ric.policy.emit_guards import (
    EmitTrace,
    GuardFailure,
    run_guard_chain,
)
from horizon_ric.policy.li_constraint import (
    LIConstraint,
    LIJurisdictionRule,
)

__all__ = [
    "AlternativeCandidate",
    "EmitTrace",
    "GuardFailure",
    "LIConstraint",
    "LIJurisdictionRule",
    "build_rejected_alternatives",
    "run_guard_chain",
]
