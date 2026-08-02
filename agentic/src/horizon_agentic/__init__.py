"""Horizon-Agentic: the deterministic action-control plane for multiple agents.

This package is **additive and experimental**. It does not modify
``horizon_ric``; it imports it read-only and composes on top. Nothing here is
wired into the shipped runtime, and nothing here is claimed in the AI-RAN
Alliance proposal — see ``agentic/README.md`` for why that separation is
deliberate.

What it adds over the single-action Shield:

``telemetry_trust``
    A gate that removes the unconditional-trust assumption on input state.
    Authenticated source, freshness, replay detection, range and
    physical-consistency bounds, cross-source corroboration for critical
    fields, and refusal when required evidence is missing.

``envelope``
    ``AgentActionEnvelope`` — who is asking, under what delegated authority,
    for which domain, and which action keys that authority permits them to
    write. Enforced at admission, not merely recorded.

``aggregate``
    Invariants over a *set* of actions. The single-action chain in
    ``horizon_ric.shield`` cannot see an interaction between two agents; these
    can.

``bundle``
    ``SafetyTransaction`` — validate, project and commit a whole multi-agent
    plan, or refuse it as a unit. Atomicity is the property that makes the
    aggregate invariants worth having.

``conflict``
    Deterministic resolution between agents whose actions cannot both stand,
    ordered by an operator-programmed priority the agents cannot influence.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
