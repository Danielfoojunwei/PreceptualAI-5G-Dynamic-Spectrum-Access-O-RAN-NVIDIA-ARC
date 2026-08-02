"""Deterministic resolution when two agents' actions cannot both stand.

O-RAN already recognises this problem: the Non-RT RIC specifications carry
analysis and service-level recommendations for detecting and avoiding A1 policy
conflict between rApps, and there is a body of work on conflict mitigation in
the Near-RT RIC. This module is not an attempt to reinvent that. Its scope is
narrower and sits underneath it: given that a conflict has been *detected* by
an aggregate invariant, produce a resolution that is deterministic, that an
operator programmed rather than an agent negotiated, and that cannot end in an
emission the invariants would have refused.

Two design choices carry the weight.

**Resolution drops requests; it never rewrites values.** The tempting design is
to compute the value that would satisfy the violated limit — restore the
bandwidth, raise the protected share. That requires inventing a number no agent
asked for and no operator authorised, and the number that satisfies one
aggregate limit routinely violates another. Dropping a member instead means
"this agent's requested change is not applied this cycle", which returns that
dimension to whatever the network already has. It is unambiguously a narrowing
operation, it needs no baseline, and it is explainable in one sentence to the
operator who has to answer for it.

**Priority comes from the operator's policy, never from the envelope.** An
agent that could assert its own precedence has, in effect, no precedence
constraint at all. :class:`~horizon_agentic.envelope.AuthorityGrant` carries
``priority``, and that is the only place it is read from.

The ordering is total, so the outcome does not depend on the order the bundle
happened to arrive in: drop the largest ``priority`` number first (least
important), breaking ties by ``agent_id`` and then by position. Determinism
here is not fastidiousness — a resolution that varies with arrival order cannot
be replayed from its certificate, which would cost the property the whole
system exists to provide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from horizon_agentic.aggregate import AggregateInvariant
from horizon_agentic.envelope import AuthorityPolicy
from horizon_ric.shield.certificate import InvariantCheck

__all__ = ["DroppedMember", "ConflictResolution", "resolve_conflicts"]

Action = Mapping[str, Any]


@dataclass(frozen=True)
class DroppedMember:
    """One agent's request that resolution declined to apply, and why."""

    index: int
    agent_id: str
    priority: int
    violated_id: str
    detail: str


@dataclass(frozen=True)
class ConflictResolution:
    """The outcome of resolution: who survived, who was dropped, and whether it worked."""

    admitted: tuple[int, ...]
    dropped: tuple[DroppedMember, ...]
    resolved: bool
    checks: tuple[InvariantCheck, ...]
    rounds: int


def _reads(invariant: AggregateInvariant) -> frozenset[str]:
    """Action keys an aggregate invariant depends on.

    An invariant that does not declare ``reads`` is treated as depending on
    everything. That is the fail-closed reading: it makes every member a
    candidate for dropping rather than silently excluding members that might in
    fact be the cause.
    """
    declared = getattr(invariant, "reads", None)
    if declared is None:
        return frozenset()
    return frozenset(declared)


def resolve_conflicts(
    actions: Sequence[Action],
    agent_ids: Sequence[str],
    invariants: Sequence[AggregateInvariant],
    policy: AuthorityPolicy,
    context: Mapping[str, Any],
    *,
    max_rounds: int = 8,
) -> ConflictResolution:
    """Drop the least important conflicting requests until the aggregates hold.

    Returns with ``resolved=False`` rather than raising when no subset
    satisfies the invariants; the caller refuses the transaction. Bounded by
    ``max_rounds`` for the same reason ``ShieldConfig.max_passes`` is bounded:
    an unbounded correction loop is a liveness hazard in a control path, and
    running out of rounds must fail closed.
    """
    if len(actions) != len(agent_ids):
        raise ValueError("actions and agent_ids must correspond one-to-one")

    admitted = list(range(len(actions)))
    dropped: list[DroppedMember] = []
    checks: tuple[InvariantCheck, ...] = ()

    for round_index in range(max_rounds + 1):
        live = [actions[i] for i in admitted]
        checks = tuple(inv.evaluate(live, context) for inv in invariants)
        violated = [c for c in checks if not c.satisfied]
        if not violated:
            return ConflictResolution(
                tuple(admitted), tuple(dropped), True, checks, round_index
            )
        if round_index == max_rounds or not admitted:
            break

        first = violated[0]
        culprit_invariant = next(
            (inv for inv in invariants if inv.id == first.invariant_id), None
        )
        keys = _reads(culprit_invariant) if culprit_invariant is not None else frozenset()

        # Candidates are members that touch what the violated invariant reads.
        # An invariant declaring no keys makes every member a candidate.
        candidates = [
            i
            for i in admitted
            if not keys or (set(actions[i]) & keys)
        ]
        if not candidates:
            break

        # Total order: least important first, then agent id, then position.
        # Every component is needed for the result to be independent of the
        # order the bundle arrived in.
        def rank(i: int) -> tuple[int, str, int]:
            try:
                priority = policy.priority_of(agent_ids[i])
            except KeyError:
                # No grant means no standing; drop before anything granted.
                priority = 1 << 30
            return (-priority, agent_ids[i], i)

        victim = min(candidates, key=rank)
        admitted.remove(victim)
        try:
            victim_priority = policy.priority_of(agent_ids[victim])
        except KeyError:
            victim_priority = 1 << 30
        dropped.append(
            DroppedMember(
                index=victim,
                agent_id=agent_ids[victim],
                priority=victim_priority,
                violated_id=first.invariant_id,
                detail=first.detail,
            )
        )

    return ConflictResolution(tuple(admitted), tuple(dropped), False, checks, max_rounds)
