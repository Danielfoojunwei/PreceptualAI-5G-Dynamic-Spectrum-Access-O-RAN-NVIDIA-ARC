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

__all__ = [
    "DroppedMember",
    "ConflictResolution",
    "resolve_conflicts",
    "RESOURCE_IDS_KEY",
]

# Context key carrying the resource id of each live action, aligned by index.
RESOURCE_IDS_KEY = "_resource_ids"

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


def _improves(before: InvariantCheck, after: InvariantCheck) -> bool:
    """Did removing a member move the violated invariant toward satisfaction?

    Margins may be ``None`` (an invariant that reports no margin) or infinite
    (not applicable), so the comparison is written to fall back to "no
    improvement" rather than raising on either.
    """
    if after.satisfied and not before.satisfied:
        return True
    if before.margin is None or after.margin is None:
        return False
    try:
        return float(after.margin) > float(before.margin)
    except (TypeError, ValueError, OverflowError):
        return False


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
    resource_ids: Sequence[str] | None = None,
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

    resources = list(resource_ids or ["" for _ in actions])
    if len(resources) != len(actions):
        raise ValueError("resource_ids must correspond one-to-one with actions")
    admitted = list(range(len(actions)))
    dropped: list[DroppedMember] = []
    checks: tuple[InvariantCheck, ...] = ()

    for round_index in range(max_rounds + 1):
        live = [actions[i] for i in admitted]
        # Resource ids travel with the surviving members, rebuilt each round so
        # a dropped member's resource goes with it.
        live_ctx = {**context, RESOURCE_IDS_KEY: [resources[i] for i in admitted]}
        checks = tuple(inv.evaluate(live, live_ctx) for inv in invariants)
        violated = [c for c in checks if not c.satisfied]
        if not violated:
            return ConflictResolution(
                tuple(admitted), tuple(dropped), True, checks, round_index
            )
        if round_index == max_rounds or not admitted:
            break

        first = violated[0]
        culpable_invariant = next(
            (inv for inv in invariants if inv.id == first.invariant_id), None
        )
        keys = (
            _reads(culpable_invariant) if culpable_invariant is not None else frozenset()
        )

        # Candidates are members that touch what the violated invariant reads.
        # An invariant declaring no keys makes every member a candidate.
        touching = [i for i in admitted if not keys or (set(actions[i]) & keys)]
        if not touching:
            break

        # Culpability before priority. Key-intersection alone is nearly a no-op
        # here: `bandwidth_hz` is in REQUIRED_ACTION_KEYS, so every well-formed
        # action "touches" what AbsoluteSliceCapacityFloor reads, and the
        # victim was decided purely by operator priority. That turns resolution
        # into a targeted denial primitive — an agent with a low priority
        # number submits an action that drives an aggregate into violation, and
        # a rival's legitimate request is dropped instead of its own.
        #
        # A member is culpable when removing it actually improves the violated
        # invariant. That is general: it needs no per-invariant introspection,
        # and it costs one extra evaluation per candidate on a path that only
        # runs when something is already wrong.
        # Tiered, because "improves it" is too weak on its own. Dropping any
        # member that lowered *some* contested value usually improves the
        # margin a little, so an improvement test alone re-admits nearly
        # everyone and priority decides again. Repair is the strong signal:
        # prefer members whose removal actually satisfies the invariant, and
        # fall back to mere improvement only when nothing repairs it.
        if culpable_invariant is not None:
            repairs: list[int] = []
            improves: list[int] = []
            for i in touching:
                keep = [j for j in admitted if j != i]
                without = [actions[j] for j in keep]
                after = culpable_invariant.evaluate(
                    without, {**context, RESOURCE_IDS_KEY: [resources[j] for j in keep]}
                )
                if after.satisfied:
                    repairs.append(i)
                elif _improves(first, after):
                    improves.append(i)
            candidates = repairs or improves or touching
        else:
            candidates = touching

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
