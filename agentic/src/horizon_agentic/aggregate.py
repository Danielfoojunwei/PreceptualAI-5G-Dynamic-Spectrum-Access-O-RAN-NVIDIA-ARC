"""Invariants over a *set* of actions, where the single-action chain is blind.

Every invariant in ``horizon_ric.shield.invariants`` takes one action and
decides it alone. That is sufficient when one planner emits one action, and it
is structurally insufficient the moment two agents act on the same cell,
because some limits are properties of the combination and of nothing else.

The clearest case is already latent in the shipped code, and it is the reason
this module exists rather than a hypothetical about future cross-domain agents.

``ProtectedSliceFloorInvariant`` guards a **fraction**: ``prb_allocation`` maps
slice id to a share of the cell's PRBs, and the floor is a share. It has no
view of what that share is a share *of*. So:

* An energy agent reduces ``bandwidth_hz`` from 100 MHz to 20 MHz. Checked
  alone this is impeccable — inside the band, under the EIRP ceiling, and it
  carries no ``prb_allocation`` at all, so the slice floor reports itself
  *vacuously satisfied*.
* A slice agent sets ``{"safety_critical": 0.20, "embb": 0.80}``. Checked
  alone this is also impeccable — the protected share is exactly its 0.20
  floor.

Both pass. Together the safety-critical slice holds 0.20 x 20 MHz = 4 MHz where
it previously held 20 MHz: a fivefold cut in the capacity the floor exists to
guarantee, with no invariant violated and two valid certificates issued. The
fraction never moved. The denominator did, and no single-action invariant can
see a denominator another agent changed.

That is not a bug in ``ProtectedSliceFloorInvariant`` — a per-action check
cannot be responsible for state it never receives. It is the boundary of what
per-action checking can mean, and crossing it needs an invariant whose argument
is the bundle.

These reuse ``InvariantCheck`` from ``horizon_ric.shield.certificate`` so
aggregate evidence has the same shape as per-action evidence and lands in the
same record. They deliberately do **not** project: a projection over a bundle
would have to decide *whose* action to alter, which is an authority question,
not a physics one. That decision lives in ``conflict.py``, and refusal is
always available when it cannot be made safely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from horizon_ric.shield.certificate import InvariantCheck

__all__ = [
    "AggregateInvariant",
    "AbsoluteSliceCapacityFloor",
    "AggregateEirpBudget",
    "effective_bandwidth_hz",
    "effective_allocation",
]

Action = Mapping[str, Any]
Context = Mapping[str, Any]


@runtime_checkable
class AggregateInvariant(Protocol):
    """A limit that is a property of several actions together.

    ``reads`` names the action keys the invariant depends on. Conflict
    resolution uses it to decide which bundle members could possibly be
    responsible for a violation; an invariant that omits it is treated as
    depending on everything, which is the fail-closed reading.
    """

    id: str
    reads: frozenset[str]

    def evaluate(
        self, actions: Sequence[Action], context: Context
    ) -> InvariantCheck: ...


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def effective_bandwidth_hz(actions: Sequence[Action]) -> float | None:
    """The bandwidth the cell actually ends up with, across the bundle.

    Several agents may each name a bandwidth. The **minimum** is taken, not the
    newest or the mean: an aggregate safety check must reason about the
    worst case the bundle could produce, and taking anything larger would let
    one optimistic agent mask another's cut. Where two agents genuinely
    disagree, that is a conflict — :mod:`horizon_agentic.conflict` resolves it,
    and this function only has to be safe in the meantime.
    """
    seen = [
        value
        for value in (_finite(a.get("bandwidth_hz")) for a in actions)
        if value is not None and value > 0.0
    ]
    return min(seen) if seen else None


def effective_allocation(actions: Sequence[Action]) -> Mapping[str, float] | None:
    """The PRB split the bundle settles on, or ``None`` if nobody sets one.

    Where more than one action carries an allocation, the one giving the
    *smallest* share to each slice is taken key-wise, for the same worst-case
    reason as :func:`effective_bandwidth_hz`. The result is not renormalised:
    an aggregate check must see what the bundle actually implies, including
    that it may no longer sum to one, rather than a tidied version of it.
    """
    allocations = [a.get("prb_allocation") for a in actions]
    present = [alloc for alloc in allocations if isinstance(alloc, Mapping) and alloc]
    if not present:
        return None
    merged: dict[str, float] = {}
    for alloc in present:
        for slice_id, share in alloc.items():
            value = _finite(share)
            if value is None:
                continue
            key = str(slice_id)
            merged[key] = value if key not in merged else min(merged[key], value)
    return merged or None


@dataclass(frozen=True)
class AbsoluteSliceCapacityFloor:
    """Capacity, in Hz, that a protected slice must retain across the bundle.

    The service commitment a protected slice actually carries is absolute — so
    many Hz of usable spectrum — while ``ProtectedSliceFloorInvariant`` can
    only express it as a share. This restates it in the unit the commitment is
    actually written in, which is what makes it immune to another agent moving
    the denominator.

    Applicability is the delicate part. This is *not* vacuously satisfied when
    only one of the two inputs is present, because that is precisely the
    single-agent case the per-action invariant already handles correctly. It
    engages when the bundle supplies both a bandwidth and an allocation — which
    is exactly the cross-agent situation nothing else covers.
    """

    slice_id: str = "safety_critical"
    min_capacity_hz: float = 20e6
    id: str = "absolute_slice_capacity_floor"
    reads: frozenset[str] = frozenset({"bandwidth_hz", "prb_allocation"})

    def evaluate(self, actions: Sequence[Action], context: Context) -> InvariantCheck:
        # Where the bundle is silent, the network's current value still
        # applies — dropping a request does not reset the cell. Falling back to
        # ``context["baseline_action"]`` is what stops conflict resolution
        # "fixing" a violation by removing the evidence of it: drop the
        # bandwidth change and the cell keeps its real bandwidth, which is
        # still what the floor is measured against. In the limit, an empty
        # bundle is evaluated against the baseline alone, so "drop everything"
        # is not an escape hatch either.
        #
        # One source of current state on purpose. An earlier draft read
        # separate ``current_bandwidth_hz`` and ``current_prb_allocation``
        # keys, which let a caller supply a bandwidth that disagreed with the
        # baseline the authority check was using — two truths about the same
        # cell, and the invariant silently believed the wrong one.
        baseline = context.get("baseline_action")
        baseline = baseline if isinstance(baseline, Mapping) else {}

        bandwidth = effective_bandwidth_hz(actions)
        if bandwidth is None:
            bandwidth = _finite(baseline.get("bandwidth_hz"))
            if bandwidth is not None and bandwidth <= 0.0:
                bandwidth = None
        allocation = effective_allocation(actions)
        if allocation is None:
            allocation = effective_allocation([baseline]) if baseline else None

        if bandwidth is None or not allocation:
            return InvariantCheck(
                invariant_id=self.id,
                satisfied=True,
                margin=float("inf"),
                unit="Hz",
                detail=(
                    "neither the bundle nor the context determines both a "
                    "bandwidth and a PRB allocation; absolute capacity is not "
                    "computable here"
                ),
            )

        share = allocation.get(self.slice_id)
        if share is None:
            return InvariantCheck(
                invariant_id=self.id,
                satisfied=False,
                margin=-self.min_capacity_hz,
                unit="Hz",
                detail=(
                    f"bundle sets a PRB allocation that omits protected slice "
                    f"{self.slice_id!r}; its capacity would be zero"
                ),
            )

        capacity = share * bandwidth
        margin = capacity - self.min_capacity_hz
        return InvariantCheck(
            invariant_id=self.id,
            satisfied=margin >= 0.0,
            margin=margin,
            unit="Hz",
            detail=(
                f"slice {self.slice_id!r} holds {share:.4f} x {bandwidth / 1e6:.3f} MHz "
                f"= {capacity / 1e6:.3f} MHz against an absolute floor of "
                f"{self.min_capacity_hz / 1e6:.3f} MHz"
            ),
        )


@dataclass(frozen=True)
class AggregateEirpBudget:
    """A site-level EIRP budget that no single carrier's ceiling expresses.

    ``MaxEirpInvariant`` bounds one emission. A site radiating four carriers
    each exactly at the per-carrier ceiling is four times the power of one, and
    per-carrier checking cannot object. Summation is in the linear domain
    because dBm do not add.
    """

    max_total_eirp_dBm: float = 36.0
    id: str = "aggregate_eirp_budget"
    reads: frozenset[str] = frozenset({"tx_power_dBm", "antenna_gain_dBi"})

    @staticmethod
    def _eirp_dBm(action: Action) -> float | None:
        power = _finite(action.get("tx_power_dBm"))
        if power is None:
            return None
        gain = _finite(action.get("antenna_gain_dBi")) or 0.0
        return power + gain

    def evaluate(self, actions: Sequence[Action], context: Context) -> InvariantCheck:
        contributions = [
            eirp for eirp in (self._eirp_dBm(a) for a in actions) if eirp is not None
        ]
        if not contributions:
            return InvariantCheck(
                invariant_id=self.id,
                satisfied=True,
                margin=float("inf"),
                unit="dBm",
                detail="no action in the bundle radiates; budget not engaged",
            )
        linear = sum(10.0 ** (eirp / 10.0) for eirp in contributions)
        total = 10.0 * math.log10(linear) if linear > 0 else float("-inf")
        margin = self.max_total_eirp_dBm - total
        return InvariantCheck(
            invariant_id=self.id,
            satisfied=margin >= 0.0,
            margin=margin,
            unit="dBm",
            detail=(
                f"{len(contributions)} emission(s) sum to {total:.3f} dBm against a "
                f"site budget of {self.max_total_eirp_dBm:.3f} dBm"
            ),
        )
