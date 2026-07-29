"""Capacity an AI planner cannot reallocate away from a safety-critical slice.

AI-RAN Alliance WG3 ("Safety and Resilience for AI-in-the-Loop Operations")
asks for automated actions that are "bounded and reversible", "fail-safe
defaults", and "dedicated capacity that AI cannot reallocate". These tests pin
that the planner is never trusted to honour the floor: it may propose any split
at all, and the projection restores the floor before anything is emitted.
"""

from __future__ import annotations

import pytest

from horizon_ric.shield.invariants import ProtectedSliceFloorInvariant

FLOOR = 0.20
SLICE = "safety_critical"


def _inv() -> ProtectedSliceFloorInvariant:
    return ProtectedSliceFloorInvariant(slice_id=SLICE, floor=FLOOR)


def test_starved_slice_is_restored_to_its_floor() -> None:
    inv = _inv()
    starved = {"prb_allocation": {SLICE: 0.02, "embb": 0.68, "best_effort": 0.30}}
    assert inv.evaluate(starved, {}).satisfied is False

    projected, corrections = inv.project(starved, {})
    alloc = projected["prb_allocation"]
    assert alloc[SLICE] == pytest.approx(FLOOR)
    assert inv.evaluate(projected, {}).satisfied is True
    assert corrections and corrections[0].constraint_id == inv.id


def test_projection_preserves_the_allocation_simplex() -> None:
    """Reclaim pro rata — a clipped allocation that no longer sums to 1 is not a
    valid PRB split and would push the error downstream."""
    inv = _inv()
    starved = {"prb_allocation": {SLICE: 0.02, "embb": 0.68, "best_effort": 0.30}}
    alloc = inv.project(starved, {})[0]["prb_allocation"]
    assert sum(alloc.values()) == pytest.approx(1.0)
    # Others shrink in proportion, preserving their relative ordering.
    assert alloc["embb"] > alloc["best_effort"]


def test_a_compliant_proposal_is_untouched() -> None:
    inv = _inv()
    ok = {"prb_allocation": {SLICE: 0.35, "embb": 0.65}}
    assert inv.evaluate(ok, {}).satisfied is True
    projected, corrections = inv.project(ok, {})
    assert corrections == []
    assert projected["prb_allocation"] == ok["prb_allocation"]


def test_unreachable_floor_fails_closed() -> None:
    """If the floor cannot be met even by zeroing every other slice, the
    invariant must stay UNSATISFIED so the guard chain refuses the emit. It must
    not fabricate an allocation that merely looks compliant."""
    inv = _inv()
    impossible = {"prb_allocation": {SLICE: 0.02, "embb": 0.05}}
    projected, corrections = inv.project(impossible, {})

    check = inv.evaluate(projected, {})
    assert check.satisfied is False, "unreachable floor must not report success"
    assert check.margin < 0
    assert projected["prb_allocation"][SLICE] == pytest.approx(0.07)
    assert "UNREACHABLE" in corrections[0].message


def test_actions_without_an_allocation_are_not_affected() -> None:
    """Most actions in this system are RF, not scheduling. The slice floor must
    be vacuously satisfied for them rather than failing every unrelated emit."""
    inv = _inv()
    rf_only = {"tx_power_dBm": 27.0, "antenna_gain_dBi": 6.0}
    assert inv.evaluate(rf_only, {}).satisfied is True
    assert inv.project(rf_only, {})[1] == []


@pytest.mark.parametrize(
    "adversarial",
    [
        {SLICE: 0.0, "embb": 1.0},          # total starvation
        {SLICE: -0.5, "embb": 1.5},         # negative share
        {"embb": 1.0},                       # slice omitted entirely
    ],
)
def test_adversarial_proposals_cannot_defeat_the_floor(adversarial: dict) -> None:
    """The guarantee must not depend on the planner being well-behaved."""
    inv = _inv()
    projected, _ = inv.project({"prb_allocation": adversarial}, {})
    assert inv.evaluate(projected, {}).satisfied is True
    assert projected["prb_allocation"][SLICE] == pytest.approx(FLOOR)
