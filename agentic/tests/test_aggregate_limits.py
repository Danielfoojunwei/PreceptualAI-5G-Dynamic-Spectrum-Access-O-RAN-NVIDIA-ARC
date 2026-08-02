"""The aggregate limits, including the two that were missing and the one that lied.

Three findings from adversarial review are pinned here.

**PRB conservation was absent.** ``AbsoluteSliceCapacityFloor`` asks whether one
slice has enough; nobody asked whether the cell had that much to give. Three
agents each raising their own slice committed 160% of the cell's PRBs with a
comfortable margin on every other check.

**Aggregate PFD was absent**, despite ``ntn`` being a first-class scope and the
single-action ``PfdCeilingInvariant`` being the one the repository's own
comments call funding-relevant.

**Summing and pair-wise limits were unsound without resource identity.** Two
agents describing the *same* cell were indistinguishable from two carriers:
``AggregateEirpBudget`` manufactured a phantom +3.01 dB from a second agent
merely declaring the power it was not changing, and ``SpectralSeparation``
would report two co-sited carriers as overlapping — a false refusal, which in a
fail-closed system costs as much as a false admission.
"""

from __future__ import annotations

import pytest
from horizon_agentic.aggregate import (
    RESOURCE_IDS_KEY,
    AggregateEirpBudget,
    AggregatePfdCeiling,
    PrbConservation,
    SpectralSeparation,
)


def ctx(*resource_ids: str) -> dict:
    return {RESOURCE_IDS_KEY: list(resource_ids)}


# ── PRB conservation ─────────────────────────────────────────────────────
def test_a_bundle_cannot_allocate_more_prbs_than_exist() -> None:
    inv = PrbConservation()
    over = [
        {"prb_allocation": {"safety_critical": 0.6}},
        {"prb_allocation": {"embb": 0.4}},
        {"prb_allocation": {"urllc": 0.6}},
    ]
    check = inv.evaluate(over, ctx("cell-1", "cell-1", "cell-1"))
    assert not check.satisfied
    assert "1.6000" in check.detail


def test_a_valid_simplex_passes() -> None:
    inv = PrbConservation()
    ok = [{"prb_allocation": {"safety_critical": 0.3, "embb": 0.7}}]
    assert inv.evaluate(ok, ctx("cell-1")).satisfied


def test_negative_shares_are_refused() -> None:
    """A negative share sums fine and is physically meaningless."""
    inv = PrbConservation()
    sneaky = [{"prb_allocation": {"safety_critical": 1.5, "embb": -0.5}}]
    check = inv.evaluate(sneaky, ctx("cell-1"))
    assert not check.satisfied
    assert "negative" in check.detail


def test_conservation_falls_back_to_the_baseline() -> None:
    inv = PrbConservation()
    check = inv.evaluate(
        [],
        {"baseline_action": {"prb_allocation": {"a": 0.9, "b": 0.9}}},
    )
    assert not check.satisfied


# ── aggregate PFD ────────────────────────────────────────────────────────
def _beam(power_dBm: float) -> dict:
    return {
        "tx_power_dBm": power_dBm,
        "sat_antenna_gain_dBi": 30.0,
        "slant_range_m": 1200e3,
        "bandwidth_hz": 20e6,
    }


def test_pfd_is_not_engaged_without_a_slant_range() -> None:
    assert AggregatePfdCeiling().evaluate([{"tx_power_dBm": 20.0}], ctx("s")).satisfied


def test_beams_add_in_the_linear_domain() -> None:
    """No per-beam ceiling expresses the total at the illuminated point.

    Each beam here sits comfortably inside the ceiling on its own. Eight of
    them add 10*log10(8) = 9.03 dB, which is the whole point: dB do not add,
    and a per-beam check cannot see the sum.
    """
    inv = AggregatePfdCeiling(max_pfd_dBW_m2_MHz=-146.0)
    one = inv.evaluate([_beam(-10.0)], ctx("beam-1"))
    assert one.satisfied, one.detail

    eight = inv.evaluate(
        [_beam(-10.0)] * 8, ctx(*[f"beam-{i}" for i in range(8)])
    )
    assert one.margin - eight.margin == pytest.approx(9.03, abs=0.05)

    sixteen = inv.evaluate(
        [_beam(-10.0)] * 16, ctx(*[f"beam-{i}" for i in range(16)])
    )
    assert not sixteen.satisfied, sixteen.detail


# ── resource identity ────────────────────────────────────────────────────
def test_two_agents_on_one_cell_do_not_manufacture_power() -> None:
    """The phantom +3.01 dB.

    Both members carry the same declared power for the same cell. Summing
    bundle members reported 29.01 dBm where one action reported 26.00 —
    3.01 dB of transmit power conjured by a second agent *declaring* a value it
    was not changing.
    """
    inv = AggregateEirpBudget(max_total_eirp_dBm=33.0)
    action = {"tx_power_dBm": 20.0, "antenna_gain_dBi": 6.0}
    one = inv.evaluate([action], ctx("cell-1"))
    same_cell = inv.evaluate([action, action], ctx("cell-1", "cell-1"))
    assert same_cell.margin == one.margin, (
        f"a second agent describing the same cell changed the total: "
        f"{one.detail} vs {same_cell.detail}"
    )


def test_distinct_resources_still_sum() -> None:
    """The limit must remain real for genuinely separate carriers."""
    inv = AggregateEirpBudget(max_total_eirp_dBm=33.0)
    action = {"tx_power_dBm": 27.0, "antenna_gain_dBi": 6.0}
    assert inv.evaluate([action], ctx("cell-1")).satisfied
    four = inv.evaluate(
        [action] * 4, ctx("cell-1", "cell-2", "cell-3", "cell-4")
    )
    assert not four.satisfied
    assert "39.0" in four.detail


def test_co_sited_carriers_are_not_reported_as_overlapping() -> None:
    """A false refusal costs as much as a false admission in a fail-closed system."""
    inv = SpectralSeparation(guard_hz=1e6)
    carrier = {"frequency_hz": 3.45e9, "bandwidth_hz": 20e6}
    assert inv.evaluate([carrier, carrier], ctx("cell-1", "cell-1")).satisfied


def test_genuinely_overlapping_carriers_are_still_caught() -> None:
    inv = SpectralSeparation(guard_hz=1e6)
    check = inv.evaluate(
        [
            {"frequency_hz": 3.41e9, "bandwidth_hz": 20e6},
            {"frequency_hz": 3.42e9, "bandwidth_hz": 20e6},
        ],
        ctx("cell-1", "cell-2"),
    )
    assert not check.satisfied
    assert "need 21.000 MHz" in check.detail


def test_missing_resource_ids_fall_back_to_one_resource_per_member() -> None:
    """Conservative for a summing limit: it over-counts rather than under-counts.

    Only reachable by a caller driving an aggregate directly; the transaction
    always supplies real ids.
    """
    inv = AggregateEirpBudget(max_total_eirp_dBm=33.0)
    action = {"tx_power_dBm": 27.0, "antenna_gain_dBi": 6.0}
    assert not inv.evaluate([action] * 4, {}).satisfied
