"""The multi-agent case: a real conflict, resolved deterministically or refused.

The scenario is chosen so the conflict cannot be staged, and the first test
proves that rather than asserting it. Two agents, each acting strictly inside
its own authority, each producing an action that is legal *by every check in
the system including the aggregate one when evaluated alone*:

======================  ==============  =====================  ==============
state                   bandwidth       safety_critical share   its capacity
======================  ==============  =====================  ==============
baseline                100 MHz         0.50                   50.0 MHz
energy agent alone       50 MHz         0.50 (declared)        25.0 MHz  ok
slice agent alone       100 MHz (decl)  0.25                   25.0 MHz  ok
**both applied**         50 MHz         0.25                   12.5 MHz  **<20**
======================  ==============  =====================  ==============

Neither agent misbehaves. Neither action is projected by the Shield. The
protected slice still holds more than double its 0.20 *fractional* floor. And
the safety-critical slice ends up with 12.5 MHz where its service commitment is
20 MHz, because the fraction is measured against a denominator the other agent
moved.

If the numbers had been chosen so that one action alone already breached the
floor, this would demonstrate nothing that single-action checking does not
already catch. ``test_each_action_alone_is_legal_at_every_layer`` is therefore
the load-bearing test in this file: it is what makes the rest evidence rather
than a tautology.
"""

from __future__ import annotations

import itertools

import pytest
from horizon_agentic.aggregate import AbsoluteSliceCapacityFloor, AggregateEirpBudget
from horizon_agentic.bundle import SafetyTransaction, TransactionRefused
from horizon_agentic.envelope import (
    AgentActionEnvelope,
    AuthorityGrant,
    AuthorityPolicy,
)
from horizon_agentic.telemetry_trust import (
    FieldBound,
    TelemetryRecord,
    TelemetryTrustGate,
    TrustPolicy,
    sign_record,
)

from horizon_ric.shield import default_terrestrial_shield

ROOT = "operator-root"
BAND_LO, BAND_HI = 3.40e9, 3.50e9
FLOOR_HZ = 20e6

BASELINE = {
    "block": "ran_control",
    "frequency_hz": 3.45e9,
    "bandwidth_hz": 100e6,
    "tx_power_dBm": 20.0,
    "antenna_gain_dBi": 6.0,
    "prb_allocation": {"safety_critical": 0.50, "embb": 0.50},
}


def policy(*, energy_priority: int = 20, slice_priority: int = 15) -> AuthorityPolicy:
    return AuthorityPolicy(
        root_principal=ROOT,
        grants={
            ROOT: AuthorityGrant(
                ROOT, frozenset({"ran"}), frozenset({"spectrum", "slice", "phy", "ntn"}), 0
            ),
            "energy-agent": AuthorityGrant(
                "energy-agent", frozenset({"ran"}), frozenset({"spectrum"}), energy_priority
            ),
            "slice-agent": AuthorityGrant(
                "slice-agent", frozenset({"ran"}), frozenset({"slice"}), slice_priority
            ),
        },
    )


def energy_envelope(bandwidth_hz: float = 50e6) -> AgentActionEnvelope:
    return AgentActionEnvelope(
        agent_id="energy-agent",
        agent_version="1.0.0",
        target_domain="ran",
        granted_scopes=frozenset({"spectrum"}),
        delegation_chain=(ROOT, "energy-agent"),
        requested_action={**BASELINE, "bandwidth_hz": bandwidth_hz},
        mutates=frozenset({"bandwidth_hz"}),
    )


def slice_envelope(share: float = 0.25) -> AgentActionEnvelope:
    return AgentActionEnvelope(
        agent_id="slice-agent",
        agent_version="1.0.0",
        target_domain="ran",
        granted_scopes=frozenset({"slice"}),
        delegation_chain=(ROOT, "slice-agent"),
        requested_action={
            **BASELINE,
            "prb_allocation": {"safety_critical": share, "embb": 1.0 - share},
        },
        mutates=frozenset({"prb_allocation"}),
    )


def transaction(pol: AuthorityPolicy | None = None, **kwargs) -> SafetyTransaction:
    return SafetyTransaction(
        shield=default_terrestrial_shield(
            band_lo_hz=BAND_LO, band_hi_hz=BAND_HI, max_eirp_dBm=33.0
        ),
        policy=pol or policy(),
        aggregates=[AbsoluteSliceCapacityFloor(min_capacity_hz=FLOOR_HZ)],
        **kwargs,
    )


def ctx() -> dict:
    return {"baseline_action": dict(BASELINE)}


# ── the non-staged proof ─────────────────────────────────────────────────
def test_each_action_alone_is_legal_at_every_layer() -> None:
    """Neither agent, acting alone, violates anything. This is the whole point.

    Checked at both layers, because passing only the per-action Shield would
    leave open that the aggregate invariant is simply a stricter version of a
    limit one agent already broke.
    """
    shield = default_terrestrial_shield(
        band_lo_hz=BAND_LO, band_hi_hz=BAND_HI, max_eirp_dBm=33.0
    )
    aggregate = AbsoluteSliceCapacityFloor(min_capacity_hz=FLOOR_HZ)

    for env in (energy_envelope(), slice_envelope()):
        disposition = shield.dispose(dict(env.requested_action), {})
        assert disposition.certificate.safe, env.agent_id
        assert not disposition.certificate.projected, (
            f"{env.agent_id} was corrected by the Shield; the conflict would be "
            "an artefact of that correction rather than of the interaction"
        )
        assert disposition.certificate.violated_ids == []

        alone = aggregate.evaluate([env.requested_action], ctx())
        assert alone.satisfied, (
            f"{env.agent_id} alone already breaches the absolute floor "
            f"({alone.detail}); the scenario would prove nothing about "
            "multi-agent interaction"
        )

    # And the transaction admits each of them on its own.
    for env in (energy_envelope(), slice_envelope()):
        result = transaction().evaluate([env], context=ctx())
        assert result.committed, (env.agent_id, result.refusals)


def test_the_two_together_are_caught() -> None:
    aggregate = AbsoluteSliceCapacityFloor(min_capacity_hz=FLOOR_HZ)
    check = aggregate.evaluate(
        [energy_envelope().requested_action, slice_envelope().requested_action], ctx()
    )
    assert not check.satisfied
    assert "12.500 MHz" in check.detail


# ── resolution ───────────────────────────────────────────────────────────
def test_resolution_drops_the_least_important_agent() -> None:
    result = transaction().evaluate(
        [energy_envelope(), slice_envelope()], context=ctx()
    )
    assert result.committed, result.refusals
    assert result.resolution is not None
    dropped = {d.agent_id for d in result.resolution.dropped}
    assert dropped == {"energy-agent"}
    assert {m.agent_id for m in result.members} == {"slice-agent"}


def test_priority_actually_decides_who_is_dropped() -> None:
    """Guards against 'deterministic' meaning 'always drops position zero'.

    Same bundle, inverted operator priorities, opposite victim. If this passed
    with the same agent dropped both ways, the ordering would be an artefact of
    position rather than of policy.
    """
    inverted = policy(energy_priority=5, slice_priority=30)
    result = transaction(inverted).evaluate(
        [energy_envelope(), slice_envelope()], context=ctx()
    )
    assert result.committed, result.refusals
    assert result.resolution is not None
    assert {d.agent_id for d in result.resolution.dropped} == {"slice-agent"}


def test_resolution_is_independent_of_arrival_order() -> None:
    """A resolution that varies with arrival order cannot be replayed.

    The certificate records what was decided; if the same bundle in a different
    order decided differently, the record would not reconstruct the decision.
    """
    outcomes = set()
    for permutation in itertools.permutations([energy_envelope(), slice_envelope()]):
        result = transaction().evaluate(list(permutation), context=ctx())
        assert result.committed
        outcomes.add(
            (
                tuple(sorted(m.agent_id for m in result.members)),
                tuple(sorted(d.agent_id for d in result.resolution.dropped)),
            )
        )
    assert len(outcomes) == 1, outcomes


def test_resolution_cannot_be_achieved_by_removing_the_evidence() -> None:
    """Dropping every request must not make the violation disappear.

    This is the failure mode the design is most exposed to. Resolution works by
    dropping members; if the aggregate invariant only ever saw the bundle, then
    dropping the last member would leave it with no bandwidth and no
    allocation, it would report itself not-applicable, and the transaction
    would commit *nothing* while declaring the floor satisfied. Resolution by
    amnesia — technically true, operationally a lie.

    The scenario: a cell already below the floor (30 MHz x 0.50 = 15 MHz
    against a 20 MHz commitment), and a slice agent proposing to take it lower
    still. No subset of the bundle repairs it, including the empty subset,
    because the empty subset is evaluated against the baseline. The transaction
    must refuse — and it must refuse citing the floor, not shrug.

    An earlier draft of this module failed exactly here, reporting "absolute
    capacity is not computable" once the bundle emptied.
    """
    starved = {**BASELINE, "bandwidth_hz": 30e6}
    context = {"baseline_action": starved}
    envelope = AgentActionEnvelope(
        agent_id="slice-agent",
        agent_version="1.0.0",
        target_domain="ran",
        granted_scopes=frozenset({"slice"}),
        delegation_chain=(ROOT, "slice-agent"),
        requested_action={
            **starved,
            "prb_allocation": {"safety_critical": 0.25, "embb": 0.75},
        },
        mutates=frozenset({"prb_allocation"}),
    )
    result = transaction().evaluate([envelope], context=context)
    assert not result.committed
    assert any("absolute_slice_capacity_floor" in r for r in result.refusals)
    assert not any("not computable" in r for r in result.refusals)
    with pytest.raises(TransactionRefused):
        _ = result.members


def test_the_empty_subset_is_still_checked_against_the_baseline() -> None:
    """Stated directly, because it is the property the test above depends on."""
    aggregate = AbsoluteSliceCapacityFloor(min_capacity_hz=FLOOR_HZ)
    check = aggregate.evaluate([], {"baseline_action": {**BASELINE, "bandwidth_hz": 30e6}})
    assert not check.satisfied
    assert "15.000 MHz" in check.detail


def test_dropping_is_enough_when_the_baseline_is_sound() -> None:
    """The other side of the same coin: a repairable conflict is repaired."""
    result = transaction().evaluate(
        [energy_envelope(bandwidth_hz=50e6), slice_envelope(share=0.35)],
        context=ctx(),
    )
    assert result.committed, result.refusals
    assert {d.agent_id for d in result.resolution.dropped} == {"energy-agent"}


# ── atomicity ────────────────────────────────────────────────────────────
def test_one_unauthorized_member_refuses_the_whole_bundle() -> None:
    """A joint proposal containing an unauthorised request is unauthorised.

    Admitting the compliant half would apply a fragment of a plan nobody
    proposed — precisely the partial application the transaction exists to
    prevent.
    """
    overreaching = AgentActionEnvelope(
        agent_id="energy-agent",
        agent_version="1.0.0",
        target_domain="ran",
        granted_scopes=frozenset({"spectrum"}),
        delegation_chain=(ROOT, "energy-agent"),
        requested_action={
            **BASELINE,
            "prb_allocation": {"safety_critical": 0.05, "embb": 0.95},
        },
        mutates=frozenset({"prb_allocation"}),
    )
    result = transaction().evaluate([slice_envelope(), overreaching], context=ctx())
    assert not result.committed
    assert any("authority.energy-agent" in r for r in result.refusals)
    with pytest.raises(TransactionRefused):
        _ = result.members


def test_refusal_keys_off_the_verdict_not_off_the_problem_text() -> None:
    """Regression: the transaction must read ``authorized``, not ``problems``.

    Found by falsifying G6's authority check. Forcing ``authorize`` to return
    ``authorized=True`` left the gate green, because the transaction was
    refusing on ``problems`` being non-empty rather than on the verdict. The
    two agree today and nothing made them agree — an ``authorize`` that ever
    reported an advisory alongside an admission would have refused it.

    Here the inverse: a verdict that refuses but says nothing must still
    refuse, and must still produce a refusal a reader can act on.
    """
    import horizon_agentic.bundle as bundle_module
    from horizon_agentic.envelope import AuthorityVerdict

    original = bundle_module.authorize_all
    bundle_module.authorize_all = lambda envs, pol, **kw: tuple(
        AuthorityVerdict(False, e.agent_id, ()) for e in envs
    )
    try:
        result = transaction().evaluate([slice_envelope()], context=ctx())
    finally:
        bundle_module.authorize_all = original

    assert not result.committed
    assert any("without a stated reason" in r for r in result.refusals)


def test_a_shield_blocked_member_refuses_the_whole_bundle() -> None:
    """The per-action guarantee is not relaxed to make room for the aggregate one.

    An action the single-action chain refuses stays refused, whatever the
    bundle says. The aggregate layer can only ever subtract from what is
    emitted.
    """
    out_of_band = AgentActionEnvelope(
        agent_id="energy-agent",
        agent_version="1.0.0",
        target_domain="ran",
        granted_scopes=frozenset({"spectrum"}),
        delegation_chain=(ROOT, "energy-agent"),
        # 41 dBm against a 33 dBm ceiling: the Shield projects, and the
        # projection is what the bundle would have to accept.
        requested_action={**BASELINE, "tx_power_dBm": 41.0},
        mutates=frozenset({"tx_power_dBm"}),
    )
    result = transaction().evaluate([out_of_band], context=ctx())
    # Either the Shield corrected it (so the emitted power is the safe one) or
    # the transaction refused. What must never happen is the requested value
    # reaching the output.
    if result.committed:
        emitted = result.members[0].action
        assert emitted["tx_power_dBm"] + emitted["antenna_gain_dBi"] <= 33.0 + 1e-9
    else:
        assert result.refusals


def test_empty_bundle_is_refused() -> None:
    assert not transaction().evaluate([], context=ctx()).committed


# ── telemetry gating ─────────────────────────────────────────────────────
def _trust_gate(now: float = 1000.0) -> TelemetryTrustGate:
    return TelemetryTrustGate(
        TrustPolicy(
            secrets={"sensor-a": b"k"},
            max_age_s=5.0,
            bounds={"interference_dBm": FieldBound(-140.0, -30.0, "dBm")},
            required_fields=frozenset({"interference_dBm"}),
        ),
        clock=lambda: now,
    )


def test_untrusted_telemetry_refuses_before_any_action_is_considered() -> None:
    """Ordering matters, not just the outcome.

    Projection computed from false state yields a safe action for a world that
    does not exist, and signs a certificate saying so. The refusal has to come
    before authority and before the Shield, and the empty authority tuple is
    what proves it did.
    """
    txn = transaction(trust_gate=_trust_gate())
    stale = sign_record(
        TelemetryRecord("sensor-a", 900.0, 1, {"interference_dBm": -95.0}), b"k"
    )
    result = txn.evaluate(
        [energy_envelope(), slice_envelope()], telemetry=[stale], context=ctx()
    )
    assert not result.committed
    assert any(r.startswith("telemetry.") for r in result.refusals)
    assert result.authority == ()


def test_trusted_telemetry_reaches_the_decision_context() -> None:
    txn = transaction(trust_gate=_trust_gate())
    fresh = sign_record(
        TelemetryRecord("sensor-a", 999.0, 1, {"interference_dBm": -95.0}), b"k"
    )
    result = txn.evaluate([slice_envelope()], telemetry=[fresh], context=ctx())
    assert result.committed, result.refusals
    assert result.trust is not None and result.trust.trusted


def test_telemetry_without_a_gate_is_refused_not_trusted() -> None:
    """Silently trusting telemetry because nobody configured a gate is the bug."""
    result = transaction().evaluate(
        [slice_envelope()],
        telemetry=[sign_record(TelemetryRecord("s", 1.0, 1, {"x": 1.0}), b"k")],
        context=ctx(),
    )
    assert not result.committed


# ── the other aggregate ──────────────────────────────────────────────────
def test_aggregate_eirp_budget_sees_what_per_carrier_checking_cannot() -> None:
    """Four carriers each at the ceiling is four times the power of one."""
    budget = AggregateEirpBudget(max_total_eirp_dBm=33.0)
    at_ceiling = {"tx_power_dBm": 27.0, "antenna_gain_dBi": 6.0}  # 33 dBm each
    assert budget.evaluate([at_ceiling], {}).satisfied
    four = budget.evaluate([at_ceiling] * 4, {})
    assert not four.satisfied
    assert "39.0" in four.detail  # 33 dBm + 10*log10(4) = 39.02 dBm
