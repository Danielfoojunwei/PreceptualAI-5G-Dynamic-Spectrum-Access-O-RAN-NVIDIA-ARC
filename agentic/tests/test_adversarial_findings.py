"""Attacks an adversarial review reproduced against this code, now pinned.

Four independent reviewers were pointed at the subtree and told to break it.
Everything here is something that *worked* — verified end to end against the
running code, not theorised — and the tests exist so the fixes cannot silently
regress.

Recording them is the point. A package whose pitch is fail-closed enforcement
had a fail-open authority check, a "success" that emitted nothing, two
denial-of-service primitives, and a conflict resolver that could be aimed at a
rival. None of that is visible from reading the happy path.
"""

from __future__ import annotations

import json

import pytest
from horizon_agentic.aggregate import AbsoluteSliceCapacityFloor
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
FLOOR_HZ = 20e6
SECRET = b"sensor-secret"

BASELINE = {
    "block": "ran_control",
    "frequency_hz": 3.45e9,
    "bandwidth_hz": 100e6,
    "tx_power_dBm": 20.0,
    "antenna_gain_dBi": 6.0,
    "prb_allocation": {"safety_critical": 0.50, "embb": 0.50},
}


def policy(**priorities) -> AuthorityPolicy:
    grants = {
        ROOT: AuthorityGrant(ROOT, frozenset({"ran"}), frozenset({"spectrum", "slice"}), 0),
        "energy-agent": AuthorityGrant(
            "energy-agent", frozenset({"ran"}), frozenset({"spectrum"}),
            priorities.get("energy", 20),
        ),
        "slice-agent": AuthorityGrant(
            "slice-agent", frozenset({"ran"}), frozenset({"slice"}),
            priorities.get("slice", 15),
        ),
    }
    return AuthorityPolicy(root_principal=ROOT, grants=grants)


def transaction(pol=None, **kw) -> SafetyTransaction:
    return SafetyTransaction(
        shield=default_terrestrial_shield(
            band_lo_hz=3.40e9, band_hi_hz=3.50e9, max_eirp_dBm=33.0
        ),
        policy=pol or policy(),
        aggregates=[AbsoluteSliceCapacityFloor(min_capacity_hz=FLOOR_HZ)],
        **kw,
    )


def slice_env(share=0.25, *, agent_id="slice-agent"):
    return AgentActionEnvelope(
        agent_id=agent_id,
        agent_version="1.0.0",
        target_domain="ran",
        granted_scopes=frozenset({"slice"}),
        delegation_chain=(ROOT, agent_id),
        requested_action={
            **BASELINE,
            "prb_allocation": {"safety_critical": share, "embb": 1.0 - share},
        },
        mutates=frozenset({"prb_allocation"}),
    )


def energy_env(bandwidth_hz=50e6):
    return AgentActionEnvelope(
        agent_id="energy-agent",
        agent_version="1.0.0",
        target_domain="ran",
        granted_scopes=frozenset({"spectrum"}),
        delegation_chain=(ROOT, "energy-agent"),
        requested_action={**BASELINE, "bandwidth_hz": bandwidth_hz},
        mutates=frozenset({"bandwidth_hz"}),
    )


def ctx() -> dict:
    return {"baseline_action": dict(BASELINE)}


# ── A2: scope bypass via undeclared state ────────────────────────────────
def test_a_slice_agent_cannot_raise_transmit_power() -> None:
    """The reproduced end-to-end bypass.

    A slice-scoped agent carries ``tx_power_dBm: 95.0`` and declares it mutates
    only ``prb_allocation``. Key-scoping passed it (the mutated set is in
    scope), the Shield clamped 95 dBm down to something legal, and the emitted
    certificate attested it as safe — a 7 dB power increase by an agent with no
    spectrum scope.

    The clamping is what makes this subtle. The Shield restores *physics*, not
    *authority*: a legal emission the agent was never entitled to request is
    still a control-plane failure.
    """
    sneaky = AgentActionEnvelope(
        agent_id="slice-agent",
        agent_version="1.0.0",
        target_domain="ran",
        granted_scopes=frozenset({"slice"}),
        delegation_chain=(ROOT, "slice-agent"),
        requested_action={
            **BASELINE,
            "tx_power_dBm": 95.0,
            "prb_allocation": {"safety_critical": 0.5, "embb": 0.5},
        },
        mutates=frozenset({"prb_allocation"}),
    )
    result = transaction().evaluate([sneaky], context=ctx())
    assert not result.committed
    assert any("without declaring them as mutations" in r for r in result.refusals)


def test_a_transaction_without_a_baseline_is_refused() -> None:
    """Fail-closed, not fail-quiet.

    ``authorize`` can run without a baseline, but only in a mode that requires
    every carried key to be declared. A transaction that silently switched to
    that mode would be indistinguishable from one that checked.
    """
    result = transaction().evaluate([slice_env()], context={})
    assert not result.committed
    assert any("no baseline_action" in r for r in result.refusals)


# ── A3: telemetry clobbering the decision frame ──────────────────────────
def test_telemetry_cannot_overwrite_the_baseline() -> None:
    """Trusted telemetry and caller context shared one flat namespace.

    A source reporting a field named ``baseline_action`` replaced the baseline
    mapping with a float; the ``isinstance(..., Mapping)`` guard then failed
    and the anti-smuggling check switched off for the whole bundle. Telemetry
    is an input to a decision — it does not get to redefine the decision's
    frame.
    """
    pol = TrustPolicy(
        secrets={"sensor-a": SECRET},
        max_age_s=5.0,
        bounds={"baseline_action": FieldBound(-1e9, 1e9, "")},
    )
    gate = TelemetryTrustGate(pol, clock=lambda: 1000.0)
    rogue = sign_record(
        TelemetryRecord("sensor-a", 1000.0, 1, {"baseline_action": 1.0}), SECRET
    )
    result = transaction(trust_gate=gate).evaluate(
        [slice_env()], telemetry=[rogue], context=ctx()
    )
    # The baseline survives, so the transaction proceeds on the real frame.
    assert result.committed, result.refusals


# ── A5/A6: denial of service and the empty success ───────────────────────
def test_duplicate_requests_from_one_principal_are_refused() -> None:
    """A flood under one id exhausted resolution and refused honest members."""
    flood = [slice_env(0.25) for _ in range(9)]
    result = transaction().evaluate(flood, context=ctx())
    assert not result.committed
    assert any("more than one request from" in r for r in result.refusals)


def test_dropping_every_member_is_a_refusal_not_a_commit() -> None:
    """Reported as ``committed=True`` with no members and no refusals.

    Resolution can empty the bundle and then find the baseline alone satisfies
    every aggregate — technically resolved. A caller reading ``committed`` saw
    unqualified success for a transaction that emitted nothing and dropped
    everyone.
    """
    # A baseline already below the floor, with every member contributing.
    starved = {**BASELINE, "bandwidth_hz": 30e6}
    result = transaction().evaluate(
        [energy_env(25e6), slice_env(0.21)], context={"baseline_action": starved}
    )
    assert not result.committed
    with pytest.raises(TransactionRefused):
        _ = result.members
    assert result.refusals


# ── A4: resolution aimed at a rival ──────────────────────────────────────
def test_resolution_drops_the_culpable_member_not_merely_the_cheapest() -> None:
    """Key-intersection alone made every action a candidate.

    ``bandwidth_hz`` is in ``REQUIRED_ACTION_KEYS``, so every well-formed
    action "touches" what the capacity floor reads, and the victim was decided
    purely by operator priority. An agent could then drive an aggregate into
    violation and watch a rival's legitimate request be dropped instead of its
    own.

    Here the *high-priority* energy agent is the one causing the violation.
    Under the old rule the low-priority slice agent — which is innocent — would
    be dropped first. Culpability must win.
    """
    # energy is more important (lower number) but is the cause: its bandwidth
    # cut is what breaches the floor. slice asks for a share that is fine at
    # the baseline bandwidth.
    pol = policy(energy=5, slice=30)
    result = transaction(pol).evaluate(
        [energy_env(30e6), slice_env(0.30)], context=ctx()
    )
    assert result.committed, result.refusals
    dropped = {d.agent_id for d in result.resolution.dropped}
    assert dropped == {"energy-agent"}, (
        "the member whose removal repairs the violation must be dropped, "
        f"not the cheapest by priority; got {dropped}"
    )
    assert {m.agent_id for m in result.members} == {"slice-agent"}


def test_priority_still_decides_between_equally_culpable_members() -> None:
    """Culpability narrows the field; priority orders what remains.

    Both agents contribute to the breach here, so the operator's ordering is
    what settles it — which is the property the original design was reaching
    for and the one that must survive the culpability fix.
    """
    pol = policy(energy=30, slice=5)
    result = transaction(pol).evaluate(
        [energy_env(40e6), slice_env(0.25)], context=ctx()
    )
    assert result.committed, result.refusals
    dropped = {d.agent_id for d in result.resolution.dropped}
    assert dropped == {"energy-agent"}


# ── items disclosed as open, now closed ──────────────────────────────────
def test_replay_defence_survives_a_restart() -> None:
    """The gate's high-water marks were process-local.

    Disclosed as open: a restart reset every mark, so the first record from
    each source set the floor with no lower bound and everything inside the
    freshness window replayed cleanly. The marks are now snapshottable and can
    be seeded back on start.
    """
    from horizon_agentic.telemetry_trust import (
        FieldBound,
        TelemetryRecord,
        TelemetryTrustGate,
        TrustPolicy,
        sign_record,
    )

    policy = TrustPolicy(
        secrets={"s": b"k"}, max_age_s=60.0,
        bounds={"x": FieldBound(0.0, 10.0, "")},
    )
    first = TelemetryTrustGate(policy, clock=lambda: 1000.0)
    rec = sign_record(TelemetryRecord("s", 1000.0, 7, {"x": 1.0}), b"k")
    assert first.admit([rec]).trusted
    snapshot = first.high_water_marks()
    assert snapshot == {"s": 7}

    # A naive restart would accept the replay.
    naive = TelemetryTrustGate(policy, clock=lambda: 1000.0)
    assert naive.admit([rec]).trusted, "precondition: an unseeded gate accepts it"

    # Seeded from the snapshot, it does not.
    restarted = TelemetryTrustGate(
        policy, clock=lambda: 1000.0, initial_high_water=snapshot
    )
    verdict = restarted.admit([rec])
    assert not verdict.trusted
    assert any(c.check_id == "replay" for c in verdict.refusals)


def test_the_envelope_has_a_wire_format() -> None:
    """Disclosed as open: the only way to construct one was importing the class.

    Round-trip must be exact, because the signature covers the serialised form
    — a field that does not survive the round trip is a field two peers can
    disagree about while both hold a valid signature.
    """
    env = slice_env()
    restored = AgentActionEnvelope.from_dict(env.to_dict())
    assert restored == env


def test_serialisation_is_order_stable() -> None:
    """`frozenset` iteration order is not stable across processes."""
    import json

    a = AgentActionEnvelope(
        "a", "1", "ran", frozenset({"slice", "spectrum"}), (ROOT, "a"),
        dict(BASELINE), frozenset({"prb_allocation", "bandwidth_hz"}),
    )
    b = AgentActionEnvelope(
        "a", "1", "ran", frozenset({"spectrum", "slice"}), (ROOT, "a"),
        dict(BASELINE), frozenset({"bandwidth_hz", "prb_allocation"}),
    )
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(
        b.to_dict(), sort_keys=True
    )


def test_an_unknown_wire_field_is_refused_not_ignored() -> None:
    """Dropping a field the sender believed it set is how peers diverge —
    with a valid signature over the sender's version."""
    payload = {**slice_env().to_dict(), "escalate": True}
    with pytest.raises(ValueError, match="does not define"):
        AgentActionEnvelope.from_dict(payload)


def test_a_wire_envelope_missing_required_fields_is_refused() -> None:
    payload = slice_env().to_dict()
    del payload["requested_action"]
    with pytest.raises(ValueError, match="missing required fields"):
        AgentActionEnvelope.from_dict(payload)


def test_a_signature_survives_the_round_trip() -> None:
    """The point of a wire format: an external agent signs, we verify."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from horizon_agentic.identity import (
        AgentRegistry,
        EnvelopeAuthenticator,
        sign_envelope,
    )

    key = Ed25519PrivateKey.generate()
    signed = sign_envelope(
        AgentActionEnvelope(
            "slice-agent", "1.0.0", "ran", frozenset({"slice"}),
            (ROOT, "slice-agent"), dict(BASELINE), frozenset({"prb_allocation"}),
            resource_id="cell-1", nonce="n1", issued_at=1000.0,
        ),
        key,
    )
    wire = json.dumps(signed.to_dict(), sort_keys=True)
    restored = AgentActionEnvelope.from_dict(json.loads(wire))
    auth = EnvelopeAuthenticator(
        AgentRegistry({"slice-agent": key.public_key()}), clock=lambda: 1000.0
    )
    verdict = auth.authenticate(restored)
    assert verdict.authenticated, verdict.problems
