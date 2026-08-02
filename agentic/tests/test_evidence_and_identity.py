"""Evidence for refusals, and identity that is verified rather than asserted.

Two gaps this file pins, both of which made earlier claims about the system
false rather than merely incomplete.

**A refused transaction produced no evidence at all.** "Signed evidence of
every correction, refusal and execution" was true of the Shield's per-action
certificates and false of the transaction layer. A refusal is precisely the
event an operator is later asked to justify, and it leaves no trace in the
network — so if nothing writes it down, there is nothing to produce.

**`agent_id` was an unauthenticated string.** Authority checking answers "may
this principal do this?", which presupposes an answer to "which principal is
this?". There was none: any process able to submit a bundle could set
`agent_id` to `slice-agent` and inherit its grant, and delegation chains were
checked for narrowing while being trivially fabricable.
"""

from __future__ import annotations

import dataclasses

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from horizon_agentic.aggregate import AbsoluteSliceCapacityFloor
from horizon_agentic.bundle import SafetyTransaction
from horizon_agentic.envelope import (
    AgentActionEnvelope,
    AuthorityGrant,
    AuthorityPolicy,
)
from horizon_agentic.evidence import (
    TransactionCertificate,
    TransactionEvidenceChain,
    canonical_transaction_bytes,
    signed_transaction,
    verify_chain,
    verify_transaction,
)
from horizon_agentic.identity import (
    AgentRegistry,
    EnvelopeAuthenticator,
    canonical_envelope_bytes,
    registry_from_keys,
    sign_envelope,
)

from horizon_ric.shield import default_terrestrial_shield

ROOT = "operator-root"
FLOOR_HZ = 20e6
NOW = 1000.0

BASELINE = {
    "block": "ran_control",
    "frequency_hz": 3.45e9,
    "bandwidth_hz": 100e6,
    "tx_power_dBm": 20.0,
    "antenna_gain_dBi": 6.0,
    "prb_allocation": {"safety_critical": 0.50, "embb": 0.50},
}


def policy() -> AuthorityPolicy:
    return AuthorityPolicy(
        root_principal=ROOT,
        grants={
            ROOT: AuthorityGrant(
                ROOT, frozenset({"ran"}), frozenset({"spectrum", "slice"}), 0
            ),
            "energy-agent": AuthorityGrant(
                "energy-agent", frozenset({"ran"}), frozenset({"spectrum"}), 20
            ),
            "slice-agent": AuthorityGrant(
                "slice-agent", frozenset({"ran"}), frozenset({"slice"}), 15
            ),
        },
    )


def envelope(agent_id="slice-agent", *, nonce="n1", issued_at=NOW, **overrides):
    base = dict(
        agent_id=agent_id,
        agent_version="1.0.0",
        target_domain="ran",
        granted_scopes=frozenset({"slice"}),
        delegation_chain=(ROOT, agent_id),
        requested_action={
            **BASELINE,
            "prb_allocation": {"safety_critical": 0.25, "embb": 0.75},
        },
        mutates=frozenset({"prb_allocation"}),
        nonce=nonce,
        issued_at=issued_at,
    )
    base.update(overrides)
    return AgentActionEnvelope(**base)


def energy_envelope(*, nonce="e1"):
    return AgentActionEnvelope(
        agent_id="energy-agent",
        agent_version="1.0.0",
        target_domain="ran",
        granted_scopes=frozenset({"spectrum"}),
        delegation_chain=(ROOT, "energy-agent"),
        requested_action={**BASELINE, "bandwidth_hz": 50e6},
        mutates=frozenset({"bandwidth_hz"}),
        nonce=nonce,
        issued_at=NOW,
    )


def transaction(**kwargs) -> SafetyTransaction:
    return SafetyTransaction(
        shield=default_terrestrial_shield(
            band_lo_hz=3.40e9, band_hi_hz=3.50e9, max_eirp_dBm=33.0
        ),
        policy=policy(),
        aggregates=[AbsoluteSliceCapacityFloor(min_capacity_hz=FLOOR_HZ)],
        clock=lambda: "2026-01-01T00:00:00Z",
        **kwargs,
    )


def ctx() -> dict:
    return {"baseline_action": dict(BASELINE)}


# ── evidence for refusals ────────────────────────────────────────────────
def test_a_refused_transaction_still_produces_a_certificate() -> None:
    """The gap this module exists to close."""
    starved = {**BASELINE, "bandwidth_hz": 30e6}
    env = dataclasses.replace(envelope(), requested_action={
        **starved, "prb_allocation": {"safety_critical": 0.25, "embb": 0.75}
    })
    result = transaction().evaluate([env], context={"baseline_action": starved})
    assert not result.committed
    cert = result.certificate
    assert cert is not None
    assert cert.committed is False
    assert cert.refusals, "a refusal certificate with no reasons is not evidence"
    assert any(
        c["id"] == "absolute_slice_capacity_floor" and not c["passed"]
        for c in cert.aggregate_checks
    )


def test_the_certificate_records_checks_that_passed_too() -> None:
    """"Evaluated and held" and "never evaluated" must be distinguishable.

    A record listing only violations cannot tell an auditor whether a limit was
    checked at all, which is the difference between a system that enforced
    something and one that forgot to.
    """
    result = transaction().evaluate([envelope()], context=ctx())
    assert result.committed
    assert result.certificate.aggregate_checks
    assert all(c["passed"] for c in result.certificate.aggregate_checks)


def test_a_dropped_request_is_recorded_with_its_priority_and_cause() -> None:
    """An auditor must be able to reconstruct why agent X was not applied."""
    result = transaction().evaluate(
        [energy_envelope(), envelope()], context=ctx()
    )
    assert result.committed
    dropped = result.certificate.dropped
    assert len(dropped) == 1
    assert dropped[0]["agent_id"] == "energy-agent"
    assert dropped[0]["priority"] == 20
    assert dropped[0]["violated_id"] == "absolute_slice_capacity_floor"
    assert "MHz" in dropped[0]["detail"]


def test_admitted_members_are_referenced_by_digest() -> None:
    result = transaction().evaluate([envelope()], context=ctx())
    admitted = result.certificate.admitted
    assert len(admitted) == 1
    assert len(admitted[0]["action_digest"]) == 64
    assert len(admitted[0]["certificate_digest"]) == 64


def test_identity_refusal_still_names_the_agent() -> None:
    """Identity refusals happen before authority runs.

    Without a fallback the certificate for that path would name no principal at
    all — the one record where knowing who tried is the entire point.
    """
    key = Ed25519PrivateKey.generate()
    auth = EnvelopeAuthenticator(
        registry_from_keys([("slice-agent", key)]), clock=lambda: NOW
    )
    result = transaction(authenticator=auth).evaluate([envelope()], context=ctx())
    assert not result.committed
    principals = result.certificate.authority_verdicts
    assert [p["agent_id"] for p in principals] == ["slice-agent"]
    assert principals[0]["authenticated"] is False
    assert principals[0]["authorized"] is None


# ── signing and the chain ────────────────────────────────────────────────
def test_certificate_signature_verifies_and_tampering_breaks_it() -> None:
    key = Ed25519PrivateKey.generate()
    result = transaction(signing_key=key).evaluate([envelope()], context=ctx())
    cert = result.certificate
    assert verify_transaction(cert, key.public_key())
    tampered = dataclasses.replace(cert, committed=False)
    assert not verify_transaction(tampered, key.public_key())


def test_canonical_bytes_exclude_the_signature() -> None:
    key = Ed25519PrivateKey.generate()
    unsigned = TransactionCertificate("t1", "2026-01-01T00:00:00Z", True)
    stamped = signed_transaction(unsigned, key)
    assert canonical_transaction_bytes(unsigned) == canonical_transaction_bytes(stamped)


def test_infinite_margins_do_not_produce_invalid_json() -> None:
    """`json.dumps` emits a bare `Infinity`, which is not valid JSON.

    Aggregate invariants report ``inf`` for "not applicable", so this is on the
    normal path, not an edge case. Signed bytes a conforming parser cannot read
    are not evidence.
    """
    import json

    from horizon_agentic.evidence import check_dict

    # A not-applicable aggregate reports inf. Reached here by a bundle that
    # sets no PRB allocation and a context with no baseline, which is an
    # ordinary spectrum-only transaction rather than a contrived input.
    spectrum_only = dataclasses.replace(
        energy_envelope(),
        requested_action={
            "block": "ran_control",
            "frequency_hz": 3.45e9,
            "bandwidth_hz": 50e6,
            "tx_power_dBm": 20.0,
            "antenna_gain_dBi": 6.0,
        },
    )
    spectrum_baseline = {
        "block": "ran_control",
        "frequency_hz": 3.45e9,
        "bandwidth_hz": 100e6,
        "tx_power_dBm": 20.0,
        "antenna_gain_dBi": 6.0,
    }
    result = transaction().evaluate(
        [spectrum_only], context={"baseline_action": spectrum_baseline}
    )
    assert result.committed, result.refusals
    margins = [c["margin"] for c in result.certificate.aggregate_checks]
    assert margins == [None], f"inf margin was not flattened: {margins}"

    raw = canonical_transaction_bytes(result.certificate)
    assert "Infinity" not in raw.decode()
    json.loads(raw)  # would raise on a bare `Infinity`

    # And directly, so the guard is pinned independently of which aggregate
    # happens to be configured.
    class _Inf:
        invariant_id = "x"
        satisfied = True
        margin = float("inf")
        unit = "Hz"
        detail = "n/a"

    assert check_dict(_Inf())["margin"] is None


def test_chain_links_entries_and_localises_tampering() -> None:
    key = Ed25519PrivateKey.generate()
    chain = TransactionEvidenceChain()
    txn = transaction(signing_key=key, chain=chain)
    for i in range(5):
        txn.evaluate([envelope(nonce=f"n{i}")], context=ctx(), transaction_id=f"t{i}")
    assert len(chain.entries) == 5
    assert verify_chain(chain.entries, public_key=key.public_key()) is None

    # Alter record 2 in place, as an attacker with store access would.
    victim = chain.entries[2]
    chain.entries[2] = dataclasses.replace(
        victim, certificate=dataclasses.replace(victim.certificate, committed=False)
    )
    broken = verify_chain(chain.entries)
    assert broken is not None
    assert broken.index == 2, "the break must be localised, not merely detected"


def test_relinking_the_chain_after_tampering_still_fails() -> None:
    """The predecessor hash is hashed, not merely stored beside the record.

    If it were only stored, an attacker could rewrite an entry and recompute
    every following link — which is the exact failure the construction exists
    to prevent.
    """
    chain = TransactionEvidenceChain()
    txn = transaction(chain=chain)
    for i in range(3):
        txn.evaluate([envelope(nonce=f"n{i}")], context=ctx(), transaction_id=f"t{i}")

    victim = chain.entries[1]
    forged = dataclasses.replace(victim.certificate, committed=False)
    # Attacker recomputes this entry's own hash so it is self-consistent.
    from horizon_agentic.evidence import chain_hash

    chain.entries[1] = dataclasses.replace(
        victim, certificate=forged, entry_hash=chain_hash(victim.prev_hash, forged)
    )
    broken = verify_chain(chain.entries)
    assert broken is not None
    assert broken.index == 2, "entry 2's prev_hash must no longer match"


def test_chain_verification_catches_a_bad_signature() -> None:
    key = Ed25519PrivateKey.generate()
    other = Ed25519PrivateKey.generate()
    chain = TransactionEvidenceChain()
    transaction(signing_key=key, chain=chain).evaluate([envelope()], context=ctx())
    assert verify_chain(chain.entries, public_key=key.public_key()) is None
    broken = verify_chain(chain.entries, public_key=other.public_key())
    assert broken is not None and "signature" in broken.reason


# ── identity ─────────────────────────────────────────────────────────────
def _authenticator(*agents: str, now: float = NOW):
    keys = {name: Ed25519PrivateKey.generate() for name in agents}
    registry = AgentRegistry({n: k.public_key() for n, k in keys.items()})
    return EnvelopeAuthenticator(registry, clock=lambda: now), keys


def test_a_signed_envelope_authenticates() -> None:
    auth, keys = _authenticator("slice-agent")
    signed = sign_envelope(envelope(), keys["slice-agent"])
    verdict = auth.authenticate(signed)
    assert verdict.authenticated, verdict.problems


def test_impersonation_is_refused() -> None:
    """The gap: `agent_id` alone was enough to inherit a grant."""
    auth, keys = _authenticator("slice-agent", "energy-agent")
    # energy-agent signs an envelope claiming to be slice-agent.
    forged = sign_envelope(envelope("slice-agent"), keys["energy-agent"])
    verdict = auth.authenticate(forged)
    assert not verdict.authenticated
    assert any("does not verify" in p for p in verdict.problems)


def test_unsigned_envelope_is_refused() -> None:
    auth, _ = _authenticator("slice-agent")
    verdict = auth.authenticate(envelope())
    assert not verdict.authenticated
    assert any("unsigned" in p for p in verdict.problems)


def test_unregistered_agent_is_refused() -> None:
    auth, keys = _authenticator("slice-agent")
    stranger = Ed25519PrivateKey.generate()
    signed = sign_envelope(envelope("ghost-agent"), stranger)
    verdict = auth.authenticate(signed)
    assert not verdict.authenticated
    assert any("no registered key" in p for p in verdict.problems)


def test_replayed_envelope_is_refused() -> None:
    """A signature alone permits replay; the nonce is what closes it."""
    auth, keys = _authenticator("slice-agent")
    signed = sign_envelope(envelope(), keys["slice-agent"])
    assert auth.authenticate(signed).authenticated
    second = auth.authenticate(signed)
    assert not second.authenticated
    assert any("seen before" in p for p in second.problems)


def test_missing_nonce_is_refused() -> None:
    auth, keys = _authenticator("slice-agent")
    signed = sign_envelope(envelope(nonce=""), keys["slice-agent"])
    verdict = auth.authenticate(signed)
    assert not verdict.authenticated
    assert any("no nonce" in p for p in verdict.problems)


def test_stale_envelope_is_refused() -> None:
    auth, keys = _authenticator("slice-agent", now=NOW)
    signed = sign_envelope(envelope(issued_at=NOW - 600.0), keys["slice-agent"])
    verdict = auth.authenticate(signed)
    assert not verdict.authenticated
    assert any("old" in p for p in verdict.problems)


def test_a_refused_envelope_does_not_burn_its_nonce() -> None:
    """Otherwise an attacker locks out a legitimate agent by replaying it first.

    Same reasoning as the telemetry gate's high-water mark: state may only
    advance on an accepted item.
    """
    auth, keys = _authenticator("slice-agent")
    stale = sign_envelope(
        envelope(nonce="shared", issued_at=NOW - 600.0), keys["slice-agent"]
    )
    assert not auth.authenticate(stale).authenticated
    fresh = sign_envelope(envelope(nonce="shared"), keys["slice-agent"])
    assert auth.authenticate(fresh).authenticated


def test_nonce_memory_is_bounded_by_the_freshness_window() -> None:
    """Unbounded nonce memory is a slow leak in a long-running control plane.

    A nonce older than the acceptance window cannot be replayed successfully —
    freshness refuses it first — so retaining it buys nothing.
    """
    now = [NOW]
    keys = {"slice-agent": Ed25519PrivateKey.generate()}
    auth = EnvelopeAuthenticator(
        AgentRegistry({"slice-agent": keys["slice-agent"].public_key()}),
        clock=lambda: now[0],
        max_age_s=30.0,
    )
    for i in range(20):
        now[0] = NOW + i
        signed = sign_envelope(
            envelope(nonce=f"n{i}", issued_at=now[0]), keys["slice-agent"]
        )
        assert auth.authenticate(signed).authenticated, i
    assert auth.seen_count("slice-agent") == 20

    now[0] = NOW + 1000.0
    signed = sign_envelope(
        envelope(nonce="late", issued_at=now[0]), keys["slice-agent"]
    )
    assert auth.authenticate(signed).authenticated
    assert auth.seen_count("slice-agent") == 1, "stale nonces must be evicted"


def test_signature_covers_mutates_and_the_delegation_chain() -> None:
    """The fields authority actually reads must be inside the signature.

    Omitting either would let an attacker take a legitimately signed envelope
    and rewrite the part that decides what it is allowed to do — which is worse
    than no signature, because it lends credibility to a modified request.
    """
    auth, keys = _authenticator("slice-agent")
    signed = sign_envelope(envelope(), keys["slice-agent"])

    widened = dataclasses.replace(signed, mutates=frozenset({"prb_allocation", "bandwidth_hz"}))
    assert not auth.authenticate(widened).authenticated

    rechained = dataclasses.replace(signed, delegation_chain=(ROOT, "x", "slice-agent"))
    assert not auth.authenticate(rechained).authenticated

    rescoped = dataclasses.replace(signed, granted_scopes=frozenset({"slice", "spectrum"}))
    assert not auth.authenticate(rescoped).authenticated


def test_canonical_envelope_bytes_are_order_stable() -> None:
    """frozenset iteration order is not stable across processes."""
    a = envelope(granted_scopes=frozenset({"slice", "spectrum"}))
    b = envelope(granted_scopes=frozenset({"spectrum", "slice"}))
    assert canonical_envelope_bytes(a) == canonical_envelope_bytes(b)


def test_transaction_refuses_an_impersonated_member_atomically() -> None:
    auth, keys = _authenticator("slice-agent", "energy-agent")
    honest = sign_envelope(envelope(), keys["slice-agent"])
    forged = sign_envelope(energy_envelope(), keys["slice-agent"])
    result = transaction(authenticator=auth).evaluate(
        [honest, forged], context=ctx()
    )
    assert not result.committed
    assert any(r.startswith("identity.energy-agent") for r in result.refusals)
    with pytest.raises(Exception):
        _ = result.members


def test_identity_runs_before_authority() -> None:
    """Ordering, not just outcome.

    Deciding what a principal may do presupposes knowing which principal it is.
    An empty authority tuple on an identity refusal is what proves the order.
    """
    auth, keys = _authenticator("slice-agent")
    result = transaction(authenticator=auth).evaluate([envelope()], context=ctx())
    assert not result.committed
    assert result.authority == ()
    assert result.identity and not result.identity[0].authenticated
