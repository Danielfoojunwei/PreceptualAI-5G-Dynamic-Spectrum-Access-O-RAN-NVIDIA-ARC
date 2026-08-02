#!/usr/bin/env python3
"""Gate G6 — the multi-agent claim, in a form that can fail.

The claim under gate:

    Two agents, each acting strictly inside its granted authority and each
    proposing an action that every existing check passes, can jointly drive a
    protected slice below its capacity commitment; Horizon-Agentic detects that
    from the bundle, resolves it by an operator-programmed ordering that does
    not depend on arrival order, and where it cannot resolve it, refuses the
    whole transaction rather than part of it.

Twelve checks. Every one of them can fail, and the falsification log in
``agentic/README.md`` records the edit that makes each fail.

The first check is the one that matters most. A "multi-agent conflict" whose
members were not individually legal proves nothing about multi-agent
interaction — it is a single-agent violation with a second actor standing
nearby. So G6 asserts non-staging *before* it asserts detection, at both
layers, and treats a Shield projection of either action as disqualifying.

Source hashes are pinned for the same reason G1 pins them: without that,
editing an invariant to make the numbers agree would pass the gate rather than
fail it.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE / "src"))

from dataclasses import dataclass  # noqa: E402

from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: E402
    Ed25519PrivateKey,
)
from horizon_agentic.aggregate import AbsoluteSliceCapacityFloor  # noqa: E402
from horizon_agentic.bundle import (  # noqa: E402
    SafetyTransaction,
    TransactionRefused,
)
from horizon_agentic.cycle import TransactionCycle  # noqa: E402
from horizon_agentic.emit import (  # noqa: E402
    TransactionBinding,
    emittable,
    verify_binding,
)
from horizon_agentic.envelope import (  # noqa: E402
    AgentActionEnvelope,
    AuthorityGrant,
    AuthorityPolicy,
)
from horizon_agentic.evidence import (  # noqa: E402
    TransactionEvidenceChain,
    digest_of,
    verify_chain,
)
from horizon_agentic.identity import (  # noqa: E402
    AgentRegistry,
    EnvelopeAuthenticator,
    sign_envelope,
)
from horizon_agentic.telemetry_trust import (  # noqa: E402
    FieldBound,
    TelemetryRecord,
    TelemetryTrustGate,
    TrustPolicy,
    sign_record,
)

from horizon_ric.shield import default_terrestrial_shield  # noqa: E402
from horizon_ric.shield.certificate import InvariantCheck  # noqa: E402

ROOT = "operator-root"
BAND_LO, BAND_HI = 3.40e9, 3.50e9
MAX_EIRP_DBM = 33.0
FLOOR_HZ = 20e6
CELL = "cell-3450-A"
NOW = 1_700_000_000.0

BASELINE: dict[str, Any] = {
    "block": "ran_control",
    "frequency_hz": 3.45e9,
    "bandwidth_hz": 100e6,
    "tx_power_dBm": 20.0,
    "antenna_gain_dBi": 6.0,
    "prb_allocation": {"safety_critical": 0.50, "embb": 0.50},
}

# Files whose content the gate's numbers depend on. Editing any of them to make
# the result come out right changes the digest, and the verifier compares
# digests against the committed result.
PINNED = [
    REPO / "src/horizon_ric/shield/shield.py",
    REPO / "src/horizon_ric/shield/invariants.py",
    REPO / "src/horizon_ric/shield/certificate.py",
    REPO / "src/horizon_ric/assurance/planner.py",
    HERE / "src/horizon_agentic/aggregate.py",
    HERE / "src/horizon_agentic/conflict.py",
    HERE / "src/horizon_agentic/envelope.py",
    HERE / "src/horizon_agentic/bundle.py",
    HERE / "src/horizon_agentic/telemetry_trust.py",
    HERE / "src/horizon_agentic/identity.py",
    HERE / "src/horizon_agentic/evidence.py",
    HERE / "src/horizon_agentic/cycle.py",
    HERE / "src/horizon_agentic/emit.py",
]


def _policy(*, energy: int = 20, slc: int = 15) -> AuthorityPolicy:
    return AuthorityPolicy(
        root_principal=ROOT,
        grants={
            ROOT: AuthorityGrant(
                ROOT, frozenset({"ran"}), frozenset({"spectrum", "slice", "phy", "ntn"}), 0
            ),
            "energy-agent": AuthorityGrant(
                "energy-agent", frozenset({"ran"}), frozenset({"spectrum"}), energy
            ),
            "slice-agent": AuthorityGrant(
                "slice-agent", frozenset({"ran"}), frozenset({"slice"}), slc
            ),
        },
    )


def _energy(bandwidth_hz: float = 50e6) -> AgentActionEnvelope:
    return AgentActionEnvelope(
        agent_id="energy-agent",
        agent_version="1.0.0",
        target_domain="ran",
        granted_scopes=frozenset({"spectrum"}),
        delegation_chain=(ROOT, "energy-agent"),
        requested_action={**BASELINE, "bandwidth_hz": bandwidth_hz},
        mutates=frozenset({"bandwidth_hz"}),
        resource_id=CELL,
        nonce="energy-1",
        issued_at=NOW,
    )


def _slice(share: float = 0.25) -> AgentActionEnvelope:
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
        resource_id=CELL,
        nonce="slice-1",
        issued_at=NOW,
    )


def _txn(policy: AuthorityPolicy, **kw) -> SafetyTransaction:
    return SafetyTransaction(
        shield=default_terrestrial_shield(
            band_lo_hz=BAND_LO, band_hi_hz=BAND_HI, max_eirp_dBm=MAX_EIRP_DBM
        ),
        policy=policy,
        aggregates=[AbsoluteSliceCapacityFloor(min_capacity_hz=FLOOR_HZ)],
        **kw,
    )


def _ctx() -> dict[str, Any]:
    return {"baseline_action": dict(BASELINE)}


# ── checks ───────────────────────────────────────────────────────────────
def check_not_staged() -> dict[str, Any]:
    """Each action alone must pass the Shield unprojected AND the aggregate."""
    shield = default_terrestrial_shield(
        band_lo_hz=BAND_LO, band_hi_hz=BAND_HI, max_eirp_dBm=MAX_EIRP_DBM
    )
    aggregate = AbsoluteSliceCapacityFloor(min_capacity_hz=FLOOR_HZ)
    per_agent = {}
    ok = True
    for env in (_energy(), _slice()):
        disposition = shield.dispose(dict(env.requested_action), {})
        alone = aggregate.evaluate([env.requested_action], _ctx())
        entry = {
            "shield_safe": bool(disposition.certificate.safe),
            "shield_projected": bool(disposition.certificate.projected),
            "shield_violated": list(disposition.certificate.violated_ids),
            "aggregate_satisfied": bool(alone.satisfied),
            "aggregate_margin_hz": alone.margin,
        }
        per_agent[env.agent_id] = entry
        ok &= (
            entry["shield_safe"]
            and not entry["shield_projected"]
            and not entry["shield_violated"]
            and entry["aggregate_satisfied"]
        )
    return {"id": "not_staged", "passed": bool(ok), "per_agent": per_agent}


def check_joint_violation() -> dict[str, Any]:
    aggregate = AbsoluteSliceCapacityFloor(min_capacity_hz=FLOOR_HZ)
    joint = aggregate.evaluate(
        [_energy().requested_action, _slice().requested_action], _ctx()
    )
    return {
        "id": "joint_violation_detected",
        "passed": not joint.satisfied,
        "margin_hz": joint.margin,
        "capacity_mhz": round((joint.margin + FLOOR_HZ) / 1e6, 6),
        "floor_mhz": FLOOR_HZ / 1e6,
    }


def check_order_independence() -> dict[str, Any]:
    outcomes = set()
    for permutation in itertools.permutations([_energy(), _slice()]):
        result = _txn(_policy()).evaluate(list(permutation), context=_ctx())
        if not result.committed or result.resolution is None:
            return {"id": "order_independent", "passed": False, "outcomes": ["refused"]}
        outcomes.add(
            (
                tuple(sorted(m.agent_id for m in result.members)),
                tuple(sorted(d.agent_id for d in result.resolution.dropped)),
            )
        )
    return {
        "id": "order_independent",
        "passed": len(outcomes) == 1,
        "distinct_outcomes": len(outcomes),
        "outcome": sorted(str(o) for o in outcomes),
    }


def check_priority_decides() -> dict[str, Any]:
    normal = _txn(_policy()).evaluate([_energy(), _slice()], context=_ctx())
    inverted = _txn(_policy(energy=5, slc=30)).evaluate(
        [_energy(), _slice()], context=_ctx()
    )
    if not (normal.committed and inverted.committed):
        return {"id": "priority_decides", "passed": False, "detail": "did not commit"}
    a = {d.agent_id for d in normal.resolution.dropped}
    b = {d.agent_id for d in inverted.resolution.dropped}
    return {
        "id": "priority_decides",
        "passed": a == {"energy-agent"} and b == {"slice-agent"},
        "dropped_normal": sorted(a),
        "dropped_inverted": sorted(b),
    }


def check_atomic_refusal() -> dict[str, Any]:
    """A cell already below the floor: no subset repairs it, so nothing emits."""
    starved = {**BASELINE, "bandwidth_hz": 30e6}
    env = AgentActionEnvelope(
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
        resource_id=CELL,
        nonce="starved-1",
        issued_at=NOW,
    )
    result = _txn(_policy()).evaluate([env], context={"baseline_action": starved})
    unreadable = False
    try:
        _ = result.members
    except TransactionRefused:
        unreadable = True
    cited = any("absolute_slice_capacity_floor" in r for r in result.refusals)
    vacuous = any("not computable" in r for r in result.refusals)
    return {
        "id": "atomic_refusal",
        "passed": (not result.committed) and unreadable and cited and not vacuous,
        "committed": result.committed,
        "actions_unreadable": unreadable,
        "cites_the_floor": cited,
        "refusals": list(result.refusals),
    }


def check_authority_is_enforced() -> dict[str, Any]:
    """An agent mutating outside its scope refuses the whole bundle."""
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
        resource_id=CELL,
        nonce="overreach-1",
        issued_at=NOW,
    )
    result = _txn(_policy()).evaluate([_slice(), overreaching], context=_ctx())
    return {
        "id": "authority_enforced",
        "passed": (not result.committed)
        and any("authority.energy-agent" in r for r in result.refusals),
        "refusals": list(result.refusals),
    }


def check_telemetry_gate_refuses() -> dict[str, Any]:
    """Every trust check must refuse something, or it is decoration."""
    secret = b"k"
    bounds = {"interference_dBm": FieldBound(-140.0, -30.0, "dBm")}
    policy = TrustPolicy(
        secrets={"sensor-a": secret, "sensor-b": b"k2"},
        max_age_s=5.0,
        bounds=bounds,
        required_fields=frozenset({"interference_dBm"}),
        corroborate=frozenset({"interference_dBm"}),
        corroboration_tolerance={"interference_dBm": 2.0},
    )
    now = 1000.0

    def gate() -> TelemetryTrustGate:
        return TelemetryTrustGate(policy, clock=lambda: now)

    def rec(source="sensor-a", *, at=999.0, seq=1, key=secret, **fields):
        if not fields:
            fields = {"interference_dBm": -95.0}
        return sign_record(TelemetryRecord(source, at, seq, fields), key)

    cases = {
        "source_authenticated": [rec(source="sensor-rogue")],
        "freshness": [rec(at=100.0), rec(source="sensor-b", key=b"k2", at=100.0)],
        "field_range": [
            rec(interference_dBm=0.0),
            rec(source="sensor-b", key=b"k2", interference_dBm=0.0),
        ],
        "required_fields": [rec(source="sensor-a", **{})],
        "corroboration": [rec()],
    }
    refused_by = {}
    for expected, records in cases.items():
        if expected == "required_fields":
            records = [
                sign_record(TelemetryRecord("sensor-a", 999.0, 1, {}), secret),
            ]
        verdict = gate().admit(records)
        refused_by[expected] = sorted(c.check_id for c in verdict.refusals)

    # Replay needs a gate with history.
    g = gate()
    first = rec()
    second = rec(source="sensor-b", key=b"k2", interference_dBm=-94.0)
    g.admit([first, second])
    refused_by["replay"] = sorted(c.check_id for c in g.admit([first, second]).refusals)

    ok = all(expected in seen for expected, seen in refused_by.items())
    return {"id": "telemetry_gate_refuses", "passed": bool(ok), "refused_by": refused_by}


def check_identity_is_enforced() -> dict[str, Any]:
    """An unsigned or impersonated envelope must refuse the transaction.

    Added after review pointed out that the gate ran the layer with
    authentication *disabled*: the artefact offered as evidence exercised
    neither identity nor signing, which are the parts that close what
    `identity.py` calls "not a subtle hole".
    """
    keys = {
        "energy-agent": Ed25519PrivateKey.generate(),
        "slice-agent": Ed25519PrivateKey.generate(),
    }
    registry = AgentRegistry({n: k.public_key() for n, k in keys.items()})

    def auth() -> EnvelopeAuthenticator:
        return EnvelopeAuthenticator(registry, clock=lambda: NOW)

    signed_ok = _txn(_policy(), authenticator=auth()).evaluate(
        [sign_envelope(_slice(), keys["slice-agent"])], context=_ctx()
    )
    unsigned = _txn(_policy(), authenticator=auth()).evaluate(
        [_slice()], context=_ctx()
    )
    impersonated = _txn(_policy(), authenticator=auth()).evaluate(
        [sign_envelope(_energy(), keys["slice-agent"])], context=_ctx()
    )
    replayed_gate = auth()
    once = sign_envelope(_slice(), keys["slice-agent"])
    first = _txn(_policy(), authenticator=replayed_gate).evaluate(
        [once], context=_ctx()
    )
    second = _txn(_policy(), authenticator=replayed_gate).evaluate(
        [once], context=_ctx()
    )
    return {
        "id": "identity_enforced",
        "passed": (
            signed_ok.committed
            and not unsigned.committed
            and not impersonated.committed
            and first.committed
            and not second.committed
        ),
        "signed_committed": signed_ok.committed,
        "unsigned_committed": unsigned.committed,
        "impersonated_committed": impersonated.committed,
        "replay_committed": second.committed,
    }


def check_evidence_is_signed_and_chained() -> dict[str, Any]:
    """Every transaction, committed or refused, must leave a verifiable record."""
    key = Ed25519PrivateKey.generate()
    chain = TransactionEvidenceChain()
    txn = _txn(_policy(), signing_key=key, chain=chain)

    committed = txn.evaluate([_slice()], context=_ctx(), transaction_id="t-commit")
    txn.evaluate([_energy()], context=_ctx(), transaction_id="t-second")
    starved = {**BASELINE, "bandwidth_hz": 30e6}
    refused = _txn(_policy(), signing_key=key, chain=chain).evaluate(
        [_slice()], context={"baseline_action": starved}, transaction_id="t-refuse"
    )
    intact = verify_chain(chain.entries, public_key=key.public_key()) is None

    # Tamper with the first record; the break must be localised to it.
    import dataclasses

    from horizon_agentic.evidence import chain_hash

    victim = chain.entries[0]
    forged = dataclasses.replace(victim.certificate, committed=False)
    chain.entries[0] = dataclasses.replace(victim, certificate=forged)
    broken = verify_chain(chain.entries)

    # The real attack: an attacker with store access rewrites the record,
    # recomputes its hash, AND rewrites the successor's `prev_hash` to match —
    # a fully re-linked chain. This is caught only because the predecessor hash
    # is *hashed into* each entry, so changing the successor's `prev_hash`
    # changes its own `entry_hash`. A first pass falsified this by removing
    # prev from the hash and the gate stayed green, because it only tested a
    # partial rewrite that the stored-prev comparison catches on its own.
    forged_hash = chain_hash(victim.prev_hash, forged)
    chain.entries[0] = dataclasses.replace(
        victim, certificate=forged, entry_hash=forged_hash
    )
    successor = chain.entries[1]
    chain.entries[1] = dataclasses.replace(successor, prev_hash=forged_hash)
    relinked = verify_chain(chain.entries)

    return {
        "id": "evidence_signed_and_chained",
        "passed": (
            committed.certificate.signature is not None
            and refused.certificate.signature is not None
            and refused.certificate.committed is False
            and bool(refused.certificate.refusals)
            and intact
            and broken is not None
            and broken.index == 0
            and relinked is not None
            and relinked.index == 1
        ),
        "commit_signed": committed.certificate.signature is not None,
        "refusal_signed": refused.certificate.signature is not None,
        "refusal_states_reasons": list(refused.certificate.refusals),
        "chain_intact_before_tamper": intact,
        "tamper_localised_to_index": broken.index if broken else None,
        "relink_detected_at_index": relinked.index if relinked else None,
    }


def check_no_silent_partial_commit() -> dict[str, Any]:
    """Dropping every member must refuse, not report an empty success.

    The envelopes are built against the starved baseline on purpose. An earlier
    version of this check reused the standard 100 MHz envelopes against a
    30 MHz baseline, so the transaction refused at *authority* — the declared
    bandwidth did not match — and the check passed without ever reaching the
    resolution path it exists to test. Falsifying the guard did not fail the
    gate, which is how the mistake surfaced.
    """
    starved = {**BASELINE, "bandwidth_hz": 30e6}

    def env(agent_id, scopes, action, mutates, nonce):
        return AgentActionEnvelope(
            agent_id=agent_id,
            agent_version="1.0.0",
            target_domain="ran",
            granted_scopes=frozenset(scopes),
            delegation_chain=(ROOT, agent_id),
            requested_action=action,
            mutates=frozenset(mutates),
            resource_id=CELL,
            nonce=nonce,
            issued_at=NOW,
        )

    # A purpose-built aggregate, because the branch is otherwise unreachable
    # with the shipped set: every single-member subset inherits the baseline
    # for the keys it does not mutate, so one member alone almost always
    # satisfies the capacity floor and resolution succeeds before emptying.
    # This one is violated by any non-empty set and satisfied by the empty set,
    # which is exactly the shape that produces "resolved, with nobody left".
    @dataclass(frozen=True)
    class _RefuseAnyMember:
        id: str = "gate_probe_refuse_any_member"
        reads: frozenset[str] = frozenset({"bandwidth_hz"})

        def evaluate(self, actions, context):
            return InvariantCheck(
                invariant_id=self.id,
                satisfied=not actions,
                margin=0.0 if not actions else -1.0,
                unit="count",
                detail=f"{len(actions)} member(s) present",
            )

    probe = SafetyTransaction(
        shield=default_terrestrial_shield(
            band_lo_hz=BAND_LO, band_hi_hz=BAND_HI, max_eirp_dBm=MAX_EIRP_DBM
        ),
        policy=_policy(),
        aggregates=[_RefuseAnyMember()],
    )
    result = probe.evaluate(
        [
            env("energy-agent", {"spectrum"},
                {**starved, "bandwidth_hz": 25e6}, {"bandwidth_hz"}, "e"),
            env("slice-agent", {"slice"},
                {**starved, "prb_allocation": {"safety_critical": 0.21, "embb": 0.79}},
                {"prb_allocation"}, "s"),
        ],
        context={"baseline_action": starved},
    )
    reached_resolution = (
        result.resolution is not None
        and result.resolution.resolved
        and not result.resolution.admitted
    )
    return {
        "id": "no_silent_partial_commit",
        "passed": (
            reached_resolution and not result.committed and bool(result.refusals)
        ),
        "reached_resolution": reached_resolution,
        "committed": result.committed,
        "refusals": list(result.refusals),
    }


def check_cycle_serialises_agents() -> dict[str, Any]:
    """Without a serialisation point, none of the rest of this gate means anything.

    Added after review put this first. Every other check is conditional on both
    agents' actions arriving in one evaluation, and nothing in the package
    caused that. Submitted separately, each agent commits and the protected
    slice is cut to 12.5 MHz anyway — the harm this layer exists to prevent,
    delivered through it.
    """
    # Separately: both commit, and the net effect breaches the floor.
    alone_e = _txn(_policy()).evaluate([_energy()], context=_ctx())
    alone_s = _txn(_policy()).evaluate([_slice()], context=_ctx())
    separately_harmful = (
        alone_e.committed
        and alone_s.committed
        and alone_e.members[0].action["bandwidth_hz"]
        * alone_s.members[0].action["prb_allocation"]["safety_critical"]
        < FLOOR_HZ
    )

    # Through a cycle: decided together, culprit dropped, floor held.
    cyc = TransactionCycle(
        _txn(_policy()),
        clock=lambda: NOW,
        baseline_source=lambda: dict(BASELINE),
        window_s=0.0,
    )
    accepted = [cyc.submit(_energy()).accepted, cyc.submit(_slice()).accepted]
    joint = cyc.close()
    dropped = {d.agent_id for d in joint.resolution.dropped} if joint.resolution else set()

    # Order independence through the cycle, not just through evaluate().
    outcomes = set()
    for order in ([_energy(), _slice()], [_slice(), _energy()]):
        c2 = TransactionCycle(
            _txn(_policy()),
            clock=lambda: NOW,
            baseline_source=lambda: dict(BASELINE),
            window_s=0.0,
        )
        for env in order:
            c2.submit(env)
        r2 = c2.close()
        outcomes.add(tuple(sorted(m.agent_id for m in r2.members)))

    return {
        "id": "cycle_serialises_agents",
        "passed": (
            separately_harmful
            and all(accepted)
            and joint.committed
            and dropped == {"energy-agent"}
            and len(outcomes) == 1
        ),
        "separate_submission_is_harmful": separately_harmful,
        "joint_committed": joint.committed,
        "joint_dropped": sorted(dropped),
        "distinct_outcomes_by_order": len(outcomes),
    }


def check_emission_binds_to_the_transaction() -> dict[str, Any]:
    """A refused bundle's members carry clean per-action certificates.

    That is the premise of the design, not a defect — which is exactly why a
    downstream gate checking only the per-action certificate would admit an
    action from a bundle the aggregate layer refused, and be right by its own
    rules. The binding is what a receiver checks instead.
    """
    key = Ed25519PrivateKey.generate()
    cyc = TransactionCycle(
        _txn(_policy(), signing_key=key),
        clock=lambda: NOW,
        baseline_source=lambda: dict(BASELINE),
        window_s=0.0,
    )
    cyc.submit(_slice())
    committed = cyc.close()
    actions = emittable(committed)
    binding_ok = verify_binding(
        actions[0].binding, committed.certificate, public_key=key.public_key()
    )

    # And a refused transaction yields nothing emittable at all.
    starved = {**BASELINE, "bandwidth_hz": 30e6}
    refused = _txn(_policy()).evaluate(
        [
            AgentActionEnvelope(
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
                resource_id=CELL,
                nonce="r",
                issued_at=NOW,
            )
        ],
        context={"baseline_action": starved},
    )
    try:
        emittable(refused)
        refused_blocked = False
    except TransactionRefused:
        refused_blocked = True

    # A binding built by hand around a genuinely refused certificate must not
    # verify. `emittable` will not produce one, so this constructs it directly
    # — which is the only way to exercise the `committed` guard. A first pass
    # mutated a committed certificate instead, and the digest comparison caught
    # that on its own, so removing the guard left the gate green.
    hand_built = TransactionBinding(
        transaction_id=refused.certificate.transaction_id,
        epoch=refused.certificate.epoch,
        certificate_digest=digest_of(refused.certificate.to_dict()),
        signature=refused.certificate.signature or "",
        signing_key_fingerprint=refused.certificate.signing_key_fingerprint or "",
    )
    refused_rejected = not verify_binding(hand_built, refused.certificate)

    return {
        "id": "emission_binds_to_the_transaction",
        "passed": binding_ok and refused_rejected and refused_blocked,
        "binding_verifies": binding_ok,
        "refused_certificate_rejected": refused_rejected,
        "refused_transaction_yields_nothing": refused_blocked,
    }


def check_certificate_is_replayable() -> dict[str, Any]:
    """A record that cannot reconstruct the decision is not evidence.

    An auditor holding a refusal must be able to determine what the cell's
    state was and which limits were in force, with what thresholds — naming the
    violated invariant is not enough, because the same invariant with a
    different floor decides differently.
    """
    cyc = TransactionCycle(
        _txn(_policy()),
        clock=lambda: NOW,
        baseline_source=lambda: dict(BASELINE),
        window_s=0.0,
    )
    cyc.submit(_slice())
    cert = cyc.close().certificate
    thresholds = {
        a["id"]: a.get("min_capacity_hz") for a in cert.aggregates
    }
    return {
        "id": "certificate_is_replayable",
        "passed": bool(
            cert.schema_version
            and len(cert.baseline_digest) == 64
            and cert.epoch == 0
            and thresholds.get("absolute_slice_capacity_floor") == FLOOR_HZ
        ),
        "schema_version": cert.schema_version,
        "baseline_digest_present": len(cert.baseline_digest) == 64,
        "epoch": cert.epoch,
        "aggregate_thresholds": thresholds,
    }


def source_digests() -> dict[str, str]:
    out = {}
    for path in PINNED:
        out[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def run() -> dict[str, Any]:
    checks = [
        check_not_staged(),
        check_joint_violation(),
        check_order_independence(),
        check_priority_decides(),
        check_atomic_refusal(),
        check_authority_is_enforced(),
        check_telemetry_gate_refuses(),
        check_identity_is_enforced(),
        check_evidence_is_signed_and_chained(),
        check_no_silent_partial_commit(),
        check_cycle_serialises_agents(),
        check_emission_binds_to_the_transaction(),
        check_certificate_is_replayable(),
    ]
    return {
        "gate": "G6",
        "claim": (
            "two individually-legal agent actions can jointly breach a protected "
            "slice's capacity commitment; the bundle detects it, resolves it "
            "deterministically by operator priority, and refuses atomically when "
            "it cannot"
        ),
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "source_digests": source_digests(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = run()
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    for check in result["checks"]:
        print(
            ("  ok    " if check["passed"] else "  FAIL  ") + check["id"],
            file=sys.stderr,
        )
    print(
        "G6 " + ("PASSED" if result["passed"] else "FAILED"),
        file=sys.stderr,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
