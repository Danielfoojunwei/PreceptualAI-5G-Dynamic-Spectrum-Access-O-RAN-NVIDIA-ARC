#!/usr/bin/env python3
"""Watch two agents collide, and watch the layer decide.

An adversarial reviewer noted that every interesting behaviour in this package
was asserted inside pytest and rendered as a dot, and that for something whose
pitch is "watch what happens when two agents collide" the absence of anything
runnable read as a choice. It was not a choice; this is the thing that was
missing.

Nothing is faked. The keys are real Ed25519 keys written to a real directory,
the envelopes cross a real TCP socket as JSON, the Shield is the unmodified one
from ``horizon_ric``, and the evidence chain is written to disk and verified by
reloading it. What the demo adds over the tests is the *sequence* — you can see
that neither agent misbehaves and the harm still happens, and then see what the
cycle does about it.

    python agentic/demo/two_agent_conflict.py --out /tmp/demo

Exits non-zero if any stage does not behave as narrated, so it is also a
smoke test of the whole subtree composed together rather than unit by unit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "agentic" / "src"))

from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: E402
    Ed25519PrivateKey,
)
from horizon_agentic.aggregate import (  # noqa: E402
    AbsoluteSliceCapacityFloor,
    PrbConservation,
)
from horizon_agentic.bundle import SafetyTransaction  # noqa: E402
from horizon_agentic.cycle import TransactionCycle  # noqa: E402
from horizon_agentic.emit import emittable, verify_binding  # noqa: E402
from horizon_agentic.envelope import (  # noqa: E402
    AgentActionEnvelope,
    AuthorityGrant,
    AuthorityPolicy,
)
from horizon_agentic.evidence import TransactionEvidenceChain  # noqa: E402
from horizon_agentic.identity import (  # noqa: E402
    EnvelopeAuthenticator,
    sign_envelope,
)
from horizon_agentic.ingress import EnvelopeIngress, serve_line_delimited  # noqa: E402
from horizon_agentic.keyring import load_registry, write_public_key  # noqa: E402
from horizon_agentic.store import JsonlChainStore  # noqa: E402

from horizon_ric.shield import default_terrestrial_shield  # noqa: E402

ROOT = "operator-root"
CELL = "cell-3450-A"
NOW = 1_700_000_000.0
FLOOR_HZ = 20e6

BASELINE: dict[str, Any] = {
    "block": "ran_control",
    "frequency_hz": 3.45e9,
    "bandwidth_hz": 100e6,
    "tx_power_dBm": 20.0,
    "antenna_gain_dBi": 6.0,
    "prb_allocation": {"safety_critical": 0.50, "embb": 0.50},
}

failures: list[str] = []


def say(text: str = "") -> None:
    print(text)


def check(ok: bool, description: str) -> bool:
    print(f"    {'ok  ' if ok else 'FAIL'}  {description}")
    if not ok:
        failures.append(description)
    return ok


def policy() -> AuthorityPolicy:
    return AuthorityPolicy(
        root_principal=ROOT,
        grants={
            ROOT: AuthorityGrant(
                ROOT, frozenset({"ran"}), frozenset({"spectrum", "slice"}), 0
            ),
            # Lower number = more important to the operator. The energy agent
            # is the *less* important of the two, which is what decides the
            # outcome later — and it is set here, by the operator, never by an
            # agent.
            "energy-agent": AuthorityGrant(
                "energy-agent", frozenset({"ran"}), frozenset({"spectrum"}), 20
            ),
            "slice-agent": AuthorityGrant(
                "slice-agent", frozenset({"ran"}), frozenset({"slice"}), 15
            ),
        },
    )


def energy_envelope(bandwidth_hz: float = 50e6) -> AgentActionEnvelope:
    """An energy-saving agent narrowing the carrier. Legal on its own."""
    return AgentActionEnvelope(
        agent_id="energy-agent",
        agent_version="1.4.0",
        target_domain="ran",
        granted_scopes=frozenset({"spectrum"}),
        delegation_chain=(ROOT, "energy-agent"),
        requested_action={**BASELINE, "bandwidth_hz": bandwidth_hz},
        mutates=frozenset({"bandwidth_hz"}),
        resource_id=CELL,
        nonce="energy-0001",
        issued_at=NOW,
    )


def slice_envelope(share: float = 0.25) -> AgentActionEnvelope:
    """A capacity agent re-slicing PRBs. Also legal on its own."""
    return AgentActionEnvelope(
        agent_id="slice-agent",
        agent_version="2.1.0",
        target_domain="ran",
        granted_scopes=frozenset({"slice"}),
        delegation_chain=(ROOT, "slice-agent"),
        requested_action={
            **BASELINE,
            "prb_allocation": {"safety_critical": share, "embb": 1.0 - share},
        },
        mutates=frozenset({"prb_allocation"}),
        resource_id=CELL,
        nonce="slice-0001",
        issued_at=NOW,
    )


def shield():
    return default_terrestrial_shield(
        band_lo_hz=3.40e9, band_hi_hz=3.50e9, max_eirp_dBm=33.0
    )


def transaction(**kw) -> SafetyTransaction:
    return SafetyTransaction(
        shield=shield(),
        policy=policy(),
        aggregates=[
            AbsoluteSliceCapacityFloor(min_capacity_hz=FLOOR_HZ),
            PrbConservation(),
        ],
        **kw,
    )


def capacity(action: dict[str, Any]) -> float:
    return action["bandwidth_hz"] * action["prb_allocation"]["safety_critical"]


# ── act one: each agent alone ────────────────────────────────────────────
def act_one() -> None:
    say("1. Each agent, acting alone. Neither misbehaves.")
    say()
    ctx = {"baseline_action": dict(BASELINE)}
    results = {}
    for env in (energy_envelope(), slice_envelope()):
        result = transaction().evaluate([env], context=ctx)
        results[env.agent_id] = result
        disposition = shield().dispose(dict(env.requested_action), {})
        say(
            f"    {env.agent_id:<14} committed={result.committed}  "
            f"shield_projected={disposition.certificate.projected}  "
            f"violated={disposition.certificate.violated_ids}"
        )
    say()
    check(
        all(r.committed for r in results.values()),
        "both actions are admitted when evaluated separately",
    )

    net = (
        results["energy-agent"].members[0].action["bandwidth_hz"]
        * results["slice-agent"].members[0].action["prb_allocation"]["safety_critical"]
    )
    say()
    say(f"    baseline capacity for 'safety_critical' : {capacity(BASELINE)/1e6:.1f} MHz")
    say(f"    after both are applied                  : {net/1e6:.1f} MHz")
    say(f"    the commitment                          : {FLOOR_HZ/1e6:.1f} MHz")
    check(
        net < FLOOR_HZ,
        "applied separately, the protected slice ends up below its floor",
    )
    say()
    say("    Nothing was violated. The fractional floor holds throughout — the")
    say("    slice still has 0.25 against a 0.20 floor. The denominator moved,")
    say("    and no per-action invariant can see that.")
    say()


# ── act two: through the cycle ───────────────────────────────────────────
async def act_two(out: Path) -> None:
    say("2. The same two agents, submitted independently, through a cycle.")
    say()

    keydir = out / "registry"
    keys = {
        "energy-agent": Ed25519PrivateKey.generate(),
        "slice-agent": Ed25519PrivateKey.generate(),
    }
    for name, key in keys.items():
        write_public_key(keydir, name, key)
    registry = load_registry(keydir).as_registry()
    say(f"    operator key directory  : {keydir}")
    say(f"    registered agents       : {sorted(keys)}")

    chain = TransactionEvidenceChain()
    operator_key = Ed25519PrivateKey.generate()
    cycle = TransactionCycle(
        transaction(
            authenticator=EnvelopeAuthenticator(registry, clock=lambda: NOW),
            signing_key=operator_key,
            chain=chain,
        ),
        clock=lambda: NOW,
        baseline_source=lambda: dict(BASELINE),
        window_s=0.0,
    )
    ingress = EnvelopeIngress(cycle, on_reject=lambda why: say(f"    refused: {why}"))
    server = await serve_line_delimited(ingress, port=0)
    host, port = server.sockets[0].getsockname()[:2]
    say(f"    ingress listening on    : {host}:{port}")
    say()

    for env, key in (
        (energy_envelope(), keys["energy-agent"]),
        (slice_envelope(), keys["slice-agent"]),
    ):
        signed = sign_envelope(env, key)
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(json.dumps(signed.to_dict()).encode() + b"\n")
        await writer.drain()
        reply = json.loads(await reader.readline())
        writer.close()
        say(
            f"    {env.agent_id:<14} submitted -> accepted={reply['accepted']} "
            f"epoch={reply['epoch']} ({reply['detail']})"
        )
    server.close()
    await server.wait_closed()

    check(cycle.pending == 2, "both envelopes are queued in one epoch, undecided")
    say()
    say("    Neither agent has been told anything but 'queued'. Nothing has been")
    say("    evaluated. That is the point: a decision at submission time would")
    say("    be a decision made in isolation.")
    say()

    result = cycle.close()
    say("3. The epoch closes, and the bundle is decided as a unit.")
    say()
    for check_ in result.aggregate_checks:
        say(
            f"    {check_.invariant_id:<32} satisfied={check_.satisfied}  "
            f"{check_.detail}"
        )
    say()
    for dropped in result.resolution.dropped if result.resolution else ():
        say(
            f"    dropped  {dropped.agent_id} (operator priority "
            f"{dropped.priority}) — {dropped.violated_id}"
        )
    admitted = {m.agent_id for m in result.members} if result.committed else set()
    say(f"    admitted {sorted(admitted)}")
    say()
    check(result.committed, "the transaction commits")
    check(
        {d.agent_id for d in result.resolution.dropped} == {"energy-agent"},
        "the culpable request is the one not applied",
    )
    emitted = {m.agent_id: m.action for m in result.members}
    check(
        capacity(emitted["slice-agent"]) >= FLOOR_HZ,
        f"the protected slice keeps "
        f"{capacity(emitted['slice-agent'])/1e6:.1f} MHz, at or above its floor",
    )
    say()

    say("4. Evidence.")
    say()
    actions = emittable(result)
    binding = actions[0].binding
    say(f"    transaction     : {binding.transaction_id} (epoch {binding.epoch})")
    say(f"    certificate     : {binding.certificate_digest[:16]}...")
    say(f"    signed by       : {binding.signing_key_fingerprint[:16]}...")
    say(f"    baseline digest : {result.certificate.baseline_digest[:16]}...")
    say(f"    limits in force : {[a['id'] for a in result.certificate.aggregates]}")
    check(
        verify_binding(binding, result.certificate, public_key=operator_key.public_key()),
        "the emission binding verifies against the committed transaction",
    )

    store = JsonlChainStore(out / "evidence.jsonl")
    store.append_chain(chain)
    anchor = chain.head
    loaded = store.load(public_key=operator_key.public_key(), expected_head=anchor)
    say(f"    evidence file   : {store.path}")
    check(loaded.intact, "the chain reloads from disk and verifies")

    lines = store.path.read_text().splitlines()
    record = json.loads(lines[0])
    record["certificate"]["committed"] = False
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    store.path.write_text("\n".join(lines) + "\n")
    tampered = store.load()
    check(
        tampered.break_at is not None and tampered.break_at.index == 0,
        "altering record 0 on disk is detected, and localised to record 0",
    )
    say()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/horizon-agentic-demo"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    say("=" * 72)
    say("Two agents, one cell, one protected slice")
    say("=" * 72)
    say()
    act_one()
    asyncio.run(act_two(args.out))

    say("=" * 72)
    if failures:
        say(f"{len(failures)} stage(s) did not behave as narrated:")
        for failure in failures:
            say(f"  - {failure}")
        return 1
    say("Every stage behaved as narrated.")
    say("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
