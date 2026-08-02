#!/usr/bin/env python3
"""Gate G6 — the multi-agent claim, in a form that can fail.

The claim under gate:

    Two agents, each acting strictly inside its granted authority and each
    proposing an action that every existing check passes, can jointly drive a
    protected slice below its capacity commitment; Horizon-Agentic detects that
    from the bundle, resolves it by an operator-programmed ordering that does
    not depend on arrival order, and where it cannot resolve it, refuses the
    whole transaction rather than part of it.

Six checks. Every one of them can fail, and the falsification log in
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

from horizon_agentic.aggregate import AbsoluteSliceCapacityFloor  # noqa: E402
from horizon_agentic.bundle import (  # noqa: E402
    SafetyTransaction,
    TransactionRefused,
)
from horizon_agentic.envelope import (  # noqa: E402
    AgentActionEnvelope,
    AuthorityGrant,
    AuthorityPolicy,
)
from horizon_agentic.telemetry_trust import (  # noqa: E402
    FieldBound,
    TelemetryRecord,
    TelemetryTrustGate,
    TrustPolicy,
    sign_record,
)

from horizon_ric.shield import default_terrestrial_shield  # noqa: E402

ROOT = "operator-root"
BAND_LO, BAND_HI = 3.40e9, 3.50e9
MAX_EIRP_DBM = 33.0
FLOOR_HZ = 20e6

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
    HERE / "src/horizon_agentic/aggregate.py",
    HERE / "src/horizon_agentic/conflict.py",
    HERE / "src/horizon_agentic/envelope.py",
    HERE / "src/horizon_agentic/bundle.py",
    HERE / "src/horizon_agentic/telemetry_trust.py",
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
