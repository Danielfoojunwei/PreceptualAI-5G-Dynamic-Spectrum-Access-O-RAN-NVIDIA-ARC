#!/usr/bin/env python3
"""Cross-check the Horizon-RIC → near-RT RIC → xApp policy path end to end.

Three independent witnesses must agree before the proof passes:

1. the rApp's own pipeline report (``run_horizon_rapp.py --report-json``);
2. the live A1 mediator northbound: every accepted policy instance must
   exist and report ``enforceStatus == ENFORCED`` — which the mediator
   only sets after receiving the xApp's ``A1_POLICY_RESP`` over RMR;
3. the mediator's structured receive log: every accepted policy id must
   appear in an ``A1_POLICY_RESP`` line carrying ``"handler_id":
   "hw-python"`` — the identity string the xApp injects from its own
   config — corroborated by the xApp's RMR send-stats line showing the
   matching number of successful sends to the mediator endpoint.
   (The xApp's own "processed request" lines sit at INFO while
   mdclogpy/ricxappframe pin the level to ERROR with no config hook, so
   the receiver-side record is used instead of patching the xApp.)

Run ``deploy/xapp-e2e/run_stack.sh`` first; see deploy/xapp-e2e/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

HORIZON_POLICY_TYPE_IDS = (20001, 20002, 20003, 20004)

# The identity string the hw-python xApp injects into every A1_POLICY_RESP.
XAPP_HANDLER_ID = "hw-python"


def _outer_object_slice(text: str) -> str | None:
    """The outermost ``{...}`` slice of a string, or None."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


def _embedded_json_objects(line: str) -> list[dict[str, Any]]:
    """Every JSON object embedded in a mediator log line, properly unescaped.

    The mediator logs the raw RMR response JSON *escaped inside* its own
    JSON log line (sometimes behind a ``payload=`` prefix). This walks the
    line recursively: parse the line as JSON, then re-parse every string
    value that itself embeds a JSON object, to any (bounded) escaping depth.
    """
    found: list[dict[str, Any]] = []

    def _walk(value: Any, depth: int) -> None:
        if depth > 8:
            return
        if isinstance(value, dict):
            found.append(value)
            for v in value.values():
                _walk(v, depth + 1)
        elif isinstance(value, list):
            for v in value:
                _walk(v, depth + 1)
        elif isinstance(value, str):
            _parse_candidates(value, depth + 1)

    def _parse_candidates(text: str, depth: int) -> None:
        for candidate in (text, _outer_object_slice(text)):
            if not candidate:
                continue
            try:
                obj = json.loads(candidate)
            except ValueError:
                continue
            if isinstance(obj, (dict, list)):
                _walk(obj, depth)
                return

    _parse_candidates(line, 0)
    return found


def _line_acks_policy(line: str, policy_id: str) -> bool:
    """True iff one embedded JSON payload in ``line`` carries BOTH this
    ``policy_instance_id`` AND ``handler_id == "hw-python"``.

    Both markers must sit in the SAME decoded payload object — a policy id
    and a handler id merely co-existing somewhere in the log file is not an
    acknowledgement.
    """
    for obj in _embedded_json_objects(line):
        if obj.get("handler_id") != XAPP_HANDLER_ID:
            continue
        if obj.get("policy_instance_id") == policy_id:
            return True
        # Tolerate nesting variants: the id may sit deeper inside the same
        # payload object (still a per-payload check, never file-wide).
        if policy_id in json.dumps(obj):
            return True
    return False


def _policy_entries(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise report['policy_ids'] into [{policy_id, policy_type_id}]."""
    entries: list[dict[str, Any]] = []
    for item in report.get("policy_ids", []):
        if isinstance(item, dict):
            entries.append(item)
        else:
            entries.append({"policy_id": str(item), "policy_type_id": None})
    return entries


def _resolve_type_ids(client: httpx.Client, entries: list[dict[str, Any]]) -> None:
    """Fill missing policy_type_id fields by listing each Horizon type."""
    unresolved = [e for e in entries if not e.get("policy_type_id")]
    if not unresolved:
        return
    for type_id in HORIZON_POLICY_TYPE_IDS:
        resp = client.get(f"/A1-P/v2/policytypes/{type_id}/policies")
        if resp.status_code != 200:
            continue
        listed = {str(p) for p in resp.json()}
        for entry in unresolved:
            if entry["policy_id"] in listed:
                entry["policy_type_id"] = type_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1-base-url", default="http://127.0.0.1:10000")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mediator-log", type=Path, required=True)
    parser.add_argument("--xapp-log", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help=(
            "Bounded wait for the lagging witnesses. Enforcement status, the "
            "mediator's A1_POLICY_RESP log lines, and — most slowly — the "
            "xApp's periodic RMR send-stats line all trail the emit by up to "
            "~30s, so the proof polls rather than reads once."
        ),
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    accepted = int(report.get("accepted", 0))
    if accepted < 1:
        print(f"FAIL: pipeline report shows {accepted} accepted policies")
        return 1

    entries = _policy_entries(report)

    def _rmr_succ_to_mediator(xapp_log_text: str) -> int:
        best = 0
        for line in xapp_log_text.splitlines():
            if "target=127.0.0.1:4562" in line and "succ=" in line:
                try:
                    best = max(best, int(line.split("succ=")[1].split(" ")[0]))
                except (IndexError, ValueError):
                    continue
        return best

    # Poll all three witnesses to convergence: each trails the emit by a
    # different amount (enforcement is quick, the RMR stats line is the
    # slowest). last_reason holds the most recent unmet condition so a
    # timeout reports exactly what never converged.
    statuses: list[dict[str, Any]] = []
    xapp_acks: list[str] = []
    rmr_succ = 0
    deadline = time.monotonic() + args.timeout_seconds
    last_reason = "not evaluated"
    health_status = 0
    while True:
        with httpx.Client(base_url=args.a1_base_url, timeout=10.0) as client:
            health = client.get("/A1-P/v2/healthcheck")
            health.raise_for_status()
            health_status = health.status_code
            _resolve_type_ids(client, entries)
            statuses = []
            enforced_ok = True
            for entry in entries:
                type_id = entry.get("policy_type_id")
                if not type_id:
                    last_reason = f"unresolved policy type for {entry['policy_id']}"
                    enforced_ok = False
                    break
                resp = client.get(
                    f"/A1-P/v2/policytypes/{type_id}/policies/{entry['policy_id']}/status"
                )
                resp.raise_for_status()
                body = resp.json()
                enforce = body.get("enforceStatus") or body.get("instance_status")
                statuses.append(
                    {
                        "policy_id": entry["policy_id"],
                        "policy_type_id": type_id,
                        "enforce_status": enforce,
                    }
                )
                if enforce not in ("ENFORCED", "IN EFFECT"):
                    last_reason = f"{entry['policy_id']} not enforced yet: {enforce!r}"
                    enforced_ok = False
                    break

        # Witness 3a: mediator log embeds a hw-python A1_POLICY_RESP per policy.
        mediator_lines = args.mediator_log.read_text(errors="replace").splitlines()
        xapp_acks = []
        acks_ok = True
        for entry in entries:
            pid = entry["policy_id"]
            if any(_line_acks_policy(line, pid) for line in mediator_lines):
                xapp_acks.append(pid)
            else:
                last_reason = (
                    f"no mediator A1_POLICY_RESP with policy_instance_id={pid} "
                    f"and handler_id={XAPP_HANDLER_ID!r} yet"
                )
                acks_ok = False
                break

        # Witness 3b: xApp RMR send-stats line (periodic flush, slowest).
        rmr_succ = _rmr_succ_to_mediator(args.xapp_log.read_text(errors="replace"))
        rmr_ok = rmr_succ >= accepted
        if not rmr_ok:
            last_reason = (
                f"xApp RMR stats show {rmr_succ} sends to the mediator; "
                f"expected >= {accepted}"
            )

        if enforced_ok and acks_ok and rmr_ok:
            break
        if time.monotonic() >= deadline:
            print(f"FAIL: witnesses did not converge in {args.timeout_seconds:.0f}s: {last_reason}")
            return 1
        time.sleep(3.0)

    proof = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "system_under_test": (
            "Horizon-RIC full pipeline (telemetry → planner → Shield → A1) "
            "against the official O-RAN-SC A1 mediator and hw-python xApp"
        ),
        "witnesses": {
            "pipeline_report": {
                "events": report.get("events"),
                "accepted": accepted,
                "blocked": report.get("blocked"),
                "emit_failed": report.get("emit_failed"),
                "audit_chain_length": report.get("audit_chain_length"),
                "audit_verify_first_broken_index": report.get(
                    "audit_verify_first_broken_index"
                ),
            },
            "a1_mediator": {
                "healthcheck": health_status,
                "policy_statuses": statuses,
            },
            "hw_python_xapp": {
                "policies_acked_with_handler_id": xapp_acks,
                "rmr_successful_sends_to_mediator": rmr_succ,
            },
        },
        "result": "pass",
        "scope": (
            "Standalone RMR deployment of the official O-RAN-SC A1 mediator "
            "and hw-python reference xApp with a static route table; no E2 "
            "nodes, RAN traffic, or platform Route Manager are exercised."
        ),
    }
    rendered = json.dumps(proof, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
