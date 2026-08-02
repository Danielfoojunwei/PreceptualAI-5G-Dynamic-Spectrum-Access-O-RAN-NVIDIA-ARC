#!/usr/bin/env python3
"""Documentation-integrity gate for the "runs unattended" demonstration.

Proposal §VII (docs/proposal/content.py:410-425) tells the Alliance that the
prototype ships with a "live demonstration — policy emission through the
production A1 mediator into a real xApp, including a Shield-corrected
over-power proposal refused rather than emitted — which runs unattended."

This script does NOT run that demonstration. It has no Docker, builds no Go
binary, starts no RMR bus. It STATICALLY verifies that the demonstration the
runbook (audit/UNATTENDED.md) points at is coherent and that every claim the
runbook makes is backed by committed evidence in the repository:

  1. the entry-point scripts and the CI workflow that launches them exist;
  2. run_stack.sh builds the components the README pin-table names, and the
     pins agree between the two;
  3. the committed xApp E2E proof (deploy/xapp-e2e/results/xapp-e2e-proof.json)
     records the ENFORCED policy count the runbook claims, with a coherent
     three-witness agreement and an intact audit chain;
  4. the committed A1-assurance proof (a1-assurance-wire-proof.json) actually
     exercises a Shield-blocked proposal REFUSED rather than emitted, with the
     refusal reaching neither the wire nor the simulator.

It exits non-zero if any of those is not backed by the committed files, so a
runbook that drifts from the evidence fails CI rather than misleading a reader.

Run::

    /home/user/venv/bin/python audit/check_unattended.py

Honest scope note (also stated in UNATTENDED.md): the "refused rather than
emitted" step is a SEPARATE proof from the xApp emission — a different script,
a different CI job, and against the OSC A1 simulator rather than through the
a1mediator+hw-python xApp. And the committed refused proposal is a
negative-bandwidth case (numeric_domain_sanity), not literally an over-power
(max_eirp) case. This checker asserts the refusal that is actually committed;
it deliberately does NOT assert the proposal's "over-power" wording, because
that specific invariant is not what the committed evidence exercises.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Pins asserted to agree between the README table and run_stack.sh.
EXPECTED_PINS = {
    "RMR (ric-plt/lib/rmr)": "8b9a214906a40b338def981d5c16b4ea247176fb",
    "A1 mediator (ric-plt/a1)": "09a757b4fd63198d8690d50b52bfd04552d47f1f",
    "hw-python xApp (ric-app/hw-python)": "a6d00525aa2f62f2457d84731da2b7d89e0b1013",
}

EXPECTED_ENFORCED = 12


class Report:
    """Accumulates PASS/FAIL lines; any failure flips the exit code."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.ok = True

    def check(self, condition: bool, label: str, detail: str = "") -> bool:
        mark = "PASS" if condition else "FAIL"
        suffix = f" — {detail}" if detail else ""
        self.lines.append(f"[{mark}] {label}{suffix}")
        if not condition:
            self.ok = False
        return bool(condition)

    def note(self, text: str) -> None:
        self.lines.append(f"[NOTE] {text}")

    def render(self) -> str:
        return "\n".join(self.lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _exists(rep: Report, rel: str, label: str) -> Path | None:
    p = REPO_ROOT / rel
    ok = rep.check(p.exists(), label, rel)
    return p if ok else None


def check_entry_points(rep: Report) -> None:
    rep.note("=== 1. Entry points and CI wiring exist ===")
    _exists(rep, "deploy/xapp-e2e/run_stack.sh", "run_stack.sh launcher present")
    _exists(rep, "scripts/run_horizon_rapp.py", "rApp daemon shim present")
    _exists(rep, "scripts/xapp_e2e_proof.py", "xApp E2E proof script present")
    _exists(rep, "deploy/xapp-e2e/a1_assurance_proof.py", "A1-assurance proof script present")
    _exists(rep, "deploy/xapp-e2e/source-replay.yaml", "replay source-config present")
    _exists(rep, "audit/run_unattended.sh", "one-command wrapper present")

    wf = _exists(rep, ".github/workflows/xapp-e2e.yml", "xApp E2E CI workflow present")
    if wf:
        text = wf.read_text()
        rep.check(
            "deploy/xapp-e2e/run_stack.sh" in text,
            "xApp E2E workflow launches run_stack.sh",
        )
        rep.check(
            "scripts/xapp_e2e_proof.py" in text,
            "xApp E2E workflow runs xapp_e2e_proof.py",
        )
    aw = _exists(
        rep, ".github/workflows/osc-a1-integration.yml", "A1-assurance CI workflow present"
    )
    if aw:
        rep.check(
            "deploy/xapp-e2e/a1_assurance_proof.py" in aw.read_text(),
            "A1-assurance workflow runs a1_assurance_proof.py",
        )


def check_pins(rep: Report) -> None:
    rep.note("=== 2. run_stack.sh builds what the README names; pins agree ===")
    run_stack = REPO_ROOT / "deploy/xapp-e2e/run_stack.sh"
    readme = REPO_ROOT / "deploy/xapp-e2e/README.md"
    if not run_stack.exists() or not readme.exists():
        rep.check(False, "run_stack.sh and README.md both present for pin cross-check")
        return
    stack_text = run_stack.read_text()
    readme_text = readme.read_text()
    for label, pin in EXPECTED_PINS.items():
        rep.check(
            pin in stack_text and pin in readme_text,
            f"pin agrees (run_stack.sh & README): {label}",
            pin[:12],
        )
    # The launcher must actually build the mediator binary and the xApp venv,
    # standing in for a compose file's "image the build defines".
    rep.check("go build -o" in stack_text and "a1mediator" in stack_text,
              "run_stack.sh builds the a1mediator binary from source")
    rep.check("venv-xapp" in stack_text and "hw-python" in stack_text,
              "run_stack.sh builds the hw-python xApp venv from source")
    rep.check("/A1-P/v2/healthcheck" in stack_text,
              "run_stack.sh waits for the mediator northbound before returning")


def check_xapp_proof(rep: Report) -> None:
    rep.note("=== 3. Committed xApp E2E proof is coherent ===")
    proof_p = REPO_ROOT / "deploy/xapp-e2e/results/xapp-e2e-proof.json"
    report_p = REPO_ROOT / "deploy/xapp-e2e/results/pipeline-report-xapp.json"
    if not proof_p.exists() or not report_p.exists():
        rep.check(False, "xApp proof and pipeline report both present")
        return
    proof = _read_json(proof_p)
    pipeline = _read_json(report_p)

    rep.check(proof.get("result") == "pass", "xApp proof result == pass",
              str(proof.get("result")))

    w = proof.get("witnesses", {})
    pr = w.get("pipeline_report", {})
    med = w.get("a1_mediator", {})
    xapp = w.get("hw_python_xapp", {})

    accepted = pr.get("accepted")
    rep.check(accepted == EXPECTED_ENFORCED,
              f"pipeline accepted == {EXPECTED_ENFORCED}", str(accepted))
    rep.check(pr.get("blocked") == 0, "no policies blocked in the emission path",
              str(pr.get("blocked")))
    rep.check(pr.get("emit_failed") == 0, "no emits failed", str(pr.get("emit_failed")))
    rep.check(pr.get("audit_verify_first_broken_index") == -1,
              "audit chain verifies intact (first_broken_index == -1)")

    # Cross-check the standalone pipeline report agrees with the proof witness.
    rep.check(pipeline.get("enforced") == EXPECTED_ENFORCED,
              f"pipeline-report enforced == {EXPECTED_ENFORCED}",
              str(pipeline.get("enforced")))
    rep.check(pipeline.get("accepted") == accepted,
              "pipeline-report accepted matches proof witness")
    rep.check(len(pipeline.get("policy_ids", [])) == EXPECTED_ENFORCED,
              f"pipeline-report carries {EXPECTED_ENFORCED} policy ids")

    statuses = med.get("policy_statuses", [])
    rep.check(med.get("healthcheck") == 200, "mediator northbound healthcheck == 200")
    rep.check(len(statuses) == EXPECTED_ENFORCED,
              f"mediator reports {EXPECTED_ENFORCED} policy statuses", str(len(statuses)))
    all_enforced = statuses and all(
        s.get("enforce_status") in ("ENFORCED", "IN EFFECT") for s in statuses
    )
    rep.check(bool(all_enforced),
              "every mediator policy status is ENFORCED (xApp ACKed over RMR)")

    acked = xapp.get("policies_acked_with_handler_id", [])
    rep.check(len(acked) == EXPECTED_ENFORCED,
              f"xApp ACKed all {EXPECTED_ENFORCED} policies with handler_id",
              str(len(acked)))
    rep.check(xapp.get("rmr_successful_sends_to_mediator", 0) >= accepted,
              "xApp RMR send-stats >= accepted policies",
              str(xapp.get("rmr_successful_sends_to_mediator")))

    # Third-witness coherence: the ACKed set must equal the enforced set.
    status_ids = {s.get("policy_id") for s in statuses}
    rep.check(status_ids == set(acked),
              "ACKed policy-id set equals ENFORCED policy-id set")


def check_refusal_proof(rep: Report) -> None:
    rep.note("=== 4. A Shield-blocked proposal is REFUSED rather than emitted ===")
    p = REPO_ROOT / "deploy/xapp-e2e/results/a1-assurance-wire-proof.json"
    if not p.exists():
        rep.check(False, "A1-assurance proof present")
        return
    proof = _read_json(p)

    rep.check(proof.get("result") == "PASS", "A1-assurance proof result == PASS",
              str(proof.get("result")))

    neg = proof.get("negative_case", {})
    rep.check(bool(neg), "negative (refusal) case present in the proof")
    rep.check(neg.get("refused") is True,
              "Shield-blocked proposal was REFUSED before emit",
              f"refused={neg.get('refused')}")
    cert = neg.get("certificate", {})
    rep.check(cert.get("emit_blocked") is True,
              "certificate carries emit_blocked=True")
    rep.check(cert.get("safe") is False, "certificate carries safe=False")
    violated = cert.get("violated_ids", [])
    rep.check(bool(violated),
              "refusal names a violated invariant", ",".join(map(str, violated)))
    rep.check(neg.get("blocked_policy_absent_from_simulator") is True,
              "refused policy never reached the simulator")
    rep.check(neg.get("policy_get_status_after") == 404,
              "GET on the refused policy id is 404 (nothing was emitted)")

    # HONEST DISCLOSURE, encoded as a NOTE not a silent pass:
    # the proposal says "over-power"; the committed refused proposal is a
    # negative-bandwidth (numeric_domain_sanity) case. We assert the refusal
    # that IS committed and surface the wording gap rather than paper over it.
    if "numeric_domain_sanity" in violated:
        rep.note(
            "GAP vs proposal wording: the committed refused proposal violates "
            "numeric_domain_sanity (negative bandwidth), NOT max_eirp "
            "(over-power). The refusal MECHANISM is genuine and proven; the "
            "specific 'over-power' invariant in §VII is not the one exercised."
        )

    # Positive case + signing must also be real for the proof to mean anything.
    pos = proof.get("positive_case", {})
    rep.check(pos.get("signature_verified") is True,
              "positive-case assurance envelope Ed25519 signature verifies")
    sim = proof.get("simulator", {})
    rep.check(sim.get("kind") == "real_vendored_submodule",
              "simulator is the real vendored OSC submodule, not a stub")
    rep.check(sim.get("fallback_used") is False,
              "no fallback simulator was used (fallback_used == False)")


def check_separation_disclosure(rep: Report) -> None:
    rep.note("=== 5. The two proofs are distinct paths (disclosed, not merged) ===")
    xapp = REPO_ROOT / "deploy/xapp-e2e/results/xapp-e2e-proof.json"
    if xapp.exists():
        blocked = _read_json(xapp).get("witnesses", {}).get(
            "pipeline_report", {}).get("blocked")
        rep.check(
            blocked == 0,
            "xApp emission path itself refuses NOTHING (blocked==0) — the "
            "refusal is proven ELSEWHERE, not in this path",
            str(blocked),
        )
        rep.note(
            "Therefore §VII's single sentence bundles TWO artifacts: the xApp "
            "emission proof (xapp-e2e-proof.json) and the separate refusal "
            "proof (a1-assurance-wire-proof.json / a1_assurance_proof.py, a "
            "different CI job). UNATTENDED.md documents both explicitly."
        )


def main() -> int:
    rep = Report()
    rep.note(f"Repo root: {REPO_ROOT}")
    check_entry_points(rep)
    check_pins(rep)
    check_xapp_proof(rep)
    check_refusal_proof(rep)
    check_separation_disclosure(rep)
    print(rep.render())
    print()
    if rep.ok:
        print("RESULT: PASS — runbook claims are backed by committed evidence.")
        return 0
    print("RESULT: FAIL — a runbook claim is not backed by committed evidence.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
