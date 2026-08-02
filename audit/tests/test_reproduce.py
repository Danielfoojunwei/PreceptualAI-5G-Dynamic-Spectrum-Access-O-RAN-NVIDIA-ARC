"""Tests for the one-command reproduction runner and its manifest.

These tests are themselves a gate: they assert the front door is honest — the
manifest is complete and points only at scripts that exist, --list names every
gate, the dependency-free gates actually RUN and PASS on a clean checkout, and a
gate with an unmet requirement is SKIPPED with a reason rather than failed or
silently passed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
AUDIT_DIR = HERE.parent
REPO_ROOT = AUDIT_DIR.parent
MANIFEST_PATH = AUDIT_DIR / "MANIFEST.json"
REPRODUCE = AUDIT_DIR / "reproduce.py"

sys.path.insert(0, str(AUDIT_DIR))
import reproduce  # noqa: E402

REQUIRED_KEYS = ("id", "title", "requirement", "command", "result_file", "ci_workflow", "notes")

VALID_SIMPLE_REQUIREMENTS = {"none", "horizon", "docker", "network"}

# Dependency-free gates with committed results: these MUST run and pass on a
# clean checkout. If any fails, that is a real finding, not a test to relax.
DEPENDENCY_FREE_EXPECTED = {
    "poisoning_shield",
    "antijam",
    "beam_management",
    "mobility_handover",
    "exploration_truncation",
    "federated_coverage",
    "credit_assignment",
    "safety_utility_frontier",
    "committed_projection_capacity_2x2",
    "committed_integrity_attack_suite",
    "committed_evasion_suite",
}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gates(manifest: dict) -> list[dict]:
    return manifest["gates"]


# --------------------------------------------------------------------------- #
# Manifest shape
# --------------------------------------------------------------------------- #


def test_manifest_parses_and_has_gates(manifest: dict) -> None:
    assert isinstance(manifest, dict)
    assert isinstance(manifest.get("gates"), list)
    assert manifest["gates"], "manifest lists no gates"


def test_every_entry_has_all_required_keys(gates: list[dict]) -> None:
    for gate in gates:
        missing = [k for k in REQUIRED_KEYS if k not in gate]
        assert not missing, f"gate {gate.get('id')!r} missing keys: {missing}"
        assert isinstance(gate["command"], list) and gate["command"], (
            f"gate {gate['id']} has an empty command"
        )
        assert all(isinstance(tok, str) for tok in gate["command"])


def test_gate_ids_are_unique(gates: list[dict]) -> None:
    ids = [g["id"] for g in gates]
    assert len(ids) == len(set(ids)), "duplicate gate ids in manifest"


def test_requirement_classes_are_valid(gates: list[dict]) -> None:
    for gate in gates:
        req = gate["requirement"]
        if req.startswith("submodule:"):
            assert req.split(":", 1)[1], f"gate {gate['id']} has empty submodule name"
        else:
            assert req in VALID_SIMPLE_REQUIREMENTS, (
                f"gate {gate['id']} has invalid requirement {req!r}"
            )


def test_every_script_path_exists_on_disk(gates: list[dict]) -> None:
    """No dangling references. Companion gates (built by a sibling task) are
    allowed to be absent, but if present must still resolve."""
    for gate in gates:
        script = REPO_ROOT / gate["command"][0]
        if gate.get("companion"):
            continue
        assert script.exists(), (
            f"gate {gate['id']} references missing script {gate['command'][0]}"
        )
        # A regen step, if declared, must also exist.
        if gate.get("regen"):
            regen_script = REPO_ROOT / gate["regen"][0]
            assert regen_script.exists(), (
                f"gate {gate['id']} regen references missing {gate['regen'][0]}"
            )


def test_result_files_exist_when_declared(gates: list[dict]) -> None:
    for gate in gates:
        rf = gate.get("result_file")
        if rf is None or gate.get("companion"):
            continue
        assert (REPO_ROOT / rf).exists(), (
            f"gate {gate['id']} declares missing result_file {rf}"
        )


# --------------------------------------------------------------------------- #
# --list
# --------------------------------------------------------------------------- #


def _run_reproduce(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPRODUCE), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_list_exits_zero_and_names_every_gate(gates: list[dict]) -> None:
    proc = _run_reproduce("--list")
    assert proc.returncode == 0, proc.stderr
    for gate in gates:
        assert gate["id"] in proc.stdout, f"--list omitted {gate['id']}"


# --------------------------------------------------------------------------- #
# Offline run: dependency-free gates actually pass
# --------------------------------------------------------------------------- #


def test_offline_gates_run_and_pass() -> None:
    """The dependency-free gates are real gates with committed results; they
    must PASS on a clean checkout. A failure here is a genuine finding."""
    manifest = reproduce.load_manifest()
    selected = reproduce.select_gates(manifest, mode="offline", only=None)
    selected_ids = {g["id"] for g in selected}
    assert DEPENDENCY_FREE_EXPECTED <= selected_ids, (
        "offline set is missing dependency-free gates: "
        f"{DEPENDENCY_FREE_EXPECTED - selected_ids}"
    )

    outcomes = {o.gate_id: o for o in (reproduce.run_gate(g) for g in selected)}
    for gid in DEPENDENCY_FREE_EXPECTED:
        outcome = outcomes[gid]
        assert outcome.status == reproduce.PASS, (
            f"dependency-free gate {gid} did not PASS: "
            f"{outcome.status} — {outcome.reason}\n{outcome.detail}"
        )


def test_offline_run_process_exits_zero() -> None:
    proc = _run_reproduce()
    assert proc.returncode == 0, (
        f"offline reproduce.py exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    )
    assert "failed" in proc.stdout  # summary line present
    assert ", 0 failed" in proc.stdout, proc.stdout


# --------------------------------------------------------------------------- #
# Unmet requirement -> loud SKIP, never a fail, never a silent pass
# --------------------------------------------------------------------------- #


def test_network_gate_is_skipped_with_reason() -> None:
    """network-class gates are never attempted automatically; they SKIP with a
    stated reason — not FAIL, not a silent pass."""
    manifest = reproduce.load_manifest()
    network_gates = [g for g in manifest["gates"] if g["requirement"] == "network"]
    assert network_gates, "expected at least one network-class gate to exist"
    for gate in network_gates:
        outcome = reproduce.run_gate(gate)
        assert outcome.status == reproduce.SKIP, (
            f"network gate {gate['id']} was {outcome.status}, expected SKIP"
        )
        assert outcome.reason.strip(), f"network gate {gate['id']} skipped without a reason"


def test_unmet_submodule_requirement_skips_with_reason(monkeypatch) -> None:
    """Simulate an unchecked-out submodule and assert the gate SKIPs loudly."""
    manifest = reproduce.load_manifest()
    sub_gates = [
        g for g in manifest["gates"] if g["requirement"].startswith("submodule:")
    ]
    assert sub_gates, "expected at least one submodule-class gate"

    # Force the submodule probe to report 'not checked out'.
    monkeypatch.setattr(
        reproduce,
        "_submodule_checked_out",
        lambda name: (False, f"submodule third_party/{name} empty (simulated)"),
    )
    for gate in sub_gates:
        outcome = reproduce.run_gate(gate)
        assert outcome.status == reproduce.SKIP, (
            f"submodule gate {gate['id']} was {outcome.status}, expected SKIP"
        )
        assert "submodule" in outcome.reason


def test_requirement_status_none_is_satisfied() -> None:
    satisfied, reason = reproduce.requirement_status("none")
    assert satisfied is True
    assert reason


def test_missing_script_skips_not_fails(monkeypatch, tmp_path) -> None:
    """A gate whose script is absent must SKIP with a reason, never FAIL and
    never silently pass."""
    fake_gate = {
        "id": "phantom",
        "title": "phantom",
        "requirement": "none",
        "command": ["audit/does_not_exist_zzz.py"],
        "result_file": None,
        "ci_workflow": None,
        "notes": "",
    }
    outcome = reproduce.run_gate(fake_gate)
    assert outcome.status == reproduce.SKIP
    assert "not present" in outcome.reason
