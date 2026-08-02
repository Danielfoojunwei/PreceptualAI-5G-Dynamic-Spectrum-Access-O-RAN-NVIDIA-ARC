"""Tests for the UNATTENDED demonstration runbook and its integrity gate.

Two things are enforced:

1. ``audit/check_unattended.py`` exits 0 against the committed evidence — the
   documentation-integrity gate agrees with the files in the repo.
2. Every file path the runbook (``audit/UNATTENDED.md``) names actually exists,
   and the one-command wrapper it advertises is a real, executable script that
   delegates to the committed proof scripts.

Run::

    /home/user/venv/bin/python -m pytest audit/tests/test_unattended.py -q
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "audit"
RUNBOOK = AUDIT / "UNATTENDED.md"
CHECKER = AUDIT / "check_unattended.py"
WRAPPER = AUDIT / "run_unattended.sh"


def test_checker_exists_and_passes_against_committed_evidence() -> None:
    """The offline gate must exit 0 on the real committed files."""
    assert CHECKER.exists(), "audit/check_unattended.py is missing"
    proc = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"check_unattended.py failed (exit {proc.returncode}):\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "RESULT: PASS" in proc.stdout


def test_checker_report_flips_on_failure() -> None:
    """The gate is not a rubber stamp: a single failed check must flip .ok,
    which is what drives the non-zero exit. Exercise the Report primitive that
    every check() call funnels through."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_unattended", CHECKER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rep = mod.Report()
    rep.check(True, "sanity true")
    assert rep.ok is True
    rep.check(False, "sanity false")
    assert rep.ok is False, "Report must flip .ok on a failed check"


# Top-level repo directories a runbook path reference can start under. Anything
# else (an /tmp/… output path, a bare shorthand) is deliberately NOT treated as
# a committed-file claim.
_REPO_TOP_DIRS = ("audit/", ".github/", "deploy/", "scripts/", "docs/", "src/")


def _referenced_paths(text: str) -> set[str]:
    """Every committed repo-relative file path the runbook names.

    Matches ``dir/sub/file.ext`` tokens (optionally with a leading dot, so
    ``.github/…`` survives) and keeps only those rooted at a real top-level
    repo directory and ending in a source/data extension.
    """
    tokens = set(re.findall(r"\.?[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+", text))
    paths: set[str] = set()
    for raw in tokens:
        tok = raw.strip("`").strip(",;:()")  # keep a leading dot (.github)
        if "://" in tok or tok.startswith("http"):
            continue
        if not re.search(r"\.(py|sh|md|json|yml|yaml)$", tok):
            continue
        if not tok.startswith(_REPO_TOP_DIRS):
            continue
        paths.add(tok)
    return paths


def test_every_file_the_runbook_references_exists() -> None:
    assert RUNBOOK.exists(), "audit/UNATTENDED.md is missing"
    text = RUNBOOK.read_text()
    missing = [p for p in sorted(_referenced_paths(text)) if not (REPO_ROOT / p).exists()]
    assert not missing, f"runbook references non-existent paths: {missing}"


def test_runbook_names_the_core_artifacts() -> None:
    """Guard against silent drift: the runbook must keep naming both proofs."""
    text = RUNBOOK.read_text()
    for required in (
        "deploy/xapp-e2e/run_stack.sh",
        "scripts/xapp_e2e_proof.py",
        "deploy/xapp-e2e/a1_assurance_proof.py",
        "deploy/xapp-e2e/results/xapp-e2e-proof.json",
        "deploy/xapp-e2e/results/a1-assurance-wire-proof.json",
        "audit/run_unattended.sh",
        "audit/check_unattended.py",
    ):
        assert required in text, f"runbook no longer references {required}"


def test_runbook_discloses_both_gaps() -> None:
    """The two honesty gaps must remain documented, not quietly dropped."""
    text = RUNBOOK.read_text().lower()
    assert "over-power" in text, "runbook must discuss the over-power wording gap"
    assert "numeric_domain_sanity" in text, (
        "runbook must name the invariant actually exercised"
    )
    assert "two" in text and "separate" in text, (
        "runbook must disclose the two-proofs-one-sentence gap"
    )


def test_wrapper_is_executable_and_delegates() -> None:
    assert WRAPPER.exists(), "audit/run_unattended.sh is missing"
    import os

    assert os.access(WRAPPER, os.X_OK), "run_unattended.sh must be executable"
    body = WRAPPER.read_text()
    # It must delegate to the committed scripts, not reimplement them.
    assert "deploy/xapp-e2e/run_stack.sh" in body
    assert "scripts/run_horizon_rapp.py" in body
    assert "scripts/xapp_e2e_proof.py" in body


def test_wrapper_bash_syntax_is_valid() -> None:
    """`bash -n` parses the wrapper without executing it."""
    proc = subprocess.run(
        ["bash", "-n", str(WRAPPER)], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"run_unattended.sh has a syntax error:\n{proc.stderr}"
