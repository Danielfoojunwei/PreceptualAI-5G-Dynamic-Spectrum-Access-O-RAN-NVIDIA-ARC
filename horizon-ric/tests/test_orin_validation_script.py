"""Tests for deploy/orin_validation.sh — Row 26 delivery-time validation gate.

Verifies:
  1. The script exists and is executable.
  2. Dry-run against the canonical good soak proof returns 0 and prints PASS.
  3. The parser handles a synthetic GOOD soak JSON and returns 0.
  4. The parser handles a synthetic BAD soak JSON (A1 = 98 %) and returns
     non-zero.
  5. The script's substrate detector picks "substitute" on this build host
     (which has no /proc/device-tree/model marker for tegra/Orin).
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy" / "orin_validation.sh"
GOOD_PROOF = REPO_ROOT / "deploy" / "SOAK_24H_PROOF.md"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file(), f"missing: {SCRIPT}"
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "script not executable by owner"
    text = SCRIPT.read_text()
    assert "#!/usr/bin/env bash" in text or text.startswith("#!/bin/bash")
    assert "/proc/device-tree/model" in text, "substrate detection missing"
    assert "WatchdogSec" not in text or "WatchdogSec=30s" not in text  # not a systemd unit


def test_dry_run_against_good_soak_passes() -> None:
    assert GOOD_PROOF.is_file(), f"prerequisite missing: {GOOD_PROOF}"
    result = _run(["--dry-run"])
    assert result.returncode == 0, f"expected exit 0, got {result.returncode}\n{result.stdout}\n{result.stderr}"
    assert "OVERALL: PASS" in result.stdout
    # All 5 bars should appear with PASS
    for bar in (
        "audit_chain_integrity",
        "a1_success_pct",
        "p99_latency_ms",
        "watchdog_silence_s",
        "zero_unhandled_exc",
    ):
        assert f"[PASS] {bar}" in result.stdout, f"bar {bar} missing PASS in:\n{result.stdout}"


def test_parser_handles_synthetic_good_json(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "a1_success_pct": 99.81,
                "p99_decision_latency_ms": 142.0,
                "audit_verifies_passed": 1440,
                "max_watchdog_silence_s": 1.2,
                "unhandled_exceptions": 0,
            }
        )
    )
    result = _run(["--dry-run", "--soak-json", str(good)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OVERALL: PASS" in result.stdout


def test_parser_rejects_synthetic_bad_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    # A1 success 98 % — below the 99.5 % constrained-Orin bar.
    bad.write_text(
        json.dumps(
            {
                "a1_success_pct": 98.0,
                "p99_decision_latency_ms": 142.0,
                "audit_verifies_passed": 1440,
                "max_watchdog_silence_s": 1.2,
                "unhandled_exceptions": 0,
            }
        )
    )
    result = _run(["--dry-run", "--soak-json", str(bad)])
    assert result.returncode != 0, "expected non-zero exit on A1=98%"
    assert "OVERALL: FAIL" in result.stdout
    assert "[FAIL] a1_success_pct" in result.stdout


def test_substrate_detection_reports_substitute_on_build_host() -> None:
    # On the GB10 build host there is no tegra-soc marker, so detection
    # must report "substitute" — this is exactly the path the substitute
    # attestation in deploy/ORIN_HARDWARE_ATTESTATION.md relies on.
    result = _run(["--dry-run"])
    assert "substrate=substitute" in result.stdout or "substrate=real-orin" in result.stdout
    # On this CI host we expect substitute. (If the test ever runs on
    # actual Orin Nano, that's a passing real-hardware validation, which
    # is also acceptable.)
    model_path = "/proc/device-tree/model"
    on_real_orin = False
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = f.read().decode("utf-8", errors="ignore").rstrip("\x00")
            on_real_orin = ("Jetson Orin Nano" in model) or ("tegra" in model)
        except OSError:
            on_real_orin = False
    expected = "real-orin" if on_real_orin else "substitute"
    assert f"substrate={expected}" in result.stdout
