"""Real `pyang --strict --canonical` gate over `deploy/yang/`.

Closes Row 9 of the gap matrix. No mocks: each test shells out to the real
`pyang` binary (installed via the `dev` extra and pinned in `.github/
workflows/lint.yml`) and asserts exit code 0 with empty stdout/stderr on the
strict-clean subset.

Modules listed in `deploy/yang/known_strict_violations.md` are skipped — they
are upstream RFC / 3GPP defects, recorded with sha256 + URL in
`deploy/yang/manifest.json` so a regulator can audit the byte-for-byte source.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
YANG_DIR = ROOT / "deploy" / "yang"
MANIFEST = YANG_DIR / "manifest.json"
KNOWN_VIOLATIONS = YANG_DIR / "known_strict_violations.md"


def _pyang() -> str:
    """Resolve the pyang binary; skip the test module if it is not installed.

    We look in (a) the current sys.executable's Scripts/bin directory (so
    `python -m pytest` from a venv finds the venv's pyang), then (b) PATH.
    """
    venv_bin = Path(sys.executable).parent
    candidate = venv_bin / "pyang"
    if candidate.exists() and os.access(candidate, os.X_OK):
        return str(candidate)
    p = shutil.which("pyang")
    if p is None:
        pytest.skip("pyang not installed; the lint CI job installs it.")
    return p


def _violations_skip_set() -> set[str]:
    """Parse the known-violations doc to recover the allowlist of file names.

    The doc enumerates each tolerated module in backticks in the Summary
    table; we re-use the same regex CI uses so test + CI cannot drift.
    Only the section between '## Summary' and '## Strict-clean modules' counts
    — the trailing table lists modules that DO pass strict and must not be
    treated as skipped.
    """
    text = KNOWN_VIOLATIONS.read_text()
    # Keep only the violations section.
    if "## Strict-clean" in text:
        text = text.split("## Strict-clean", 1)[0]
    return set(re.findall(r"`([\w@.\-]+\.yang)`", text))


def _all_modules() -> list[Path]:
    return sorted(YANG_DIR.glob("*.yang"))


def _strict_clean_modules() -> list[Path]:
    skip = _violations_skip_set()
    return [m for m in _all_modules() if m.name not in skip]


def test_yang_dir_is_populated() -> None:
    """We must vendor a non-trivial number of real YANG modules."""
    mods = _all_modules()
    assert len(mods) >= 11, f"expected >=11 vendored modules, got {len(mods)}"


def test_manifest_records_every_module() -> None:
    """Every .yang file under deploy/yang must appear in manifest.json with
    a SHA-256 that matches the on-disk bytes (tamper-evident vendoring)."""
    manifest = json.loads(MANIFEST.read_text())
    recorded = manifest["modules"]
    on_disk = {m.name for m in _all_modules()}
    assert on_disk <= set(recorded.keys()), (
        f"manifest missing entries for: {on_disk - set(recorded.keys())}"
    )
    for name, meta in recorded.items():
        if name not in on_disk:
            continue
        digest = hashlib.sha256((YANG_DIR / name).read_bytes()).hexdigest()
        assert digest == meta["sha256"], (
            f"sha256 drift for {name}: manifest={meta['sha256']} disk={digest}"
        )
        assert meta["url"].startswith(("http://", "https://")), (
            f"manifest entry for {name} missing source URL"
        )


@pytest.mark.parametrize(
    "module",
    _strict_clean_modules(),
    ids=lambda p: p.name,
)
def test_module_passes_pyang_strict(module: Path) -> None:
    """Each strict-clean module must validate without diagnostics."""
    pyang = _pyang()
    proc = subprocess.run(
        [pyang, "--strict", "--canonical", "-p", str(YANG_DIR), str(module)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"pyang exit={proc.returncode} on {module.name}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert not proc.stdout.strip() and not proc.stderr.strip(), (
        f"pyang produced diagnostics for {module.name}:\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )


def test_known_violations_doc_lists_only_existing_modules() -> None:
    """The skip-list must not drift from the on-disk vendored set."""
    skip = _violations_skip_set()
    on_disk = {m.name for m in _all_modules()}
    stale = skip - on_disk
    assert not stale, f"known_strict_violations.md references missing modules: {stale}"


def test_at_least_one_ietf_one_oran_one_3gpp_module_strict_clean() -> None:
    """Sanity: we vendor real modules from each of the three SDOs the chart
    declares support for, and at least one from each must pass strict."""
    clean = {m.name for m in _strict_clean_modules()}
    assert any(n.startswith("ietf-") for n in clean), "no IETF strict-clean module"
    assert any(n.startswith("o-ran-") for n in clean), "no O-RAN strict-clean module"
    assert any(n.startswith("_3gpp-") for n in clean), "no 3GPP strict-clean module"
