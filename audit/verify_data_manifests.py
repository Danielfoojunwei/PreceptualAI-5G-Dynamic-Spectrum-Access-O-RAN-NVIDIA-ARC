#!/usr/bin/env python3
"""Gate — the committed checksum manifests, and whether they still hold.

The claim under gate:

    ``datasets/DATASETS_INDEX.md`` tells a reviewer they can "verify committed
    data at any time" with three commands. One of those three commands does
    not pass, and nothing in the repository notices.

What this found
---------------
``benchmarks/results/SHA256SUMS`` pins 15 result files. **All 15 mismatch**
at HEAD. The manifest was last written 2026-07-27; the results it pins were
regenerated afterwards, and no step recomputes the manifest when that
happens. It also covers 15 of the 31 ``.json`` files in that directory, while
the index row describes it as "one line per file".

The other two commands do pass: ``datasets/spectrum_dsa/SHA256SUMS`` verifies,
and ``examples/replay_telemetry.jsonl`` matches the digest the index prints.

Why this gate goes green on a known defect
------------------------------------------
Regenerating ``benchmarks/results/SHA256SUMS`` is a one-line fix, but it is a
fix to a file this branch is not permitted to touch — the standing constraint
on this work is additive-only. So the gate does the next most useful thing: it
measures the drift exactly, pins the measurement, and fails the moment the
measurement changes in *either* direction. If someone regenerates the manifest,
this gate goes red and says so, and the expectation below is updated to the
repaired state. Until then the defect is a tracked number rather than an
invisible one.

Read ``expected.benchmark_results.known_defect`` in the emitted JSON before
reading ``passed``. A green run here means "the manifests are in exactly the
state we recorded", not "the manifests are correct".

Additive: reads committed data, writes nothing outside ``--out``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

INDEX = REPO / "datasets" / "DATASETS_INDEX.md"

# The state this gate pins. Every number here was measured, not chosen.
EXPECTED: dict[str, Any] = {
    "manifest_count": 2,
    "manifest_paths": [
        "benchmarks/results/SHA256SUMS",
        "datasets/spectrum_dsa/SHA256SUMS",
    ],
    "spectrum_dsa": {"pinned": 1, "matched": 1, "mismatched": 0, "missing": 0},
    "benchmark_results": {
        "pinned": 15,
        "matched": 0,
        "mismatched": 15,
        "missing": 0,
        "json_files_present": 31,
        "json_files_unpinned": 16,
        "known_defect": (
            "every pinned digest is stale — the index advertises "
            "`(cd benchmarks/results && sha256sum -c SHA256SUMS)` and it "
            "fails 15/15. Fix: regenerate the manifest over all 31 result "
            "files and add the regeneration to whatever writes them."
        ),
    },
    # Datasets that exist on disk but have no row in DATASETS_INDEX.md.
    "datasets_missing_from_index": ["5gad_inl"],
}


@dataclass
class Check:
    id: str
    passed: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "passed": self.passed,
            "detail": self.detail,
            **({"data": self.data} if self.data else {}),
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _parse_manifest(path: Path) -> list[tuple[str, str]]:
    """Parse a ``sha256sum``-format manifest into (digest, name) pairs."""
    out: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # `sha256sum` writes "<hex>  <name>"; the second space becomes '*' in
        # binary mode. Accept both rather than assuming the writer's mode.
        m = re.match(r"^([0-9a-fA-F]{64})\s+[*]?(.+)$", line)
        if not m:
            raise ValueError(f"{path}: unparseable line {raw!r}")
        out.append((m.group(1).lower(), m.group(2)))
    return out


def _audit_manifest(manifest: Path) -> dict[str, Any]:
    base = manifest.parent
    matched, mismatched, missing = [], [], []
    for digest, name in _parse_manifest(manifest):
        target = base / name
        if not target.exists():
            missing.append(name)
        elif _sha256(target) == digest:
            matched.append(name)
        else:
            mismatched.append(name)
    return {
        "pinned": len(matched) + len(mismatched) + len(missing),
        "matched": len(matched),
        "mismatched": len(mismatched),
        "missing": len(missing),
        "mismatched_names": sorted(mismatched),
        "missing_names": sorted(missing),
    }


def _find_manifests() -> list[Path]:
    found = [
        p
        for p in REPO.rglob("SHA256SUMS")
        if "third_party" not in p.parts and ".git" not in p.parts
    ]
    return sorted(found)


def run_checks() -> list[Check]:
    checks: list[Check] = []

    # ── discovery ───────────────────────────────────────────────────────
    manifests = _find_manifests()
    rel = sorted(str(p.relative_to(REPO)) for p in manifests)
    checks.append(
        Check(
            "every_committed_manifest_is_discovered",
            rel == EXPECTED["manifest_paths"],
            f"found {len(rel)} manifest(s): {rel}",
            {"found": rel, "expected": EXPECTED["manifest_paths"]},
        )
    )

    by_rel = {str(p.relative_to(REPO)): p for p in manifests}

    # ── the manifest that holds ─────────────────────────────────────────
    sd_path = by_rel.get("datasets/spectrum_dsa/SHA256SUMS")
    if sd_path is None:
        checks.append(
            Check(
                "spectrum_dsa_manifest_verifies",
                False,
                "datasets/spectrum_dsa/SHA256SUMS is missing entirely",
            )
        )
    else:
        sd = _audit_manifest(sd_path)
        want = EXPECTED["spectrum_dsa"]
        ok = all(sd[k] == want[k] for k in want)
        checks.append(
            Check(
                "spectrum_dsa_manifest_verifies",
                ok,
                f"{sd['matched']}/{sd['pinned']} pinned file(s) match",
                sd,
            )
        )

    # ── the manifest that does not ──────────────────────────────────────
    br_path = by_rel.get("benchmarks/results/SHA256SUMS")
    want_br = EXPECTED["benchmark_results"]
    if br_path is None:
        checks.append(
            Check(
                "benchmark_results_drift_is_exactly_as_recorded",
                False,
                "benchmarks/results/SHA256SUMS is missing entirely",
            )
        )
        br = None
    else:
        br = _audit_manifest(br_path)
        ok = all(br[k] == want_br[k] for k in ("pinned", "matched", "mismatched", "missing"))
        checks.append(
            Check(
                "benchmark_results_drift_is_exactly_as_recorded",
                ok,
                f"{br['mismatched']} of {br['pinned']} pinned digest(s) stale "
                f"(recorded: {want_br['mismatched']} of {want_br['pinned']})",
                br,
            )
        )
        # This is the finding restated as its own check, so a reader scanning
        # ids sees it without opening `data`.
        checks.append(
            Check(
                "the_advertised_verify_command_still_fails",
                br["mismatched"] == want_br["mismatched"] and br["mismatched"] > 0,
                "`(cd benchmarks/results && sha256sum -c SHA256SUMS)` exits "
                f"non-zero: {br['mismatched']} mismatch(es). "
                + str(want_br["known_defect"]),
                {"known_defect": want_br["known_defect"]},
            )
        )

    # ── coverage ────────────────────────────────────────────────────────
    results_dir = REPO / "benchmarks" / "results"
    present = sorted(p.name for p in results_dir.glob("*.json"))
    pinned_names = (
        {n for _, n in _parse_manifest(br_path)} if br_path is not None else set()
    )
    unpinned = sorted(n for n in present if n not in pinned_names)
    checks.append(
        Check(
            "benchmark_result_coverage_is_exactly_as_recorded",
            len(present) == want_br["json_files_present"]
            and len(unpinned) == want_br["json_files_unpinned"],
            f"{len(present) - len(unpinned)} of {len(present)} result file(s) "
            f"pinned; {len(unpinned)} unpinned",
            {"present": len(present), "unpinned": unpinned},
        )
    )

    # ── the third advertised command ────────────────────────────────────
    # Read the expected digest out of the index rather than hardcoding it, so
    # this fails if the doc and the file disagree in either direction.
    index_text = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
    m = re.search(
        r"replay_telemetry\.jsonl`.*?`([0-9a-f]{64})`", index_text, re.S
    )
    replay = REPO / "examples" / "replay_telemetry.jsonl"
    if m is None or not replay.exists():
        checks.append(
            Check(
                "replay_telemetry_matches_the_digest_the_index_prints",
                False,
                "could not find the documented digest or the file itself",
            )
        )
    else:
        actual = _sha256(replay)
        checks.append(
            Check(
                "replay_telemetry_matches_the_digest_the_index_prints",
                actual == m.group(1),
                f"documented {m.group(1)[:12]}…, actual {actual[:12]}…",
            )
        )

    # ── index coverage of datasets/ ─────────────────────────────────────
    ds_root = REPO / "datasets"
    dirs = sorted(p.name for p in ds_root.iterdir() if p.is_dir()) if ds_root.exists() else []
    absent = [d for d in dirs if d not in index_text]
    checks.append(
        Check(
            "datasets_absent_from_the_index_are_exactly_as_recorded",
            absent == EXPECTED["datasets_missing_from_index"],
            f"{len(absent)} of {len(dirs)} dataset dir(s) have no row in "
            f"DATASETS_INDEX.md: {absent}",
            {"dataset_dirs": dirs, "absent_from_index": absent},
        )
    )

    return checks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="verify_data_manifests.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", help="write the JSON result here")
    args = p.parse_args(argv)

    checks = run_checks()
    passed = all(c.passed for c in checks)
    result = {
        "gate": "data-manifests",
        "claim": (
            "The three verification commands DATASETS_INDEX.md advertises are "
            "in exactly the state recorded here: two pass, and "
            "benchmarks/results/SHA256SUMS fails 15 of 15."
        ),
        "reading_note": (
            "passed=true means 'unchanged since we measured it', NOT 'the "
            "manifests are correct'. One of them is knowingly stale; see "
            "expected.benchmark_results.known_defect."
        ),
        "expected": EXPECTED,
        "checks": [c.to_dict() for c in checks],
        "passed": passed,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    for c in checks:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.id}: {c.detail}")
    print(f"data-manifests: {sum(c.passed for c in checks)}/{len(checks)} passed")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
