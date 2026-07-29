#!/usr/bin/env python3
"""Verify a fresh poisoning->Shield run against the committed real result.

This gates the proposal's single load-bearing number: on 8000 decisions drawn
from measured DeepMIMO campus geometry, 4658 requested actions are illegal and
0 survive the Shield. Until now that number lived only in a committed JSON with
no automated regeneration, which makes it an assertion rather than evidence.

Unlike the anti-jam and DeepMIMO-DSA verifiers, every quantity checked here is
an integer COUNT of decisions, not a float aggregate over per-receiver values.
Counts are host-stable: the decision stream is seeded, and a receiver either
requests more than the licensed EIRP or it does not. So this verifier compares
exactly, with no tolerance band — a single decision changing class is a real
change and must fail.

What it checks:

* the fresh run consumed the same measured scenario as the committed result
  (source_tree_sha256), so a passing run cannot be one that quietly fell back
  to synthetic input;
* every decision count reproduces exactly, including the split between
  violations caused by the poisoned planner and violations demanded by the
  real geometry alone -- that split is the paper's claim that the exposure is
  structural rather than adversarial, and it is the field most likely to move
  silently if the decision stream is ever regenerated;
* the safety invariants themselves still hold: zero illegal emissions and zero
  in-spec-but-harmful emissions after projection;
* the adjudicating oracle is still independent of the Shield's own constants,
  because a benchmark that grades its own work proves nothing.

Exit non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Exact-match fields. All are integer decision counts (see module docstring).
COUNT_FIELDS = (
    "decisions",
    "poisoned_inputs",
    "eirp_projections_applied",
    "unguarded_illegal_from_poisoning",
    "unguarded_illegal_from_real_geometry",
    "unguarded_illegal_emits",
    "shielded_illegal_emits",
    "shielded_blocked_emits",
    "shield_prevented",
    "in_spec_but_harmful_unguarded",
    "in_spec_but_harmful_after_shield",
)

# The oracle must not share the Shield's constants, or the result is circular.
INDEPENDENT_ORACLE_MARKER = "NOT Shield constants"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(committed: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    # 1. Same measured scenario. A run that silently fell back to synthetic
    #    input would otherwise be free to reproduce nothing and still pass.
    for key in ("source_tree_sha256", "dataset", "scenario"):
        if committed.get(key) != fresh.get(key):
            errors.append(
                f"{key}: committed {committed.get(key)!r} != fresh {fresh.get(key)!r}"
            )
    if fresh.get("receivers") != committed.get("receivers"):
        errors.append(
            f"receivers: committed {committed.get('receivers')} "
            f"!= fresh {fresh.get('receivers')}"
        )

    c_shield = committed.get("shield") or {}
    f_shield = fresh.get("shield") or {}
    if not f_shield:
        errors.append("fresh result has no 'shield' block")
        return errors

    # 2. Every decision count, exactly.
    for field in COUNT_FIELDS:
        if field not in c_shield:
            continue
        if c_shield[field] != f_shield.get(field):
            errors.append(
                f"shield.{field}: committed {c_shield[field]} "
                f"!= fresh {f_shield.get(field)}"
            )

    # 3. The safety invariants, asserted directly rather than inferred from
    #    the diff above -- if BOTH results drifted the same way, the equality
    #    check would pass while the guarantee was broken.
    if f_shield.get("shielded_illegal_emits") != 0:
        errors.append(
            f"SAFETY: {f_shield.get('shielded_illegal_emits')} illegal emissions "
            "survived the Shield (must be 0)"
        )
    if f_shield.get("in_spec_but_harmful_after_shield") != 0:
        errors.append(
            f"SAFETY: {f_shield.get('in_spec_but_harmful_after_shield')} "
            "in-spec-but-harmful emissions survived the Shield (must be 0)"
        )
    if not f_shield.get("unguarded_illegal_emits", 0) > 0:
        errors.append(
            "the unguarded path emitted nothing illegal, so the run proves "
            "nothing about the Shield -- check the decision stream"
        )

    # 4. The oracle stays independent of the Shield.
    oracle = str(f_shield.get("oracle", ""))
    if INDEPENDENT_ORACLE_MARKER not in oracle:
        errors.append(
            f"oracle is no longer declared independent of the Shield: {oracle!r}"
        )

    if f_shield.get("result") != "PASS":
        errors.append(
            f"fresh run did not self-report PASS: {f_shield.get('result')!r}"
        )

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--committed",
        type=Path,
        default=Path("benchmarks/results/poisoning_shield.json"),
    )
    ap.add_argument("--fresh", type=Path, required=True)
    args = ap.parse_args()

    errors = verify(_load(args.committed), _load(args.fresh))
    if errors:
        print("poisoning-shield verification FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    shield = _load(args.fresh)["shield"]
    print(
        "poisoning-shield verification PASSED: "
        f"{shield['decisions']} decisions, "
        f"{shield['unguarded_illegal_emits']} illegal unguarded "
        f"({shield['unguarded_illegal_from_real_geometry']} from real geometry alone), "
        f"{shield['shielded_illegal_emits']} after the Shield; "
        f"{shield['in_spec_but_harmful_unguarded']} in-spec-but-harmful -> "
        f"{shield['in_spec_but_harmful_after_shield']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
