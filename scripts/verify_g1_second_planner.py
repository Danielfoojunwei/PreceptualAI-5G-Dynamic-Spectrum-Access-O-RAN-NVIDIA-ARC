#!/usr/bin/env python3
"""Verify a fresh G1 run against the committed result — WP1 deliverable C.

This gates one claim, and it is a claim about generality rather than about any
single planner:

    a second, independently written planner is shielded without modification to
    either the planner or the Shield.

The planner is a UCB1 bandit over a (centre-frequency x transmit-power) arm grid
(``src/horizon_ric/planners/ucb_spectrum.py``): a different algorithm and a
different interface style from the production risk-band rules planner and from
the existing tabular Q-learner, pure stdlib, importing nothing from the Shield.
The Shield is the stock ``default_terrestrial_shield`` chain driven through its
ordinary public ``dispose`` contract. One five-line adapter translates between
them and contains no safety logic.

Every quantity compared here is an integer COUNT of decisions, not a float
aggregate. Counts are host-stable: the decision stream is seeded, there is no
numpy on this path (so no ``Generator`` bit-stream drift), and an arm either
requests more than the licensed EIRP / a carrier outside the band or it does not.
So this verifier compares exactly, with no tolerance band — a single decision
changing class is a real change and must fail. The two peak-EIRP floats are exact
dB sums over a discrete grid, and are compared to 1e-9.

Why each check is non-vacuous:

* comparing counts alone would pass if BOTH the committed and the fresh run
  drifted the same way, so the safety property (zero illegal emissions) and the
  physical bounds (peak shielded EIRP under the ceiling, every emitted carrier
  inside the licensed channel less its guard band) are asserted *directly* on the
  fresh run as well;
* a run in which the planner never requested anything illegal would satisfy
  "zero illegal emissions" trivially, so the unguarded illegal-request count and
  the number of illegal arms in the planner's grid are both required to be
  positive. This is what stops the gate being passed by quietly making the arm
  grid safe;
* the adjudicating oracle must remain independent of the Shield's own constants,
  because a benchmark that grades its own work with the grader's constants proves
  nothing — the marker string is required in the emitted oracle note;
* and the claim itself is about *unmodified* code, so the SHA-256 of the Shield
  (``shield.py``, ``invariants.py``) and of the planner (``ucb_spectrum.py``) must
  match the pins recorded in the committed result. Editing either side to make
  the other pass fails the gate rather than satisfying it.

Exit non-zero on any violation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Exact-match fields. All are integer decision counts (see module docstring).
COUNT_FIELDS = (
    "decisions",
    "unguarded_illegal_requests",
    "unguarded_over_eirp",
    "unguarded_out_of_band",
    "shielded_illegal_emits",
    "shield_blocked_emits",
    "projections_applied",
    "emitted_decisions",
    "shield_prevented",
)

# Exact dB sums over a discrete arm grid, not statistical aggregates.
PEAK_FIELDS = ("peak_unguarded_eirp_dBm", "peak_shielded_eirp_dBm")
PEAK_TOL_DB = 1e-9

# The claim is about UNMODIFIED code on both sides of the adapter.
PINNED_SOURCES = (
    "src/horizon_ric/shield/shield.py",
    "src/horizon_ric/shield/invariants.py",
    "src/horizon_ric/planners/ucb_spectrum.py",
)

# The oracle must not share the Shield's constants, or the result is circular.
INDEPENDENT_ORACLE_MARKER = "NOT Shield constants"

# Enforcement tolerances, asserted directly against the fresh run.
EIRP_TOL_DB = 1e-6
BAND_TOL_HZ = 1e-6


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(
    committed: dict[str, Any], fresh: dict[str, Any], repo_root: Path = Path(".")
) -> list[str]:
    errors: list[str] = []

    c_gate = committed.get("gate") or {}
    f_gate = fresh.get("gate") or {}
    if not f_gate:
        errors.append("fresh result has no 'gate' block")
        return errors

    # 1. Every decision count, exactly — plus the declared licence the run was
    #    graded against. A fresh run free to lower the EIRP ceiling or widen the
    #    band could reproduce every count while policing nothing.
    for field in COUNT_FIELDS:
        if field not in c_gate:
            continue
        fresh_value = f_gate.get(field)
        if not isinstance(fresh_value, int) or isinstance(fresh_value, bool):
            errors.append(f"gate.{field}: fresh value {fresh_value!r} is not an integer count")
            continue
        if c_gate[field] != fresh_value:
            errors.append(
                f"gate.{field}: committed {c_gate[field]} != fresh {fresh_value}"
            )
    for field in PEAK_FIELDS:
        if field not in c_gate:
            continue
        try:
            delta = abs(float(c_gate[field]) - float(f_gate.get(field)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errors.append(f"gate.{field}: fresh value {f_gate.get(field)!r} is not a number")
            continue
        if delta > PEAK_TOL_DB:
            errors.append(
                f"gate.{field}: committed {c_gate[field]} != fresh {f_gate.get(field)}"
            )
    if committed.get("declared_constants") != fresh.get("declared_constants"):
        errors.append(
            "declared_constants changed between the committed and the fresh run: "
            f"committed {committed.get('declared_constants')!r} != "
            f"fresh {fresh.get('declared_constants')!r}"
        )

    # 2. The safety property, asserted directly rather than inferred from the
    #    diff above — if BOTH results drifted the same way, the equality check
    #    would pass while the guarantee was broken.
    if f_gate.get("shielded_illegal_emits") != 0:
        errors.append(
            f"SAFETY: {f_gate.get('shielded_illegal_emits')} illegal emissions survived "
            "the Shield (must be 0)"
        )

    # 3. The two physical bounds the Shield exists to enforce, recomputed here
    #    from the declared licence rather than read back from the run's own
    #    summary: peak emitted EIRP under the ceiling, and every emitted carrier
    #    inside the usable channel.
    declared = fresh.get("declared_constants") or {}
    band = declared.get("band_hz") or []
    guard = declared.get("guard_band_hz")
    ceiling = declared.get("max_eirp_dBm")
    if len(band) != 2 or guard is None or ceiling is None:
        errors.append(
            "fresh result does not declare band_hz / guard_band_hz / max_eirp_dBm, "
            "so the enforcement bounds cannot be checked independently"
        )
    else:
        usable_lo = float(band[0]) + float(guard)
        usable_hi = float(band[1]) - float(guard)
        peak = f_gate.get("peak_shielded_eirp_dBm")
        if peak is None or float(peak) > float(ceiling) + EIRP_TOL_DB:
            errors.append(
                f"SAFETY: peak shielded EIRP {peak} dBm exceeds the licensed ceiling "
                f"{ceiling} dBm"
            )
        lower = f_gate.get("shielded_min_lower_edge_hz")
        upper = f_gate.get("shielded_max_upper_edge_hz")
        if lower is None or float(lower) < usable_lo - BAND_TOL_HZ:
            errors.append(
                f"SAFETY: an emitted carrier's lower edge {lower} Hz is below the usable "
                f"channel floor {usable_lo} Hz"
            )
        if upper is None or float(upper) > usable_hi + BAND_TOL_HZ:
            errors.append(
                f"SAFETY: an emitted carrier's upper edge {upper} Hz is above the usable "
                f"channel ceiling {usable_hi} Hz"
            )
        graded_window = f_gate.get("usable_channel_hz") or []
        if len(graded_window) != 2 or (
            abs(float(graded_window[0]) - usable_lo) > BAND_TOL_HZ
            or abs(float(graded_window[1]) - usable_hi) > BAND_TOL_HZ
        ):
            errors.append(
                f"the run graded its emissions against {graded_window!r}, not against the "
                f"declared usable channel [{usable_lo}, {usable_hi}]"
            )

    # 4. NON-VACUITY. Zero illegal emissions is trivially true of a planner that
    #    never asked for anything illegal, so the unguarded path must actually
    #    have requested illegal emissions, and the planner's arm grid must still
    #    contain illegal arms. Without this the run proves nothing about the
    #    Shield and the gate could be passed by quietly making the grid safe.
    if not f_gate.get("unguarded_illegal_requests", 0) > 0:
        errors.append(
            "the unguarded planner requested nothing illegal, so the run proves nothing "
            "about the Shield -- check the arm grid and the reward callback"
        )
    illegal_arms = (f_gate.get("arm_grid") or {}).get("illegal_arms", 0)
    if not illegal_arms > 0:
        errors.append(
            f"the planner's arm grid contains {illegal_arms} illegal arms, so it can no "
            "longer request an illegal emission -- the gate would pass vacuously"
        )

    # 5. The oracle stays independent of the Shield.
    oracle = str(f_gate.get("oracle", ""))
    if INDEPENDENT_ORACLE_MARKER not in oracle:
        errors.append(f"oracle is no longer declared independent of the Shield: {oracle!r}")

    # 6. THE G1 CLAIM ITSELF: neither side was modified. The Shield's enforcement
    #    code and the planner must hash to the pins recorded in the committed
    #    result, so "the planner is shielded" cannot be achieved by editing the
    #    planner to behave, or by editing the Shield to accept it.
    c_pins = committed.get("source_sha256") or {}
    f_pins = fresh.get("source_sha256") or {}
    for rel in PINNED_SOURCES:
        pinned = c_pins.get(rel)
        if not pinned:
            errors.append(f"committed result has no source_sha256 pin for {rel}")
            continue
        path = repo_root / rel
        if not path.is_file():
            errors.append(f"{rel} not found under --repo-root {repo_root}")
            continue
        actual = _sha256(path)
        if actual != pinned:
            errors.append(
                f"{rel} was MODIFIED since the committed run: pinned {pinned} != "
                f"actual {actual} -- the G1 claim is that neither the planner nor the "
                "Shield changed"
            )
        if f_pins.get(rel) != pinned:
            errors.append(
                f"{rel}: fresh run recorded pin {f_pins.get(rel)} != committed {pinned}"
            )

    if f_gate.get("result") != "PASS":
        errors.append(f"fresh run did not self-report PASS: {f_gate.get('result')!r}")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--committed",
        type=Path,
        default=Path("benchmarks/results/g1_second_planner.json"),
    )
    ap.add_argument("--fresh", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    args = ap.parse_args()

    errors = verify(_load(args.committed), _load(args.fresh), args.repo_root)
    if errors:
        print("g1-second-planner verification FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    fresh = _load(args.fresh)
    gate = fresh["gate"]
    print(
        "g1-second-planner verification PASSED: "
        f"{fresh['planner']['planner_id']} ({gate['arm_grid']['arms']} arms, "
        f"{gate['arm_grid']['illegal_arms']} illegal) over {gate['decisions']} decisions -> "
        f"{gate['unguarded_illegal_requests']} illegal requests unguarded "
        f"({gate['unguarded_over_eirp']} over-EIRP, {gate['unguarded_out_of_band']} "
        f"out-of-band, peak {gate['peak_unguarded_eirp_dBm']} dBm), "
        f"{gate['shielded_illegal_emits']} after the Shield "
        f"(peak {gate['peak_shielded_eirp_dBm']} dBm vs {gate['max_eirp_dBm']} dBm ceiling, "
        f"{gate['projections_applied']} projections, {gate['shield_blocked_emits']} blocked); "
        "planner and Shield sources unmodified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
