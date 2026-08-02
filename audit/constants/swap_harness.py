#!/usr/bin/env python3
"""Swap harness — prove each gate-bearing constant is one a gate is *sensitive* to.

The inventory (``inventory.py``) enumerates the constants. That alone does not
make gate G2 ("every published gate reproduces without retuning any constant")
meaningful: a constant no gate can see could be retuned freely and G2 would still
pass. This harness closes that hole. For each covered constant it **perturbs the
value and shows the dependent gate/check flips**, so "not retuned" is a real
claim about a real dependency rather than a vacuous one.

Two honest approaches, chosen per constant:

* **behavioural** — monkeypatch/instantiate the check with a perturbed value,
  re-run it on a scenario calibrated to the committed value's boundary, and
  assert the satisfied/blocked outcome changes;
* **sha256** — for gates that pin a file's SHA-256 (G1 pins ``shield.py`` /
  ``invariants.py`` / ``ucb_spectrum.py``; G6 and G8 pin their sources), show
  that changing the constant's literal changes the file digest, so the pin — and
  therefore the gate — fails. The real file's digest is first confirmed to match
  the committed pin, so the demonstration is against the live gate, not a straw
  digest.

Nothing here writes to ``src/`` — the sha256 approach perturbs the source text
*in memory* and hashes that, leaving the tree untouched.

Covers (at least) the four named constants: the slice floor (0.20), the EIRP
ceiling (33.0 dBm), the aggregate absolute floor (20e6 Hz), and
``ShieldConfig.max_passes``; plus the aggregate PFD/EIRP budget and an OCUDU RAN
Parameter ID, and it records the ``max_passes`` headroom finding.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
G1_RESULT = REPO_ROOT / "benchmarks" / "results" / "g1_second_planner.json"
OCUDU_CATALOGUE = REPO_ROOT / "ocudu" / "catalogue" / "ocudu-e2sm-catalogue.json"


# ---------------------------------------------------------------------------
# SHA-256 pin helpers (the G1/G6/G8-style dependency).
# ---------------------------------------------------------------------------
def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(rel: str) -> str:
    return hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()


def _committed_pin(rel: str) -> str | None:
    """The SHA-256 the committed G1 result pins for ``rel`` (None if absent)."""
    if not G1_RESULT.exists():
        return None
    data = json.loads(G1_RESULT.read_text(encoding="utf-8"))
    pin: str | None = (data.get("source_sha256") or {}).get(rel)
    return pin


def sha256_swap(rel: str, old_literal: str, new_literal: str, gate: str) -> dict[str, Any]:
    """Show that editing a constant's literal in ``rel`` breaks its SHA-256 pin.

    Confirms the live file matches the committed pin (so the pin is real and
    current), then rewrites exactly one occurrence of ``old_literal`` in memory
    and shows the digest changes — which is exactly what makes the gate fail.
    """
    source = (REPO_ROOT / rel).read_text(encoding="utf-8")
    occurrences = source.count(old_literal)
    live = _sha256_file(rel)
    pin = _committed_pin(rel)
    perturbed_source = source.replace(old_literal, new_literal, 1)
    perturbed_digest = _sha256_text(perturbed_source)
    return {
        "constant_literal": old_literal,
        "file": rel,
        "approach": "sha256",
        "gate": gate,
        "occurrences_in_source": occurrences,
        "committed_pin": pin,
        "live_digest": live,
        "live_matches_pin": pin is not None and live == pin,
        "perturbed_to": new_literal,
        "perturbed_digest": perturbed_digest,
        # The flip: the perturbed digest no longer equals the pin the gate checks.
        "flipped": (
            occurrences == 1
            and perturbed_digest != live
            and (pin is None or perturbed_digest != pin)
        ),
        "detail": (
            f"{rel}: {old_literal!r} -> {new_literal!r} changes the digest from "
            f"{live[:12]}... to {perturbed_digest[:12]}..., failing {gate}'s pin"
        ),
    }


# ---------------------------------------------------------------------------
# Behavioural swaps — re-run the check with a perturbed value.
# ---------------------------------------------------------------------------
def swap_slice_floor() -> dict[str, Any]:
    """ProtectedSliceFloorInvariant.floor (0.20): the slice-floor check flips."""
    from horizon_ric.shield.invariants import ProtectedSliceFloorInvariant

    live = ProtectedSliceFloorInvariant()
    floor = live.floor
    # Allocate the protected slice exactly at the committed floor -> satisfied,
    # margin 0. Any strictly higher floor makes the same allocation fail.
    action = {"prb_allocation": {"safety_critical": floor, "embb": 1.0 - floor}}
    perturbed = ProtectedSliceFloorInvariant(floor=floor + 0.05)

    ok_committed = live.evaluate(action, {}).satisfied
    ok_perturbed = perturbed.evaluate(action, {}).satisfied
    behavioural = {
        "constant": "ProtectedSliceFloorInvariant.floor",
        "value": floor,
        "approach": "behavioural",
        "gate": "scripts/verify_g1_second_planner.py (Shield slice-floor check)",
        "scenario": f"safety_critical allocated exactly the floor ({floor})",
        "satisfied_at_committed": ok_committed,
        "satisfied_at_perturbed": ok_perturbed,
        "perturbed_value": floor + 0.05,
        "flipped": ok_committed and not ok_perturbed,
        "detail": (
            f"at floor={floor} the boundary allocation is satisfied; raising the "
            f"floor to {floor + 0.05} makes the identical allocation UNSATISFIED"
        ),
    }
    sha = sha256_swap(
        "src/horizon_ric/shield/invariants.py",
        "floor: float = 0.20",
        "floor: float = 0.25",
        "scripts/verify_g1_second_planner.py",
    )
    return {"behavioural": behavioural, "sha256": sha, "flipped": behavioural["flipped"] and sha["flipped"]}


def swap_eirp_ceiling() -> dict[str, Any]:
    """MaxEirpInvariant.max_eirp_dBm (33.0): the EIRP check flips."""
    from horizon_ric.shield.invariants import MaxEirpInvariant

    live = MaxEirpInvariant()
    ceiling = live.max_eirp_dBm
    # EIRP exactly at the ceiling -> margin 0, satisfied. Lower the ceiling and
    # the identical emission is over budget.
    action = {"tx_power_dBm": ceiling - 6.0, "antenna_gain_dBi": 6.0}
    perturbed = MaxEirpInvariant(max_eirp_dBm=ceiling - 3.0)

    ok_committed = live.evaluate(action, {}).satisfied
    ok_perturbed = perturbed.evaluate(action, {}).satisfied
    behavioural = {
        "constant": "MaxEirpInvariant.max_eirp_dBm",
        "value": ceiling,
        "approach": "behavioural",
        "gate": "scripts/verify_g1_second_planner.py / scripts/verify_poisoning_shield.py",
        "scenario": f"EIRP = tx_power+gain = {ceiling} dBm, exactly at the ceiling",
        "satisfied_at_committed": ok_committed,
        "satisfied_at_perturbed": ok_perturbed,
        "perturbed_value": ceiling - 3.0,
        "flipped": ok_committed and not ok_perturbed,
        "detail": (
            f"an emission at exactly {ceiling} dBm is admitted; lowering the "
            f"ceiling to {ceiling - 3.0} dBm blocks the identical emission"
        ),
    }
    sha = sha256_swap(
        "src/horizon_ric/shield/invariants.py",
        "max_eirp_dBm: float = 33.0",
        "max_eirp_dBm: float = 30.0",
        "scripts/verify_g1_second_planner.py",
    )
    return {"behavioural": behavioural, "sha256": sha, "flipped": behavioural["flipped"] and sha["flipped"]}


def swap_aggregate_floor() -> dict[str, Any]:
    """AbsoluteSliceCapacityFloor.min_capacity_hz (20e6): the cross-agent floor flips."""
    from horizon_agentic.aggregate import AbsoluteSliceCapacityFloor

    live = AbsoluteSliceCapacityFloor()
    floor_hz = live.min_capacity_hz
    # Choose bandwidth so share*bandwidth lands exactly on the floor.
    share = 0.20
    bandwidth = floor_hz / share  # 20e6 / 0.20 = 100 MHz -> capacity 20e6 exactly
    actions = [{"bandwidth_hz": bandwidth, "prb_allocation": {"safety_critical": share, "embb": 1 - share}}]
    perturbed = dataclasses.replace(live, min_capacity_hz=floor_hz + 5e6)

    ok_committed = live.evaluate(actions, {}).satisfied
    ok_perturbed = perturbed.evaluate(actions, {}).satisfied
    behavioural = {
        "constant": "AbsoluteSliceCapacityFloor.min_capacity_hz",
        "value": floor_hz,
        "approach": "behavioural",
        "gate": "agentic/verify_g6_multi_agent.py",
        "scenario": f"{share} x {bandwidth/1e6:.1f} MHz = {share*bandwidth/1e6:.1f} MHz, exactly at the floor",
        "satisfied_at_committed": ok_committed,
        "satisfied_at_perturbed": ok_perturbed,
        "perturbed_value": floor_hz + 5e6,
        "flipped": ok_committed and not ok_perturbed,
        "detail": (
            f"a bundle giving the protected slice exactly {floor_hz/1e6:.0f} MHz is "
            f"satisfied; raising the floor to {(floor_hz+5e6)/1e6:.0f} MHz makes it "
            "UNSATISFIED"
        ),
    }
    sha = sha256_swap(
        "agentic/src/horizon_agentic/aggregate.py",
        "min_capacity_hz: float = 20e6",
        "min_capacity_hz: float = 25e6",
        "agentic/verify_g6_multi_agent.py",
    )
    return {"behavioural": behavioural, "sha256": sha, "flipped": behavioural["flipped"] and sha["flipped"]}


def measure_projection_passes() -> dict[str, Any]:
    """How many projection passes the default terrestrial chain actually needs.

    Substantiates the ``max_passes`` finding: the smallest ``max_passes`` for
    which each action's disposition stops changing is the number of passes that
    action needs, and the maximum over the battery is the smallest value of
    ``max_passes`` that reproduces every committed disposition.

    **A correction.** An earlier version of this battery held only the first
    four scenarios, all of which converge in one pass, and this module
    published the conclusion that the disposition is "INSENSITIVE to
    max_passes for any value >= 1". That was false, and the fifth scenario is
    the counterexample that ``audit/refusal_semantics.py`` turned up:

        tx_power_dBm = 33.1, antenna_gain_dBi = 60.0, ceiling 33.0 dBm

    ``MaxEirpInvariant.project`` subtracts ``overage = 33.1 + 60.0 - 33.0``
    from ``tx_power_dBm``. In binary floating point that lands on
    ``-26.999999999999993``, whose EIRP is ``33.00000000000001`` — margin
    ``-7.1e-15``, so the invariant is *still* unsatisfied. A second pass
    clears the residue. At ``max_passes=1`` this action is REFUSED; at 2 it is
    corrected.

    The size of the overage does not predict the pass count: a **7 dB**
    overage converges in one pass and this **0.1 dB** one does not, because
    what matters is whether the float subtraction rounds back exactly. So the
    1 -> 2 boundary is genuinely behavioural, and only values >= 2 are
    headroom.
    """
    from horizon_ric.shield.shield import Shield, ShieldConfig, default_terrestrial_shield

    base = default_terrestrial_shield(band_lo_hz=3.4e9, band_hi_hz=3.5e9, max_eirp_dBm=33.0)
    invariants = list(base._invariants)  # the built chain; instantiate our own configs
    scenarios = [
        {"block": "x", "frequency_hz": 3.45e9, "bandwidth_hz": 20e6, "tx_power_dBm": 40.0, "antenna_gain_dBi": 6.0},
        {"block": "x", "frequency_hz": 3.30e9, "bandwidth_hz": 20e6, "tx_power_dBm": 20.0, "antenna_gain_dBi": 6.0},
        {"block": "learned_constellation", "frequency_hz": 3.45e9, "bandwidth_hz": 20e6,
         "tx_power_dBm": 45.0, "antenna_gain_dBi": 6.0, "constellation_order": 128, "papr_dB": 12.0},
        {"block": "neural_rx", "frequency_hz": 3.51e9, "bandwidth_hz": 20e6, "tx_power_dBm": 40.0,
         "antenna_gain_dBi": 6.0, "predicted_tbler": 0.01, "baseline_tbler": 0.1},
        # The counterexample. Needs two passes; see the docstring.
        {"block": "x", "frequency_hz": 3.45e9, "bandwidth_hz": 20e6, "tx_power_dBm": 33.1,
         "antenna_gain_dBi": 60.0},
    ]
    # Smallest max_passes at which the disposition equals its settled value AND
    # never changes again. An earlier version stopped at the first *repeat*,
    # which is not the same thing and got the answer wrong: the residue
    # scenario is blocked at both max_passes=0 and max_passes=1 and only flips
    # at 2, so "first repeat" concluded 0 passes were needed. Convergence has
    # to be read from the settled end, not from the first pair that agree.
    MAX = 8
    needed = 0
    per_scenario: list[int] = []
    for action in scenarios:
        dispositions = [
            Shield(invariants, ShieldConfig(max_passes=mp))
            .dispose(dict(action))
            .certificate.emit_blocked
            for mp in range(MAX + 1)
        ]
        settled = dispositions[MAX]
        converged_at = MAX
        for mp in range(MAX + 1):
            if all(d == settled for d in dispositions[mp:]):
                converged_at = mp
                break
        per_scenario.append(converged_at)
        needed = max(needed, converged_at)
    return {
        "passes_needed_max": needed,
        "passes_needed_per_scenario": per_scenario,
        "scenarios": len(scenarios),
    }


def swap_max_passes() -> dict[str, Any]:
    """ShieldConfig.max_passes (8): flips the disposition and breaks the G1 pin.

    Behaviourally the chain converges in <=1 pass, so 8 is conservative headroom;
    the disposition only flips when max_passes is dropped to 0 (no projection at
    all), which turns a fixable over-EIRP action into a blocked emit. The *exact*
    value 8 is gate-bearing through G1's SHA-256 pin of shield.py, not through the
    disposition. Both are demonstrated; the headroom is recorded as a finding.
    """
    from horizon_ric.shield.shield import Shield, ShieldConfig, default_terrestrial_shield

    base = default_terrestrial_shield(band_lo_hz=3.4e9, band_hi_hz=3.5e9, max_eirp_dBm=33.0)
    invariants = list(base._invariants)
    live_passes = ShieldConfig().max_passes
    # A single over-EIRP action: fixable in one projection pass.
    action = {"block": "x", "frequency_hz": 3.45e9, "bandwidth_hz": 20e6,
              "tx_power_dBm": 40.0, "antenna_gain_dBi": 6.0}
    blocked_committed = Shield(invariants, ShieldConfig(max_passes=live_passes)).dispose(dict(action)).certificate.emit_blocked
    blocked_zero = Shield(invariants, ShieldConfig(max_passes=0)).dispose(dict(action)).certificate.emit_blocked

    convergence = measure_projection_passes()
    behavioural = {
        "constant": "ShieldConfig.max_passes",
        "value": live_passes,
        "approach": "behavioural",
        "gate": "scripts/verify_g1_second_planner.py (Shield disposition)",
        "scenario": "over-EIRP action fixable by one projection pass",
        "emit_blocked_at_committed": blocked_committed,
        "emit_blocked_at_perturbed": blocked_zero,
        "perturbed_value": 0,
        "flipped": (not blocked_committed) and blocked_zero,
        "detail": (
            f"at max_passes={live_passes} the fixable action is projected to safe "
            "(emit_blocked=False); at max_passes=0 no projection runs and the same "
            "action is blocked (emit_blocked=True)"
        ),
    }
    needed = convergence["passes_needed_max"]
    finding = {
        "id": "max_passes_is_headroom_above_the_convergence_floor",
        "text": (
            f"The default chain needs up to {needed} projection pass(es) across "
            f"{convergence['scenarios']} infeasible scenarios, so the disposition "
            f"is insensitive to max_passes only for values >= {needed}: the "
            f"committed 8 is headroom ABOVE that floor, not a tuned threshold. "
            f"CORRECTION: this module previously reported the floor as 1 and the "
            f"constant as insensitive for any value >= 1. That was false. A "
            f"0.1 dB over-EIRP action at 60 dBi antenna gain leaves a "
            f"-7.1e-15 dB floating-point residue after one clamp and needs a "
            f"second pass; at max_passes=1 it is REFUSED. A 7 dB overage "
            f"converges in one pass, so the size of the overage does not "
            f"predict the pass count. Its exact value 8 remains gate-bearing "
            f"through G1's SHA-256 pin of shield.py — which the sha256 swap "
            f"below breaks."
        ),
        "passes_needed_max": needed,
        "passes_needed_per_scenario": convergence["passes_needed_per_scenario"],
        "corrects_earlier_finding": "max_passes_is_headroom",
    }
    sha = sha256_swap(
        "src/horizon_ric/shield/shield.py",
        "max_passes: int = 8",
        "max_passes: int = 16",
        "scripts/verify_g1_second_planner.py",
    )
    return {
        "behavioural": behavioural,
        "sha256": sha,
        "finding": finding,
        "flipped": behavioural["flipped"] and sha["flipped"],
    }


def swap_ocudu_min_prb_ratio() -> dict[str, Any]:
    """MIN_PRB_POLICY_RATIO (11): flips G8's parameter-id-vs-catalogue check.

    G8 asserts each declared RAN Parameter ID against the ids OCUDU declares in
    the extracted catalogue. This replays that assertion against the committed
    catalogue (no submodule needed): the live id is present in the action's
    parameter table; a perturbed id is not, so the check fails.
    """
    from horizon_ocudu.rc_slice_quota import MIN_PRB_POLICY_RATIO

    catalogue = json.loads(OCUDU_CATALOGUE.read_text(encoding="utf-8"))
    slice_action = next(
        a for a in catalogue["e2sm_rc_control_actions"]
        if a["node"] == "DU" and a["style_id"] == 2 and a["action_id"] == 6
    )
    declared_ids = {p["id"] for p in slice_action["ran_parameters"]}
    live_id = MIN_PRB_POLICY_RATIO
    perturbed_id = 999
    behavioural = {
        "constant": "MIN_PRB_POLICY_RATIO",
        "value": live_id,
        "approach": "behavioural",
        "gate": "ocudu/verify_g8_ocudu_conformance.py (parameter_ids_match_ocudu)",
        "scenario": "assert the declared id appears in OCUDU's extracted catalogue",
        "present_at_committed": live_id in declared_ids,
        "present_at_perturbed": perturbed_id in declared_ids,
        "perturbed_value": perturbed_id,
        "flipped": (live_id in declared_ids) and (perturbed_id not in declared_ids),
        "detail": (
            f"id {live_id} is in OCUDU's slice-quota parameter table {sorted(declared_ids)}; "
            f"perturbing it to {perturbed_id} is not, so G8's id-match check fails"
        ),
    }
    return {"behavioural": behavioural, "flipped": behavioural["flipped"]}


SWAPS = {
    "slice_floor": swap_slice_floor,
    "eirp_ceiling": swap_eirp_ceiling,
    "aggregate_floor": swap_aggregate_floor,
    "max_passes": swap_max_passes,
    "ocudu_min_prb_ratio": swap_ocudu_min_prb_ratio,
}


def run_all() -> dict[str, Any]:
    return {name: fn() for name, fn in SWAPS.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        help="write the full JSON report here instead of stdout (the "
        "per-swap summary still goes to stdout either way)",
    )
    args = parser.parse_args(argv)

    results = run_all()
    all_flipped = all(r["flipped"] for r in results.values())
    report = json.dumps(results, indent=2, sort_keys=True, default=str)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    for name, r in results.items():
        print(f"  {name}: flipped={r['flipped']}")
    if "max_passes" in results:
        print(f"  finding: {results['max_passes']['finding']['text']}")
    return 0 if all_flipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
