#!/usr/bin/env python3
"""Gate — what the Shield refuses, what it corrects, and what §VII claims.

The claim under gate:

    Refusal in this Shield is reserved for actions no projection can repair.
    An over-power proposal is always repairable, so it is always corrected and
    never refused — which means §VII's promise of "a Shield-corrected
    over-power proposal refused rather than emitted" describes behaviour the
    chain deliberately does not have.

Why a gate and not a note
-------------------------
``audit/check_unattended.py`` already disclosed that the committed refusal
proof exercises ``numeric_domain_sanity`` rather than ``max_eirp``, and
recorded it as *the proof is missing*. Whether it is missing or impossible
changes what the author has to do: write a test, or change a sentence. This
gate settles it by exhausting the over-power space.

Eleven checks. Three are load-bearing:

* ``max_eirp_never_refuses`` — the finding. 2688 over-power actions, crossed
  with every field the other invariants read, and not one is refused.
* ``convergence_floor_is_two_passes_not_one`` — corrects a claim this
  repository previously published. ``audit/constants/swap_harness.py`` said
  the chain "converges in <= 1 pass" and was therefore "INSENSITIVE to
  max_passes for any value >= 1". False: one action needs two.
* ``overage_size_does_not_predict_pass_count`` — why the above was missed. A
  7 dB overage converges in one pass and a 0.1 dB overage does not, so a
  battery of large, obvious violations will never find the boundary.

Additive: reads the shipping Shield, writes nothing to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# `audit.constants` imports itself by its fully-qualified name, so REPO has to
# be importable too, not just HERE.
for p in (REPO / "src", REPO, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from refusal_semantics import (  # noqa: E402
    BAND_HI_HZ,
    BAND_LO_HZ,
    DEFERRING,
    MAX_EIRP_DBM,
    REFUSING,
    build_shield,
    characterise,
)

from audit.constants.swap_harness import measure_projection_passes  # noqa: E402
from horizon_ric.shield.shield import Shield, ShieldConfig  # noqa: E402

# The exact §VII sentence this gate is about, quoted from docs/proposal/content.py.
SECTION_VII_CLAIM = (
    "a Shield-corrected over-power proposal refused rather than emitted"
)

# What the author can paste in its place. Describes what the demonstration
# actually does, which is the stronger claim anyway: the emitted action is not
# the proposed one.
PROPOSED_REWORDING = (
    "an over-power proposal corrected to the EIRP ceiling before emission, "
    "and a non-physical proposal refused outright"
)


class Check:
    def __init__(self, cid: str, passed: bool, detail: str) -> None:
        self.id, self.passed, self.detail = cid, passed, detail

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "passed": self.passed, "detail": self.detail}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# The counterexample, named once so every check that uses it agrees.
RESIDUE_ACTION = {
    "block": "spectrum",
    "frequency_hz": (BAND_LO_HZ + BAND_HI_HZ) / 2.0,
    "bandwidth_hz": 20e6,
    "tx_power_dBm": 33.1,
    "antenna_gain_dBi": 60.0,
}
BIG_OVERAGE_ACTION = {
    "block": "spectrum",
    "frequency_hz": (BAND_LO_HZ + BAND_HI_HZ) / 2.0,
    "bandwidth_hz": 20e6,
    "tx_power_dBm": 40.0,
    "antenna_gain_dBi": 0.0,
}


def _blocked_at(action: dict[str, Any], max_passes: int) -> bool:
    base = build_shield()
    shield = Shield(list(base._invariants), ShieldConfig(max_passes=max_passes))
    return bool(shield.dispose(dict(action)).certificate.emit_blocked)


def _passes_needed(action: dict[str, Any], limit: int = 8) -> int:
    dispositions = [_blocked_at(action, mp) for mp in range(limit + 1)]
    settled = dispositions[limit]
    for mp in range(limit + 1):
        if all(d == settled for d in dispositions[mp:]):
            return mp
    return limit


def run_checks() -> list[Check]:
    logging.disable(logging.CRITICAL)
    try:
        return _run()
    finally:
        logging.disable(logging.NOTSET)


def _run() -> list[Check]:
    checks: list[Check] = []

    def add(cid: str, passed: bool, detail: str) -> None:
        checks.append(Check(cid, passed, detail))

    by_id, summary = characterise()

    # ── 1. the probe covers the chain that actually ships ────────────────
    chain_ids = set(build_shield().invariant_ids)
    probed_ids = set(by_id)
    add(
        "probe_covers_the_shipping_chain",
        chain_ids == probed_ids,
        f"default_terrestrial_shield builds {sorted(chain_ids)}; this gate "
        f"probes {sorted(probed_ids)}. Equal={chain_ids == probed_ids} — if a "
        "chain member is added upstream, this fails rather than quietly "
        "characterising four of five invariants.",
    )

    # ── 2. nothing was left unprobed ─────────────────────────────────────
    unprobed = [i for i, c in by_id.items() if c.violations_probed == 0]
    add(
        "every_invariant_was_actually_made_to_fail",
        not unprobed,
        "an invariant this sweep never got to fail would be classified by "
        f"default, not by evidence. Unprobed: {unprobed or 'none'}. "
        + "; ".join(
            f"{i}={c.violations_probed}" for i, c in sorted(by_id.items())
        ),
    )

    # ── 3. the refusing invariants refuse ────────────────────────────────
    nds = by_id["numeric_domain_sanity"]
    add(
        "numeric_domain_sanity_refuses",
        nds.verdict == REFUSING and nds.refused == nds.violations_probed,
        f"{nds.refused}/{nds.violations_probed} non-physical actions set "
        "emit_blocked in its own projection — nothing about a negative "
        "bandwidth or a NaN frequency can be repaired, so it refuses",
    )
    mask = by_id["spectral_mask_ts38104"]
    add(
        "spectral_mask_corrects_or_refuses_by_repairability",
        mask.verdict == REFUSING and mask.repaired > 0 and mask.refused > 0,
        f"{mask.repaired} off-band carriers were shifted back inside the "
        f"band; {mask.refused} carriers wider than the usable channel were "
        "refused. The same invariant does both, decided by whether a "
        "projection exists — which is the rule the whole design follows.",
    )

    # ── 4. THE FINDING: max_eirp never refuses ───────────────────────────
    add(
        "max_eirp_never_refuses",
        summary["over_power_refused"] == 0 and summary["over_power_probed"] > 2000,
        f"{summary['over_power_probed']} over-power actions probed — powers "
        f"from 0.1 dB to 967 dB above the {MAX_EIRP_DBM} dBm ceiling, crossed "
        "with antenna gain, constellation order, PAPR and TBLER so any "
        "interaction that could block the clamp is included. "
        f"REFUSED: {summary['over_power_refused']}. Largest clamp applied: "
        f"{summary['largest_power_clamp_dB']} dB. MaxEirpInvariant.project "
        "subtracts the overage unconditionally, so there is no over-power "
        "action it cannot repair.",
    )

    # ── 5. but it defers rather than fully correcting ────────────────────
    eirp = by_id["max_eirp"]
    add(
        "max_eirp_defers_on_floating_point_residue",
        eirp.verdict == DEFERRING and eirp.deferred == 96,
        f"{eirp.deferred} of {eirp.violations_probed} probes are NOT repaired "
        "by a single pass of its own projection: clamping 33.1 dBm at 60 dBi "
        "lands on tx=-26.999999999999993, whose EIRP is 33.00000000000001 — "
        "margin -7.1e-15 dB, still violated. It neither repairs nor refuses, "
        "so it is DEFERRING: it relies on the Shield's fixed-point loop.",
    )

    # ── 6. the loop does absorb it, in exactly two passes ────────────────
    residue_passes = _passes_needed(RESIDUE_ACTION)
    add(
        "residue_case_needs_two_passes",
        residue_passes == 2
        and _blocked_at(RESIDUE_ACTION, 1)
        and not _blocked_at(RESIDUE_ACTION, 2),
        f"tx=33.1 dBm at 60 dBi converges at max_passes={residue_passes}: "
        f"REFUSED at 1 ({_blocked_at(RESIDUE_ACTION, 1)}), corrected at 2. "
        "The fixed-point loop is what makes this safe, and it is doing real "
        "work here rather than being unused headroom.",
    )

    # ── 7. why nobody found this: overage size predicts nothing ──────────
    big_passes = _passes_needed(BIG_OVERAGE_ACTION)
    add(
        "overage_size_does_not_predict_pass_count",
        big_passes == 1 and residue_passes == 2,
        f"a 7.0 dB overage converges in {big_passes} pass; a 0.1 dB overage "
        f"needs {residue_passes}. What matters is whether the float "
        "subtraction rounds back exactly, not how large the violation is — "
        "so a battery of large, obvious violations will never find the "
        "boundary, which is precisely how the earlier claim was missed.",
    )

    # ── 8. correcting a claim this repository published ──────────────────
    convergence = measure_projection_passes()
    add(
        "convergence_floor_is_two_passes_not_one",
        convergence["passes_needed_max"] == 2
        and convergence["passes_needed_per_scenario"].count(2) == 1,
        "audit/constants/swap_harness.py previously reported that the chain "
        "'converges in <= 1 pass' and was therefore 'INSENSITIVE to "
        "max_passes for any value >= 1'. That was FALSE. Per-scenario "
        f"convergence is now {convergence['passes_needed_per_scenario']} over "
        f"{convergence['scenarios']} scenarios, floor "
        f"{convergence['passes_needed_max']}. Two defects produced the wrong "
        "answer: the battery held only large overages, and the measurement "
        "stopped at the first pair of agreeing dispositions rather than "
        "reading convergence from the settled end (the residue case is "
        "blocked at BOTH 0 and 1, so 'first repeat' concluded 0).",
    )

    # ── 9. the correction §VII CAN claim is real ─────────────────────────
    shield = build_shield()
    disposition = shield.dispose(dict(BIG_OVERAGE_ACTION), {})
    clamped = float(disposition.safe_action["tx_power_dBm"])
    add(
        "over_power_correction_is_real_and_emitted",
        not disposition.certificate.emit_blocked
        and disposition.certificate.projected
        and abs(clamped - MAX_EIRP_DBM) < 1e-9,
        f"a {BIG_OVERAGE_ACTION['tx_power_dBm']} dBm proposal is corrected to "
        f"{clamped} dBm and EMITTED (emit_blocked="
        f"{disposition.certificate.emit_blocked}, projected="
        f"{disposition.certificate.projected}). This is what the "
        "demonstration actually shows, and it is the stronger claim: the "
        "action that reaches the RAN is not the action the planner proposed.",
    )

    # ── 10. so the §VII sentence is unobtainable, not merely unproven ────
    add(
        "section_vii_claim_is_unobtainable_as_written",
        summary["over_power_refused"] == 0,
        f"§VII promises {SECTION_VII_CLAIM!r}. No over-power proposal is "
        "refused, by construction, so this cannot be demonstrated and no "
        "test will produce it. The fix is a wording change, which is the "
        "author's to make — this gate deliberately does not edit the "
        f"proposal. Proposed replacement: {PROPOSED_REWORDING!r}",
    )

    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write the result JSON here")
    args = parser.parse_args(argv)

    checks = run_checks()
    by_id, summary = (lambda r: r)(characterise())
    passed = all(c.passed for c in checks)
    result = {
        "gate": "refusal-semantics",
        "claim": (
            "Refusal is reserved for actions no projection can repair. An "
            "over-power proposal is always repairable, so it is always "
            "corrected and never refused."
        ),
        "section_vii_claim": SECTION_VII_CLAIM,
        "proposed_rewording": PROPOSED_REWORDING,
        "characterisation": {
            "summary": summary,
            "by_invariant": {k: v.to_dict() for k, v in by_id.items()},
        },
        "source_digests": {
            "src/horizon_ric/shield/invariants.py": _sha256(
                REPO / "src" / "horizon_ric" / "shield" / "invariants.py"
            ),
            "src/horizon_ric/shield/shield.py": _sha256(
                REPO / "src" / "horizon_ric" / "shield" / "shield.py"
            ),
            "audit/refusal_semantics.py": _sha256(HERE / "refusal_semantics.py"),
        },
        "checks": [c.to_dict() for c in checks],
        "passed": passed,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    for c in checks:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.id}", file=sys.stderr)
    print(
        f"refusal-semantics: {sum(c.passed for c in checks)}/{len(checks)} "
        "checks passed",
        file=sys.stderr,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
