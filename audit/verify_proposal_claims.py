#!/usr/bin/env python3
"""Gate — proposal sentences against the result files they cite.

``docs/proposal/validate_docx.py`` checks 89 things about the document and not
one of them is whether a sentence agrees with the JSON behind it. Every finding
below sits in that blind spot, and every one was found by reading a result file
the proposal itself points at.

Six disagreements, each of them arithmetic rather than judgement:

* **Fig. 2(c)** is labelled "peak EIRP (dBm)" and plots 46.0 — but 46.0 is
  ``ru_max_tx_dBm``, a transmit power. The benchmark defines EIRP as
  ``tx_power_dBm + antenna_gain_dBi`` (poisoning_shield_benchmark.py:191), and
  the run's declared gain is 6.0 dBi, so the peak requested EIRP is **52.0 dBm**.
  The figure understates the violation by exactly the antenna gain and puts a
  transmit power and an EIRP ceiling on the same axis.
* **"four benchmarks instead reject physically inconsistent input"** — three do.
  ``scripts/verify_data_dependence.py``'s own docstring says "Three of these
  benchmarks".
* **"up to 85 times the classical baseline"**, attributed to white-box
  perturbation — 84.5x is the ``boundary`` attack, which the evasion suite
  documents as decision-based black-box. The white-box maximum is **28.6x**.
* **"enforcement surrenders 0.0 dB"** — true only where the licence does not
  bind. ``safety_utility_frontier.json`` carries
  ``utility_forgone_at_operational_cap = 0.6526`` and a scope note whose
  purpose is to say the claim is not that the cost is zero.
* **"Eight verification workflows gate every change"** — ten run on
  ``pull_request`` today; twelve files exist.
* **"six 100 MHz subbands"** — six subbands *across* 100 MHz.
  ``carrier_bw_hz`` is 16 MHz.

Why this gate goes green on known-wrong prose
---------------------------------------------
The proposal is not this branch's to edit. So each check passes when the
disagreement is **exactly** as recorded and fails when either side moves —
including when the author fixes the sentence, which is the intended way for
this gate to die. Read ``checks[].data.proposed`` for the wording each one
suggests.

A green run means "the prose and the data disagree in precisely the six ways
we measured", not "the proposal is correct".

Additive: reads the proposal and the committed results, writes nothing to
either.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

CONTENT = REPO / "docs" / "proposal" / "content.py"
MKFIGS = REPO / "docs" / "proposal" / "mkfigs.py"
RESULTS = REPO / "benchmarks" / "results"
DATA_DEP = REPO / "scripts" / "verify_data_dependence.py"
POISON_BENCH = REPO / "benchmarks" / "poisoning_shield_benchmark.py"


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


def proposal_prose() -> str:
    """Every string literal in content.py, concatenated and whitespace-flat.

    The sentences under gate are split across source lines by the formatter, so
    a naive grep of the file misses them. Rendering the module would be the
    other option, but that imports it; this stays inert.
    """
    src = CONTENT.read_text(encoding="utf-8")
    blob = " ".join(re.findall(r'"((?:[^"\\]|\\.)*)"', src))
    return re.sub(r"\s+", " ", blob)


def _load(name: str) -> dict[str, Any]:
    obj = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"{name} is not a JSON object")
    return obj


def run_checks() -> list[Check]:
    prose = proposal_prose()
    checks: list[Check] = []

    def claims(text: str) -> bool:
        return text in prose

    # ── 1. Fig 2(c): a transmit power under an EIRP label ────────────────
    poison = _load("poisoning_shield.json")
    consts = poison["data_provenance"]["declared_constants"]
    tx_max = float(consts["ru_max_tx_dBm"])
    gain = float(consts["antenna_gain_dBi"])
    ceiling = float(consts["max_eirp_dBm"])
    peak_eirp = tx_max + gain

    figsrc = MKFIGS.read_text(encoding="utf-8")
    # The panel-(c) block, from its section comment to the next one.
    m = re.search(r"# ---- \(c\).*?(?=# ---- \(d\))", figsrc, re.S)
    panel_c = m.group(0) if m else ""
    plots_tx = f"[{tx_max}]" in panel_c
    labelled_eirp = "peak EIRP (dBm)" in panel_c
    # The benchmark's own definition, read back rather than assumed.
    bench = POISON_BENCH.read_text(encoding="utf-8")
    eirp_is_sum = bool(
        re.search(
            r"eirp_dBm\s*=\s*float\(action\.get\(\"tx_power_dBm\".*?\+\s*float\("
            r"action\.get\(\"antenna_gain_dBi\"",
            bench,
            re.S,
        )
    )
    checks.append(
        Check(
            "fig2c_plots_a_transmit_power_under_an_eirp_label",
            plots_tx and labelled_eirp and eirp_is_sum and gain > 0,
            f"panel (c) plots {tx_max} dBm on an axis labelled 'peak EIRP "
            f"(dBm)', but the benchmark defines EIRP as tx_power + "
            f"antenna_gain, and the declared gain is {gain} dBi — the peak "
            f"requested EIRP is {peak_eirp} dBm, {peak_eirp - ceiling:g} dB "
            f"over the {ceiling} dBm ceiling, not {tx_max - ceiling:g} dB",
            {
                "plotted_dBm": tx_max,
                "declared_antenna_gain_dBi": gain,
                "actual_peak_eirp_dBm": peak_eirp,
                "licence_ceiling_dBm": ceiling,
                "understated_by_dB": gain,
                "proposed": (
                    "plot the unguarded bar at 52.0 dBm EIRP, or relabel the "
                    "axis 'peak transmit power' and add the gain separately"
                ),
            },
        )
    )

    # ── 2. Fig 2(c) is hardcoded, contradicting the build README ─────────
    hardcoded = re.findall(r"ax\.bar\(\[\d\], \[([\d.]+)\]", panel_c)
    checks.append(
        Check(
            "fig2c_numbers_are_typed_in_not_read_from_the_result",
            sorted(hardcoded) == sorted([str(tx_max), str(ceiling)]),
            f"panel (c) bar heights {hardcoded} are literals in mkfigs.py; "
            f"both happen to equal declared constants today, so the figure is "
            f"not wrong on that count — but it will not follow the JSON if "
            f"either constant changes",
            {
                "literals": hardcoded,
                "proposed": (
                    "read ru_max_tx_dBm and max_eirp_dBm from "
                    "poisoning_shield.json, or drop the docstring and README "
                    "claim that no number in this figure is typed in"
                ),
            },
        )
    )

    # ── 3. three rejecters, not four ─────────────────────────────────────
    dd = DATA_DEP.read_text(encoding="utf-8")
    says_three = "Three of these benchmarks reconstruct" in dd
    checks.append(
        Check(
            "ray_reconstruction_rejecters_is_three_not_four",
            claims("four benchmarks instead reject physically inconsistent")
            and says_three,
            "the proposal says four benchmarks reject physically inconsistent "
            "input; verify_data_dependence.py's own docstring says three",
            {
                "proposed": (
                    "three benchmarks instead reject physically inconsistent "
                    "input"
                )
            },
        )
    )

    # ── 4. 85x is black-box, not white-box ───────────────────────────────
    ev = _load("evasion_suite.json")
    attacks = ev["awgn"]["attacks"]
    ratios = {
        k: float(v["neural_over_classical_x"])
        for k, v in attacks.items()
        if isinstance(v, dict) and "neural_over_classical_x" in v
    }
    headline = max(ratios.values())
    headline_attack = max(ratios, key=lambda k: ratios[k])
    white_box = {k: v for k, v in ratios.items() if k != headline_attack}
    checks.append(
        Check(
            "the_85x_figure_is_the_black_box_boundary_attack",
            claims("85 times the classical baseline")
            and claims("White-box adversarial")
            and headline_attack == "boundary"
            and round(headline, 1) == 84.5,
            f"the proposal attributes {headline:.1f}x to white-box "
            f"perturbation; it is the {headline_attack!r} attack, which the "
            f"suite documents as decision-based black-box. The white-box "
            f"maximum is {max(white_box.values()):.1f}x",
            {
                "ratios_by_attack": ratios,
                "white_box_max": round(max(white_box.values()), 1),
                "proposed": (
                    "Adversarial perturbation drives the neural receiver's "
                    "symbol error up to 85 times the classical baseline (29 "
                    "times for the strongest white-box attack)"
                ),
            },
        )
    )

    # ── 5. "surrenders 0.0 dB" is the non-binding case only ──────────────
    suf = _load("safety_utility_frontier.json")
    forgone = float(suf["utility_forgone_at_operational_cap"])
    anchors = {
        a["anchor_eirp_dbm"]: a["utility_forgone_vs_operational_cap"]
        for a in suf["reference_anchors"]
    }
    checks.append(
        Check(
            "zero_utility_cost_holds_only_where_the_licence_does_not_bind",
            claims("surrenders 0.0 dB") and forgone > 0.0,
            f"utility_forgone_at_operational_cap = {forgone:.4f}, and the "
            f"result's own scope note exists to say the claim is not that the "
            f"cost is zero; against deployable anchors the cost is "
            f"{anchors.get(46.0)} (46 dBm) and {anchors.get(52.0)} (52 dBm)",
            {
                "utility_forgone_at_operational_cap": forgone,
                "anchors": anchors,
                "scope_note": suf.get("scope_note", "")[:400],
                "proposed": (
                    "Where the licence does not bind, enforcement surrenders "
                    "nothing; where it does bind, the cost is explicit and "
                    "measured — 0.20 of served fraction against a 40 W "
                    "small-cell anchor"
                ),
            },
        )
    )

    # ── 6. workflow count ────────────────────────────────────────────────
    wf_dir = REPO / ".github" / "workflows"
    files = sorted(p.name for p in wf_dir.glob("*.yml"))
    on_pr = sorted(
        p.name
        for p in wf_dir.glob("*.yml")
        if "pull_request" in p.read_text(encoding="utf-8")
    )
    checks.append(
        Check(
            "verification_workflow_count_has_grown_past_eight",
            claims("Eight verification workflows") and len(on_pr) != 8,
            f"the proposal says eight; {len(on_pr)} workflow(s) run on "
            f"pull_request and {len(files)} files exist",
            {
                "on_pull_request": on_pr,
                "all_files": files,
                "proposed": f"{len(on_pr)} verification workflows gate every change",
            },
        )
    )

    # ── 7. subband width ─────────────────────────────────────────────────
    carrier_bw = float(consts["carrier_bw_hz"])
    checks.append(
        Check(
            "subbands_are_across_100_mhz_not_100_mhz_each",
            claims("six 100 MHz subbands") and carrier_bw < 100e6,
            f"the proposal reads as six subbands of 100 MHz each; the run's "
            f"declared carrier_bw_hz is {carrier_bw / 1e6:g} MHz",
            {
                "carrier_bw_hz": carrier_bw,
                "proposed": "six subbands across 100 MHz",
            },
        )
    )

    return checks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="verify_proposal_claims.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", help="write the JSON result here")
    args = p.parse_args(argv)

    checks = run_checks()
    passed = all(c.passed for c in checks)
    result = {
        "gate": "proposal-claims",
        "claim": (
            "Six sentences in docs/proposal/content.py disagree with the "
            "result files they cite, in exactly the ways recorded here."
        ),
        "reading_note": (
            "passed=true means the disagreements are unchanged, NOT that the "
            "proposal is correct. Each check carries the wording it suggests "
            "under data.proposed. Fixing a sentence turns this gate red, "
            "which is how it is meant to end."
        ),
        "checks": [c.to_dict() for c in checks],
        "passed": passed,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    for c in checks:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.id}: {c.detail}")
    print(f"proposal-claims: {sum(c.passed for c in checks)}/{len(checks)} recorded")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
