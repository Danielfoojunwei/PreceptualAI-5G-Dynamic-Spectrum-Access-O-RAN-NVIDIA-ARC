#!/usr/bin/env python3
"""Gate — proposal sentences against the result files they cite.

``docs/proposal/validate_docx.py`` checks the document; it has never checked
whether a sentence agrees with the JSON behind it. Seven did not. All seven
have now been corrected at source, and this gate holds the corrections in
place: each check passes on the corrected wording and fails if the old one
comes back or the underlying number moves.

The seven, and what each was:

* **Fig. 1(c)** was labelled "peak EIRP (dBm)" and plotted 46.0 — but 46.0 is
  ``ru_max_tx_dBm``, a transmit power. ``poisoning_shield_benchmark.py``
  defines EIRP as ``tx_power_dBm + antenna_gain_dBi``, and the declared gain is
  6.0 dBi, so the peak requested EIRP is **52.0 dBm**. The figure drew a 19 dB
  violation as 13 dB, understating its own result by exactly the antenna gain.
  Both bar heights are now read from the result rather than typed in.
* **"four benchmarks instead reject physically inconsistent input"** — three
  do, as ``scripts/verify_data_dependence.py``'s own docstring says.
* **"White-box adversarial perturbation ... up to 85 times"** — 84.5x is the
  ``boundary`` attack, which the evasion suite documents as decision-based
  black-box. The white-box maximum is 28.6x.
* **"enforcement surrenders 0.0 dB"** — true only where the licence does not
  bind. ``safety_utility_frontier.json`` carries
  ``utility_forgone_at_operational_cap = 0.6526`` and a scope note whose stated
  purpose is that the claim is not that the cost is zero.
* **"Eight verification workflows"** — eleven run on ``pull_request``.
* **"six 100 MHz subbands"** — six subbands *across* 100 MHz;
  ``carrier_bw_hz`` is 16 MHz.
* **The executive summary** welded "driven by ray-traced propagation for 4096
  receivers" onto the 12/12 ENFORCED result. Those are disjoint evidence
  bases: the A1/xApp run replays authored telemetry.

Falsified rather than asserted: restoring any of the original wordings, or
returning panel (c) to a literal, turns the corresponding check red.

Additive: reads the proposal and the committed results, writes nothing to
either.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

CONTENT = REPO / "docs" / "proposal" / "content.py"
DOCX = (REPO / "docs" / "proposal"
        / "Horizon-RIC_AI-RAN_Call-for-Innovation_Proposal.docx")
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
    """Every word of the BUILT document, flattened.

    Two earlier versions of this function graded the wrong thing, in the same
    direction both times — they searched text that a reader never sees.

    The first matched double-quoted literals in ``content.py`` with a regex and
    was blind to every single-quoted one, so ``not says(...)`` checks passed
    without seeing what they searched. The second walked the module's AST,
    which fixed the quoting but swept in the module docstring, comments-as-
    strings and any other constant no renderer emits: an adversarial reviewer
    put a corrected caption back to its broken form, hid the gate's search
    string in the docstring, and got 9/9 green over a ``.docx`` that carried
    the original defect.

    Both were the same mistake. The claim under gate is about the *document*,
    so the document is what gets read: paragraphs, table cells and inline
    shapes of the built ``.docx``. If the document has not been built, that is
    a failure rather than a silent fallback — a gate on a file that is not
    there proves nothing.
    """
    if not DOCX.exists():
        raise FileNotFoundError(
            f"{DOCX} has not been built; run docs/proposal/build_docx.py. This "
            f"gate reads the rendered document on purpose."
        )
    from docx import Document

    doc = Document(str(DOCX))
    parts = [para.text for para in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return re.sub(r"\s+", " ", " ".join(parts))


def _load(name: str) -> dict[str, Any]:
    obj = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"{name} is not a JSON object")
    return obj


def run_checks() -> list[Check]:
    prose = proposal_prose()
    checks: list[Check] = []

    def says(text: str) -> bool:
        return text in prose

    # ── 1. Fig 1(c): the EIRP is the sum, and the figure now plots it ────
    poison = _load("poisoning_shield.json")
    consts = poison["data_provenance"]["declared_constants"]
    tx_max = float(consts["ru_max_tx_dBm"])
    gain = float(consts["antenna_gain_dBi"])
    ceiling = float(consts["max_eirp_dBm"])
    peak_eirp = tx_max + gain

    bench = POISON_BENCH.read_text(encoding="utf-8")
    eirp_is_sum = bool(
        re.search(
            r"eirp_dBm\s*=\s*float\(action\.get\(\"tx_power_dBm\".*?\+\s*float\("
            r"action\.get\(\"antenna_gain_dBi\"",
            bench,
            re.S,
        )
    )
    # Render the figure and read the bars that were actually drawn. Grepping
    # the source for a literal was defeated by one space — `ax.bar([0], [ 46.0 ])`
    # passed a check whose own docstring promised it would fail, while the
    # annotation still read 52.0 over a bar drawn at 46.0.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(REPO / "docs" / "proposal"))
    import mkfigs  # noqa: E402

    mkfigs.enforcement_figure(*mkfigs._load())
    fig = mkfigs.LAST_ENFORCEMENT_FIGURE
    if fig is None:  # pragma: no cover — mkfigs always sets it
        raise RuntimeError("mkfigs did not record the rendered figure")
    panel_c_ax = fig.axes[2]
    drawn = [round(patch.get_height(), 6) for patch in panel_c_ax.patches]
    ylabel = panel_c_ax.get_ylabel()
    panel_d_title = fig.axes[3].get_title()
    plt.close("all")

    checks.append(
        Check(
            "fig1c_plots_eirp_read_from_the_result_not_a_typed_transmit_power",
            eirp_is_sum and gain > 0 and drawn == [peak_eirp, ceiling]
            and "EIRP" in ylabel,
            f"panel (c) draws bars at {drawn} on an axis labelled {ylabel!r}; "
            f"the benchmark's own EIRP is ru_max_tx_dBm + antenna_gain_dBi = "
            f"{peak_eirp} dBm against the {ceiling} dBm ceiling",
            {
                "peak_eirp_dBm": peak_eirp,
                "licence_ceiling_dBm": ceiling,
                "bars_as_drawn": drawn,
                "y_axis_label": ylabel,
                "was": "labelled 'peak EIRP' but plotted 46.0, a transmit "
                       "power — a 19 dB violation drawn as 13 dB",
            },
        )
    )

    # ── 2. the caption says where 52.0 comes from ────────────────────────
    checks.append(
        Check(
            "the_caption_derives_52_dBm_rather_than_asserting_it",
            says("52.0 dBm is the 46.0 dBm transmit-power request plus the "
                 "declared 6.0 dBi antenna gain"),
            "Fig. 1's caption states the arithmetic, so a reader can check the "
            "bar against the benchmark's own definition of EIRP",
        )
    )

    # ── 3. three rejecters, not four ─────────────────────────────────────
    dd = DATA_DEP.read_text(encoding="utf-8")
    checks.append(
        Check(
            "ray_reconstruction_rejecters_stated_as_three",
            says("three instead reject physically inconsistent input outright")
            and "Three of these benchmarks reconstruct" in dd
            and not says("four benchmarks instead reject"),
            "the proposal says three, matching verify_data_dependence.py's "
            "own docstring",
            {"was": "four benchmarks instead reject physically inconsistent input"},
        )
    )

    # ── 4. 85x attributed to the black-box attack ────────────────────────
    ev = _load("evasion_suite.json")
    attacks = ev["awgn"]["attacks"]
    ratios = {
        k: float(v["neural_over_classical_x"])
        for k, v in attacks.items()
        if isinstance(v, dict) and "neural_over_classical_x" in v
    }
    headline_attack = max(ratios, key=lambda k: ratios[k])
    # Named explicitly rather than "everything except the headline". The old
    # form assumed `boundary` was the only black-box attack; `transfer` is one
    # too, at 28.5x, and one nudge would have had this gate print a black-box
    # number as "the white-box maximum".
    WHITE_BOX = {"fgsm", "bim", "mim"}
    white_box_max = max(v for k, v in ratios.items() if k in WHITE_BOX)
    checks.append(
        Check(
            "the_85x_figure_is_no_longer_attributed_to_a_white_box_attack",
            not says("White-box adversarial perturbation")
            # ...and neither is the FIGURE, which titled panel (d) "white-box
            # attack" over the boundary result — the same misattribution, in
            # the artefact the proposal actually embeds.
            and "white-box" not in panel_d_title.lower()
            and headline_attack == "boundary"
            and round(ratios[headline_attack], 1) == 84.5,
            f"{ratios[headline_attack]:.1f}x is the {headline_attack!r} attack, "
            f"which the suite documents as decision-based black-box; the "
            f"white-box maximum is {white_box_max:.1f}x",
            {
                "ratios_by_attack": ratios,
                "white_box_attacks": sorted(WHITE_BOX),
                "white_box_max": round(white_box_max, 1),
                "panel_d_title": panel_d_title,
                "was": "White-box adversarial perturbation ... up to 85 times, "
                       "and a panel titled 'white-box attack' over the "
                       "black-box boundary result",
            },
        )
    )

    # ── 5. zero utility cost stated conditionally ────────────────────────
    suf = _load("safety_utility_frontier.json")
    forgone = float(suf["utility_forgone_at_operational_cap"])
    anchors = {
        a["anchor_eirp_dbm"]: a["utility_forgone_vs_operational_cap"]
        for a in suf["reference_anchors"]
    }
    checks.append(
        Check(
            "utility_cost_is_stated_conditionally_with_the_binding_case_priced",
            not says("surrenders 0.0 dB")
            and says("Where the licence does not bind, enforcement surrenders "
                     "nothing")
            and says("0.20 of served fraction against a 40 W small-cell anchor")
            and forgone > 0.0,
            f"the non-binding case says 'surrenders nothing' and the binding "
            f"case is priced at {anchors.get(46.0)} against the 46 dBm anchor; "
            f"utility_forgone_at_operational_cap = {forgone:.4f}",
            {
                "utility_forgone_at_operational_cap": forgone,
                "anchors": anchors,
                "was": "Utility is not the price ... enforcement surrenders 0.0 dB",
            },
        )
    )

    # ── 6. workflow count, and the right predicate ──────────────────────
    # Counted by parsing `on.pull_request`, not by grepping for the string:
    # the old method counted a workflow that merely mentioned it in a comment
    # and missed the .yaml spelling entirely. The claim was also reworded —
    # it used to say these workflows "gate every change", and only three do.
    # The other eight are path-scoped and stay green on a change they do not
    # cover.
    import yaml

    wf_dir = REPO / ".github" / "workflows"
    on_pr, gate_everything = [], []
    for wf in sorted(list(wf_dir.glob("*.yml")) + list(wf_dir.glob("*.yaml"))):
        try:
            spec = yaml.safe_load(wf.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        # PyYAML resolves an unquoted `on:` key to the boolean True.
        triggers = spec.get("on", spec.get(True)) or {}
        if not isinstance(triggers, dict) or "pull_request" not in triggers:
            continue
        on_pr.append(wf.name)
        pr = triggers["pull_request"] or {}
        if not isinstance(pr, dict) or not pr.get("paths"):
            gate_everything.append(wf.name)
    checks.append(
        Check(
            "verification_workflow_count_matches_the_workflows_that_run",
            says(f"{len(on_pr)} verification workflows run on every pull request")
            or says("Eleven verification workflows run on every pull request")
            and len(on_pr) == 11,
            f"{len(on_pr)} workflow(s) run on pull_request; only "
            f"{len(gate_everything)} of them are unfiltered and therefore "
            f"actually gate every change ({', '.join(gate_everything)})",
            {
                "on_pull_request": on_pr,
                "unfiltered": gate_everything,
                "was": "Eight verification workflows gate every change — wrong "
                       "count, and the wrong predicate: eight of the eleven "
                       "are path-scoped",
            },
        )
    )

    # ── 7. subband width ─────────────────────────────────────────────────
    carrier_bw = float(consts["carrier_bw_hz"])
    checks.append(
        Check(
            "subbands_stated_as_across_100_mhz",
            says("six subbands across 100 MHz")
            and not says("six 100 MHz subbands")
            and carrier_bw < 100e6,
            f"six subbands across 100 MHz, consistent with the run's declared "
            f"carrier_bw_hz of {carrier_bw / 1e6:g} MHz",
            {"carrier_bw_hz": carrier_bw, "was": "six 100 MHz subbands"},
        )
    )

    # ── 8. the executive summary keeps the two evidence bases apart ──────
    checks.append(
        Check(
            "the_summary_separates_the_a1_result_from_the_ray_traced_benchmarks",
            says("Two independent results")
            and not says("ENFORCED by the production ric-plt/a1 mediator — "
                         "driven by ray-traced propagation"),
            "the A1/xApp interop run replays authored telemetry; the "
            "ray-traced measurements drive the benchmarks, and the summary now "
            "says so rather than welding them into one sentence",
            {"was": "12 of 12 policies ENFORCED ... — driven by ray-traced "
                    "propagation for 4096 receivers"},
        )
    )

    # ── 9. §VII's refusal promise matches what the Shield does ───────────
    checks.append(
        Check(
            "section_vii_promises_correction_and_refusal_separately",
            says("an over-power proposal corrected to the EIRP ceiling before "
                 "emission and a non-physical proposal refused outright")
            and not says("over-power proposal refused rather than emitted"),
            "over-power is corrected, not refused — audit/refusal_semantics.py "
            "probes 2688 over-power actions and finds zero refusals — and the "
            "refusal that IS proven comes from numeric_domain_sanity",
            {"was": "a Shield-corrected over-power proposal refused rather "
                    "than emitted"},
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
            "Nine statements in docs/proposal/content.py and mkfigs.py agree "
            "with the result files they cite. Each was wrong before; the "
            "wording it replaced is recorded under data.was."
        ),
        "reading_note": (
            "passed=true means the corrections are in place and the numbers "
            "behind them still hold. Restoring an old wording, or moving a "
            "constant the corrected sentence quotes, turns this red."
        ),
        "checks": [c.to_dict() for c in checks],
        "passed": passed,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    for c in checks:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.id}: {c.detail}")
    print(f"proposal-claims: {sum(c.passed for c in checks)}/{len(checks)} passed")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
