#!/usr/bin/env python3
"""Gate G2 — a second band, and every constant that is not the band edges.

The claim under gate:

    Moving the Shield to a second operating band in a different frequency
    range changes the derived band edges and NOTHING ELSE. Every safety
    constant — the EIRP ceiling, the protected-slice floor, the PAPR limit,
    the neural-RX tolerance, the projection budget — is byte-identical
    between the two, and the chain behaves structurally the same in both.

§V defines G2 as "every published gate reproduces without retuning any
constant, or the discrepancy is published". The vehicle WP2 gives it is a
second scenario *and band*. This gate covers the band half in full and states
the scenario half plainly as not done.

Why the band half is the part with teeth
----------------------------------------
"Without retuning any constant" is only meaningful if you can say which
constants exist and what they are. ``audit/constants/inventory.json`` answers
that — 22 of them, AST-derived from source. This gate builds the Shield twice,
once per band, and asserts that the only thing that differs is the pair of
numbers G9 *derives* from a cell configuration. A constant that had been
quietly nudged to make the second band work would show up here as a
difference.

Moving to FR2 was not a formality. The FR1 transmission-bandwidth table has no
entry for 100 MHz at 120 kHz SCS, so the second band was **refused** until
TS 38.104 Table 5.3.2-2 was added — check ``fr2_carrier_refused_by_fr1_table``
reproduces that refusal, so the FR2 support is load-bearing rather than
decorative.

WHAT THIS DOES NOT DO
---------------------
It does not add a second ray-traced dataset. WP2 asks for "an independent
ray-traced scenario with genuine frequency selectivity, at a different
carrier"; the DeepMIMO ``o1_28`` archive is 6.8 GB and did not fit alongside
the OCUDU build on this host. ``second_scenario_is_disclosed_as_absent``
records that rather than letting the band change stand in for it.
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
for p in (REPO / "src", REPO, HERE / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from horizon_ocudu.cell_config import (  # noqa: E402
    CellConfigError,
    OcuduCell,
    derive_shield_envelope,
    n_rb,
    parse_gnb_config,
)

from horizon_ric.shield.shield import default_terrestrial_shield  # noqa: E402

INVENTORY = REPO / "audit" / "constants" / "inventory.json"
FR1_CONFIG = REPO / "third_party" / "ocudu" / "configs" / (
    "gnb_ru_ran550_tdd_n78_100mhz_4x2.yml"
)

# The second band. n257 is FR2 (26.5-29.5 GHz); ARFCN 2079167 puts the carrier
# centre at 28.00008 GHz, which is where DeepMIMO ray-traces its `*_28`
# scenarios. 100 MHz at 120 kHz SCS is the standard FR2 NR configuration.
SECOND_BAND_CELL = OcuduCell(
    band=257,
    dl_arfcn=2079167,
    channel_bandwidth_mhz=100,
    common_scs_khz=120,
    source_path="<n257 28 GHz cell, WP2 second band>",
    pci=1,
    plmn="00101",
    tac=7,
    nof_antennas_dl=4,
)

# The constants that must NOT move when the band does. Read from the live
# Shield rather than named by value here, so this gate cannot pass by agreeing
# with a number typed into itself.
BAND_DERIVED_KEYS = frozenset({"band_lo_hz", "band_hi_hz"})


class Check:
    def __init__(self, cid: str, passed: bool, detail: str) -> None:
        self.id, self.passed, self.detail = cid, passed, detail

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "passed": self.passed, "detail": self.detail}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _invariant_parameters(shield: Any) -> dict[str, dict[str, Any]]:
    """Every numeric field of every invariant in the chain, by invariant id.

    Reflected off the live objects, so a parameter added upstream appears here
    automatically instead of being silently excluded from the comparison.
    """
    out: dict[str, dict[str, Any]] = {}
    for inv in shield._invariants:
        fields: dict[str, Any] = {}
        for name, value in vars(inv).items():
            if name.startswith("_") or name == "id":
                continue
            if isinstance(value, (int, float, str, bool)):
                fields[name] = value
        out[inv.id] = fields
    return out


def _mask_violated(shield: Any, action: dict[str, Any]) -> bool:
    return any(
        c.invariant_id == "spectral_mask_ts38104"
        for c in shield.violations(action, {})
    )


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

    # ── 1. two bands, both derived from a cell rather than typed ─────────
    fr1_cell = parse_gnb_config(FR1_CONFIG)
    fr1 = derive_shield_envelope(fr1_cell)
    fr2 = derive_shield_envelope(SECOND_BAND_CELL)
    add(
        "two_bands_derive_from_cell_configurations",
        fr1.band.number == 78
        and fr2.band.number == 257
        and fr1.n_rb == 273
        and fr2.n_rb == 66,
        f"band 1: n{fr1.band.number} "
        f"{fr1.band_lo_hz / 1e6:.2f}-{fr1.band_hi_hz / 1e6:.2f} MHz, "
        f"{fr1.n_rb} PRB at {fr1_cell.common_scs_khz} kHz (FR1). "
        f"band 2: n{fr2.band.number} "
        f"{fr2.band_lo_hz / 1e9:.5f}-{fr2.band_hi_hz / 1e9:.5f} GHz, "
        f"{fr2.n_rb} PRB at {SECOND_BAND_CELL.common_scs_khz} kHz (FR2). "
        f"Carrier separation {(fr2.centre_hz - fr1.centre_hz) / 1e9:.2f} GHz.",
    )

    # ── 2. FR2 support is load-bearing, not decorative ──────────────────
    try:
        n_rb(100, 120)  # FR1 table
        fr1_refused = False
        fr1_detail = "the FR1 table accepted an FR2-only configuration"
    except CellConfigError as exc:
        fr1_refused = True
        fr1_detail = str(exc)
    add(
        "fr2_carrier_refused_by_fr1_table",
        fr1_refused,
        "100 MHz at 120 kHz SCS has no FR1 entry, so the second band was "
        f"REFUSED until Table 5.3.2-2 was added: {fr1_detail}",
    )
    add(
        "the_two_tables_disagree_where_they_overlap",
        n_rb(50, 60) == 65 and n_rb(50, 60, fr2=True) == 66,
        f"50 MHz at 60 kHz SCS is a valid key in BOTH tables and they give "
        f"different answers: FR1 {n_rb(50, 60)} PRB, FR2 "
        f"{n_rb(50, 60, fr2=True)} PRB. Merging them would silently mis-size "
        "a carrier by one resource block, which is why they are separate.",
    )

    # ── 3. THE GATE: nothing but the band edges moved ───────────────────
    shield_1 = default_terrestrial_shield(
        band_lo_hz=fr1.band_lo_hz, band_hi_hz=fr1.band_hi_hz
    )
    shield_2 = default_terrestrial_shield(
        band_lo_hz=fr2.band_lo_hz, band_hi_hz=fr2.band_hi_hz
    )
    params_1 = _invariant_parameters(shield_1)
    params_2 = _invariant_parameters(shield_2)

    differing: dict[str, list[str]] = {}
    for inv_id, fields in params_1.items():
        other = params_2.get(inv_id, {})
        moved = [
            k
            for k, v in fields.items()
            if other.get(k) != v and k not in BAND_DERIVED_KEYS
        ]
        if moved:
            differing[inv_id] = moved
    add(
        "no_constant_moved_except_the_derived_band_edges",
        not differing and params_1.keys() == params_2.keys(),
        f"compared every numeric parameter of all {len(params_1)} invariants "
        f"across the two bands. Differing (excluding {sorted(BAND_DERIVED_KEYS)}): "
        f"{differing or 'NONE'}. The band edges themselves DO differ, and they "
        "are the two values G9 derives from the cell rather than tunes.",
    )

    # And the band edges really did change, or the check above is vacuous.
    add(
        "the_band_edges_actually_changed",
        params_1["spectral_mask_ts38104"]["band_lo_hz"]
        != params_2["spectral_mask_ts38104"]["band_lo_hz"],
        "a comparison that excludes the band edges proves nothing unless the "
        "band edges moved: "
        f"{params_1['spectral_mask_ts38104']['band_lo_hz'] / 1e9:.5f} GHz -> "
        f"{params_2['spectral_mask_ts38104']['band_lo_hz'] / 1e9:.5f} GHz",
    )

    # ── 4. and the chain behaves the same way in both ───────────────────
    behaviours = {}
    for name, env, shield in (("n78", fr1, shield_1), ("n257", fr2, shield_2)):
        in_carrier = {
            "block": "spectrum",
            "frequency_hz": env.centre_hz,
            "bandwidth_hz": 20e6,
            "tx_power_dBm": 20.0,
        }
        out_of_carrier = dict(in_carrier, frequency_hz=env.band_lo_hz - 50e6)
        disposition = shield.dispose(dict(out_of_carrier), {})
        behaviours[name] = {
            "in_carrier_passes": not _mask_violated(shield, in_carrier),
            "out_of_carrier_violates": _mask_violated(shield, out_of_carrier),
            "out_of_carrier_projected_back": (
                env.band_lo_hz
                <= float(disposition.safe_action["frequency_hz"])
                <= env.band_hi_hz
            ),
        }
    same = behaviours["n78"] == behaviours["n257"]
    add(
        "the_chain_behaves_identically_in_both_bands",
        same and all(behaviours["n78"].values()),
        f"n78: {behaviours['n78']}; n257: {behaviours['n257']}. Identical="
        f"{same}. In both, an in-carrier emission passes, an emission 50 MHz "
        "below the lower edge violates the mask, and projection returns it "
        "inside the carrier.",
    )

    # ── 5. the inventory is the thing that says what a constant IS ──────
    inventory_ok = INVENTORY.exists()
    n_constants = 0
    if inventory_ok:
        inv_doc = json.loads(INVENTORY.read_text(encoding="utf-8"))
        rows = inv_doc.get("constants") or inv_doc.get("rows") or []
        n_constants = len(rows)
    add(
        "the_constant_set_is_enumerated_not_assumed",
        inventory_ok and n_constants >= 20,
        f"audit/constants/inventory.json enumerates {n_constants} gate-bearing "
        "constants, AST-derived from source and drift-guarded. 'Without "
        "retuning any constant' is only checkable because that list exists.",
    )

    # ── 6. the half that is NOT done, said plainly ──────────────────────
    add(
        "second_scenario_is_disclosed_as_absent",
        True,
        "NOT DONE: WP2 also asks for an independent ray-traced scenario with "
        "genuine frequency selectivity at the second carrier. The DeepMIMO "
        "`o1_28` archive reached 6.8 GB before being abandoned — it did not "
        "fit alongside the OCUDU build on this host (3.2 GB free at the "
        "point it was killed). This gate covers the BAND change only. The "
        "committed ray-traced data is still the single 3.5 GHz "
        "`asu_campus_3p5` scenario.",
    )

    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write the result JSON here")
    args = parser.parse_args(argv)

    checks = run_checks()
    passed = all(c.passed for c in checks)
    result = {
        "gate": "G2",
        "claim": (
            "Moving the Shield to a second operating band in a different "
            "frequency range changes the derived band edges and nothing else; "
            "every safety constant is byte-identical between the two."
        ),
        "covers": "the band half of WP2",
        "does_not_cover": (
            "the second ray-traced scenario; the committed dataset is still "
            "the single 3.5 GHz asu_campus_3p5"
        ),
        "source_digests": {
            "ocudu/src/horizon_ocudu/cell_config.py": _sha256(
                HERE / "src" / "horizon_ocudu" / "cell_config.py"
            ),
            "src/horizon_ric/shield/invariants.py": _sha256(
                REPO / "src" / "horizon_ric" / "shield" / "invariants.py"
            ),
            "audit/constants/inventory.json": _sha256(INVENTORY),
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
        f"G2: {sum(c.passed for c in checks)}/{len(checks)} checks passed",
        file=sys.stderr,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
