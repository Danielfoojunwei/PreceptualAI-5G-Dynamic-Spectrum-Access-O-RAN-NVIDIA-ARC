#!/usr/bin/env python3
"""Gate G9 — the Shield's band comes from the radio, not from a keyboard.

The claim under gate:

    The frequency envelope the Shield enforces is DERIVED, by the normative
    TS 38.104 formulae, from the same OCUDU cell configuration the gNB itself
    boots from — and the derivation is wired to the environment variables the
    unmodified pipeline already reads, so no product source has to change for
    it to take effect.

Why this needed a gate rather than a patch
------------------------------------------
``horizon_ric.rapp.pipeline.PipelineConfig`` hand-types the band as
3.40-3.50 GHz. The OCUDU cell in ``gnb_ru_ran550_tdd_n78_100mhz_4x2.yml``
transmits 3508.18-3608.18 MHz. Those two intervals **do not overlap at all**.
Check ``default_envelope_excludes_the_real_carrier`` states that as a finding
rather than quietly correcting it, and check
``default_envelope_admits_what_the_real_cell_must_refuse`` shows the sharp
edge: a 3.45 GHz action is *inside* the hand-typed default and *outside* the
real carrier — the old envelope would have waved through the one emission the
real cell is not licensed to make.

Eleven checks. Two are load-bearing:

* ``env_var_names_are_read_from_pipeline_source`` — the derivation is only
  wired to the pipeline if the names agree, and asserting agreement against
  names copied into this file would prove nothing. It reads them out of
  ``pipeline.py`` itself.
* ``every_committed_cell_derives`` — a derivation that works on the one
  config I aimed it at is a demo. This runs all of them.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE / "src"))

from horizon_ocudu.cell_config import (  # noqa: E402
    CellConfigError,
    derive_eirp_ceiling_dBm,
    derive_shield_envelope,
    n_rb,
    nr_arfcn_to_hz,
    parse_gnb_config,
)

from horizon_ric.rapp.pipeline import PipelineConfig  # noqa: E402
from horizon_ric.shield.shield import default_terrestrial_shield  # noqa: E402

OCUDU_CONFIGS = REPO / "third_party" / "ocudu" / "configs"
REFERENCE_CONFIG = OCUDU_CONFIGS / "gnb_ru_ran550_tdd_n78_100mhz_4x2.yml"
PIPELINE_SOURCE = REPO / "src" / "horizon_ric" / "rapp" / "pipeline.py"

# TS 38.104 §5.4.2.1 worked by hand for the reference cell:
#   F_REF = 3 000 000 000 + 15 000 × (637212 − 600 000) = 3 558 180 000 Hz
EXPECTED_CENTRE_HZ = 3_558_180_000.0
EXPECTED_N_RB = 273
EXPECTED_TX_BW_HZ = 273 * 12 * 30_000  # 98.28 MHz


class Check:
    def __init__(self, cid: str, passed: bool, detail: str) -> None:
        self.id = cid
        self.passed = passed
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "passed": self.passed, "detail": self.detail}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pipeline_env_names() -> set[str]:
    """Every ``HORIZON_SHIELD_*`` name pipeline.py passes to os.environ.

    Parsed out of the AST, not grepped and not copied: the point is that this
    gate learns the names from the product source, so renaming one there
    fails here instead of silently unwiring the derivation.
    """
    tree = ast.parse(PIPELINE_SOURCE.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("HORIZON_SHIELD_"):
                names.add(node.value)
    return names


def _action(freq_hz: float, bw_hz: float, tx_dBm: float) -> dict[str, Any]:
    """A minimal action in the shape the pipeline builds (pipeline.py:328-330)."""
    return {
        "frequency_hz": freq_hz,
        "bandwidth_hz": bw_hz,
        "tx_power_dBm": tx_dBm,
    }


def _mask_violated(shield: Any, action: dict[str, Any]) -> bool:
    """True iff the spectral-mask invariant specifically rejects ``action``."""
    return any(
        c.invariant_id == "spectral_mask_ts38104"
        for c in shield.violations(action, {})
    )


def run_checks() -> list[Check]:
    checks: list[Check] = []

    def add(cid: str, passed: bool, detail: str) -> None:
        checks.append(Check(cid, passed, detail))

    # ── 1. the ARFCN formula, exactly ────────────────────────────────────
    centre = nr_arfcn_to_hz(637212)
    add(
        "arfcn_formula_is_ts38104_exact",
        centre == EXPECTED_CENTRE_HZ,
        f"NR-ARFCN 637212 -> {centre / 1e6:.3f} MHz "
        f"(expected {EXPECTED_CENTRE_HZ / 1e6:.3f}); "
        "F_REF = 3000 MHz + 15 kHz x (637212 - 600000), TS 38.104 §5.4.2.1",
    )

    # An ARFCN off the end of the raster must raise, not extrapolate.
    try:
        nr_arfcn_to_hz(9_999_999)
        raster_guarded = False
        raster_detail = "an out-of-range ARFCN was accepted"
    except CellConfigError as exc:
        raster_guarded = True
        raster_detail = f"out-of-range ARFCN refused: {exc}"
    add("arfcn_raster_is_bounded", raster_guarded, raster_detail)

    # ── 2. the transmission-bandwidth table ──────────────────────────────
    prb = n_rb(100, 30)
    add(
        "n_rb_matches_ts38104_table_5_3_2_1",
        prb == EXPECTED_N_RB,
        f"100 MHz at 30 kHz SCS -> {prb} PRB (expected {EXPECTED_N_RB})",
    )
    try:
        n_rb(100, 15)
        table_guarded = False
        table_detail = "an N/A bandwidth/SCS combination was accepted"
    except CellConfigError as exc:
        table_guarded = True
        table_detail = f"undefined combination refused: {exc}"
    add("undefined_bandwidth_scs_is_refused", table_guarded, table_detail)

    # ── 3. the reference cell ────────────────────────────────────────────
    cell = parse_gnb_config(REFERENCE_CONFIG)
    env = derive_shield_envelope(cell)
    ok_env = (
        env.centre_hz == EXPECTED_CENTRE_HZ
        and env.n_rb == EXPECTED_N_RB
        and abs(env.transmission_bandwidth_hz - EXPECTED_TX_BW_HZ) < 1.0
        and env.band.number == 78
    )
    add(
        "reference_cell_derives_its_carrier",
        ok_env,
        f"{REFERENCE_CONFIG.name}: band n{env.band.number}, "
        f"{env.band_lo_hz / 1e6:.2f}-{env.band_hi_hz / 1e6:.2f} MHz, "
        f"{env.n_rb} PRB, transmission bandwidth "
        f"{env.transmission_bandwidth_hz / 1e6:.2f} MHz",
    )

    # ── 4. THE FINDING: the hand-typed default is not this carrier ───────
    d_lo, d_hi = PipelineConfig.band_lo_hz, PipelineConfig.band_hi_hz
    disjoint = d_hi <= env.band_lo_hz or d_lo >= env.band_hi_hz
    add(
        "default_envelope_excludes_the_real_carrier",
        disjoint,
        f"PipelineConfig defaults police {d_lo / 1e6:.0f}-{d_hi / 1e6:.0f} MHz; "
        f"the OCUDU cell transmits {env.band_lo_hz / 1e6:.2f}-"
        f"{env.band_hi_hz / 1e6:.2f} MHz. The intervals are DISJOINT — the "
        "hand-typed band was never this radio's band. Recorded as a finding, "
        "not silently corrected: the fix is to export the derived values, "
        "which needs no product-source change.",
    )

    # ── 5. and it admits exactly what the real cell must refuse ──────────
    default_shield = default_terrestrial_shield(band_lo_hz=d_lo, band_hi_hz=d_hi)
    derived_shield = default_terrestrial_shield(
        band_lo_hz=env.band_lo_hz, band_hi_hz=env.band_hi_hz
    )
    # 3.45 GHz, 10 MHz wide: comfortably inside the hand-typed default, and
    # 58 MHz below the real carrier's lower edge.
    trap = _action(3.45e9, 10e6, 20.0)
    admitted_by_default = not _mask_violated(default_shield, trap)
    refused_by_derived = _mask_violated(derived_shield, trap)
    add(
        "default_envelope_admits_what_the_real_cell_must_refuse",
        admitted_by_default and refused_by_derived,
        "a 3450.0 MHz / 10 MHz emission passes the hand-typed envelope's "
        f"spectral mask (admitted={admitted_by_default}) and is rejected by "
        f"the OCUDU-derived one (refused={refused_by_derived}); the derived "
        "envelope is strictly the more truthful of the two",
    )

    # ── 6. and it admits what the real cell may in fact emit ─────────────
    legal = _action(env.centre_hz, 20e6, 20.0)
    add(
        "derived_envelope_admits_an_in_carrier_action",
        not _mask_violated(derived_shield, legal),
        f"a {env.centre_hz / 1e6:.2f} MHz / 20 MHz emission at the carrier "
        "centre passes the derived envelope's spectral mask",
    )

    # ── 7. the wiring — names read from the product source ───────────────
    pipeline_names = _pipeline_env_names()
    derived_names = set(env.env())
    wired = derived_names <= pipeline_names
    add(
        "env_var_names_are_read_from_pipeline_source",
        wired,
        f"pipeline.py declares {sorted(pipeline_names)}; the derivation emits "
        f"{sorted(derived_names)}; subset={wired}. Names are parsed from the "
        "pipeline AST, so a rename there fails here rather than silently "
        "unwiring the derivation.",
    )

    # ── 8. every committed cell, not just the one I aimed at ─────────────
    derived: list[str] = []
    refused: list[str] = []
    for path in sorted(OCUDU_CONFIGS.glob("*.yml")):
        try:
            c = parse_gnb_config(path)
        except CellConfigError:
            continue  # no cell_cfg: cu_cp.yml and friends, correctly skipped
        try:
            e = derive_shield_envelope(c)
        except CellConfigError as exc:
            refused.append(f"{path.name}: {exc}")
            continue
        derived.append(
            f"{path.name}: n{e.band.number} "
            f"{e.band_lo_hz / 1e6:.2f}-{e.band_hi_hz / 1e6:.2f} MHz "
            f"({e.n_rb} PRB)"
        )
    add(
        "every_committed_cell_derives",
        len(derived) >= 9 and not refused,
        f"{len(derived)} committed OCUDU cell configuration(s) derived a "
        f"carrier; {len(refused)} refused. "
        + ("; ".join(derived) if not refused else f"REFUSED: {refused}"),
    )

    # ── 9. EIRP is not invented ──────────────────────────────────────────
    try:
        derive_eirp_ceiling_dBm(env, antenna_gain_dBi=float("nan"))
        nan_guarded = False
        nan_detail = "a non-finite antenna gain was accepted"
    except CellConfigError as exc:
        nan_guarded = True
        nan_detail = str(exc)
    # And a cell with no SSB power must refuse outright.
    bare = parse_gnb_config(OCUDU_CONFIGS / "gnb_rf_b210_fdd_srsUE.yml")
    bare_env = derive_shield_envelope(bare)
    try:
        derive_eirp_ceiling_dBm(bare_env, antenna_gain_dBi=8.0)
        bare_guarded = False
        bare_detail = "an EIRP ceiling was produced from a config with no SSB power"
    except CellConfigError as exc:
        bare_guarded = True
        bare_detail = str(exc)
    add(
        "eirp_is_never_invented",
        nan_guarded and bare_guarded,
        "antenna gain is keyword-only with no default and no OCUDU config "
        f"carries one. non-finite gain: {nan_detail} | "
        f"config without ssb_block_power_dbm: {bare_detail}",
    )

    # ── 10. a misconfigured carrier is refused, not laundered ────────────
    from dataclasses import replace as _replace

    # Same 100 MHz cell, but an ARFCN that puts it past n78's upper edge.
    off_band = _replace(cell, dl_arfcn=653_334)  # 3800.01 MHz centre
    try:
        derive_shield_envelope(off_band)
        containment = False
        containment_detail = "a carrier outside its declared band was accepted"
    except CellConfigError as exc:
        containment = True
        containment_detail = str(exc)
    add("carrier_must_fit_its_declared_band", containment, containment_detail)

    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write the result JSON here")
    args = parser.parse_args(argv)

    checks = run_checks()
    passed = all(c.passed for c in checks)
    result = {
        "gate": "G9",
        "claim": (
            "The Shield's frequency envelope is derived by the normative "
            "TS 38.104 formulae from the same OCUDU cell configuration the "
            "gNB boots from, and is wired to the environment variables the "
            "unmodified pipeline already reads."
        ),
        "reference_config": str(REFERENCE_CONFIG.relative_to(REPO)),
        "source_digests": {
            "ocudu/src/horizon_ocudu/cell_config.py": _sha256(
                HERE / "src" / "horizon_ocudu" / "cell_config.py"
            ),
            "src/horizon_ric/rapp/pipeline.py": _sha256(PIPELINE_SOURCE),
            "src/horizon_ric/shield/invariants.py": _sha256(
                REPO / "src" / "horizon_ric" / "shield" / "invariants.py"
            ),
            str(REFERENCE_CONFIG.relative_to(REPO)): _sha256(REFERENCE_CONFIG),
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
        f"G9: {sum(c.passed for c in checks)}/{len(checks)} checks passed",
        file=sys.stderr,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
