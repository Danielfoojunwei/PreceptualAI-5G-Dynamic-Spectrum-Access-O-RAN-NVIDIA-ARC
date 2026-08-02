"""Tests for the gate-bearing constant inventory and swap harness (gate G2).

Three properties are asserted:

1. ``inventory.json`` regenerates identically from source (the drift guard —
   same pattern as OCUDU's ``extract_catalogue``).
2. Every constant's value read from the committed ``inventory.json`` equals the
   value the *running* code imports from source (no transcription rot).
3. The swap harness demonstrably flips the gate/check each covered constant
   feeds — including the four named ones: slice floor (0.20), EIRP ceiling
   (33.0 dBm), aggregate absolute floor (20e6 Hz), and ShieldConfig.max_passes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from audit.constants import inventory as inv
from audit.constants import swap_harness as swap

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. Drift guard: committed JSON == fresh regeneration from source.
# ---------------------------------------------------------------------------
def test_inventory_json_regenerates_identically():
    committed = inv.INVENTORY_JSON.read_text(encoding="utf-8")
    fresh = inv.inventory_json_text()
    assert committed == fresh, (
        "audit/constants/inventory.json is STALE relative to source — regenerate "
        "with: python audit/constants/inventory.py --out audit/constants/inventory.json"
    )


def test_inventory_is_nonempty_and_well_formed():
    data = inv.load_committed_inventory()
    assert data["schema"] == inv.SCHEMA
    assert data["constant_count"] == len(data["constants"]) == len(inv.CONSTANT_SPECS)
    for row in data["constants"]:
        assert row["symbol"]
        assert (REPO_ROOT / row["file"]).is_file()
        assert isinstance(row["line"], int) and row["line"] > 0
        assert row["unit"]


# ---------------------------------------------------------------------------
# 2. Every committed value equals the live value imported from source.
# ---------------------------------------------------------------------------
def _committed_row(symbol: str, container: str):
    data = inv.load_committed_inventory()
    for row in data["constants"]:
        if row["symbol"] == symbol and row["container"] == container:
            return row
    raise AssertionError(f"no committed row for {container}.{symbol}")


@pytest.mark.parametrize("spec", inv.CONSTANT_SPECS, ids=lambda s: f"{s.container}.{s.symbol}")
def test_committed_value_matches_live_source(spec):
    row = _committed_row(spec.symbol, spec.container)
    live = inv.live_value(spec)
    assert row["value"] == pytest.approx(live) if isinstance(live, float) else row["value"] == live, (
        f"{spec.container}.{spec.symbol}: committed {row['value']!r} != live {live!r}"
    )


@pytest.mark.parametrize("spec", inv.CONSTANT_SPECS, ids=lambda s: f"{s.container}.{s.symbol}")
def test_committed_line_points_at_the_literal(spec):
    """The recorded line actually contains the symbol's assignment in source."""
    row = _committed_row(spec.symbol, spec.container)
    source_line = (REPO_ROOT / spec.file).read_text(encoding="utf-8").splitlines()[row["line"] - 1]
    assert spec.symbol in source_line, (
        f"{spec.file}:{row['line']} does not mention {spec.symbol!r}: {source_line!r}"
    )


def test_ast_and_import_paths_agree():
    """resolve_record (AST) and live_value (import) are independent — assert they agree."""
    for spec in inv.CONSTANT_SPECS:
        ast_value = inv.resolve_record(spec).value
        live = inv.live_value(spec)
        if isinstance(live, float):
            assert ast_value == pytest.approx(live)
        else:
            assert ast_value == live


# ---------------------------------------------------------------------------
# 3. The four named constants: perturbation flips the dependent gate/check.
# ---------------------------------------------------------------------------
def test_slice_floor_flips_the_gate():
    r = swap.swap_slice_floor()
    assert r["behavioural"]["satisfied_at_committed"] is True
    assert r["behavioural"]["satisfied_at_perturbed"] is False
    assert r["sha256"]["live_matches_pin"] is True
    assert r["sha256"]["flipped"] is True
    assert r["flipped"] is True


def test_eirp_ceiling_flips_the_gate():
    r = swap.swap_eirp_ceiling()
    assert r["behavioural"]["satisfied_at_committed"] is True
    assert r["behavioural"]["satisfied_at_perturbed"] is False
    assert r["sha256"]["live_matches_pin"] is True
    assert r["sha256"]["flipped"] is True
    assert r["flipped"] is True


def test_aggregate_absolute_floor_flips_the_gate():
    r = swap.swap_aggregate_floor()
    assert r["behavioural"]["satisfied_at_committed"] is True
    assert r["behavioural"]["satisfied_at_perturbed"] is False
    assert r["sha256"]["flipped"] is True
    assert r["flipped"] is True


def test_max_passes_flips_and_records_headroom_finding():
    r = swap.swap_max_passes()
    # Behavioural: a fixable action is safe at the committed value, blocked at 0.
    assert r["behavioural"]["emit_blocked_at_committed"] is False
    assert r["behavioural"]["emit_blocked_at_perturbed"] is True
    # SHA-256 pin: editing the literal 8 breaks G1's pin of shield.py.
    assert r["sha256"]["live_matches_pin"] is True
    assert r["sha256"]["flipped"] is True
    # The honest finding: 8 is headroom, chain converges in <=1 pass.
    assert r["finding"]["passes_needed_max"] <= 1
    assert r["flipped"] is True


def test_ocudu_ran_parameter_id_flips_g8_check():
    r = swap.swap_ocudu_min_prb_ratio()
    assert r["behavioural"]["present_at_committed"] is True
    assert r["behavioural"]["present_at_perturbed"] is False
    assert r["flipped"] is True


def test_every_swap_flips_its_gate():
    results = swap.run_all()
    for name, r in results.items():
        assert r["flipped"] is True, f"swap {name!r} did not flip its gate: {r}"


# ---------------------------------------------------------------------------
# 4. Non-vacuity of the sha256 approach: the perturbed literal is real & unique.
# ---------------------------------------------------------------------------
def test_sha256_swaps_target_a_unique_existing_literal():
    for name in ("slice_floor", "eirp_ceiling", "aggregate_floor", "max_passes"):
        sha = swap.SWAPS[name]()["sha256"]
        assert sha["occurrences_in_source"] == 1, (
            f"{name}: literal {sha['constant_literal']!r} occurs "
            f"{sha['occurrences_in_source']} times, not exactly once"
        )
        assert sha["perturbed_digest"] != sha["live_digest"]


# ---------------------------------------------------------------------------
# 5. Every covered constant is actually in the inventory (harness <-> inventory).
# ---------------------------------------------------------------------------
def test_harness_constants_are_inventoried():
    symbols = {row["symbol"] for row in inv.load_committed_inventory()["constants"]}
    for required in ("floor", "max_eirp_dBm", "min_capacity_hz", "max_passes", "MIN_PRB_POLICY_RATIO"):
        assert required in symbols, f"{required} covered by the harness but not inventoried"


def test_finding_constant_has_no_dependent_gate():
    """SliceQuota.max_ratio is recorded as sensitive to no gate — an honest finding."""
    row = _committed_row("max_ratio", "SliceQuota")
    assert row["dependent_gates"] == []
    assert "no published gate" in row["note"].lower()


def test_inventory_file_is_valid_python_ast_source_paths():
    """Every inventoried file parses (guards against a typo'd path in the spec)."""
    for spec in inv.CONSTANT_SPECS:
        ast.parse((REPO_ROOT / spec.file).read_text(encoding="utf-8"))
