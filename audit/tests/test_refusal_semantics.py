"""Unit coverage for the refusal-semantics characterisation.

Run with::

    PYTHONPATH=src:agentic/src:ocudu/src:audit/src:audit \\
        /home/user/venv/bin/python -m pytest audit/tests/test_refusal_semantics.py -q
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO, REPO / "audit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from refusal_semantics import (  # noqa: E402
    DEFERRING,
    MAX_EIRP_DBM,
    REFUSING,
    build_shield,
    characterise,
    over_power_actions,
)

from audit.constants.swap_harness import measure_projection_passes  # noqa: E402
from horizon_ric.shield.invariants import MaxEirpInvariant  # noqa: E402
from horizon_ric.shield.shield import Shield, ShieldConfig  # noqa: E402

RESIDUE = {
    "block": "spectrum",
    "frequency_hz": 3.45e9,
    "bandwidth_hz": 20e6,
    "tx_power_dBm": 33.1,
    "antenna_gain_dBi": 60.0,
}


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _blocked_at(action, max_passes):
    base = build_shield()
    return Shield(
        list(base._invariants), ShieldConfig(max_passes=max_passes)
    ).dispose(dict(action)).certificate.emit_blocked


# ── the finding ─────────────────────────────────────────────────────────────


def test_no_over_power_action_is_ever_refused():
    inv = MaxEirpInvariant(max_eirp_dBm=MAX_EIRP_DBM)
    refused = 0
    probed = 0
    for action in over_power_actions():
        if inv.evaluate(action, {}).satisfied:
            continue
        probed += 1
        projected, _ = inv.project(dict(action), {})
        if projected.get("emit_blocked"):
            refused += 1
    assert probed > 2000, "the sweep must actually exercise the space"
    assert refused == 0


def test_the_clamp_is_unconditional_even_at_absurd_power():
    inv = MaxEirpInvariant(max_eirp_dBm=MAX_EIRP_DBM)
    a = dict(RESIDUE, tx_power_dBm=1e6)
    projected, corrections = inv.project(dict(a), {})
    assert corrections, "a violated ceiling must record a correction"
    assert not projected.get("emit_blocked")
    assert projected["tx_power_dBm"] < a["tx_power_dBm"]


# ── the residue, and the correction it forced ───────────────────────────────


def test_residue_case_is_unrepaired_by_a_single_projection():
    inv = MaxEirpInvariant(max_eirp_dBm=MAX_EIRP_DBM)
    assert not inv.evaluate(RESIDUE, {}).satisfied
    projected, _ = inv.project(dict(RESIDUE), {})
    check = inv.evaluate(projected, {})
    assert not check.satisfied, "this is the whole point of the counterexample"
    assert check.margin is not None and -1e-13 < check.margin < 0


def test_residue_case_is_refused_at_one_pass_and_corrected_at_two():
    assert _blocked_at(RESIDUE, 1) is True
    assert _blocked_at(RESIDUE, 2) is False


def test_a_large_overage_converges_in_one_pass():
    # The reason the residue was missed: violation size does not predict the
    # pass count.
    big = dict(RESIDUE, tx_power_dBm=40.0, antenna_gain_dBi=0.0)
    assert _blocked_at(big, 1) is False


def test_convergence_floor_is_two_not_one():
    m = measure_projection_passes()
    assert m["passes_needed_max"] == 2
    assert m["passes_needed_per_scenario"] == [1, 1, 1, 1, 2]


def test_convergence_is_read_from_the_settled_end():
    # The earlier measurement stopped at the first pair of agreeing
    # dispositions. For the residue case those are max_passes 0 and 1 — both
    # blocked — so it concluded zero passes were needed. This pins the shape
    # that made the old method wrong.
    dispositions = [_blocked_at(RESIDUE, mp) for mp in range(4)]
    assert dispositions[0] is True and dispositions[1] is True
    assert dispositions[2] is False and dispositions[3] is False


# ── classification ──────────────────────────────────────────────────────────


def test_classification_matches_the_shipping_chain():
    by_id, _ = characterise()
    assert set(by_id) == set(build_shield().invariant_ids)


def test_every_invariant_is_classified_by_evidence_not_default():
    by_id, _ = characterise()
    unprobed = [i for i, c in by_id.items() if c.violations_probed == 0]
    assert not unprobed, f"classified without evidence: {unprobed}"


def test_max_eirp_is_deferring_and_numeric_sanity_is_refusing():
    by_id, _ = characterise()
    assert by_id["max_eirp"].verdict == DEFERRING
    assert by_id["max_eirp"].refused == 0
    assert by_id["numeric_domain_sanity"].verdict == REFUSING
    assert by_id["numeric_domain_sanity"].repaired == 0


def test_spectral_mask_both_repairs_and_refuses():
    by_id, _ = characterise()
    mask = by_id["spectral_mask_ts38104"]
    assert mask.repaired > 0 and mask.refused > 0


# ── CI wiring ───────────────────────────────────────────────────────────────


def test_ci_runs_this_gate():
    wf = (REPO / ".github" / "workflows" / "audit.yml").read_text(encoding="utf-8")
    assert "verify_refusal_semantics.py" in wf
