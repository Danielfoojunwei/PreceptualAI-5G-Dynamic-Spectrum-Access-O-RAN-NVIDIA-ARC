"""Checks for the real-DeepMIMO safety-utility frontier evidence.

The committed ``benchmarks/results/safety_utility_frontier.json`` is the real
4096-receiver run of ``benchmarks/safety_utility_frontier_loop.py`` on the
licence-gated DeepMIMO ASU 3.5 GHz feature set. The raw feature rows are not
redistributed (see the dataset manifest), so this test validates the committed
real result and its byte-level provenance;
``scripts/verify_safety_utility_frontier.py`` re-checks a fresh rebuild against
it with tolerance bands. No synthetic data, no synthetic fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _result() -> dict:
    return json.loads(
        (ROOT / "benchmarks/results/safety_utility_frontier.json").read_text()
    )


def test_result_is_bound_to_the_real_deepmimo_build() -> None:
    result = _result()
    manifest = json.loads(
        (ROOT / "datasets/deepmimo_asu_3p5/manifest.json").read_text()
    )
    # The frontier run consumed exactly the canonical, checksum-pinned build.
    assert result["features_sha256"] == manifest["features_sha256"]
    assert result["source_tree_sha256"] == manifest["source_tree_sha256"]
    assert result["receivers"] == manifest["sampled_receivers"] == 4096
    assert "not over-the-air" in result["data_kind"]
    # The reward physics is the shared substrate's, not a re-derivation.
    assert abs(result["env"]["noise_dbm"] - (-93.9897)) < 1e-3
    assert result["env"]["reward"].startswith("served fraction")


def test_frontier_is_monotone_and_binds_at_the_operational_cap() -> None:
    result = _result()
    frontier = result["frontier"]
    caps = [f["eirp_cap_dbm"] for f in frontier]
    assert set([20.0, 26.0, 33.0, 40.0, 46.0, 52.0]).issubset(caps)
    best = [f["best_feasible_reward"] for f in frontier]
    # A looser cap never hurts: best feasible served fraction is non-decreasing.
    assert all(b2 >= b1 for b1, b2 in zip(best, best[1:]))
    # utility_forgone is measured against the true unconstrained optimum (1.0,
    # full service) and the constraint genuinely binds at the operational cap.
    assert result["unconstrained_optimum"]["reward"] == 1.0
    for f in frontier:
        assert abs(f["utility_forgone"] - (1.0 - f["best_feasible_reward"])) < 1e-9
    op = next(f for f in frontier if f["eirp_cap_dbm"] == 33.0)
    assert result["operational_cap_dbm"] == 33.0
    assert result["utility_forgone_at_operational_cap"] == op["utility_forgone"]
    assert result["utility_forgone_at_operational_cap"] > 0.0
    # The Shield cannot beat the feasible optimum; on this deterministic loop
    # it realises exactly the best feasible reward at every cap.
    for f in frontier:
        assert abs(f["realised_mean_reward"] - f["best_feasible_reward"]) < 1e-9


def test_marginal_price_of_safety_is_positive_and_consistent() -> None:
    result = _result()
    frontier = {f["eirp_cap_dbm"]: f for f in result["frontier"]}
    marginal = result["marginal_price_per_db"]
    assert len(marginal) == len(frontier) - 1
    for m in marginal:
        lo, hi = frontier[m["from_cap"]], frontier[m["to_cap"]]
        expected = (hi["best_feasible_reward"] - lo["best_feasible_reward"]) / (
            m["to_cap"] - m["from_cap"]
        )
        assert abs(m["served_fraction_lost_per_db"] - expected) < 1e-6
    # Tightening the cap loses served fraction between every adjacent pair on
    # this data (the spec-level invariant is "positive somewhere"; the committed
    # real run is strictly positive everywhere).
    assert all(m["served_fraction_lost_per_db"] > 0.0 for m in marginal)


def test_every_executed_action_is_legal_under_its_own_cap() -> None:
    result = _result()
    band_lo = result["env"]["band_lo_hz"]
    band_hi = result["env"]["band_hi_hz"]
    for f in result["frontier"]:
        cap = f["eirp_cap_dbm"]
        # The optimiser always proposed above the cap; the Shield projected
        # every single step, and nothing ever emitted above the cap or out of
        # band.
        assert f["projection_rate"] == 1.0
        assert f["illegal_proposal_rate"] == 1.0
        assert f["mean_proposed_eirp_dbm"] > cap
        assert f["executed_eirp_dbm"] <= cap + 1e-9
        assert f["max_executed_eirp_dbm"] <= cap + 1e-9
        assert band_lo <= f["executed_frequency_hz"] <= band_hi
    free = result["free_constraint_case"]
    assert free["max_executed_eirp_dbm"] <= result["operational_cap_dbm"] + 1e-9
    assert band_lo <= free["executed_frequency_hz"] <= band_hi


def test_free_constraint_case_costs_nothing() -> None:
    result = _result()
    free = result["free_constraint_case"]
    # The task optimum is feasible, so the Shield applies no correction and
    # forgoes zero utility: safety is free exactly when the optimum is legal.
    assert free["fixed_eirp_dbm"] == 32.0
    assert free["projection_rate"] == 0.0
    assert free["guard_refused_rate"] == 0.0
    assert free["corrected"] is False
    assert free["utility_cost"] == 0.0
    assert free["realised_mean_reward"] == free["task_optimum_reward"]
    assert free["best_subband"] in free["candidate_subbands"]
    # Cross-evidence: the committed DSA benchmark reached the same conclusion
    # over 4096 best-subband decisions — cite it and re-check the source file.
    cross = free["committed_cross_evidence"]
    dsa = json.loads((ROOT / "benchmarks/results/deepmimo_dsa.json").read_text())
    best = dsa["strategies"]["best_subband"]
    assert best["illegal_emits_after_shield"] == 0
    assert best["mean_regret_db"] == 0.0
    assert cross["illegal_emits_after_shield"] == best["illegal_emits_after_shield"]
    assert cross["mean_regret_db"] == best["mean_regret_db"]
    assert cross["decisions"] == best["decisions"] == 4096


def test_trust_chain_holds_and_catches_tamper() -> None:
    result = _result()
    tc = result["trust_chain"]
    # Every closed-loop step is on the hash chain; the chain verifies intact
    # and replays; after a single-byte tamper of record #1 on disk the replay
    # gate refuses the chain and re-verification pinpoints the tamper.
    assert tc["evidence_chain_length"] >= 2
    assert tc["verify_first_broken_index"] == -1
    assert tc["replayed_transitions_intact"] == tc["evidence_chain_length"]
    assert tc["tampered_chain_refused"] is True
    assert tc["verify_after_tamper_index"] not in (None, -1)
