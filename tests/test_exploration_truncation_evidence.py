"""Checks for the real-DeepMIMO exploration-truncation / graduation evidence.

The committed ``benchmarks/results/exploration_truncation.json`` is the real
4096-receiver run of ``benchmarks/exploration_truncation_loop.py`` on the
licence-gated DeepMIMO ASU 3.5 GHz feature set. The raw feature rows are not
redistributed (see the dataset manifest), so this test validates the committed
real result and its provenance; ``scripts/verify_exploration_truncation.py``
re-checks a freshly re-run result against it. No synthetic data, no synthetic
fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _result() -> dict:
    return json.loads(
        (ROOT / "benchmarks/results/exploration_truncation.json").read_text()
    )


def test_result_is_bound_to_the_real_deepmimo_build() -> None:
    result = _result()
    manifest = json.loads(
        (ROOT / "datasets/deepmimo_asu_3p5/manifest.json").read_text()
    )
    # The run consumed exactly the canonical, checksum-pinned build.
    assert result["features_sha256"] == manifest["features_sha256"]
    assert result["source_tree_sha256"] == manifest["source_tree_sha256"]
    assert result["receivers"] == manifest["sampled_receivers"] == 4096
    assert "not over-the-air" in result["data_kind"]
    # Measured anchors, not assumptions.
    assert abs(result["env"]["noise_dbm"] - (-93.9897)) < 0.01
    assert result["env"]["eirp_cap_dbm"] == 33.0
    assert result["env"]["band_lo_hz"] == 3.45e9
    assert result["env"]["band_hi_hz"] == 3.55e9


def test_learner_is_genuinely_pulled_toward_illegality() -> None:
    result = _result()
    ro = result["learners"]["reward_only"]
    # The unconstrained optimum lies outside the feasible set: the true reward
    # at EIRP 50 dBm is far above the best feasible reward, so exploration is
    # actively pulled over the cap and the Q1 illegal rate is substantial.
    belief = ro["infeasible_belief"]
    assert belief["true_unconstrained_reward_at_eirp_50_dbm"] > result["optimal_feasible_reward"]
    assert ro["illegal_proposal_rate_q1"] > 0.2


def test_reward_only_never_graduates() -> None:
    result = _result()
    ro = result["learners"]["reward_only"]
    # Every above-cap proposal returns the identical (capped) reward — a flat
    # plateau with zero gradient — so the truncated learner stays illegal to
    # the end and the Shield stays load-bearing throughout Q4.
    assert ro["graduated"] is False
    assert ro["illegal_proposal_rate_q4"] > 0.5
    assert ro["illegal_proposal_rate_q4"] >= ro["illegal_proposal_rate_q1"] - 0.05
    assert ro["shield_interventions_q4"] > 0


def test_correction_aware_graduates() -> None:
    result = _result()
    ro = result["learners"]["reward_only"]
    ca = result["learners"]["correction_aware"]
    # Told (via the chain's projected/guard_refused flags) that it was
    # corrected, the learner drives its own illegal-proposal rate to ~0 and
    # the Shield stops having to intervene.
    assert ca["graduated"] is True
    assert ca["illegal_proposal_rate_q4"] < 0.05
    assert ca["illegal_proposal_rate_q4"] < 0.25 * ca["illegal_proposal_rate_q1"]
    assert ca["illegal_proposal_rate_q4"] < ro["illegal_proposal_rate_q4"]
    assert ca["shield_interventions_total"] < ro["shield_interventions_total"]
    assert ca["shield_interventions_q4"] == 0


def test_shield_kept_every_executed_action_legal() -> None:
    result = _result()
    harm = result["counterfactual_harm_prevented"]
    for learner in result["learners"].values():
        # Safety holds unconditionally, however badly the learner behaves.
        assert learner["max_executed_eirp_dbm"] <= 33.0 + 1e-9
        assert learner["executed_out_of_band_steps"] == 0
    # And the Shield had real harm to prevent: over a thousand illegal actions
    # never reached the RAN, the worst of them 19 dB over the EIRP cap.
    assert harm["illegal_actions_blocked_total"] > 0
    assert harm["over_cap_proposals_total"] > 0
    assert harm["out_of_band_proposals_total"] > 0
    assert harm["max_over_cap_excess_db"] >= 15.0
    assert harm["worst_unshielded_eirp_dbm"] > 33.0


def test_truncation_leaves_the_infeasible_region_unobserved() -> None:
    result = _result()
    n_infeasible = result["action_space"]["n_infeasible_arms"]
    assert n_infeasible > 0
    for learner in result["learners"].values():
        # The Shield projects every illegal proposal, so no infeasible bin
        # ever yields a real observation — for either learner. Neither can
        # know the true above-cap value; the honest claim is ignorance, and
        # both learners' value estimates are far from the oracle.
        assert learner["unobserved_infeasible_bins"] == n_infeasible
        belief = learner["infeasible_belief"]
        gap = abs(
            belief["value_estimate_at_eirp_50_dbm"]
            - belief["true_unconstrained_reward_at_eirp_50_dbm"]
        )
        assert gap > 0.1
        assert belief["mean_abs_error_vs_oracle_above_cap"] > 0.1


def test_realised_utility_is_shield_rescued_for_both() -> None:
    result = _result()
    optimal = result["optimal_feasible_reward"]
    ro = result["learners"]["reward_only"]
    ca = result["learners"]["correction_aware"]
    # Both realise close to the feasible optimum (the Shield rescues every
    # illegal proposal onto the cap); neither exceeds it.
    for learner in (ro, ca):
        assert 0.0 < learner["realised_mean_reward"] <= optimal + 1e-9
        assert learner["realised_mean_reward"] > 0.8 * optimal


def test_trust_chain_holds_and_catches_tamper() -> None:
    result = _result()
    tc = result["trust_chain"]
    steps = result["training"]["steps_per_learner"]
    # Every decision of both learners is on the chain; it verifies intact,
    # and once tampered it is refused as training data (load_transitions
    # raises EvidenceIntegrityError) with the tamper pinpointed.
    assert tc["evidence_chain_length"] == 2 * steps
    assert tc["verify_first_broken_index"] == -1
    assert tc["tampered_chain_refused"] is True
    assert tc["verify_after_tamper_index"] not in (None, -1)
