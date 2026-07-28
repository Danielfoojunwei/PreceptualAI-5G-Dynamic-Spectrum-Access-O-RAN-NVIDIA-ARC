"""Checks for the real-DeepMIMO exploration-truncation / graduation evidence.

The committed ``benchmarks/results/exploration_truncation.json`` is the real
4096-receiver run of ``benchmarks/exploration_truncation_loop.py`` on the
licence-gated DeepMIMO ASU 3.5 GHz feature set. The raw feature rows are not
redistributed (see the dataset manifest), so this test validates the committed
real result and its provenance; ``scripts/verify_exploration_truncation.py``
re-checks a freshly re-run result against it. No synthetic data, no synthetic
fixture.

Several of these tests exist because the first published version of this
artifact was wrong. Legality was scored by a benchmark-private
centre-frequency-in-band test that disagreed with the Shield on 28 of 297 arms,
so every illegal / graduation / blocked rate was measured against the wrong
feasible set. The tests below pin the corrected predicate, the zero-data
baseline that makes the (absent) utility gain visible, and the masked control
that shows the graduation curriculum was never necessary.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORRECTION_PENALTY = 0.25  # benchmarks/exploration_truncation_loop.py


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


def test_legality_is_the_shields_own_predicate() -> None:
    result = _result()
    fp = result["feasibility_predicate"]
    # The corrected predicate is the Shield's, and it is stricter than the
    # retracted one: TS 38.104's occupied-bandwidth mask makes the two edge
    # subbands infeasible at every EIRP, which a centre-frequency test misses.
    assert "Shield.is_feasible" in fp["source"]
    assert fp["n_infeasible_arms_shield"] == 241
    assert fp["n_infeasible_arms_superseded"] == 213
    assert fp["arms_where_predicates_disagree"] == 28
    assert fp["disagreement_invariants"] == ["spectral_mask_ts38104"]
    assert result["action_space"]["n_arms"] == 297
    assert result["action_space"]["n_infeasible_arms"] == 241
    assert result["action_space"]["n_feasible_arms"] == 56


def test_illegal_rate_and_projection_rate_are_the_same_quantity() -> None:
    result = _result()
    # Under the Shield's own predicate an action is infeasible IFF the Shield
    # projects it, so these two rates cannot differ. They did in the retracted
    # result (0.878125 vs 0.909375 for reward_only) — that gap was the bug.
    assert result["feasibility_predicate"]["illegal_rate_equals_projection_rate"] is True
    for learner in result["learners"].values():
        summary = learner["closed_loop_summary"]
        assert summary["illegal_proposal_rate"] == summary["projection_rate"]
        assert learner["projection_flag_mismatches"] == 0


def test_learner_is_genuinely_pulled_toward_illegality() -> None:
    result = _result()
    ro = result["learners"]["reward_only"]
    # The unconstrained optimum lies outside the feasible set: the true reward
    # at EIRP 50 dBm is far above the best feasible reward, so exploration is
    # actively pulled over the cap and the Q1 illegal rate is substantial.
    belief = ro["infeasible_belief"]
    assert belief["true_unconstrained_reward_at_eirp_50_dbm"] > result["optimal_feasible_reward"]
    assert ro["illegal_proposal_rate_q1"] > 0.2


def test_the_above_cap_plateau_belongs_to_the_shield_not_the_reward() -> None:
    result = _result()
    geom = result["above_cap_reward_geometry"]
    # RETRACTED CLAIM: "every proposal above the cap returns the identical real
    # reward — a flat plateau with zero gradient" was stated as a property of
    # the reward. It is not. The true served fraction keeps climbing above the
    # cap at ~83% of its sub-cap slope; what is flat is what the learner is
    # SERVED, because the Shield projects every above-cap proposal onto 33 dBm.
    assert geom["true_slope_above_cap_per_db"] > 0.0
    assert 0.8 < geom["true_slope_ratio_above_over_below"] < 0.9
    assert geom["reward_at_eirp_52_dbm"] > geom["reward_at_eirp_33_dbm"]
    assert "claim_retracted" in geom


def test_reward_only_never_graduates() -> None:
    result = _result()
    ro = result["learners"]["reward_only"]
    # The learner is served the identical (capped) reward for every above-cap
    # proposal, so it stays illegal to the end and the Shield stays
    # load-bearing throughout Q4.
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


def test_the_shields_own_mask_makes_the_curriculum_unnecessary() -> None:
    result = _result()
    ca = result["learners"]["correction_aware"]
    sm = result["learners"]["shield_masked"]
    # The free-lunch control. Handed the Shield's analytic feasibility
    # predicate as an action mask, the same bandit is at zero illegal
    # proposals and zero interventions from Q1 — no curriculum, no penalty, no
    # 255 corrections to learn from. correction_aware's "graduation" is
    # therefore the cost of rediscovering a constraint that was already known,
    # not a capability the Shield conferred.
    assert sm["action_mask_applied"] is True
    assert sm["illegal_proposals_blocked"] == 0
    assert sm["shield_interventions_total"] == 0
    assert all(sm[f"illegal_proposal_rate_q{q}"] == 0.0 for q in (1, 2, 3, 4))
    assert sm["graduation_is_vacuous"] is True
    # And it costs nothing in utility — it is strictly ahead on realised regret.
    assert sm["realised_regret"] <= ca["realised_regret"]
    assert ca["shield_interventions_total"] > 0


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
    assert harm["centre_frequency_out_of_band_proposals_total"] > 0
    # The spectral-mask category is strictly larger than the centre-frequency
    # one — that difference is exactly the class of proposal the retracted
    # predicate scored as legal.
    assert (
        harm["occupied_bandwidth_out_of_band_proposals_total"]
        > harm["centre_frequency_out_of_band_proposals_total"]
    )
    assert harm["max_over_cap_excess_db"] >= 15.0
    assert harm["worst_unshielded_eirp_dbm"] > 33.0
    assert harm["shield_masked_blocked"] == 0


def test_truncation_leaves_the_infeasible_region_unobserved() -> None:
    result = _result()
    n_infeasible = result["action_space"]["n_infeasible_arms"]
    assert n_infeasible > 0
    for learner in result["learners"].values():
        # The Shield projects every illegal proposal, so no infeasible bin
        # ever yields a real observation — for any learner. None can know the
        # true above-cap value; the honest claim is ignorance.
        assert learner["unobserved_infeasible_bins"] == n_infeasible


def test_correction_aware_reports_a_shaped_score_not_a_value_estimate() -> None:
    result = _result()
    ca = result["learners"]["correction_aware"]["infeasible_belief"]
    ro = result["learners"]["reward_only"]["infeasible_belief"]
    optimal = result["optimal_feasible_reward"]
    # RETRACTED CLAIM: correction_aware's 0.0974 at EIRP 50 was published as a
    # "value estimate" whose 0.4057 error against the oracle showed it was
    # "even further from the truth". It is neither. It is
    #     optimal_feasible_reward - CORRECTION_PENALTY
    # on an arm that is corrected every time — an arithmetic identity of the
    # penalty, not something the learner discovered. So it is now named
    # shaped_score_*, and no oracle error is reported for it: comparing a
    # penalised score with an unpenalised oracle is a category error.
    assert "shaped_score_at_eirp_50_dbm" in ca
    assert "value_estimate_at_eirp_50_dbm" not in ca
    assert "mean_abs_error_vs_oracle_above_cap" not in ca
    assert ca["shaped_score_identity_holds"] is True
    assert abs(ca["shaped_score_at_eirp_50_dbm"] - (optimal - CORRECTION_PENALTY)) < 1e-6
    # reward_only's number IS an honest (unpenalised) value estimate, so it
    # keeps both fields: it believes the capped reward it was actually served.
    assert abs(ro["value_estimate_at_eirp_50_dbm"] - optimal) < 1e-6
    assert ro["mean_abs_error_vs_oracle_above_cap"] > 0.1


def test_no_learner_beats_the_zero_data_constant_cap_policy() -> None:
    result = _result()
    optimal = result["optimal_feasible_reward"]
    # RETRACTED CLAIM: "realised utility is identical for both (0.3334) against
    # a feasible optimum of 0.3474" was reported without its baseline. A policy
    # that proposes the cap forever — no data, no learning, no exploration —
    # realises 0.3474 exactly. Every learner here is strictly WORSE than it
    # over the full run; the measured utility gain from learning is negative.
    for name, learner in result["learners"].items():
        assert abs(learner["constant_cap_policy_reward"] - optimal) < 1e-3
        assert learner["realised_regret"] > 0.0, f"{name} regret must be reported and positive"
        assert learner["realised_mean_reward"] < learner["constant_cap_policy_reward"]
    # The masked learner pays the smallest exploration cost, because it never
    # wastes a step on an arm the Shield was always going to refuse.
    regrets = {n: la["realised_regret"] for n, la in result["learners"].items()}
    assert regrets["shield_masked"] < regrets["reward_only"]
    assert regrets["shield_masked"] < regrets["correction_aware"]


def test_trust_chain_holds_and_catches_tamper() -> None:
    result = _result()
    tc = result["trust_chain"]
    steps = result["training"]["steps_per_learner"]
    # Every decision of every learner is on the chain; it verifies intact,
    # and once tampered it is refused as training data (load_transitions
    # raises EvidenceIntegrityError) with the tamper pinpointed.
    assert tc["evidence_chain_length"] == len(result["learners"]) * steps
    assert tc["verify_first_broken_index"] == -1
    assert tc["tampered_chain_refused"] is True
    assert tc["verify_after_tamper_index"] not in (None, -1)
