"""Checks for the real-DeepMIMO credit-assignment evidence.

The committed ``benchmarks/results/credit_assignment.json`` is the real
4096-receiver run of ``benchmarks/credit_assignment_loop.py`` on the
licence-gated DeepMIMO ASU 3.5 GHz feature set: the same tabular bandit trained
under four credit regimes against the Shield-wrapped spectrum environment, in
two action-space arms (unmasked, and masked by the Shield's own feasibility
predicate). The raw feature rows are not redistributed (see the dataset
manifest), so this test validates the committed real result and its byte-level
provenance; the ``realdata`` CI workflow rebuilds the data deterministically and
re-runs the loop end to end. No synthetic data.

These tests were rewritten after the adversarial post-mortem
(``deploy/shield-learning/ERRATA.md``). Several assertions they used to make were
vacuous and are now asserted the other way round:

* ``value_mae_infeasible == 0.0`` for the projection/correction-aware regimes was
  an EMPTY-SET artifact, not accuracy. It must now be ``None`` with an explicit
  abstention flag.
* ``argmax_tie_count == 20`` was asserted as evidence about the Shield. It is a
  pure function of the grid top. It is now asserted only as the *before* half of
  the masking measurement.
* No test claimed the learner beat doing nothing. The zero-data constant-cap
  baseline is now asserted to tie the best learner exactly, and to beat every
  learner over the full run.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SHIELDED = ("naive_proposed_credit", "projection_aware", "correction_aware")
ALL_REGIMES = SHIELDED + ("unconstrained_oracle",)
ARMS = ("unmasked", "shield_masked")


def _result() -> dict:
    return json.loads((ROOT / "benchmarks/results/credit_assignment.json").read_text())


def _unmasked() -> dict:
    return _result()["arms"]["unmasked"]["regimes"]


def _masked() -> dict:
    return _result()["arms"]["shield_masked"]["regimes"]


# --------------------------------------------------------------------------
# Provenance and environment
# --------------------------------------------------------------------------


def test_result_is_bound_to_the_real_deepmimo_build() -> None:
    result = _result()
    manifest = json.loads((ROOT / "datasets/deepmimo_asu_3p5/manifest.json").read_text())
    # The run consumed exactly the canonical, checksum-pinned build.
    assert result["features_sha256"] == manifest["features_sha256"]
    assert result["source_tree_sha256"] == manifest["source_tree_sha256"]
    assert result["receivers"] == manifest["sampled_receivers"] == 4096
    assert "not over-the-air" in result["data_kind"]


def test_environment_anchors_are_the_measured_ones() -> None:
    env = _result()["env"]
    # Thermal noise floor at 20 MHz + 7 dB NF; the regulatory EIRP cap; and the
    # best feasible served fraction the cap allows on the real channels.
    assert abs(env["noise_dbm"] - (-94.0)) < 0.1
    assert env["eirp_cap_dbm"] == 33.0
    assert env["bandwidth_hz"] == 20e6
    assert env["sinr_threshold_db"] == 0.0
    assert "served fraction" in env["reward"]
    assert abs(env["optimal_feasible_reward"] - 0.3474) < 1e-3


def test_shield_genuinely_binds_and_safety_has_a_price() -> None:
    result = _result()
    truth = result["true_unconstrained_value_by_eirp"]
    env = result["env"]
    # Reward is monotone increasing in EIRP across the grid, so the
    # unconstrained optimum is outside the feasible set: the projection binds.
    eirps = sorted(int(e) for e in truth)
    vals = [truth[str(e)] for e in eirps]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert truth["52"] > env["optimal_feasible_reward"] + 0.2
    assert result["safety_utility_gap"] > 0.2


# --------------------------------------------------------------------------
# THE HARD GUARANTEE — the claim that survived the post-mortem
# --------------------------------------------------------------------------


def test_no_illegal_action_was_ever_executed_in_any_arm_or_regime() -> None:
    arms = _result()["arms"]
    for arm in ARMS:
        for name in ALL_REGIMES:
            r = arms[arm]["regimes"][name]
            assert r["max_executed_eirp_dbm"] == 33.0, (arm, name)
            assert r["executed_out_of_band_steps"] == 0, (arm, name)


def test_trust_chain_holds_replays_and_catches_tamper() -> None:
    result = _result()
    tc = result["trust_chain"]
    rounds = result["learner"]["rounds_per_regime"]
    # Every decision of every arm and regime is on the hash chain, the chain
    # verifies intact, and the training tables rebuilt from the verify-gated
    # replay buffer match online training bit-for-bit.
    assert tc["evidence_chain_length"] == len(ARMS) * len(ALL_REGIMES) * rounds
    assert tc["replay_transitions"] == tc["evidence_chain_length"]
    assert tc["verify_first_broken_index"] == -1
    assert tc["replay_matches_online"] is True
    # After a tamper of record #1 on disk, replay REFUSES to produce training
    # data and re-verification pinpoints the tamper at its exact index.
    assert tc["tampered_chain_refused"] is True
    assert tc["verify_after_tamper_index"] == 1


def test_sample_record_carries_both_actions_for_credit_assignment() -> None:
    sample = json.loads(
        (ROOT / "benchmarks/results/credit_assignment_sample_record.json").read_text()
    )
    action = sample["record"]["chosen_action"]
    # The evidence schema records BOTH the proposed and the executed action, so
    # a replayed learner can attribute reward to what was executed.
    for key in (
        "requested_frequency_hz",
        "requested_tx_power_dBm",
        "safe_frequency_hz",
        "safe_tx_power_dBm",
        "reward",
        "projected",
        "guard_refused",
    ):
        assert key in action
    assert sample["record"]["decision_id"].startswith("credit-")
    assert len(sample["hash"]) == 64


# --------------------------------------------------------------------------
# P1 — the feasibility mask is the Shield's own predicate
# --------------------------------------------------------------------------


def test_feasibility_mask_comes_from_the_shield_not_a_restatement() -> None:
    am = _result()["action_space_masking"]
    assert "Shield.is_feasible" in am["mask_source"]
    # The mask is not merely asserted to be the Shield's: the loop checks, at
    # run time, that it agrees bin-for-bin with what the Shield actually
    # projects. That is what stops a mask from silently drifting.
    assert am["mask_agrees_with_projection_on_every_bin"] is True
    assert am["mask_matches_analytic_eirp_cap"] is True
    assert am["feasible_bins"] == 14 and am["infeasible_bins"] == 19


def test_masking_the_action_space_removes_the_whole_pathology() -> None:
    result = _result()
    am = result["action_space_masking"]
    unmasked, masked = _unmasked(), _masked()

    # BEFORE: the naive regime's terminal value table cannot distinguish the
    # legal cap from any illegal EIRP — a 20-way argmax tie — so it proposes
    # illegal actions to the very end.
    assert am["argmax_tie_count_before_after"]["naive_proposed_credit"]["unmasked"] == 20
    assert unmasked["naive_proposed_credit"]["illegal_proposals"] == 201
    assert unmasked["naive_proposed_credit"]["infeasible_bins_claimed"] == 19

    # AFTER: masked bins are never proposed, so they never enter the value
    # table. The tie collapses, illegal proposals go to zero, and no regime
    # claims a value for an infeasible bin. The credit rule stops mattering.
    for name in ALL_REGIMES:
        assert masked[name]["argmax_tie_count"] == 1, name
        assert masked[name]["illegal_proposals"] == 0, name
        assert masked[name]["infeasible_bins_claimed"] == 0, name
        assert masked[name]["projection_rate_first_quarter"] == 0.0, name
        assert masked[name]["projection_rate_last_quarter"] == 0.0, name
        assert masked[name]["argmax_proposed_eirp_dbm"] == 33.0, name


def test_masking_is_defence_in_depth_not_a_substitute() -> None:
    result = _result()
    am = result["action_space_masking"]
    assert "does not replace the guarantee" in am["defence_in_depth_note"]
    # The unmasked arm still executed nothing illegal despite proposing 201
    # illegal actions: the projection operator, not the mask, is the guarantee.
    unmasked = _unmasked()
    assert unmasked["naive_proposed_credit"]["illegal_proposals"] > 0
    assert unmasked["naive_proposed_credit"]["max_executed_eirp_dbm"] == 33.0


def test_masking_costs_realised_reward_because_clipping_was_subsidising_us() -> None:
    """Masking makes realised regret WORSE, and that is honest, not a bug.

    Unmasked, 20 of the 33 grid bins clip onto the single optimal executed
    action, so uniform exploration hits the optimum with probability 0.606.
    Masking drops that to 1/14 = 0.071, so the learner pays the normal price of
    exploring. The projection operator was silently subsidising the unmasked
    arm's realised reward.
    """
    am = _result()["action_space_masking"]
    regret = am["mean_realised_regret_over_deployable_regimes"]
    assert regret["shield_masked"] > regret["unmasked"]
    assert am["exploration_cost_of_masking"] > 0.0
    assert abs(am["exploration_cost_of_masking"] - 0.016486) < 1e-3


def test_legality_is_judged_by_the_shield_and_the_substrate_agreement_is_checked() -> None:
    """The shared substrate's ``illegal_without_shield`` uses a centre-frequency
    test that is wrong at the band edges. This loop runs one interior subband, so
    the two predicates happen to agree — but the agreement is measured and
    reported, not assumed. (Handoff: the substrate flag still needs fixing for
    the loops that do sweep the band edges.)
    """
    arms = _result()["arms"]
    for arm in ARMS:
        for name in ALL_REGIMES:
            r = arms[arm]["regimes"][name]
            assert r["illegal_predicate_agrees_with_substrate"] is True, (arm, name)
            assert r["illegal_proposals"] == r["illegal_proposals_substrate_predicate"]


# --------------------------------------------------------------------------
# P2 — the correction penalty must actually change behaviour
# --------------------------------------------------------------------------


def test_correction_penalty_is_load_bearing_in_the_unmasked_arm() -> None:
    """The committed run re-trains regime C at penalty 0 and at 0.2 with the
    SAME rng and compares the proposal trajectory. Before the tie-break was made
    uniform this probe came back identical at every penalty from 0.0 to 5.0 —
    the penalty was inert and the published "the penalty fixes the behaviour"
    claim was false (ERRATA correction 2).
    """
    probe = _result()["correction_penalty_probe"]["unmasked"]
    assert probe["proposal_trajectory_differs"] is True
    assert probe["realised_scalar_differs"] is True
    assert probe["penalty_bites"] is True
    off, on = probe["penalty_off"], probe["penalty_on"]
    # Switching the penalty off costs the Shield 6.5x the correction load.
    assert off["projected_steps"] == 208 and on["projected_steps"] == 32
    assert off["projection_rate_last_quarter"] == 0.933333
    assert on["projection_rate_last_quarter"] == 0.016667
    assert off["proposal_trajectory_sha256"] != on["proposal_trajectory_sha256"]


def test_the_penalty_probe_does_not_key_on_argmax_tie_count() -> None:
    """A gate written against ``argmax_tie_count`` would pass while proving
    nothing: the tie count moved 20 -> 1 with the penalty even in the era when
    the trajectory did not move at all. The field is published so the trap is
    visible; the criterion deliberately ignores it.
    """
    probe = _result()["correction_penalty_probe"]["unmasked"]
    assert probe["penalty_off"]["argmax_tie_count"] != probe["penalty_on"]["argmax_tie_count"]
    assert "NOT used as the criterion" in probe["note"]


def test_correction_penalty_is_provably_a_no_op_once_the_action_space_is_masked() -> None:
    """With the Shield's mask applied, nothing is ever projected, so the penalty
    has nothing to penalise. Regime C is then a strict no-op — measured, not
    argued: identical trajectory hash, identical realised reward.
    """
    probe = _result()["correction_penalty_probe"]["shield_masked"]
    assert probe["penalty_bites"] is False
    assert probe["proposal_trajectory_differs"] is False
    assert probe["realised_scalar_differs"] is False
    off, on = probe["penalty_off"], probe["penalty_on"]
    assert off["proposal_trajectory_sha256"] == on["proposal_trajectory_sha256"]
    assert off["projected_steps"] == on["projected_steps"] == 0
    assert off["realised_mean_reward"] == on["realised_mean_reward"]


def test_the_tie_break_rule_is_uniform_and_documented() -> None:
    rule = _result()["learner"]["tie_break_rule"]
    assert "uniform across all regimes" in rule
    assert "seeded random choice" in rule


# --------------------------------------------------------------------------
# P3(a) — accuracy claims are not satisfiable by abstention
# --------------------------------------------------------------------------


def test_abstention_is_reported_as_null_not_as_a_perfect_score() -> None:
    """RETRACTION under test. This file used to assert
    ``value_mae_infeasible == 0.0`` for the projection- and correction-aware
    regimes and read it as accuracy. Both regimes claim ZERO infeasible bins;
    the 0.0 was ``_mae_over`` scoring an empty set. It is now ``None``.
    """
    for name in ("projection_aware", "correction_aware"):
        r = _unmasked()[name]
        assert r["infeasible_bins_claimed"] == 0, name
        assert r["value_mae_infeasible"] is None, name
        assert r["value_mae_infeasible_vs_projected_truth"] is None, name
        assert r["abstained_on_infeasible_bins"] is True, name
    # A regime that DOES claim is scored, and its abstention flag is false.
    naive = _unmasked()["naive_proposed_credit"]
    assert naive["abstained_on_infeasible_bins"] is False
    assert naive["value_mae_infeasible"] is not None


def test_naive_proposed_credit_is_measurably_biased_against_the_unconstrained_truth() -> None:
    result = _result()
    naive = _unmasked()["naive_proposed_credit"]
    # Naive credit books the capped reward (~0.347) as the value of EVERY
    # illegal EIRP: its claims over the 19 infeasible bins are wrong by ~0.156
    # served fraction on average against the unconstrained truth R.
    assert naive["infeasible_bins_claimed"] == 19
    assert abs(naive["value_mae_infeasible"] - 0.155697) < 1e-6
    for eirp in range(34, 53):
        claim = naive["learned_value_by_eirp"][str(eirp)]
        assert abs(claim - result["env"]["optimal_feasible_reward"]) < 1e-3
        assert result["true_unconstrained_value_by_eirp"][str(eirp)] > claim
    # Feasible-bin credit is untouched by the projection, so it stays exact.
    assert naive["value_mae_feasible"] == 0.0


# --------------------------------------------------------------------------
# P3(c) — what the bias number is measured against, and how arbitrary it is
# --------------------------------------------------------------------------


def test_against_the_realisable_truth_the_naive_regime_is_exactly_right() -> None:
    """The other half of the story, and the more damning half.

    ``value_mae_infeasible`` scores claims against R — the reward the PROPOSED
    action would earn with no Shield, a counterfactual no deployed rApp can
    observe. Against R∘Pi — what the proposal actually causes — the naive
    regime is exact and the ORACLE is the one in error. Both are published.
    """
    unmasked = _unmasked()
    naive, oracle = unmasked["naive_proposed_credit"], unmasked["unconstrained_oracle"]
    assert naive["value_mae_infeasible_vs_projected_truth"] == 0.0
    assert abs(naive["value_mae_infeasible"] - 0.155697) < 1e-6
    # Exactly mirrored for the oracle.
    assert oracle["value_mae_infeasible"] == 0.0
    assert abs(oracle["value_mae_infeasible_vs_projected_truth"] - 0.155697) < 1e-6
    # The realisable value function is flat above the cap, by construction of
    # the projection — every over-cap proposal causes the same executed action.
    realisable = _result()["realisable_value_by_eirp_after_projection"]
    assert len({realisable[str(e)] for e in range(33, 53)}) == 1


def test_the_headline_bias_number_is_a_free_parameter_of_the_grid_top() -> None:
    """RETRACTION under test: 0.1557 is not a measured property of the Shield.

    It is exactly ``mean_{b>cap} |R(cap) - R(b)|``, a pure function of where the
    top of the EIRP grid was placed, and it has zero effect on realised reward.
    """
    sens = _result()["value_mae_infeasible_sensitivity"]
    top = sens["grid_top_dbm"]
    assert abs(top["34"]["value_mae_infeasible_if_naive"] - 0.018311) < 1e-6
    assert abs(top["52"]["value_mae_infeasible_if_naive"] - 0.155697) < 1e-6
    assert abs(top["110"]["value_mae_infeasible_if_naive"] - 0.461023) < 1e-6
    # A 25x span on the headline number, chosen by a constant in the runner.
    assert top["110"]["value_mae_infeasible_if_naive"] > 25 * top["34"]["value_mae_infeasible_if_naive"]
    # And the published number is reproduced exactly by that pure formula.
    assert (
        top["52"]["value_mae_infeasible_if_naive"]
        == _unmasked()["naive_proposed_credit"]["value_mae_infeasible"]
    )
    # argmax_tie_count is the same free parameter: grid_top - cap + 1.
    assert _unmasked()["naive_proposed_credit"]["argmax_tie_count"] == 52 - 33 + 1


# --------------------------------------------------------------------------
# P3(b) — the zero-data baseline
# --------------------------------------------------------------------------


def test_a_zero_data_constant_cap_policy_matches_the_best_learner_exactly() -> None:
    """The single most damning measured fact in this benchmark.

    Proposing a constant 33 dBm every step — no data, no learning, no
    exploration — realises EXACTLY the feasible optimum, which is exactly what
    the best deployable learner converges to. Measured learning gain: 0.000000.
    """
    result = _result()
    zdb = result["zero_data_baseline"]
    assert zdb["equals_optimal_feasible"] is True
    assert zdb["realised_mean_reward"] == result["env"]["optimal_feasible_reward"] == 0.347412
    assert zdb["projected_steps"] == 0
    assert zdb["measured_learning_gain_last_quarter"] == 0.0


def test_over_the_full_run_every_learner_is_strictly_worse_than_not_learning() -> None:
    result = _result()
    zdb = result["zero_data_baseline"]
    assert zdb["measured_learning_gain_all"] < 0.0
    assert abs(zdb["measured_learning_gain_all"] - (-0.011577)) < 1e-6
    # Regime by regime, in both arms: nobody beats the zero-data policy.
    arms = result["arms"]
    for arm in ARMS:
        for name in SHIELDED:  # the oracle is not deployable and is excluded
            r = arms[arm]["regimes"][name]
            assert r["learning_gain_over_constant_cap_policy_all"] < 0.0, (arm, name)
            assert r["learning_gain_over_constant_cap_policy_last_quarter"] <= 0.0, (arm, name)


def test_every_regime_reports_realised_regret_against_the_baseline() -> None:
    arms = _result()["arms"]
    for arm in ARMS:
        for name in ALL_REGIMES:
            r = arms[arm]["regimes"][name]
            assert r["constant_cap_policy_reward"] == 0.347412, (arm, name)
            for field in ("realised_regret_last_quarter", "realised_regret_all"):
                assert isinstance(r[field], float), (arm, name, field)
    # The best deployable regime reaches exactly zero last-quarter regret — it
    # ties the zero-data policy, it does not beat it.
    assert _unmasked()["projection_aware"]["realised_regret_last_quarter"] == 0.0


# --------------------------------------------------------------------------
# What is left of the original story, stated honestly
# --------------------------------------------------------------------------


def test_correction_aware_graduates_in_the_unmasked_arm_and_naive_never_does() -> None:
    unmasked = _unmasked()
    ca = unmasked["correction_aware"]
    assert ca["projection_rate_last_quarter"] < ca["projection_rate_first_quarter"]
    assert ca["projection_rate_last_quarter"] < 0.1
    # Naive credit keeps leaning on the Shield forever: its terminal tie makes
    # it propose illegal EIRPs at high rate to the very end.
    naive = unmasked["naive_proposed_credit"]
    assert naive["projection_rate_last_quarter"] > 0.5
    assert naive["illegal_proposals"] > 100


def test_realised_rewards_are_honest_about_what_the_shield_hides() -> None:
    unmasked = _unmasked()
    # With an exact projection, the shielded regimes realise near-identical
    # reward — the bias costs value-function truth and projection load, not
    # realised reward. Say so with numbers rather than manufacture an effect.
    realised = [unmasked[name]["realised_mean_reward"] for name in SHIELDED]
    assert max(realised) - min(realised) < 0.02
    # The oracle's realised (no-Shield world) reward clears the feasible optimum,
    # but what it actually emitted through the Shield stayed capped.
    oracle = unmasked["unconstrained_oracle"]
    assert oracle["oracle"] is True and oracle["deployable"] is False
    assert oracle["realised_mean_reward_last_quarter"] > 0.5
    assert oracle["shielded_execution_mean_reward"] < 0.4


def test_the_scope_note_says_the_pathology_needs_the_unmasked_arm() -> None:
    note = _result()["scope_note"]
    assert "measurable ONLY in the unmasked arm" in note
    assert "no deployable learner beats a zero-data constant-cap policy" in note
