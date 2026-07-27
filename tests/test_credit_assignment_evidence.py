"""Checks for the real-DeepMIMO credit-assignment bias evidence.

The committed ``benchmarks/results/credit_assignment.json`` is the real
4096-receiver run of ``benchmarks/credit_assignment_loop.py`` on the
licence-gated DeepMIMO ASU 3.5 GHz feature set: the same tabular bandit trained
under four credit regimes against the Shield-wrapped spectrum environment. The
raw feature rows are not redistributed (see the dataset manifest), so this test
validates the committed real result and its byte-level provenance; the
``realdata`` CI workflow rebuilds the data deterministically and re-runs the
loop end to end. No synthetic data.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SHIELDED = ("naive_proposed_credit", "projection_aware", "correction_aware")
ALL_REGIMES = SHIELDED + ("unconstrained_oracle",)


def _result() -> dict:
    return json.loads(
        (ROOT / "benchmarks/results/credit_assignment.json").read_text()
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


def test_environment_anchors_are_the_measured_ones() -> None:
    result = _result()
    env = result["env"]
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
    # What the cap costs on this scenario: the oracle's converged reward minus
    # the best feasible reward. Positive, and large.
    assert result["safety_utility_gap"] > 0.2


def test_naive_proposed_credit_is_measurably_biased() -> None:
    result = _result()
    r = result["regimes"]
    naive = r["naive_proposed_credit"]
    # The headline: naive credit books the capped reward (~0.347) as the value
    # of EVERY illegal EIRP, so its claims over the 19 infeasible bins are wrong
    # by ~0.156 served fraction on average (up to ~0.258 at the grid top).
    assert naive["infeasible_bins_claimed"] == 19
    assert naive["value_mae_infeasible"] > 0.1
    assert naive["value_mae_infeasible"] > r["projection_aware"]["value_mae_infeasible"]
    # Every infeasible claim is exactly the capped reward — censored feedback.
    for eirp in range(34, 53):
        claim = naive["learned_value_by_eirp"][str(eirp)]
        assert abs(claim - result["env"]["optimal_feasible_reward"]) < 1e-3
        assert result["true_unconstrained_value_by_eirp"][str(eirp)] > claim
    # Terminal indifference: the naive table cannot distinguish the legal cap
    # from any illegal EIRP — a 20-way argmax tie spanning all 19 illegal bins.
    assert naive["argmax_tie_count"] == 20
    # Feasible-bin credit is untouched by the projection, so it stays exact.
    assert naive["value_mae_feasible"] < 1e-6


def test_projection_and_correction_aware_credit_do_not_claim_illegal_values() -> None:
    result = _result()
    r = result["regimes"]
    for name in ("projection_aware", "correction_aware"):
        assert r[name]["infeasible_bins_claimed"] == 0
        assert r[name]["value_mae_infeasible"] == 0.0
        assert r[name]["value_mae_feasible"] < 1e-6
        assert r[name]["argmax_proposed_eirp_dbm"] == 33.0
        assert r[name]["argmax_tie_count"] == 1
    # The oracle claims the illegal region and is exactly right — but only via
    # counterfactual access no deployable rApp has.
    oracle = r["unconstrained_oracle"]
    assert oracle["oracle"] is True and oracle["deployable"] is False
    assert oracle["infeasible_bins_claimed"] == 19
    assert oracle["value_mae_infeasible"] == 0.0
    assert oracle["argmax_proposed_eirp_dbm"] == 52.0


def test_correction_aware_graduates_and_naive_never_does() -> None:
    result = _result()
    r = result["regimes"]
    ca = r["correction_aware"]
    assert ca["projection_rate_last_quarter"] < ca["projection_rate_first_quarter"]
    assert ca["projection_rate_last_quarter"] < 0.1
    # Naive credit keeps leaning on the Shield forever: its terminal tie makes
    # it propose illegal EIRPs at high rate to the very end.
    naive = r["naive_proposed_credit"]
    assert naive["projection_rate_last_quarter"] > 0.5
    assert naive["illegal_proposals"] > 100


def test_realised_rewards_are_honest_about_what_the_shield_hides() -> None:
    result = _result()
    r = result["regimes"]
    # With an exact projection, the shielded regimes realise near-identical
    # reward — the bias costs value-function truth and projection load, not
    # realised reward. Say so with numbers rather than manufacture an effect.
    realised = [r[name]["realised_mean_reward"] for name in SHIELDED]
    assert max(realised) - min(realised) < 0.02
    # The oracle's realised (no-Shield world) reward clears the feasible optimum,
    # but what it actually emitted through the Shield stayed capped.
    oracle = r["unconstrained_oracle"]
    assert oracle["realised_mean_reward_last_quarter"] > 0.5
    assert oracle["shielded_execution_mean_reward"] < 0.4
    # No regime ever emitted an illegal action to the RAN.
    for name in ALL_REGIMES:
        assert r[name]["max_executed_eirp_dbm"] <= 33.0 + 1e-9


def test_trust_chain_holds_replays_and_catches_tamper() -> None:
    result = _result()
    tc = result["trust_chain"]
    rounds = result["learner"]["rounds_per_regime"]
    # Every decision of every regime is on the hash chain, the chain verifies
    # intact, and the training tables rebuilt from the verify-gated replay
    # buffer match online training bit-for-bit.
    assert tc["evidence_chain_length"] == 4 * rounds
    assert tc["replay_transitions"] == tc["evidence_chain_length"]
    assert tc["verify_first_broken_index"] == -1
    assert tc["replay_matches_online"] is True
    # After a tamper of record #1 on disk, replay REFUSES to produce training
    # data and re-verification pinpoints the tamper.
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
