"""Checks for the real-DeepMIMO federated coverage-map evidence.

The committed ``benchmarks/results/federated_coverage.json`` is the real
4096-receiver run of ``benchmarks/federated_coverage_loop.py`` on the
licence-gated DeepMIMO ASU 3.5 GHz feature set. The raw feature rows are not
redistributed (see the dataset manifest), so this test validates the committed
real result and its byte-level provenance; the ``realdata`` CI workflow rebuilds
the data deterministically and re-runs the loop end to end. No synthetic data.

These assertions were re-based after an adversarial post-mortem. Two of the
previous ones gated on quantities the underlying claims never needed and failed
under a partition reseed (``krum_rmse < baseline``: 9/12 seeds;
``robust_target_receiver == 9``: 5/12). Everything asserted here is checked at
every one of the run's 12 partition seeds by the loop's own seed sweep, and
:mod:`scripts.verify_federated_coverage` re-checks it on freshly rebuilt data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Kept in sync with benchmarks/federated_coverage_loop.py.
SEPARATION_FACTOR = 1e4
ROBUST_RMSE_CEILING_DB = 22.0
FLTRUST_PARITY_TOLERANCE = 1.10
DP_TARGET_EPSILON = 4.1447


def _result() -> dict:
    return json.loads(
        (ROOT / "benchmarks/results/federated_coverage.json").read_text()
    )


def test_result_is_bound_to_the_real_deepmimo_build() -> None:
    result = _result()
    manifest = json.loads(
        (ROOT / "datasets/deepmimo_asu_3p5/manifest.json").read_text()
    )
    # The federated run consumed exactly the canonical, checksum-pinned build.
    assert result["features_sha256"] == manifest["features_sha256"]
    assert result["source_tree_sha256"] == manifest["source_tree_sha256"]
    assert result["receivers"] == manifest["sampled_receivers"] == 4096
    assert "not over-the-air" in result["data_kind"]
    # The committed result must not be a deliberately-broken diagnostic run.
    assert result["regression_injected"] == "none"


def test_coverage_signal_is_real_and_learned() -> None:
    result = _result()
    base = result["baseline_predict_mean_rmse_db"]
    clean = result["outcomes"]["clean_fedavg"]
    # A ~100 dB dynamic-range coverage field with a real, learnable structure:
    # clean federated learning beats the predict-the-mean baseline handily.
    assert result["coverage_dynamic_range_db"] > 50.0
    assert result["closed_form_rmse_db"] < base
    assert clean["rmse_db"] < base
    assert not clean["diverged"]


def test_server_root_set_is_held_out_of_every_client() -> None:
    result = _result()
    root = result["server_root_set"]
    # FLTrust's root data and the DP clip calibration both come from receivers no
    # client holds, which is what makes neither of them leak the private cohort.
    assert root["receivers"] + root["client_receivers"] == result["receivers"]
    assert root["receivers"] > 0


def test_poison_breaks_naive_fedavg_and_robust_aggregation_bounds_it() -> None:
    result = _result()
    o = result["outcomes"]
    naive = o["poisoned_fedavg"]["rmse_db"]
    # Model-replacement poison diverges plain FedAvg (RMSE explodes).
    assert o["poisoned_fedavg"]["diverged"]
    assert naive > result["baseline_predict_mean_rmse_db"]
    # Both robust aggregators are separated from it by orders of magnitude and
    # sit under an absolute ceiling. Neither claim is a sub-decibel margin.
    for method in ("krum", "fltrust"):
        r = o[f"poisoned_{method}"]
        assert not r["diverged"]
        assert r["rmse_db"] < naive / SEPARATION_FACTOR
        assert r["rmse_db"] <= ROBUST_RMSE_CEILING_DB


def test_krums_cost_is_mostly_an_aggregation_tax_not_a_poison_tax() -> None:
    result = _result()
    tax = result["defence_cost_decomposition"]["krum"]
    # Measured with ZERO adversaries, Krum already costs multiple dB against
    # clean FedAvg. That cost is the price of discarding K-1 updates on a
    # non-IID split, not the price of the attack.
    assert tax["aggregation_tax_db"] > 1.0
    assert tax["poison_tax_db"] is not None
    assert tax["poison_tax_db"] < tax["aggregation_tax_db"]


def test_fltrust_reaches_clean_parity_under_every_attack_in_the_battery() -> None:
    result = _result()
    clean = result["outcomes"]["clean_fedavg"]["rmse_db"]
    battery = result["attack_battery"]
    # More than the shipped scaling attack: sign-flip, ALIE and Min-Max too.
    assert {"scaling", "sign_flip", "alie", "min_max"} <= set(battery)
    for attack, row in battery.items():
        assert row["fltrust"] <= clean * FLTRUST_PARITY_TOLERANCE, attack


def test_fltrust_survives_adversary_counts_where_krum_is_undefined() -> None:
    result = _result()
    breakdown = result["byzantine_breakdown"]["by_n_malicious"]
    # Krum needs n > 2f+2, so at K=10 it is undefined from f=4 up; FLTrust has no
    # such counting bound and stays bounded all the way to f=7 of 10.
    assert breakdown["4"]["krum"] is None
    assert breakdown["7"]["krum"] is None
    for f_str in ("4", "5", "6", "7"):
        assert breakdown[f_str]["fltrust"] <= ROBUST_RMSE_CEILING_DB, f_str


def test_dp_is_fixed_at_an_unchanged_certified_epsilon() -> None:
    result = _result()
    dp = result["dp_at_fixed_epsilon"]
    # The whole point: same epsilon, same delta, same adjacency, same client
    # count — only the clip norm and the round count move.
    assert dp["legacy"]["epsilon"] == dp["tuned"]["epsilon"] == DP_TARGET_EPSILON
    assert dp["delta"] == 1e-5
    assert dp["adjacency"] == "replace_one"  # NOT relaxed to add/remove-one
    # And the result is a utility fix, not a trade-off: below the do-nothing
    # baseline, where the legacy configuration was ~17x worse than it.
    assert dp["tuned"]["rmse_db"] < result["baseline_predict_mean_rmse_db"]
    assert dp["tuned"]["rmse_db"] < dp["legacy"]["rmse_db"] / 10.0


def test_the_dp_defect_diagnosis_is_backed_by_the_measured_norms() -> None:
    result = _result()
    calib = result["dp_at_fixed_epsilon"]["clip_calibration"]
    # The legacy clip of 1.0 never binds on the clean trajectory, so its noise
    # was calibrated to a sensitivity the data never attains.
    assert calib["legacy_clip_binds_count"] == 0
    assert calib["clean_trajectory_update_l2_max"] < 1.0
    assert calib["n_updates_observed"] > 0


def test_dp_clip_is_chosen_without_touching_the_private_cohort() -> None:
    result = _result()
    tuned = result["dp_at_fixed_epsilon"]["tuned"]
    sweep = tuned["public_clip_sweep"]
    # The shipped clip is the argmin of a sweep scored entirely on server-held
    # root receivers; a clip read off the private updates would itself leak.
    assert len(sweep) >= 5
    best = min(sweep, key=lambda r: r["root_set_rmse_db"])
    assert tuned["clip_norm"] == best["clip_norm"]


def test_rounds_are_a_privacy_cost_not_a_free_parameter() -> None:
    result = _result()
    sweep = result["dp_at_fixed_epsilon"]["round_sweep_at_fixed_epsilon"]
    by_rounds = {r["rounds"]: r for r in sweep}
    # z grows like sqrt(R) at fixed epsilon, so running far past convergence
    # strictly costs utility. The shipped round count is the sweep's argmin.
    assert by_rounds[40]["noise_multiplier"] > by_rounds[1]["noise_multiplier"]
    assert by_rounds[40]["rmse_db"] > min(r["rmse_db"] for r in sweep)
    assert result["dp_at_fixed_epsilon"]["tuned"]["rounds"] in by_rounds


def test_one_shot_release_exploits_the_public_whitening() -> None:
    result = _result()
    one = result["dp_at_fixed_epsilon"]["one_shot_release"]
    # Whitening makes A^T A / N exactly the identity, so the task is a single
    # 6-dim sum releasable once instead of once per round.
    assert one["gram_identity_max_abs_error"] < 1e-8
    sizes = {r["n_clients"]: r for r in one["by_federation_size"]}
    # Every release is charged at the same certified budget...
    assert all(r["epsilon"] == DP_TARGET_EPSILON for r in sizes.values())
    # ...and utility improves monotonically with federation size, because the
    # per-client sensitivity falls like 1/K while the noise stays fixed.
    ordered = [sizes[k]["rmse_db"] for k in sorted(sizes)]
    assert ordered == sorted(ordered, reverse=True)
    # At a large federation it lands within a couple of dB of non-private FedAvg.
    assert sizes[max(sizes)]["rmse_db"] < result["outcomes"]["clean_fedavg"]["rmse_db"] + 2.0


def test_poison_misdirects_the_fill_and_the_defence_recovers_it() -> None:
    result = _result()
    m = result["coverage_fill_misdirection"]
    # Without robustness the power-fill is aimed at a different cell, and that
    # cell is NOT badly covered — the concrete operational harm.
    assert m["poison_moved_the_fill"] is True
    assert m["by_aggregator"]["fedavg"]["in_bottom_decile"] is False
    # The claim is a predicate on coverage, not an equality on a cell id.
    assert m["robust_target_in_bottom_decile"] is True
    assert m["robust_recovers_clean_target"] is True
    assert (
        m["by_aggregator"][m["robust_defence"]]["predicted_coverage_dbw"]
        <= m["bottom_decile_threshold_dbw"]
    )


def test_every_gate_holds_at_every_partition_seed() -> None:
    result = _result()
    sr = result["seed_robustness"]
    assert sr["n_seeds"] >= 12
    # A gate that a reseed can flip is not evidence. All of them must hold at all
    # of the seeds, and the run's own gate block must agree.
    assert sr["all_gates_hold_every_seed"] is True
    assert all(v == sr["n_seeds"] for v in sr["gates_held"].values())
    assert result["all_gates_pass"] is True
    assert all(result["gates"].values())


@pytest.mark.parametrize(
    "name", ["krum_rmse_below_baseline", "krum_target_receiver_exact_match"]
)
def test_the_retired_gates_are_recorded_as_seed_fragile(name: str) -> None:
    result = _result()
    retired = result["seed_robustness"]["retired_gates"][name]
    # These are the two assertions this file used to make. They are kept as
    # measured evidence of *why* they were retired, not as passing claims.
    assert retired["held"] < retired["of"]
    assert retired["why_retired"]


def test_the_document_retracts_the_claims_the_errata_flagged() -> None:
    result = _result()
    retractions = " ".join(result["retractions"]).lower()
    for phrase in ("trade-off", "krum is the correct defense", "0.3 dB".lower()):
        assert phrase in retractions
    assert "0.0850" in " ".join(result["retractions"])


def test_trust_chain_holds_and_catches_tamper() -> None:
    result = _result()
    tc = result["trust_chain"]
    # An over-EIRP coverage-fill is projected to the legal cap and refused emit.
    assert tc["overpower_fill_corrected"] is True
    assert tc["overpower_fill_safe_eirp_dbm"] <= 33.0 + 1e-6
    assert tc["overpower_fill_reached_ran"] is False
    # The evidence chain verifies intact, then the tamper is pinpointed.
    assert tc["verify_first_broken_index"] == -1
    assert tc["evidence_chain_length"] >= 2
    assert tc["verify_after_tamper_index"] not in (None, -1)
