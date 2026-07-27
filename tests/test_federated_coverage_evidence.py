"""Checks for the real-DeepMIMO federated coverage-map evidence.

The committed ``benchmarks/results/federated_coverage.json`` is the real
4096-receiver run of ``benchmarks/federated_coverage_loop.py`` on the
licence-gated DeepMIMO ASU 3.5 GHz feature set. The raw feature rows are not
redistributed (see the dataset manifest), so this test validates the committed
real result and its byte-level provenance; the ``realdata`` CI workflow rebuilds
the data deterministically and re-runs the loop end to end. No synthetic data.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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


def test_poison_breaks_naive_fedavg_and_krum_defends() -> None:
    result = _result()
    base = result["baseline_predict_mean_rmse_db"]
    o = result["outcomes"]
    # Model-replacement poison diverges plain FedAvg (RMSE explodes).
    assert o["poisoned_fedavg"]["diverged"]
    assert o["poisoned_fedavg"]["rmse_db"] > base
    # Krum bounds it below the baseline and far below the poisoned FedAvg.
    assert not o["poisoned_krum"]["diverged"]
    assert o["poisoned_krum"]["rmse_db"] < base
    assert o["poisoned_krum"]["rmse_db"] < o["poisoned_fedavg"]["rmse_db"]
    # DP release carries a finite certified budget.
    assert o["dp_fedavg_clean"]["epsilon"] is not None
    assert o["dp_fedavg_clean"]["epsilon"] > 0.0


def test_poison_misdirects_the_coverage_fill() -> None:
    result = _result()
    m = result["coverage_fill_misdirection"]
    # Without robustness the power-fill is aimed at a different (garbage) cell;
    # Krum recovers the clean model's target cell.
    assert m["poison_moved_the_fill"] is True
    assert m["robust_target_receiver"] == m["clean_target_receiver"]
    assert m["naive_poisoned_target_receiver"] != m["clean_target_receiver"]


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
