"""Checks for the real-DeepMIMO anti-jam null-steering evidence.

The committed ``benchmarks/results/antijam.json`` is the real 4096-receiver run
of ``benchmarks/antijam_loop.py`` on the licence-gated DeepMIMO ASU 3.5 GHz
angular feature set. The raw feature rows are not redistributed (see the
angular manifest), so this test validates the committed real result and its
byte-level provenance; the ``realdata`` CI workflow rebuilds the data
deterministically and re-runs the loop end to end. No synthetic data: the
desired channel per receiver is its real ray-traced path sum — only the jammer
is a modeled interferer (single-BS scenario, honestly labelled as such).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _result() -> dict:
    return json.loads((ROOT / "benchmarks/results/antijam.json").read_text())


def test_result_is_bound_to_the_real_angular_build() -> None:
    result = _result()
    manifest = json.loads(
        (ROOT / "datasets/deepmimo_asu_3p5/angular_manifest.json").read_text()
    )
    # The anti-jam run consumed exactly the canonical, checksum-pinned build.
    assert result["features_sha256"] == manifest["features_sha256"]
    assert result["source_tree_sha256"] == manifest["source_tree_sha256"]
    assert result["receivers"] == manifest["sampled_receivers"] == 4096
    assert "not over-the-air" in result["data_kind"]
    # The array the loop beamforms over is the manifest's shared definition.
    assert result["bs_array"]["elements"] == 8
    assert result["bs_array"]["spacing_wavelengths"] == 0.5


def test_jammer_is_honestly_labelled_as_modeled() -> None:
    result = _result()
    # The desired channel is real; the jammer is not — and must say so.
    assert "modeled" in result["jammer"]["note"]
    assert "MODELED" in result["scope_note"] or "modeled" in result["scope_note"]
    assert result["jammer"]["jammer_to_signal_db"] == 30.0


def test_jammer_collapses_and_mvdr_restores_the_real_links() -> None:
    result = _result()
    # The jammer-blind serving beam collapses under the jammer: mean SINR is
    # deeply negative and almost no receiver stays usable.
    assert result["mean_sinr_jammed_db"] < 0.0
    assert result["restored_fraction_jammed"] < 0.5
    # MVDR null-steering materially restores SINR on the real channels...
    assert result["mean_sinr_gain_db"] > 5.0
    assert result["mean_sinr_antijam_db"] > result["mean_sinr_jammed_db"]
    # ...and more receivers are usable after mitigation than under the jammer.
    assert result["restored_fraction_antijam"] > result["restored_fraction_jammed"]


def test_jammer_is_genuinely_nulled() -> None:
    result = _result()
    # The MVDR pattern suppresses the jammer well beyond the DFT serving beam
    # (idealized analytic covariance — see the scope note).
    assert result["mean_null_depth_db"] > 10.0
    assert result["min_null_depth_db"] > 10.0


def test_trust_chain_holds_and_catches_tamper() -> None:
    result = _result()
    tc = result["trust_chain"]
    # An over-EIRP null-steer is projected to the legal cap and refused emit.
    assert tc["overpower_nullsteer_corrected"] is True
    assert tc["overpower_nullsteer_safe_eirp_dbm"] <= 33.0 + 1e-6
    assert tc["overpower_nullsteer_reached_ran"] is False
    # The evidence chain verifies intact, then the tamper is pinpointed.
    assert tc["verify_first_broken_index"] == -1
    assert tc["evidence_chain_length"] >= 2
    assert tc["verify_after_tamper_index"] not in (None, -1)
