"""Checks for the real-DeepMIMO analog beam-management evidence.

The committed ``benchmarks/results/beam_management.json`` is the real
4096-receiver run of ``benchmarks/beam_management_loop.py`` on the
licence-gated DeepMIMO ASU 3.5 GHz angular feature set. The raw feature rows
are not redistributed (see the dataset manifest), so this test validates the
committed real result and its byte-level provenance; the ``realdata`` CI
workflow rebuilds the data deterministically and re-runs the loop end to end.
No synthetic data.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COHERENT_BOUND_DB = 9.0309  # 10*log10(8): the hard 8-element combining ceiling.


def _result() -> dict:
    return json.loads(
        (ROOT / "benchmarks/results/beam_management.json").read_text()
    )


def test_result_is_bound_to_the_real_angular_build() -> None:
    result = _result()
    manifest = json.loads(
        (ROOT / "datasets/deepmimo_asu_3p5/angular_manifest.json").read_text()
    )
    # The beam sweep consumed exactly the canonical, checksum-pinned build.
    assert result["features_sha256"] == manifest["features_sha256"]
    assert result["source_tree_sha256"] == manifest["source_tree_sha256"]
    assert result["receivers"] == manifest["sampled_receivers"] == 4096
    assert result["bs_array"] == manifest["bs_array"]
    assert "not over-the-air" in result["data_kind"]


def test_array_gain_is_real_and_physically_bounded() -> None:
    result = _result()
    # Coherent 8-element combining shows up (> 6 dB mean over the
    # single-element reference) and no receiver exceeds the 10*log10(8)
    # physical ceiling — the sweep measures real combining, not an artefact.
    assert result["mean_array_gain_db"] > 6.0
    assert result["mean_array_gain_db"] <= COHERENT_BOUND_DB + 0.1
    assert result["median_array_gain_db"] > 6.0
    assert result["max_array_gain_db"] <= COHERENT_BOUND_DB + 0.01


def test_beam_selection_beats_a_fixed_beam() -> None:
    result = _result()
    # Steering to the swept-and-selected beam recovers real dB over a fixed
    # broadside beam — the whole point of the SSB sweep on wide AoD geometry.
    assert result["mean_misalignment_recovery_db"] > 1.0


def test_real_aod_spread_exercises_the_codebook() -> None:
    result = _result()
    hist = result["beam_selection_histogram"]
    assert len(hist) == 8
    assert sum(hist) == 4096
    # The real angle-of-departure spread makes beam choice a real decision:
    # at least half the codebook wins somewhere, and most receivers are best
    # served by a non-boresight beam.
    assert result["distinct_beams_selected"] >= 4
    assert sum(1 for c in hist if c > 0) == result["distinct_beams_selected"]
    boresight = result["codebook"]["boresight_beam_index"]
    assert result["fraction_non_boresight"] > 0.5
    assert abs(result["fraction_non_boresight"] - (1 - hist[boresight] / 4096)) < 1e-3
    assert hist[result["modal_beam_index"]] == max(hist)


def test_trust_chain_holds_and_catches_tamper() -> None:
    result = _result()
    tc = result["trust_chain"]
    # An over-EIRP beam activation is projected to the legal cap and refused emit.
    assert tc["overpower_beam_corrected"] is True
    assert tc["overpower_beam_safe_eirp_dbm"] <= 33.0 + 1e-6
    assert tc["overpower_beam_reached_ran"] is False
    # The evidence chain verifies intact, then the tamper is pinpointed.
    assert tc["verify_first_broken_index"] == -1
    assert tc["evidence_chain_length"] >= 2
    assert tc["verify_after_tamper_index"] not in (None, -1)
