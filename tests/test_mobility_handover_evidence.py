"""Checks for the real-DeepMIMO beam-level mobility / handover evidence.

The committed ``benchmarks/results/mobility_handover.json`` is the real
4096-receiver run of ``benchmarks/mobility_handover_loop.py`` on the
licence-gated DeepMIMO ASU 3.5 GHz angular feature set (per-path AoD/AoA ray
geometry). The raw feature rows are not redistributed (see the angular
manifest), so this test validates the committed real result and its byte-level
provenance; ``scripts/verify_mobility_handover.py`` re-checks a fresh rebuild
against it on host-stable invariants. No synthetic data, no synthetic fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _result() -> dict:
    return json.loads(
        (ROOT / "benchmarks/results/mobility_handover.json").read_text()
    )


def test_result_is_bound_to_the_real_angular_build() -> None:
    result = _result()
    manifest = json.loads(
        (ROOT / "datasets/deepmimo_asu_3p5/angular_manifest.json").read_text()
    )
    # The mobility run consumed exactly the canonical, checksum-pinned build.
    assert result["features_sha256"] == manifest["features_sha256"]
    assert result["source_tree_sha256"] == manifest["source_tree_sha256"]
    assert result["receivers"] == manifest["sampled_receivers"] == 4096
    assert "not over-the-air" in result["data_kind"]
    assert result["bs_position_m"] == manifest["tx_position_m"]


def test_doppler_is_genuinely_on() -> None:
    result = _result()
    dop = result["doppler"]
    traj = result["trajectory"]
    assert dop["enabled"] is True
    assert dop["max_shift_hz"] > 0.0
    assert dop["coherence_time_s"] > 0.0
    # Tc ~ 0.423 / f_d_max, and f_d_max <= |v| / lambda (lambda ~ 0.0857 m).
    assert abs(dop["coherence_time_s"] - 0.423 / dop["max_shift_hz"]) < 1e-4
    assert dop["max_shift_hz"] <= traj["ue_speed_mps"] / (3e8 / 3.5e9) + 1e-6


def test_trajectory_is_a_real_drive() -> None:
    result = _result()
    traj = result["trajectory"]
    # A real, ordered corridor of ray-traced receiver positions, not a mock.
    assert traj["points"] >= 150
    assert traj["total_distance_m"] > 500.0
    assert traj["ue_speed_mps"] > 0.0
    assert traj["delta_t_s"] > 0.0
    assert traj["corridor"]["x_min_m"] < traj["corridor"]["x_max_m"]


def test_a3_hysteresis_beats_greedy_on_ping_pong() -> None:
    result = _result()
    g, h = result["greedy"], result["hysteresis"]
    # Greedy argmax tracking flaps between adjacent DFT beams in the
    # Doppler-driven fading; the A3 event (hysteresis + TTT) cuts ping-pongs.
    assert g["ping_pongs"] > h["ping_pongs"]
    # Real mobility across beams occurred, and the A3 policy still tracks the
    # best beam acceptably (bounded outage), with fewer raw handovers.
    assert h["handovers"] >= 1
    assert h["handovers"] <= g["handovers"]
    assert h["outage_fraction"] <= 0.25
    # Greedy serves the argmax by construction, so its outage is exactly zero.
    assert g["outage_fraction"] == 0.0
    # The drive genuinely crosses several beams of the 8-beam codebook.
    assert len(g["beams_visited"]) >= 2
    assert len(h["beams_visited"]) >= 2


def test_trust_chain_holds_and_catches_tamper() -> None:
    result = _result()
    tc = result["trust_chain"]
    # An over-EIRP handover emit is projected to the legal cap and refused emit.
    assert tc["overpower_handover_corrected"] is True
    assert tc["overpower_handover_safe_eirp_dbm"] <= 33.0 + 1e-6
    assert tc["overpower_handover_reached_ran"] is False
    # The evidence chain verifies intact, then the tamper is pinpointed.
    assert tc["verify_first_broken_index"] == -1
    assert tc["evidence_chain_length"] >= 2
    assert tc["verify_after_tamper_index"] not in (None, -1)
    # The gated action is the confirmed A3 handover into the final serving beam.
    clean = tc["gate_records"][0]
    assert clean["label"] == "a3-confirmed-handover"
    assert clean["target_beam"] == result["hysteresis"]["final_serving_beam"]
    assert clean["emitted_clean"] is True
