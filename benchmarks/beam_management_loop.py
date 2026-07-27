#!/usr/bin/env python3
"""Analog beam management on **real DeepMIMO** ray-traced angle-of-departure geometry.

This drives an O-RAN *analog beam management* rApp function end to end on real
measured-physics data — no synthetic RF anywhere in the loop:

    real ray-traced multipath geometry (DeepMIMO ASU 3.5 GHz, 4096 receivers)
        -> per-receiver array channel from real per-path power/phase/AoD azimuth
        -> SSB-style sweep of an 8-beam DFT codebook (sin-space tiling)
        -> best serving-beam selection + array-gain / misalignment accounting
        -> network-wide beam-selection policy for the modal beam
        -> Decision Safety Shield (EIRP legality, projection operator)
        -> hash-chained, tamper-evident evidence record

The task is the real 5G NR beam-establishment procedure in miniature: the gNB
sweeps a codebook of analog beams (as it does with SSB bursts), each UE reports
the strongest beam, and the RIC gates the resulting beam-selection policy before
it may touch the RAN. Every physical quantity comes from the ray tracer: each
receiver's channel to the 8-element half-wavelength ULA is

    h = sum_l  sqrt(10^(P_l/10)) * e^{j*phase_l} * a(aod_az_l),
    a(phi)[n] = e^{j*2*pi*0.5*n*sin(phi)},   n = 0..7,

built from the receiver's own ray-traced per-path power (dBW), phase (deg) and
azimuth angle-of-departure. The codebook is the 8-beam DFT whose steering
targets u_b = (2b+1)/8 - 1 tile sin-space [-1, 1); beam b's received power is
|w_b^H h|^2 and the serving beam is the argmax — exactly a P-1 sweep followed by
selection.

Why this is the *right* real task on this data. The dominant angles of
departure span essentially the full azimuth circle across the campus (receivers
surround the site), so beam selection is a real decision: different receivers
genuinely need different beams, all 8 codebook beams win somewhere, and ~77% of
receivers are best served by a non-boresight beam. Two physically meaningful
figures quantify the win on real geometry:

* ``array_gain_db`` — best-beam power over the single-element average
  ``mean_n |h[n]|^2``. Coherent 8-element combining is hard-bounded at
  10*log10(8) ≈ 9.03 dB per receiver; the real multipath mean lands ~7 dB,
  honest for angularly spread channels (a single-path channel would hit the
  bound exactly).
* ``misalignment_recovery_db`` — best-beam power over the fixed broadside DFT
  beam. This is what *sweeping* buys over never steering at all; it is >= 0 dB
  by construction and is large exactly because the real AoD geometry is wide.

Everything downstream (Shield EIRP legality, guard chain, evidence chain,
tamper detection) runs on the actually-selected modal beam, so the whole
topology is exercised on real data, not a mock.

Pure numpy + the shipped ``horizon_ric`` modules. No torch, no GPU. The real
DeepMIMO angular feature file is licence-gated (not redistributed in-repo);
build it with ``datasets/deepmimo_asu_3p5/build_angular.py`` (deterministic,
checksum-pinned) or point ``--features`` at a cached copy. The committed result
JSON is the real 4096-Rx run; the ``realdata`` CI workflow rebuilds the data
and re-runs this loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.evidence.schema import (
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.policy.emit_guards import run_guard_chain
from horizon_ric.shield import default_terrestrial_shield

# --- Spectrum / EIRP geometry (shared with the DSA benchmark). -----------------
BAND_LO_HZ = 3.45e9
BAND_HI_HZ = 3.55e9
MAX_EIRP_DBM = 33.0
ANTENNA_GAIN_DBI = 6.0
BEAM_FREQUENCY_HZ = 3.50e9  # the beam-selection policy rides the band centre.
BEAM_BANDWIDTH_HZ = 20e6
SAFE_TX_POWER_DBM = 26.0  # legal request (EIRP 32 dBm < 33): emits without a fix.

# --- Array / codebook geometry (matches the dataset manifest bs_array). --------
N_ELEMENTS = 8  # uniform linear array, half-wavelength spacing, 3.5 GHz.
N_BEAMS = 8  # one DFT beam per element: full-rank sin-space tiling.
COHERENT_BOUND_DB = 10.0 * np.log10(N_ELEMENTS)  # ~9.03 dB combining ceiling.


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"no feature rows in {path}")
    return rows


def _dft_codebook() -> tuple[np.ndarray, np.ndarray, int]:
    """8-beam DFT codebook ``W`` [elements, beams], its sin-space centres, boresight.

    ``u_b = (2b+1)/8 - 1`` tiles sin-space [-1, 1) with one beam per element,
    the standard analog-beamforming codebook for an N-element half-wavelength
    ULA. The *boresight* beam is the codebook entry nearest broadside (u ~ 0) —
    the reference a never-steering deployment would be stuck with.
    """
    n = np.arange(N_ELEMENTS)
    u = (2.0 * np.arange(N_BEAMS) + 1.0) / N_BEAMS - 1.0
    codebook = np.exp(1j * 2.0 * np.pi * 0.5 * np.outer(n, u)) / np.sqrt(N_ELEMENTS)
    return codebook, u, int(np.argmin(np.abs(u)))


def _receiver_channel(paths: list[dict[str, Any]]) -> np.ndarray:
    """Complex array channel ``h`` [8] from the receiver's real ray-traced paths.

    Per-path complex amplitude from the ray tracer's power (dBW) and phase
    (deg); per-path array response from the azimuth angle-of-departure (the
    ray-traced elevations sit within ~2 deg of the horizontal plane, so azimuth
    carries the geometry for a horizontal ULA).
    """
    power_dbw = np.asarray([p["power_dbw"] for p in paths], dtype=np.float64)
    phase = np.radians(np.asarray([p["phase_deg"] for p in paths], dtype=np.float64))
    aod_az = np.radians(np.asarray([p["aod_az"] for p in paths], dtype=np.float64))
    alpha = np.sqrt(10.0 ** (power_dbw / 10.0)) * np.exp(1j * phase)
    steering = np.exp(1j * 2.0 * np.pi * 0.5 * np.outer(np.arange(N_ELEMENTS), np.sin(aod_az)))
    return steering @ alpha


def _beam_sweep(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sweep the codebook over every receiver (the SSB P-1 sweep + selection).

    Returns per-receiver best beams, array gains (dB over the single-element
    average) and misalignment recoveries (dB over the fixed boresight beam),
    plus the boresight beam index.
    """
    codebook, _u, boresight = _dft_codebook()
    best_beams = np.zeros(len(rows), dtype=np.int64)
    array_gain_db = np.zeros(len(rows), dtype=np.float64)
    recovery_db = np.zeros(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        h = _receiver_channel(row["paths"])
        beam_gains = np.abs(codebook.conj().T @ h) ** 2
        single_element_ref = float(np.mean(np.abs(h) ** 2))
        b = int(np.argmax(beam_gains))
        best_beams[i] = b
        array_gain_db[i] = 10.0 * np.log10(beam_gains[b] / single_element_ref)
        recovery_db[i] = 10.0 * np.log10(beam_gains[b] / beam_gains[boresight])
    if not (np.all(np.isfinite(array_gain_db)) and np.all(np.isfinite(recovery_db))):
        raise ValueError("beam sweep produced non-finite gains")
    return best_beams, array_gain_db, recovery_db, boresight


def _shield_gate(
    store: JsonlEvidenceStore,
    *,
    decision_id: str,
    label: str,
    beam_index: int,
    tx_power_dbm: float = SAFE_TX_POWER_DBM,
    frequency_hz: float = BEAM_FREQUENCY_HZ,
) -> dict[str, Any]:
    """Emit the beam-selection policy for the chosen serving beam through the Shield.

    The Shield is a projection operator: an over-EIRP proposal is corrected onto
    the legal cap (33 dBm) rather than passed; a legal one emits cleanly. When a
    correction is applied, the raw action trips ``corrections_not_audited`` so it
    cannot reach the RAN until the correction is on the evidence chain.
    """
    shield = default_terrestrial_shield(
        band_lo_hz=BAND_LO_HZ, band_hi_hz=BAND_HI_HZ, max_eirp_dBm=MAX_EIRP_DBM
    )
    proposed = {
        "block": "policy_emit",
        "policy_type": "horizon.beam.selection",
        "frequency_hz": frequency_hz,
        "bandwidth_hz": BEAM_BANDWIDTH_HZ,
        "tx_power_dBm": tx_power_dbm,
        "antenna_gain_dBi": ANTENNA_GAIN_DBI,
    }
    disp = shield.dispose(proposed, {}, decision_id=decision_id)
    cert = disp.certificate
    guard_fails = run_guard_chain(certificate=cert, elapsed_ms=1.0)
    safe = disp.safe_action
    req_eirp = tx_power_dbm + ANTENNA_GAIN_DBI
    safe_eirp = safe["tx_power_dBm"] + safe["antenna_gain_dBi"]
    store.append(
        DecisionRecord.new(
            decision_id=decision_id,
            rapp_instance_id="beam-management",
            state_hash=f"beam-{beam_index}",
            chosen_action={
                "label": label,
                "policy_type": "horizon.beam.selection",
                "serving_beam_index": beam_index,
                "requested_eirp_dbm": req_eirp,
                "safe_eirp_dbm": safe_eirp,
                "projected": bool(cert.projected),
                "guard_refused": bool(guard_fails),
            },
            predicted_outcome_chosen=PredictedOutcome(
                sla_risk_30s=0.1, sla_risk_1min=0.1, sla_risk_5min=0.1
            ),
            rejected_alternatives=[],
            model_versions=ModelVersions(
                encoder="none",
                risk_heads="beam-codebook",
                dyna="none",
                policy="beam-mgmt",
                constraint_layer="terrestrial-shield",
                rapp="0.2.0",
            ),
        )
    )
    return {
        "label": label,
        "serving_beam_index": beam_index,
        "requested_eirp_dbm": round(req_eirp, 2),
        "safe_eirp_dbm": round(safe_eirp, 2),
        "safe": bool(cert.safe),
        "projected": bool(cert.projected),
        "corrected": bool(safe_eirp < req_eirp - 1e-6),
        "emitted_clean": not bool(guard_fails),
        "guard_refused": [f.guard_id for f in guard_fails],
    }


def run(
    features: Path,
    manifest_path: Path,
    *,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    rows = _load_rows(features)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "sampled_receivers" in manifest and len(rows) != manifest["sampled_receivers"]:
        raise ValueError("feature row count does not match manifest")

    best_beams, array_gain_db, recovery_db, boresight = _beam_sweep(rows)
    histogram = np.bincount(best_beams, minlength=N_BEAMS)
    distinct_beams = int(np.count_nonzero(histogram))
    fraction_non_boresight = float(np.mean(best_beams != boresight))
    modal_beam = int(np.argmax(histogram))
    dominant_aod = np.asarray(
        [r["paths"][0]["aod_az"] for r in rows], dtype=np.float64
    )  # strongest path first in each row.

    # --- Trust chain: gate the modal serving beam + prove tamper-evidence. ----
    ap = audit_path or Path("benchmarks/results/beam_management_audit.jsonl")
    if ap.exists():
        ap.unlink()
    ap.parent.mkdir(parents=True, exist_ok=True)
    store = JsonlEvidenceStore(ap)
    gate_records = [
        _shield_gate(
            store, decision_id="beam-select-modal", label="modal-serving-beam",
            beam_index=modal_beam,
        ),
        # A beam activation that naively requests EIRP 46 dBm the Shield must cap.
        _shield_gate(
            store, decision_id="beam-overpower-select", label="over-eirp-beam",
            beam_index=modal_beam, tx_power_dbm=40.0,
        ),
    ]
    overpower = gate_records[1]
    chain_len = len(store)
    verify_intact = store.verify()

    result: dict[str, Any] = {
        "benchmark": "Analog beam management on real DeepMIMO ray-traced AoD geometry",
        "task": "O-RAN beam-management rApp (SSB-style 8-beam DFT sweep + selection)",
        "dataset": manifest.get("dataset", "DeepMIMO ASU Campus 3.5 GHz (angular ray geometry)"),
        "scenario": manifest.get("scenario", "asu_campus_3p5"),
        "data_kind": manifest.get(
            "data_kind", "site-specific Wireless InSite ray tracing"
        ),
        "features_sha256": manifest.get("features_sha256"),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "receivers": len(rows),
        "bs_array": manifest.get("bs_array"),
        "codebook": {
            "beams": N_BEAMS,
            "kind": "DFT, sin-space tiling u_b=(2b+1)/8-1",
            "boresight_beam_index": boresight,
        },
        "dominant_aod_az_deg_min": round(float(dominant_aod.min()), 4),
        "dominant_aod_az_deg_max": round(float(dominant_aod.max()), 4),
        "coherent_combining_bound_db": round(float(COHERENT_BOUND_DB), 4),
        "mean_array_gain_db": round(float(array_gain_db.mean()), 4),
        "median_array_gain_db": round(float(np.median(array_gain_db)), 4),
        "max_array_gain_db": round(float(array_gain_db.max()), 4),
        "mean_misalignment_recovery_db": round(float(recovery_db.mean()), 4),
        "median_misalignment_recovery_db": round(float(np.median(recovery_db)), 4),
        "beam_selection_histogram": [int(c) for c in histogram],
        "distinct_beams_selected": distinct_beams,
        "fraction_non_boresight": round(fraction_non_boresight, 4),
        "modal_beam_index": modal_beam,
        "trust_chain": {
            "gate_records": gate_records,
            "overpower_beam_corrected": bool(overpower["corrected"]),
            "overpower_beam_safe_eirp_dbm": overpower["safe_eirp_dbm"],
            "overpower_beam_reached_ran": bool(overpower["emitted_clean"]),
            "evidence_chain_length": chain_len,
            "verify_first_broken_index": verify_intact,
        },
        "scope_note": (
            "Single-BS analog beam management: an 8-beam DFT codebook swept over "
            "real ray-traced per-path power/phase/AoD azimuth from DeepMIMO ASU "
            "3.5 GHz, with Shield legality and a tamper-evident evidence chain on "
            "the selected beam. This is site-specific ray tracing, not OTA "
            "capture; one scenario, one base station, azimuth-plane ULA."
        ),
    }

    # --- Tamper-evidence proof on the freshly written chain. ------------------
    lines = ap.read_text(encoding="utf-8").splitlines()
    if len(lines) >= 2:
        lines[1] = lines[1].replace(
            '"serving_beam_index": ', '"serving_beam_index": 999, "_t": ', 1
        )
        ap.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result["trust_chain"]["verify_after_tamper_index"] = JsonlEvidenceStore(
            ap
        ).verify()

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/generated/angular_features.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/angular_manifest.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("benchmarks/results/beam_management.json")
    )
    args = parser.parse_args()

    result = run(args.features, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

    tc = result["trust_chain"]
    # Coherent combining must show up (>6 dB mean) and respect the physical
    # 10*log10(8) bound; steering to the selected beam must beat a fixed beam;
    # the real AoD spread must exercise the codebook; the trust chain must hold
    # and catch the tamper.
    combining_real = 6.0 < result["mean_array_gain_db"] <= COHERENT_BOUND_DB + 0.1
    steering_pays = result["mean_misalignment_recovery_db"] > 1.0
    codebook_exercised = result["distinct_beams_selected"] >= 4
    chain_ok = (
        tc["verify_first_broken_index"] == -1
        and not tc["overpower_beam_reached_ran"]
        and tc.get("verify_after_tamper_index") not in (None, -1)
    )
    return 0 if (combining_real and steering_pays and codebook_exercised and chain_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
