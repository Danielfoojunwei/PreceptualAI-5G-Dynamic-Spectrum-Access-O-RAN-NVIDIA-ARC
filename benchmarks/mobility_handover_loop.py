#!/usr/bin/env python3
"""Beam-level mobility / handover with Doppler ON, on **real DeepMIMO** ray geometry.

This drives Horizon's mobility decision path end to end on real measured-physics
angle-of-departure/arrival geometry — no synthetic RF anywhere in the loop:

    real per-path ray geometry (DeepMIMO ASU 3.5 GHz, 4096 receivers)
        -> real spatial corridor: receivers with y in [75, 85] m sorted by x
           (a 198-point, ~1 km west-to-east drive past the BS at [166, 104, 22])
        -> 8-element ULA channel h = sum_l alpha_l * a(aod_az_l)  (real AoD)
        -> per-path Doppler from the real AoA azimuths + the UE velocity vector
           (phase advance 2*pi*f_d,l*dt accumulated along the drive)
        -> 8-beam DFT codebook -> per-beam gain timeline
        -> A3-event handover policy (hysteresis + time-to-trigger) vs greedy
        -> Decision Safety Shield (EIRP legality, projection operator)
        -> hash-chained, tamper-evident evidence record

The task is a real O-RAN Mobility-Robustness-Optimization (MRO) rApp function:
track the best serving beam for a moving UE and hand over only on a confirmed
A3 event (candidate beam better than serving by HYST dB for TTT consecutive
samples), instead of greedily chasing the instantaneous argmax beam. On this
real corridor the greedy tracker flaps between adjacent DFT beams in the
Doppler-driven fading (many ping-pongs); the A3 policy cuts ping-pongs to zero
while keeping the UE within 3 dB of the best available beam for ~90% of samples.

Honest framing — read this before citing numbers:

* This is **intra-cell BEAM handover**: the scenario has a single base station
  at [166, 104, 22] with an 8-element ULA, and the "handover" is between beams
  of its 8-beam DFT codebook. It is NOT inter-gNB handover.
* The trajectory, per-path powers/phases and AoD/AoA angles are real Wireless
  InSite ray tracing. The **Doppler is synthesized** from that real per-path
  AoA geometry plus a chosen UE velocity (12 m/s along the corridor); the ray
  tracer itself is static. Note the per-sample phase advance
  2*pi*f_d,l*dt = 2*pi*(step_m/lambda)*cos(heading - aoa_az_l) is
  distance-driven (speed cancels), so the beam decisions depend only on the
  real geometry; the speed choice sets the reported Doppler shift and
  coherence time.
* Ray tracing, not over-the-air capture; one scenario.

Pure numpy + the shipped ``horizon_ric`` modules. No torch, no GPU. The real
DeepMIMO angular feature file is licence-gated (not redistributed in-repo);
build it with ``datasets/deepmimo_asu_3p5/build_angular.py`` (deterministic,
checksum-pinned) or point ``--features`` at a cached copy. The committed result
JSON is the real 4096-Rx run; ``scripts/verify_mobility_handover.py`` re-checks
a fresh rebuild against it on host-stable invariants.
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

# --- Spectrum / EIRP geometry (shared with the DSA + coverage benchmarks). -----
BAND_LO_HZ = 3.45e9
BAND_HI_HZ = 3.55e9
MAX_EIRP_DBM = 33.0
ANTENNA_GAIN_DBI = 6.0
HO_FREQUENCY_HZ = 3.50e9  # the beam-handover policy rides the band centre.
HO_BANDWIDTH_HZ = 20e6
SAFE_TX_POWER_DBM = 26.0  # legal request (EIRP 32 dBm < 33): emits without a fix.

# --- Array / codebook geometry (matches the angular manifest bs_array). --------
N_ANT = 8
CARRIER_HZ = 3.5e9
LAMBDA_M = 3e8 / CARRIER_HZ  # ~0.0857 m
MAX_PATHS = 10

# --- Trajectory / mobility defaults. -------------------------------------------
CORRIDOR_Y_M = 80.0  # dense east-west receiver band passing ~24 m south of the BS
CORRIDOR_HALFWIDTH_M = 5.0
UE_SPEED_MPS = 12.0  # ~43 km/h campus drive
HYST_DB = 3.0  # A3 hysteresis
TTT_SAMPLES = 3  # A3 time-to-trigger, in trajectory samples
PP_WINDOW_SAMPLES = 5  # A -> B -> A within this window counts as a ping-pong
OUTAGE_MARGIN_DB = 3.0  # serving > 3 dB below the best beam = outage sample


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"no feature rows in {path}")
    return rows


def _steering(az_deg: np.ndarray) -> np.ndarray:
    """ULA steering vectors ``a(az)[n] = exp(j*2*pi*0.5*n*sin(az))``; shape [L, 8]."""
    n = np.arange(N_ANT)
    return np.exp(1j * 2 * np.pi * 0.5 * n[None, :] * np.sin(np.radians(az_deg))[:, None])


def _dft_codebook() -> np.ndarray:
    """8-beam DFT codebook W (shape [8, 8]); beam b points at u_b = (2b+1)/8 - 1."""
    n = np.arange(N_ANT)
    u = (2 * np.arange(N_ANT) + 1) / N_ANT - 1
    return (1 / np.sqrt(N_ANT)) * np.exp(1j * 2 * np.pi * 0.5 * np.outer(n, u))


def _beam_boresights_deg() -> list[float]:
    u = (2 * np.arange(N_ANT) + 1) / N_ANT - 1
    return [round(float(np.degrees(np.arcsin(v))), 2) for v in u]


def _corridor_trajectory(
    rows: list[dict[str, Any]], y_center: float, halfwidth: float
) -> list[dict[str, Any]]:
    """Real drive: receivers with y in the corridor band, ordered west-to-east.

    Every trajectory point is a real ray-traced receiver position; nothing is
    interpolated or invented. Sorting by x turns the band into a monotone
    west-to-east drive (in this band each receiver has a unique x).
    """
    sel = [r for r in rows if y_center - halfwidth <= r["position_m"][1] <= y_center + halfwidth]
    sel.sort(key=lambda r: (r["position_m"][0], r["position_m"][1]))
    if len(sel) < 2:
        raise ValueError("corridor selects fewer than 2 receivers")
    return sel


def _beam_gain_timeline(
    traj: list[dict[str, Any]], speed_mps: float
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Per-beam gain (dB) along the drive with Doppler-driven phase evolution.

    Heading is the finite-difference direction of travel; each path's Doppler
    ``f_d,l = (|v|/lambda) * cos(heading - aoa_az_l)`` uses the path's real AoA
    azimuth, and its phase is advanced by the accumulated ``2*pi*f_d,l*dt``
    (dt = step distance / speed). Returns (gains_db [T, 8], max |f_d| Hz,
    per-step dt [T], step distances [T]).
    """
    pos = np.asarray([r["position_m"][:2] for r in traj], dtype=np.float64)
    diffs = np.diff(pos, axis=0)
    steps = np.linalg.norm(diffs, axis=1)
    if not np.all(steps > 0):
        raise ValueError("trajectory contains a zero-length step")
    heading = np.degrees(np.arctan2(diffs[:, 1], diffs[:, 0]))
    heading = np.concatenate([heading, heading[-1:]])  # final point keeps its heading
    dt = np.concatenate([[0.0], steps / speed_mps])
    steps = np.concatenate([[0.0], steps])

    codebook = _dft_codebook()
    psi = np.zeros(MAX_PATHS)  # accumulated Doppler phase per path slot (strongest-first)
    gains_db = np.zeros((len(traj), N_ANT))
    fd_max = 0.0
    for k, row in enumerate(traj):
        paths = row["paths"]
        aoa_az = np.asarray([p["aoa_az"] for p in paths], dtype=np.float64)
        fd = (speed_mps / LAMBDA_M) * np.cos(np.radians(heading[k] - aoa_az))
        fd_max = max(fd_max, float(np.abs(fd).max()))
        psi[: len(paths)] += 2 * np.pi * fd * dt[k]
        amp = np.asarray([np.sqrt(10 ** (p["power_dbw"] / 10)) for p in paths])
        phase = np.asarray([np.radians(p["phase_deg"]) for p in paths]) + psi[: len(paths)]
        alpha = amp * np.exp(1j * phase)
        h = (alpha[:, None] * _steering(np.asarray([p["aod_az"] for p in paths]))).sum(axis=0)
        beam_power = np.abs(codebook.conj().T @ h) ** 2
        gains_db[k] = 10 * np.log10(np.maximum(beam_power, 1e-30))
    if not np.all(np.isfinite(gains_db)):
        raise ValueError("non-finite beam gains")
    return gains_db, fd_max, dt, steps


def _serve_greedy(gains_db: np.ndarray) -> np.ndarray:
    """Baseline: serve the instantaneous argmax beam every sample (0 dB, TTT=1)."""
    return np.argmax(gains_db, axis=1)


def _serve_a3(gains_db: np.ndarray, hyst_db: float, ttt: int) -> np.ndarray:
    """A3-event tracker: switch to the best beam only after it beats the serving
    beam by ``hyst_db`` for ``ttt`` consecutive samples."""
    t_len = gains_db.shape[0]
    serving = int(np.argmax(gains_db[0]))
    candidate, count = -1, 0
    out = np.zeros(t_len, dtype=np.int64)
    for k in range(t_len):
        best = int(np.argmax(gains_db[k]))
        if best != serving and gains_db[k, best] > gains_db[k, serving] + hyst_db:
            count = count + 1 if best == candidate else 1
            candidate = best
            if count >= ttt:
                serving, candidate, count = best, -1, 0
        else:
            candidate, count = -1, 0
        out[k] = serving
    return out


def _mobility_metrics(serving: np.ndarray, gains_db: np.ndarray) -> dict[str, Any]:
    """Handover count, ping-pong count (A->B->A within PP_WINDOW_SAMPLES) and
    outage fraction (serving beam > OUTAGE_MARGIN_DB below the best beam)."""
    t_len = len(serving)
    events = [
        (k, int(serving[k - 1]), int(serving[k]))
        for k in range(1, t_len)
        if serving[k] != serving[k - 1]
    ]
    ping_pongs = sum(
        1
        for i in range(1, len(events))
        if events[i][2] == events[i - 1][1]
        and events[i][0] - events[i - 1][0] <= PP_WINDOW_SAMPLES
    )
    best_gain = gains_db.max(axis=1)
    serving_gain = gains_db[np.arange(t_len), serving]
    outage = float(np.mean(serving_gain < best_gain - OUTAGE_MARGIN_DB))
    return {
        "handovers": len(events),
        "ping_pongs": int(ping_pongs),
        "outage_fraction": round(outage, 4),
        "beams_visited": sorted({int(b) for b in serving}),
    }


def _shield_gate(
    store: JsonlEvidenceStore,
    *,
    decision_id: str,
    label: str,
    target_beam: int,
    from_beam: int,
    tx_power_dbm: float = SAFE_TX_POWER_DBM,
) -> dict[str, Any]:
    """Emit the beam-handover policy for the selected target beam through the Shield.

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
        "policy_type": "horizon.mobility.handover",
        "frequency_hz": HO_FREQUENCY_HZ,
        "bandwidth_hz": HO_BANDWIDTH_HZ,
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
            rapp_instance_id="mobility-handover",
            state_hash=f"beam-{from_beam}-to-{target_beam}",
            chosen_action={
                "label": label,
                "policy_type": "horizon.mobility.handover",
                "from_beam": from_beam,
                "target_beam": target_beam,
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
                risk_heads="beam-tracker",
                dyna="none",
                policy="mro",
                constraint_layer="terrestrial-shield",
                rapp="0.2.0",
            ),
        )
    )
    return {
        "label": label,
        "from_beam": from_beam,
        "target_beam": target_beam,
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
    speed_mps: float = UE_SPEED_MPS,
    hyst_db: float = HYST_DB,
    ttt: int = TTT_SAMPLES,
) -> dict[str, Any]:
    rows = _load_rows(features)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "sampled_receivers" in manifest and len(rows) != manifest["sampled_receivers"]:
        raise ValueError("feature row count does not match manifest")

    traj = _corridor_trajectory(rows, CORRIDOR_Y_M, CORRIDOR_HALFWIDTH_M)
    gains_db, fd_max, dt, steps = _beam_gain_timeline(traj, speed_mps)

    greedy_serving = _serve_greedy(gains_db)
    a3_serving = _serve_a3(gains_db, hyst_db, ttt)
    greedy = _mobility_metrics(greedy_serving, gains_db)
    hysteresis = _mobility_metrics(a3_serving, gains_db)

    # The confirmed A3 handover the rApp would actually emit: the last beam
    # switch of the drive (into the final serving beam).
    final_beam = int(a3_serving[-1])
    prev_beams = a3_serving[a3_serving != final_beam]
    from_beam = int(prev_beams[-1]) if len(prev_beams) else final_beam

    # --- Trust chain: gate the target beam + prove tamper-evidence. -----------
    ap = audit_path or Path("benchmarks/results/mobility_handover_audit.jsonl")
    if ap.exists():
        ap.unlink()
    ap.parent.mkdir(parents=True, exist_ok=True)
    store = JsonlEvidenceStore(ap)
    gate_records = [
        _shield_gate(
            store, decision_id="mro-beam-handover", label="a3-confirmed-handover",
            target_beam=final_beam, from_beam=from_beam,
        ),
        # A handover emit that naively requests EIRP 46 dBm the Shield must cap.
        _shield_gate(
            store, decision_id="mro-overpower-handover", label="over-eirp-handover",
            target_beam=final_beam, from_beam=from_beam, tx_power_dbm=40.0,
        ),
    ]
    overpower = gate_records[1]
    chain_len = len(store)
    verify_intact = store.verify()

    positions = [r["position_m"] for r in traj]
    result: dict[str, Any] = {
        "benchmark": "Beam-level mobility / handover with Doppler on real DeepMIMO ray geometry",
        "task": "O-RAN Mobility-Robustness-Optimization (MRO) rApp, intra-cell beam handover",
        "dataset": manifest.get("dataset", "DeepMIMO ASU Campus 3.5 GHz (angular ray geometry)"),
        "scenario": manifest.get("scenario", "asu_campus_3p5"),
        "data_kind": manifest.get(
            "data_kind", "site-specific Wireless InSite ray tracing"
        ),
        "features_sha256": manifest.get("features_sha256"),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "receivers": len(rows),
        "bs_position_m": manifest.get("tx_position_m", [166.0, 104.0, 22.0]),
        "beam_codebook": {
            "type": "8-beam DFT over an 8-element half-wavelength ULA",
            "boresights_deg": _beam_boresights_deg(),
        },
        "trajectory": {
            "points": len(traj),
            "corridor": {
                "y_center_m": CORRIDOR_Y_M,
                "y_halfwidth_m": CORRIDOR_HALFWIDTH_M,
                "x_min_m": round(float(positions[0][0]), 3),
                "x_max_m": round(float(positions[-1][0]), 3),
                "direction": "west-to-east",
            },
            "ue_speed_mps": speed_mps,
            "delta_t_s": round(float(dt[1:].mean()), 4),
            "total_distance_m": round(float(steps.sum()), 1),
            "total_time_s": round(float(dt.sum()), 1),
        },
        "doppler": {
            "enabled": True,
            "max_shift_hz": round(fd_max, 2),
            "coherence_time_s": round(0.423 / fd_max, 6),
            "model": (
                "per-path f_d = (|v|/lambda) * cos(heading - aoa_az); phase advanced "
                "by accumulated 2*pi*f_d*dt along the real trajectory"
            ),
        },
        "greedy": {
            "policy": "serve argmax beam every sample (hysteresis 0 dB, TTT 1)",
            "handovers": greedy["handovers"],
            "ping_pongs": greedy["ping_pongs"],
            "outage_fraction": greedy["outage_fraction"],
            "beams_visited": greedy["beams_visited"],
        },
        "hysteresis": {
            "policy": "A3 event: candidate > serving + HYST for TTT consecutive samples",
            "hyst_db": hyst_db,
            "ttt_samples": ttt,
            "ping_pong_window_samples": PP_WINDOW_SAMPLES,
            "outage_margin_db": OUTAGE_MARGIN_DB,
            "handovers": hysteresis["handovers"],
            "ping_pongs": hysteresis["ping_pongs"],
            "outage_fraction": hysteresis["outage_fraction"],
            "beams_visited": hysteresis["beams_visited"],
            "final_serving_beam": final_beam,
        },
        "trust_chain": {
            "gate_records": gate_records,
            "overpower_handover_corrected": bool(overpower["corrected"]),
            "overpower_handover_safe_eirp_dbm": overpower["safe_eirp_dbm"],
            "overpower_handover_reached_ran": bool(overpower["emitted_clean"]),
            "evidence_chain_length": chain_len,
            "verify_first_broken_index": verify_intact,
        },
        "scope_note": (
            "Intra-cell BEAM handover: a single base station at [166, 104, 22] with an "
            "8-element ULA; the handover is between beams of its 8-beam DFT codebook, "
            "not between gNBs. The trajectory, path powers/phases and AoD/AoA angles are "
            "real Wireless InSite ray tracing; the Doppler is synthesized from that real "
            "per-path AoA geometry plus a chosen UE velocity (12 m/s along the corridor). "
            "Site-specific ray tracing, not OTA capture; one scenario."
        ),
    }

    # --- Tamper-evidence proof on the freshly written chain. ------------------
    lines = ap.read_text(encoding="utf-8").splitlines()
    if len(lines) >= 2:
        lines[1] = lines[1].replace('"target_beam": ', '"target_beam": 999999, "_t": ', 1)
        ap.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result["trust_chain"]["verify_after_tamper_index"] = JsonlEvidenceStore(ap).verify()

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
        "--out", type=Path, default=Path("benchmarks/results/mobility_handover.json")
    )
    parser.add_argument("--speed", type=float, default=UE_SPEED_MPS)
    parser.add_argument("--hyst-db", type=float, default=HYST_DB)
    parser.add_argument("--ttt", type=int, default=TTT_SAMPLES)
    args = parser.parse_args()

    result = run(
        args.features, args.manifest,
        speed_mps=args.speed, hyst_db=args.hyst_db, ttt=args.ttt,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

    g, h = result["greedy"], result["hysteresis"]
    tc = result["trust_chain"]
    # Doppler must be genuinely on; the A3 policy must cut ping-pongs while still
    # tracking the best beam and making at least one real handover; the trust
    # chain must hold, refuse the over-EIRP emit, and catch the tamper.
    doppler_on = result["doppler"]["max_shift_hz"] > 0.0
    pp_reduced = g["ping_pongs"] > h["ping_pongs"]
    tracks = h["outage_fraction"] <= 0.25
    moved = h["handovers"] >= 1
    chain_ok = (
        tc["verify_first_broken_index"] == -1
        and not tc["overpower_handover_reached_ran"]
        and tc.get("verify_after_tamper_index") not in (None, -1)
    )
    return 0 if (doppler_on and pp_reduced and tracks and moved and chain_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
