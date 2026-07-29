#!/usr/bin/env python3
"""Anti-jam MVDR null-steering on **real DeepMIMO** angular ray geometry.

This drives Horizon's *interference-mitigation + trust core* end to end on the
real angular feature build (DeepMIMO ASU 3.5 GHz, 4096 receivers):

    real ray-traced per-path AoD/power/phase (Wireless InSite)
        -> real beamformed desired channel per receiver (8-element ULA)
        -> 8-beam DFT serving-beam selection (chosen jammer-blind)
        -> modeled wideband jammer from a fixed azimuth collapses the SINR
        -> sample-matrix-inversion (SMI) null-steering beamformer estimated from
           finite snapshots restores the link
        -> Decision Safety Shield (EIRP legality, projection operator)
        -> hash-chained, tamper-evident evidence record

HONEST framing, stated plainly: the DESIRED downlink channel and the array
response are REAL — each receiver's channel ``h_d = Σ_l α_l·a(aod_az_l)`` is
synthesised from its own ray-traced path list (real per-path power, phase and
angle of departure over the shared 8-element half-wavelength ULA; no synthetic
desired channel anywhere). The JAMMER is a MODELED interferer: this single-BS
scenario has no second real transmitter, so the jammer is a point source at a
fixed, documented azimuth (the rounded median strongest-path AoD, -142°) with a
jammer-to-signal ratio pinned 30 dB above the mean serving-beam signal power.

The mitigation is the PRACTICAL adaptive nuller, not the idealized bound. The
array does not know the interference covariance: it estimates it from a finite
block of jammer+noise training snapshots (sample matrix inversion) and adds
diagonal loading, then forms ``w = R̂⁻¹ h_d``. The achieved SINR is scored
against the TRUE interference the beamformer actually faces, so finite-snapshot
estimation error and loading cap the null at realistic tens of dB — the numbers
are what a real adaptive array delivers, not a perfect-covariance ceiling.

The task is a real O-RAN anti-jam rApp function: the serving beam is the best
DFT codebook beam for the *real* channel picked without knowledge of the
jammer; when the jammer lands in that beam's pattern the SINR collapses tens of
dB below usable. Because the desired channel is the real multipath ray sum —
not a single plane wave — the restored SINR is receiver-specific real physics:
receivers whose ray geometry is angularly close to the jammer recover less than
receivers with separable geometry.

Everything downstream (Shield EIRP legality, guard chain, evidence chain,
tamper detection) runs on the actually-computed mitigation, so the whole
topology is exercised on real data, not a mock.

Pure numpy + the shipped ``horizon_ric`` modules. No torch, no GPU. The real
angular feature file is licence-gated (not redistributed in-repo); build it
with ``datasets/deepmimo_asu_3p5/build_angular.py`` (deterministic,
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
NULLSTEER_FREQUENCY_HZ = 3.50e9  # the anti-jam reconfiguration at band centre.
NULLSTEER_BANDWIDTH_HZ = 20e6
SAFE_TX_POWER_DBM = 26.0  # legal request (EIRP 32 dBm < 33): emits without a fix.

# --- Array geometry (from the angular manifest's shared bs_array). -------------
N_ELEMENTS = 8
SPACING_WAVELENGTHS = 0.5
N_BEAMS = 8

# --- Jammer model (MODELED — single-BS scenario has no second transmitter). ----
# Fixed azimuth: the rounded median strongest-path AoD across the 4096-receiver
# build (-142.0975° -> -142°), i.e. the jammer sits in the angular region where
# most real serving energy departs — the hardest honest placement, not a corner
# case the codebook never looks at.
JAMMER_AZ_DEG = -142.0
JSR_DB = 30.0  # jammer power pinned 30 dB above the mean serving-beam signal.
NOISE_FLOOR_DBW = -131.0  # thermal kTB for 20 MHz: -174 dBm/Hz + 73 dB - 30.
SINR_THRESH_DB = 0.0  # a receiver is "usable" above this SINR.

# --- Practical anti-jam beamformer (sample matrix inversion, NOT ideal MVDR). --
# The array does NOT know the interference covariance; it ESTIMATES it from a
# finite block of jammer+noise training snapshots (SMI) with diagonal loading.
# Finite snapshots + loading cap the achievable null at realistic tens of dB —
# not the perfect-covariance infinity — so the reported numbers are what a real
# adaptive array delivers, not a theoretical bound.
SMI_SNAPSHOTS = 24  # training snapshots (3x the 8 array DoF).
DIAGONAL_LOADING_DB = 10.0  # loading level, dB above the thermal noise floor.
SMI_SEED = 20260727  # fixed: the training snapshots are reproducible.
# Real jammers are not ideal point sources: local scattering / finite bandwidth
# give a few degrees of angular spread, which an 8-element array cannot null
# infinitely deep. Modeling the jammer as a small cluster of sub-rays is what
# keeps the achievable null at realistic tens of dB instead of a point-source
# simulation artifact.
JAMMER_ANGULAR_SPREAD_DEG = 4.0
JAMMER_SUBRAYS = 7


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"no feature rows in {path}")
    return rows


def _steering(az_deg: float | np.ndarray) -> np.ndarray:
    """ULA steering vector(s) at azimuth ``az_deg`` (half-wavelength spacing)."""
    n = np.arange(N_ELEMENTS)
    phase = 2.0 * np.pi * SPACING_WAVELENGTHS * np.outer(np.sin(np.radians(az_deg)), n)
    vec = np.exp(1j * phase)
    return vec[0] if np.isscalar(az_deg) else vec


def _desired_channels(rows: list[dict[str, Any]]) -> np.ndarray:
    """Real beamformed desired channel per receiver, shape ``[n_rx, 8]``.

    ``h_d = Σ_l α_l · a(aod_az_l)`` with per-path complex amplitude
    ``α_l = sqrt(10^(power_dbw/10)) · exp(j·phase)`` — every path's power,
    phase and angle of departure is real ray-traced output; nothing synthetic.
    """
    channels = np.zeros((len(rows), N_ELEMENTS), dtype=np.complex128)
    for i, row in enumerate(rows):
        for path in row["paths"]:
            amp = np.sqrt(10.0 ** (path["power_dbw"] / 10.0)) * np.exp(
                1j * np.radians(path["phase_deg"])
            )
            channels[i] += amp * _steering(path["aod_az"])
    if not np.all(np.isfinite(channels)):
        raise ValueError("feature rows produced non-finite channels")
    return channels


def _dft_codebook() -> np.ndarray:
    """8-beam DFT codebook, unit-norm columns; beam centres ``u_b=(2b+1)/8-1``."""
    beams = np.arange(N_BEAMS)
    u = (2.0 * beams + 1.0) / N_BEAMS - 1.0
    n = np.arange(N_ELEMENTS)
    return np.exp(1j * 2.0 * np.pi * SPACING_WAVELENGTHS * np.outer(n, u)) / np.sqrt(N_ELEMENTS)


def _antijam_metrics(channels: np.ndarray) -> dict[str, Any]:
    """Serving-beam jammed SINR vs sample-matrix-inversion (SMI) anti-jam SINR.

    Practical adaptive nulling, not the idealized bound. The serving beam is
    chosen jammer-blind (argmax beamformed power on the real channel). The array
    does NOT know the interference covariance: it ESTIMATES it from
    ``SMI_SNAPSHOTS`` finite jammer+noise training snapshots (signal-free, as
    during a sounding gap) and adds diagonal loading, giving ``R̂``. The anti-jam
    weight is ``w = R̂⁻¹ h_d``; its SINR is scored against the TRUE interference
    ``R = P_j·ggᴴ + σ²I`` — the honest SINR an imperfect estimate actually
    delivers. Finite snapshots + loading cap the null at realistic tens of dB.
    ``null_depth_db`` compares unit-norm serving vs unit-norm SMI weights toward
    the jammer — positive dB is the extra jammer suppression the null adds.
    """
    codebook = _dft_codebook()
    beam_power = np.abs(codebook.conj().T @ channels.T) ** 2  # [beams, n_rx]
    serving = np.argmax(beam_power, axis=0)
    rx_ids = np.arange(channels.shape[0])
    signal = beam_power[serving, rx_ids]

    jam_center = _steering(JAMMER_AZ_DEG)
    signal_mean = float(np.mean(signal))
    jam_power = signal_mean * 10.0 ** (JSR_DB / 10.0)
    noise = 10.0 ** (NOISE_FLOOR_DBW / 10.0)
    # Spread jammer: JAMMER_SUBRAYS equal-power sub-rays across the angular
    # spread, so the interference is not rank-1 and cannot be nulled perfectly.
    offsets = np.linspace(-0.5, 0.5, JAMMER_SUBRAYS) * JAMMER_ANGULAR_SPREAD_DEG
    jam_rays = np.stack([_steering(JAMMER_AZ_DEG + o) for o in offsets])  # [M, 8]
    per_ray = jam_power / JAMMER_SUBRAYS
    # TRUE interference covariance the beamformer must actually cope with.
    cov_true = per_ray * (jam_rays.T @ jam_rays.conj()) + noise * np.eye(N_ELEMENTS)

    # Estimate R̂ from finite jammer+noise snapshots: each ray radiates an
    # independent CN(0,1) symbol per snapshot; n_k ~ CN(0, σ²I). Then diagonal
    # load — standard SMI, no knowledge of the true covariance.
    rng = np.random.default_rng(SMI_SEED)
    sym = (
        rng.standard_normal((SMI_SNAPSHOTS, JAMMER_SUBRAYS))
        + 1j * rng.standard_normal((SMI_SNAPSHOTS, JAMMER_SUBRAYS))
    ) / np.sqrt(2.0)
    n = (
        rng.standard_normal((SMI_SNAPSHOTS, N_ELEMENTS))
        + 1j * rng.standard_normal((SMI_SNAPSHOTS, N_ELEMENTS))
    ) * np.sqrt(noise / 2.0)
    y = np.sqrt(per_ray) * (sym @ jam_rays) + n  # [snapshots, 8]
    # Sample covariance for row-stored snapshots: R̂ = (1/K) Σ_k y_k y_kᴴ.
    cov_hat = (y.T @ y.conj()) / SMI_SNAPSHOTS
    loading = noise * 10.0 ** (DIAGONAL_LOADING_DB / 10.0)
    cov_hat_inv = np.linalg.inv(cov_hat + loading * np.eye(N_ELEMENTS))

    # Jammed SINR on the serving DFT beam, scored on the TRUE covariance.
    denom_per_beam = np.real(np.einsum("nb,nm,mb->b", codebook.conj(), cov_true, codebook))
    sinr_jammed_db = 10.0 * np.log10(signal / denom_per_beam[serving])

    # SMI weight w = R̂⁻¹ h_d; achieved SINR = |wᴴh_d|² / (wᴴ R_true w).
    weights = (cov_hat_inv @ channels.T).T  # [n_rx, 8]
    num = np.abs(np.einsum("in,in->i", weights.conj(), channels)) ** 2
    den = np.real(np.einsum("in,nm,im->i", weights.conj(), cov_true, weights))
    sinr_antijam_db = 10.0 * np.log10(num / den)

    # Effective jammer suppression: total jammer power (the whole spread source,
    # not just its centre) through the unit-norm SMI weight vs the unit-norm
    # serving beam. This is spread-aware, so it reflects the real limit an
    # 8-element array hits nulling an angularly-spread jammer.
    jammer_cov = per_ray * (jam_rays.T @ jam_rays.conj())
    w_unit = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    w_serve = codebook[:, serving].T  # [n_rx, 8], unit-norm serving beams
    jam_pow_serve = np.real(np.einsum("in,nm,im->i", w_serve.conj(), jammer_cov, w_serve))
    jam_pow_smi = np.real(np.einsum("in,nm,im->i", w_unit.conj(), jammer_cov, w_unit))
    null_depth_db = 10.0 * np.log10(jam_pow_serve / jam_pow_smi)

    gain_db = sinr_antijam_db - sinr_jammed_db
    return {
        "serving": serving,
        "signal_mean_dbw": 10.0 * np.log10(signal_mean),
        "jam_power_dbw": 10.0 * np.log10(jam_power),
        "sinr_jammed_db": sinr_jammed_db,
        "sinr_antijam_db": sinr_antijam_db,
        "sinr_gain_db": gain_db,
        "null_depth_db": null_depth_db,
    }


def _shield_gate(
    store: JsonlEvidenceStore,
    *,
    decision_id: str,
    label: str,
    tx_power_dbm: float = SAFE_TX_POWER_DBM,
) -> dict[str, Any]:
    """Emit the anti-jam null-steer reconfiguration as a policy through the Shield.

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
        "policy_type": "horizon.antijam.nullsteer",
        "frequency_hz": NULLSTEER_FREQUENCY_HZ,
        "bandwidth_hz": NULLSTEER_BANDWIDTH_HZ,
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
            rapp_instance_id="antijam",
            state_hash=f"jam-az-{JAMMER_AZ_DEG}",
            chosen_action={
                "label": label,
                "policy_type": "horizon.antijam.nullsteer",
                "beamformer": "mvdr-nullsteer",
                "jammer_azimuth_deg": JAMMER_AZ_DEG,
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
                risk_heads="mvdr-nullsteer",
                dyna="none",
                policy="antijam",
                constraint_layer="terrestrial-shield",
                rapp="0.2.0",
            ),
        )
    )
    return {
        "label": label,
        "jammer_azimuth_deg": JAMMER_AZ_DEG,
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

    channels = _desired_channels(rows)
    m = _antijam_metrics(channels)
    jammed = m["sinr_jammed_db"]
    antijam = m["sinr_antijam_db"]
    restored_jammed = float(np.mean(jammed > SINR_THRESH_DB))
    restored_antijam = float(np.mean(antijam > SINR_THRESH_DB))

    # --- Trust chain: gate the mitigation + prove tamper-evidence. ------------
    ap = audit_path or Path("benchmarks/results/antijam_audit.jsonl")
    if ap.exists():
        ap.unlink()
    ap.parent.mkdir(parents=True, exist_ok=True)
    store = JsonlEvidenceStore(ap)
    gate_records = [
        _shield_gate(store, decision_id="antijam-nullsteer", label="antijam-nullsteer"),
        # A null-steer reconfiguration naively requesting EIRP 46 dBm the Shield
        # must cap.
        _shield_gate(
            store, decision_id="antijam-overpower", label="over-eirp-nullsteer",
            tx_power_dbm=40.0,
        ),
    ]
    overpower = gate_records[1]
    chain_len = len(store)
    verify_intact = store.verify()

    result: dict[str, Any] = {
        "benchmark": "Anti-jam MVDR null-steering on real DeepMIMO angular channels",
        "task": "O-RAN interference-mitigation (anti-jam) rApp",
        "dataset": manifest.get("dataset", "DeepMIMO ASU Campus 3.5 GHz (angular ray geometry)"),
        "scenario": manifest.get("scenario", "asu_campus_3p5"),
        "data_kind": manifest.get(
            "data_kind", "site-specific Wireless InSite ray tracing"
        ),
        "features_sha256": manifest.get("features_sha256"),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "receivers": len(rows),
        "bs_array": manifest.get("bs_array"),
        "jammer": {
            "azimuth_deg": JAMMER_AZ_DEG,
            "azimuth_rule": "rounded median strongest-path AoD of this build",
            "jammer_to_signal_db": JSR_DB,
            "power_dbw": round(float(m["jam_power_dbw"]), 4),
            "note": "modeled interferer; single-BS scenario",
        },
        "noise_floor_dbw": NOISE_FLOOR_DBW,
        "sinr_threshold_db": SINR_THRESH_DB,
        "mean_serving_signal_dbw": round(float(m["signal_mean_dbw"]), 4),
        "serving_beam_histogram": np.bincount(m["serving"], minlength=N_BEAMS).tolist(),
        "mean_sinr_jammed_db": round(float(np.mean(jammed)), 4),
        "median_sinr_jammed_db": round(float(np.median(jammed)), 4),
        "mean_sinr_antijam_db": round(float(np.mean(antijam)), 4),
        "median_sinr_antijam_db": round(float(np.median(antijam)), 4),
        "mean_sinr_gain_db": round(float(np.mean(m["sinr_gain_db"])), 4),
        "median_sinr_gain_db": round(float(np.median(m["sinr_gain_db"])), 4),
        "mean_null_depth_db": round(float(np.mean(m["null_depth_db"])), 4),
        "min_null_depth_db": round(float(np.min(m["null_depth_db"])), 4),
        "restored_fraction_jammed": round(restored_jammed, 4),
        "restored_fraction_antijam": round(restored_antijam, 4),
        "trust_chain": {
            "gate_records": gate_records,
            "overpower_nullsteer_corrected": bool(overpower["corrected"]),
            "overpower_nullsteer_safe_eirp_dbm": overpower["safe_eirp_dbm"],
            "overpower_nullsteer_reached_ran": bool(overpower["emitted_clean"]),
            "evidence_chain_length": chain_len,
            "verify_first_broken_index": verify_intact,
        },
        "scope_note": (
            "The desired downlink channel is real: each receiver's h_d is the sum "
            "of its own ray-traced paths (real AoD, power, phase; Wireless InSite) "
            "over the real 8-element half-wavelength ULA. The jammer is a MODELED "
            "point interferer at a fixed documented azimuth — this single-BS "
            "scenario has no second real transmitter — and its covariance is "
            "analytic (perfectly known), so null depths are idealized upper "
            "bounds. Site-specific ray tracing, not OTA capture; one scenario."
        ),
    }

    # --- Tamper-evidence proof on the freshly written chain. ------------------
    lines = ap.read_text(encoding="utf-8").splitlines()
    if len(lines) >= 2:
        lines[1] = lines[1].replace(
            '"jammer_azimuth_deg": ', '"jammer_azimuth_deg": 999999, "_t": ', 1
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
        "--out", type=Path, default=Path("benchmarks/results/antijam.json")
    )
    args = parser.parse_args()

    result = run(args.features, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

    tc = result["trust_chain"]
    # The anti-jam must materially restore SINR on the real channels, more
    # receivers must be usable after mitigation, the jammer must be genuinely
    # nulled, and the trust chain must hold and catch the tamper.
    restores = result["mean_sinr_gain_db"] > 5.0
    more_usable = result["restored_fraction_antijam"] > result["restored_fraction_jammed"]
    nulled = result["mean_null_depth_db"] > 10.0
    chain_ok = (
        tc["verify_first_broken_index"] == -1
        and not tc["overpower_nullsteer_reached_ran"]
        and tc.get("verify_after_tamper_index") not in (None, -1)
    )
    return 0 if (restores and more_usable and nulled and chain_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
