#!/usr/bin/env python3
"""Federated coverage-map learning on **real DeepMIMO** ray-traced channels.

This drives Horizon's *federated learning + trust core* end to end on real
measured-physics data — no synthetic gradient vectors anywhere in the loop:

    real ray-traced channels (DeepMIMO ASU 3.5 GHz, 4096 receivers)
        -> geographic, non-IID federated clients (partitioned by real position)
        -> rounds of FedProx local ridge solves on each client's own channels
        -> Byzantine-robust aggregation (Krum / coordinate-median / trimmed-mean)
        -> differentially-private release (RDP accountant, certified epsilon)
        -> global coverage/RSRP predictor  ->  worst-coverage cell selection
        -> Decision Safety Shield (EIRP legality, projection operator)
        -> hash-chained, tamper-evident evidence record

The task is a real O-RAN Coverage-and-Capacity-Optimization (CCO) rApp function:
learn a map from receiver position to received channel gain (coverage, dBW), then
pick the worst-covered cell for a power-fill action. Each receiver contributes
its real mean per-subband gain `y = mean(g) ∈ ℝ` (dBW, Wireless InSite ray
tracing) and real position. The model is a linear regressor over whitened
quadratic position features `[x, y, x², y², xy]`; whitening is a fixed,
data-published preprocessing so the least-squares problem is well conditioned.
Each client runs a stable FedProx proximal ridge step on its own receivers and
sends the update; the FedAvg fixed point is the global coverage predictor.

Why this is the *right* real task on this data. The received-gain field spans
~100 dB across the campus and is strongly spatially structured (path loss), so a
position→coverage model has real, large, learnable signal (closed-form
RMSE ≈ 11.8 dB vs a 20.1 dB predict-the-mean baseline). That is what makes the
security result unambiguous on real data:

* WITHOUT robust aggregation (plain FedAvg): a model-replacement poison from a
  few clients *diverges* the coverage predictor — RMSE explodes to millions of
  dB and the worst-coverage cell it picks is meaningless.
* WITH robust aggregation (Krum): the poison is bounded, RMSE stays near the
  clean optimum, and the power-fill targets a genuinely poor-coverage cell.

Everything downstream (Shield EIRP legality, guard chain, evidence chain, tamper
detection) runs on the actually-selected cell, so the whole topology is
exercised on real data, not a mock.

Pure numpy + the shipped ``horizon_ric`` modules. No torch, no GPU. The real
DeepMIMO feature file is licence-gated (not redistributed in-repo); build it with
``datasets/deepmimo_asu_3p5/build.py`` (deterministic, checksum-pinned) or point
``--features`` at a cached copy. The committed result JSON is the real 4096-Rx
run; the ``realdata`` CI workflow rebuilds the data and re-runs this loop.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from horizon_ric.evidence.schema import (
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
)
from horizon_ric.evidence.store import JsonlEvidenceStore
from horizon_ric.federated import robust
from horizon_ric.federated.dp import DPConfig, RDPAccountant, dp_fedavg
from horizon_ric.federated.poison_attacks import scaling_attack
from horizon_ric.policy.emit_guards import run_guard_chain
from horizon_ric.runtime_env import stamp
from horizon_ric.shield import default_terrestrial_shield

# --- Spectrum / EIRP geometry (shared with the DSA benchmark). -----------------
BAND_LO_HZ = 3.45e9
BAND_HI_HZ = 3.55e9
MAX_EIRP_DBM = 33.0
ANTENNA_GAIN_DBI = 6.0
FILL_FREQUENCY_HZ = 3.50e9  # a coverage-fill reservation at band centre.
FILL_BANDWIDTH_HZ = 20e6
SAFE_TX_POWER_DBM = 26.0  # legal request (EIRP 32 dBm < 33): emits without a fix.

# --- Federation defaults. ------------------------------------------------------
N_SITES = 10
N_MALICIOUS = 3  # below the Krum breakpoint f < (K-2)/2 and the median f < K/2.
ROUNDS = 40
PROX_LAMBDA = 2.0  # FedProx proximal strength (keeps local solves stable).
PROX_ETA = 0.6  # local step size on the proximal solution.
POISON_BOOST = 6.0
DP_CLIP = 1.0
DP_NOISE = 8.0  # certified eps ~ 3.2 at delta=1e-5 over ROUNDS rounds.
DP_DELTA = 1e-5


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"no feature rows in {path}")
    return rows


def _coverage_and_positions(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Real coverage target ``y`` (mean received gain, dBW) and 2D positions."""
    gains = np.asarray([r["subband_gain_dbw"] for r in rows], dtype=np.float64)
    pos = np.asarray([r["position_m"][:2] for r in rows], dtype=np.float64)
    if not np.all(np.isfinite(gains)) or not np.all(np.isfinite(pos)):
        raise ValueError("feature rows have non-finite gains or positions")
    return gains.mean(axis=1), pos


def _whiten_design(pos: np.ndarray) -> np.ndarray:
    """Whitened quadratic position design matrix ``[whitened(x,y,x²,y²,xy), 1]``.

    Centering + SVD whitening makes the global least-squares problem well
    conditioned (global feature covariance = I), so local proximal solves are
    stable across geographically concentrated, non-IID clients. It is a fixed,
    deterministic preprocessing computed from the published feature set.
    """
    x, y = pos[:, 0], pos[:, 1]
    feats = np.column_stack([x, y, x * x, y * y, x * y])
    feats = feats - feats.mean(axis=0)
    u, _s, _vt = np.linalg.svd(feats, full_matrices=False)
    whitened = u * np.sqrt(len(feats))
    return np.column_stack([whitened, np.ones(len(feats))])


def _partition_geographic(pos: np.ndarray, n_sites: int) -> list[np.ndarray]:
    """Partition receivers into ``n_sites`` clients by real 2D position (Lloyd).

    Ray tracing is site-specific, so clustering by position yields genuinely
    non-IID clients: each covers a spatial region with its own local slice of the
    global coverage function.
    """
    rng = np.random.default_rng(1234)
    centers = [pos[int(rng.integers(len(pos)))]]
    for _ in range(1, n_sites):
        d = np.min([np.linalg.norm(pos - c, axis=1) for c in centers], axis=0)
        centers.append(pos[int(np.argmax(d))])
    centers = np.asarray(centers)
    assign = np.zeros(len(pos), dtype=np.int64)
    for _ in range(50):
        assign = np.argmin(
            np.stack([np.linalg.norm(pos - c, axis=1) for c in centers], axis=1),
            axis=1,
        )
        new = np.asarray(
            [
                pos[assign == k].mean(axis=0) if np.any(assign == k) else centers[k]
                for k in range(n_sites)
            ]
        )
        if np.allclose(new, centers):
            centers = new
            break
        centers = new
    return [np.flatnonzero(assign == k) for k in range(n_sites)]


def _local_prox(
    w: np.ndarray, design: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """FedProx proximal ridge update for one client; returns the update vector.

    Solves ``argmin_v ½·E‖Av − b‖² + ½λ‖v − w‖²`` in closed form and returns a
    damped step ``η·(v* − w)``. The proximal term keeps the step bounded and
    stable regardless of the client's local conditioning — essential on
    geographically concentrated non-IID clients.
    """
    n = design.shape[0]
    hessian = design.T @ design / n + PROX_LAMBDA * np.eye(design.shape[1])
    grad = design.T @ (design @ w - target) / n
    return -PROX_ETA * np.linalg.solve(hessian, grad)


@dataclass
class FederationOutcome:
    method: str
    poisoned: bool
    rmse_db: float
    w_norm: float
    diverged: bool
    worst_cov_receiver_index: int
    worst_cov_position: list[float]
    epsilon: float | None


def _rmse_db(w: np.ndarray, design: np.ndarray, y: np.ndarray, ym: float, ys: float) -> float:
    pred = (design @ w) * ys + ym
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def _worst_cell(
    w: np.ndarray, design: np.ndarray, ym: float, ys: float,
    rows: list[dict[str, Any]], pos: np.ndarray,
) -> tuple[int, list[float]]:
    pred = (design @ w) * ys + ym
    j = int(np.argmin(pred))
    return int(rows[j]["receiver_index"]), [round(float(v), 1) for v in pos[j]]


def _run_federation(
    design: np.ndarray,
    y: np.ndarray,
    target_norm: np.ndarray,
    ym: float,
    ys: float,
    sites: list[np.ndarray],
    rows: list[dict[str, Any]],
    pos: np.ndarray,
    *,
    method: str,
    poisoned: bool,
    rounds: int,
    n_malicious: int,
    dp: bool,
) -> FederationOutcome:
    dim = design.shape[1]
    w = np.zeros(dim, dtype=np.float64)
    n_sites = len(sites)
    mal_ids = set(range(n_malicious)) if poisoned else set()
    rng = np.random.default_rng(11)
    accountant = RDPAccountant() if dp else None
    dp_cfg = DPConfig(clip_norm=DP_CLIP, noise_multiplier=DP_NOISE)

    for _ in range(rounds):
        benign = [
            _local_prox(w, design[sites[c]], target_norm[sites[c]])
            for c in range(n_sites)
            if c not in mal_ids
        ]
        if mal_ids:
            mal = scaling_attack(benign, len(mal_ids), boost=POISON_BOOST)
            updates = benign + list(mal)
        else:
            updates = benign
        if dp:
            delta = dp_fedavg(updates, dp_cfg, rng=rng, accountant=accountant)
        else:
            delta = robust.aggregate(
                updates, method=method, f=max(n_malicious, 1), beta=max(n_malicious, 1)
            )
        w = w + delta

    w_norm = float(np.linalg.norm(w))
    rmse = _rmse_db(w, design, y, ym, ys)
    idx, position = _worst_cell(w, design, ym, ys, rows, pos)
    eps = accountant.get_epsilon(DP_DELTA) if accountant is not None else None
    return FederationOutcome(
        method=("dp_fedavg" if dp else method),
        poisoned=poisoned,
        rmse_db=round(rmse, 4) if rmse < 1e6 else float(f"{rmse:.3e}"),
        w_norm=round(w_norm, 4) if w_norm < 1e6 else float(f"{w_norm:.3e}"),
        diverged=bool(w_norm > 1e2),
        worst_cov_receiver_index=idx,
        worst_cov_position=position,
        epsilon=(round(eps, 4) if eps is not None else None),
    )


def _shield_gate(
    store: JsonlEvidenceStore,
    *,
    decision_id: str,
    label: str,
    target_index: int,
    tx_power_dbm: float = SAFE_TX_POWER_DBM,
    frequency_hz: float = FILL_FREQUENCY_HZ,
) -> dict[str, Any]:
    """Emit a coverage-fill power policy for the selected cell through the Shield.

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
        "policy_type": "horizon.coverage.power_fill",
        "frequency_hz": frequency_hz,
        "bandwidth_hz": FILL_BANDWIDTH_HZ,
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
            rapp_instance_id="federated-coverage",
            state_hash=f"cell-{target_index}",
            chosen_action={
                "label": label,
                "policy_type": "horizon.coverage.power_fill",
                "target_receiver_index": target_index,
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
                risk_heads="fed-coverage-map",
                dyna="none",
                policy="cco",
                constraint_layer="terrestrial-shield",
                rapp="0.2.0",
            ),
        )
    )
    return {
        "label": label,
        "target_receiver_index": target_index,
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
    n_sites: int = N_SITES,
    n_malicious: int = N_MALICIOUS,
    rounds: int = ROUNDS,
) -> dict[str, Any]:
    rows = _load_rows(features)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "sampled_receivers" in manifest and len(rows) != manifest["sampled_receivers"]:
        raise ValueError("feature row count does not match manifest")

    y, pos = _coverage_and_positions(rows)
    design = _whiten_design(pos)
    ym, ys = float(y.mean()), float(y.std())
    target_norm = (y - ym) / ys
    sites = _partition_geographic(pos, n_sites)

    baseline_rmse = ys  # predicting the mean coverage everywhere.
    closed_form = np.linalg.lstsq(design, target_norm, rcond=None)[0]
    closed_rmse = _rmse_db(closed_form, design, y, ym, ys)

    specs = [
        ("clean_fedavg", "fedavg", False, False),
        ("poisoned_fedavg", "fedavg", True, False),
        ("poisoned_krum", "krum", True, False),
        ("poisoned_median", "median", True, False),
        ("poisoned_trimmed_mean", "trimmed_mean", True, False),
        ("dp_fedavg_clean", "fedavg", False, True),
    ]
    outcomes: dict[str, FederationOutcome] = {
        name: _run_federation(
            design, y, target_norm, ym, ys, sites, rows, pos,
            method=method, poisoned=poisoned, rounds=rounds,
            n_malicious=n_malicious, dp=dp,
        )
        for name, method, poisoned, dp in specs
    }

    # The poison misdirects the coverage-fill to a different cell than the clean
    # model would pick — quantify that on real positions.
    clean_cell = outcomes["clean_fedavg"].worst_cov_receiver_index
    naive_cell = outcomes["poisoned_fedavg"].worst_cov_receiver_index
    krum_cell = outcomes["poisoned_krum"].worst_cov_receiver_index

    # --- Trust chain: gate the selected cell + prove tamper-evidence. ---------
    ap = audit_path or Path("benchmarks/results/federated_coverage_audit.jsonl")
    if ap.exists():
        ap.unlink()
    ap.parent.mkdir(parents=True, exist_ok=True)
    store = JsonlEvidenceStore(ap)
    gate_records = [
        _shield_gate(
            store, decision_id="cov-robust-fill", label="robust-selected-cell",
            target_index=krum_cell,
        ),
        # A coverage-fill that naively requests EIRP 46 dBm the Shield must cap.
        _shield_gate(
            store, decision_id="cov-overpower-fill", label="over-eirp-fill",
            target_index=krum_cell, tx_power_dbm=40.0,
        ),
    ]
    overpower = gate_records[1]
    chain_len = len(store)
    verify_intact = store.verify()

    result: dict[str, Any] = {
        "benchmark": "Federated coverage-map learning on real DeepMIMO channels",
        "task": "O-RAN Coverage-and-Capacity-Optimization (CCO) rApp",
        "dataset": manifest.get("dataset", "DeepMIMO ASU Campus 3.5 GHz"),
        "scenario": manifest.get("scenario", "asu_campus_3p5"),
        "data_kind": manifest.get(
            "data_kind", "site-specific Wireless InSite ray tracing"
        ),
        "features_sha256": manifest.get("features_sha256"),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "receivers": len(rows),
        "coverage_dynamic_range_db": round(float(y.max() - y.min()), 4),
        "baseline_predict_mean_rmse_db": round(baseline_rmse, 4),
        "closed_form_rmse_db": round(closed_rmse, 4),
        "federation": {
            "n_sites": n_sites,
            "n_malicious": n_malicious,
            "rounds": rounds,
            "local_solver": "FedProx proximal ridge",
            "prox_lambda": PROX_LAMBDA,
            "prox_eta": PROX_ETA,
            "model": "linear over whitened quadratic position features",
            "poison": "model-replacement / scaling (Bagdasaryan 2020)",
            "poison_boost": POISON_BOOST,
            "dp_noise_multiplier": DP_NOISE,
            "dp_delta": DP_DELTA,
        },
        "outcomes": {k: asdict(v) for k, v in outcomes.items()},
        "coverage_fill_misdirection": {
            "clean_target_receiver": clean_cell,
            "robust_target_receiver": krum_cell,
            "naive_poisoned_target_receiver": naive_cell,
            "poison_moved_the_fill": bool(naive_cell != clean_cell),
        },
        "trust_chain": {
            "gate_records": gate_records,
            "overpower_fill_corrected": bool(overpower["corrected"]),
            "overpower_fill_safe_eirp_dbm": overpower["safe_eirp_dbm"],
            "overpower_fill_reached_ran": bool(overpower["emitted_clean"]),
            "evidence_chain_length": chain_len,
            "verify_first_broken_index": verify_intact,
        },
        "scope_note": (
            "Real DeepMIMO ray tracing drives real federated learning (FedProx), "
            "real Byzantine-robust aggregation, a real RDP privacy accountant, "
            "real Shield legality and a real tamper-evident evidence chain. "
            "This is site-specific ray tracing, not OTA capture; one scenario."
        ),
    }

    # --- Tamper-evidence proof on the freshly written chain. ------------------
    lines = ap.read_text(encoding="utf-8").splitlines()
    if len(lines) >= 2:
        lines[1] = lines[1].replace(
            '"target_receiver_index": ', '"target_receiver_index": 999999, "_t": ', 1
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
        default=Path("datasets/deepmimo_asu_3p5/generated/channel_features.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/deepmimo_asu_3p5/manifest.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("benchmarks/results/federated_coverage.json")
    )
    parser.add_argument("--sites", type=int, default=N_SITES)
    parser.add_argument("--malicious", type=int, default=N_MALICIOUS)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()

    result = run(
        args.features, args.manifest,
        n_sites=args.sites, n_malicious=args.malicious, rounds=args.rounds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stamp(result)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

    o = result["outcomes"]
    tc = result["trust_chain"]
    base = result["baseline_predict_mean_rmse_db"]
    # Clean FL must learn (beat the predict-mean baseline); the poison must break
    # plain FedAvg (divergence); Krum must bound it below baseline and below the
    # poisoned FedAvg; the trust chain must hold and catch the tamper.
    learns = o["clean_fedavg"]["rmse_db"] < base
    poison_breaks = o["poisoned_fedavg"]["diverged"]
    krum_defends = (
        not o["poisoned_krum"]["diverged"]
        and o["poisoned_krum"]["rmse_db"] < base
        and o["poisoned_krum"]["rmse_db"] < o["poisoned_fedavg"]["rmse_db"]
    )
    chain_ok = (
        tc["verify_first_broken_index"] == -1
        and not tc["overpower_fill_reached_ran"]
        and tc.get("verify_after_tamper_index") not in (None, -1)
    )
    return 0 if (learns and poison_breaks and krum_defends and chain_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
